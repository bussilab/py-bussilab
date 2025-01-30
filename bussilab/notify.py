"""
Module implementing Slack notifications.

This module sends notification through an App installed in the Slack workspace.
Some settings are needed first for authentication. It is recommended to add a
file named `.bussilabrc` to your home directory with the following content:
```bash
notify:
  token: xoxb-00000
  channel: U00000
```
The `token` here should be provided by the administrator of your workspace.
The channel should be the Slack ID associated to your user. It can be found
looking in your Slack profile. With these settings, the tool will send
notifications to you by default.

Notifications can then be sent using either the command line:
```bash
bussilab notify --message "text here"
```
or from python:
```python
from bussilab.notify import notify
notify("text here")
```

Notice that the message is optional. Even with an empty message, the footer
will allow you to reconstruct from which machine and directory the message was
sent from. This might be sufficient for your goal.

You can also indicate a specific channel for the notification using the
`channel` option:
```bash
bussilab notify --message "text here" --channel "project-myproject"
```
or from python:
```python
from bussilab.notify import notify
notify("text here", channel="project-myproject")
```
This will only work if the App has been added to the specified channel.

The following syntax can be used to upload a file:
```bash
bussilab notify --message "text here" --file /path/to/file
```
or from python:
```python
from bussilab.notify import notify
notify("text here",file="/path/to/file")
```

The commands above will return the URL of the message. This URL can be used
later to update or delete them or to post reactions:
```bash
url=$(bussilab notify --message "text here")
bussilab notify --update $url --message "revised message"
bussilab notify --react $url:heart

# this will remove only the reaction:
bussilab notify --delete $url:heart

# this will remove the entire message:
bussilab notify --delete $url

url=$(bussilab notify --message "text here")
```
or from python:
```python
from bussilab.notify import notify
url=notify("text here")
notify("revised message", update=url)
notify(react=url+":heart")
notify(delete=url+":heart")
notify(delete=url)
```
In these cases, the channel is not needed and should not be provided.
Notice that you will only be able to update or delete messages sent through the
App.
"""

import datetime
import re
import os
import socket
import time
import warnings
import random

try:
    # slack client 3
    from slack_sdk import WebClient
    from slack_sdk.web.base_client import SlackResponse
    from slack_sdk.errors import SlackApiError
except ModuleNotFoundError:
    # slack client 2
    from slack import WebClient
    from slack.web.base_client import SlackResponse
    from slack.errors import SlackApiError

from . import coretools

from typing import cast

def _try_multiple_times(func,*args,**kwargs):
     max_attempts=10
     num_attempts=0
     num_attempts_delay=3
     jittering=0.2
     while True:
        try:
            num_attempts+=1
            return func(*args,**kwargs)
        except SlackApiError as e:
            if num_attempts>=max_attempts:
                raise

            if "error" in e.response and e.response["error"]=="ratelimited" and "Retry-After" in e.response.headers:
                wait=float(e.response.headers["Retry-After"])
                if num_attempts>num_attempts_delay:
                  wait*=2**(num_attempts-num_attempts_delay)
                wait*=random.uniform(1,1+jittering)
                warnings.warn("Slack API, retry-after "
                              +str(wait)
                              +" seconds"+
                              " ["+str(num_attempts)+"/"+str(max_attempts)+"]",
                              UserWarning)
                time.sleep(wait)
            elif not hasattr(e.response,"status_code") or e.response.status_code!=200:
                wait=30
                if num_attempts>num_attempts_delay:
                  wait*=2**(num_attempts-num_attempts_delay)
                wait*=random.uniform(1,1+jittering)
                warnings.warn("Slack API, server-side problem: "
                              +str(e.response)+"\n"+
                              "retrying after "
                              +str(wait)
                              +" seconds"+
                              " ["+str(num_attempts)+"/"+str(max_attempts)+"]",
                              UserWarning)
                time.sleep(wait)
            else:
                raise

_match_message=r"https://[^/]*\.slack\.com/archives/.*"

def _parse_url(url: str):
    if re.match(r"^https://[^/]*\.slack\.com/archives/.*:.*",url):
        organization = re.sub(r"^https://","", re.sub(r"\.slack\.com/archives/.*","",url))
        url=re.sub(r"^https://[^/]*\.slack\.com/archives/","",url)
        url1=re.sub(r":.*","",url)
        url1=url1[:-6]+"."+url1[-6:]
        channel=re.sub("/p.*","",url1)
        ts=re.sub(":.*$","",re.sub("^.*/p","",url1))
        react=re.sub(r".*:","",url)
        return { "type":"reaction", "ts":ts, "channel":channel, "organization":organization, "reaction": react}
    if re.match(r"^https://[^/]*\.slack\.com/archives/.*",url):
        organization = re.sub(r"^https://","", re.sub(r"\.slack\.com/archives/.*","",url))
        url=re.sub(r"^https://[^/]*\.slack\.com/archives/","",url)
        url=re.sub(r"\?.*","",url)
        url=url[:-6]+"."+url[-6:]
        channel=re.sub("/p.*","",url)
        ts=re.sub(".*/p","",url)
        return { "type":"message", "ts":ts, "channel":channel, "organization":organization }
    if re.match(r"^https://[^/]*\.slack\.com/files/.*",url):
        organization = re.sub(r"^https://","", re.sub(r"\.slack\.com/files/.*","",url))
        url=re.sub(r"^https://[^/]*\.slack\.com/files/","",url)
        user=re.sub("/.*","",url)
        id=re.sub("^"+user+"/","",url)
        id=re.sub("/.*","",id)
        return { "type":"file", "id":id, "user":user, "organization":organization }
    return {}

def notify(message: str = "",
           channel: str = None,
           *,
           react: str = None,
           update: str = None,
           delete: str = None,
           reply: str = None,
           reply_broadcast: str = None,
           title: str = "",
           screenlog: str = "",
           screenlog_maxlines: int = 0,
           footer: bool = True,
           type: str = "mrkdwn",
           file: str = "",
           token: str = None):
    """Tool to send notifications to Slack.

       Parameters
       ----------

       message: str

           A string that will form the body of the message.

       channel: None or str

           The channel. By default, taken from your `~/.bussilabrc`
           configuration file.

       update: None or str

           The URL of a message to be updated.

       delete: None or str

           The URL of a message to be deleted. By passing a URL
           concatenated with the string `":name_of_reaction"` you can
           delete a reaction. Buy passing two comma-separated URLs
           you can delete both a file and the message with which it was
           shared.
       
       reply: None or str
       
           The URL of a message to be replied
       
       reply_broadcast: None or str
       
           The URL of a message to be broadcast-replied
       
       react: None or str
       
           The URL of a message to which you want to add a reaction,
           followed by the string `:name_of_the_reaction`
           
       file: None or str
       
           The path of a file to be uploaded

       title: str

           The title of the notification.

       footer: bool

           If True, a footer is added with current user, machine, and
           directory.

       type: str

           The type of message. Can be "mrkdwn" or "plain_text".

       token: None or str

           The token. By default, taken from your `~/.bussilabrc`
           configuration file.

       Returns
       -------

           str
               A string with the URL of the sent message.
               In case the `delete` keyword is used, it returns an empty
               string.
               In case a file is uploaded, it returns two comma-separated
               URLs corresponding to the message and to the file.


       Example
       -------

       ```python
       from bussilab.notify import notify
       notify("send this message")
       ```
       See `bussilab.notify` for more examples.
    """

    if [bool(channel),
        bool(update),
        bool(react),
        bool(delete),
        bool(reply),
        bool(reply_broadcast)
       ].count(True)>1:
        raise TypeError("channel/update/delete/reply/reply_broadcast are mutually incompatible")

    if len(file)>0 and (update or react or delete or reply_broadcast):
        raise TypeError("files cannot be updated")

    config = None
    if token is None:
        config = coretools.config()
        token=config["notify"]["token"]

    client = WebClient(token=token)

    if delete:
        # this is to enable deletion of both a message and a file:
        delete_multi=delete.split(",")
        if len(delete_multi)>1:
            for d in delete_multi:
                notify(message,channel,delete=d,token=token)
            return ""
        delete_dict=_parse_url(delete)
        if not delete_dict:
            raise TypeError("cannot parse delete URL")
        if delete_dict["type"]=="message":
            _try_multiple_times(client.chat_delete,
                                channel=delete_dict["channel"],
                                ts=delete_dict["ts"])
        elif delete_dict["type"]=="file":
            _try_multiple_times(client.files_delete,
                                file=delete_dict["id"])
        elif delete_dict["type"]=="reaction":
            _try_multiple_times(client.reactions_remove,
                                channel=delete_dict["channel"],
                                timestamp=delete_dict["ts"],
                                name=delete_dict["reaction"])
        else:
            raise RuntimeError("unknown type")
        # delete always returns an empty string
        return ""

    if react:
        react_dict=_parse_url(react)
        response = _try_multiple_times(client.reactions_add,
          name=react_dict["reaction"],
          timestamp=react_dict["ts"],
          channel=react_dict["channel"])
        return react

    screenlog_message=""
    if len(screenlog)>0:
        # we manually removed "deleted" lines.
        # this is very useful for tdqm-like logs
        with open(screenlog,'rb') as handler:
            screenlog_message=handler.read().decode()
       	    screenlog_message=re.sub(r'.*\r([^\n])', r'\1', screenlog_message, flags=re.M)
        if screenlog_maxlines>0:
            screenlog_message_lines=screenlog_message.split("\n")
            if len(screenlog_message_lines) > screenlog_maxlines:
                screenlog_message_lines = screenlog_message_lines[-screenlog_maxlines:]
            screenlog_message="\n".join(screenlog_message_lines)

    if len(screenlog_message)>2900:
        screenlog_message=screenlog_message[:2900] + " [truncated]"
        
    if len(message)>2900:
        message=message[:2900] + " [truncated]"

    if len(title)>2900:
        title=title[:2900] + " [truncated]"

    if update:
        update_dict=_parse_url(update)
        if not update_dict:
           raise TypeError("")
        organization=update_dict["organization"]
    elif reply:
        reply_dict=_parse_url(reply)
        organization=reply_dict["organization"]
    elif reply_broadcast:
        reply_dict=_parse_url(reply_broadcast)
        organization=reply_dict["organization"]
    else:
        if channel is None:
            if config is None:
                config = coretools.config()
            channel=config["notify"]["channel"]
        if re.match(r"^https://[^/]*\.slack\.com/archives/.*", channel):
            organization = re.sub("^https://","", re.sub(r"\.slack\.com/archives/.*","",channel))
            channel=re.sub(r"^https://[^/]*\.slack\.com/archives/","",channel)
        else:
            # this is needed to set organization correctly (so as to build the
            # proper link) when passing the name of a channel
            organization = ""

    blocks=[]
    text=""

    if len(title) > 0:
        text+="*" + title+"*\n"
        blocks.append(
           {
               "type": "section",
               "text": {"type": "mrkdwn", "text": "*" + title + "*"},
           }
           )

    if len(message) > 0:
        text+=message+"\n"
        blocks.append(
           {
               "type": "section",
               "text": {
                         "type": type,
                         "text": message
                       },
           }
           )
        
    if len(screenlog_message) > 0:
        text+=screenlog_message+"\n"
        blocks.append(
           {
               "type": "section",
               "text": {
                         "type": "mrkdwn",
                         "text": "```\n" + screenlog_message + "\n```\n"
                       },
           }
           )

    if footer:
        footer_text = ""
        if update:
            footer_text += "Updated"
        else:
            footer_text += "Sent"
        footer_text += " by "+ os.environ['USER']
        footer_text += " at " + socket.gethostname() +'\n'
        footer_text += "pwd: " + os.getcwd() + '\n'
        footer_text += datetime.datetime.now().isoformat(sep=' ',timespec='milliseconds')
        text+=footer_text+"\n"
        blocks.append({
                          "type": "context",
                          "elements": [
                              {  # type: ignore
                                "type": "mrkdwn",
                                "text": footer_text
                              }
                          ]
                      })
    if len(blocks)==0:
        text+="(empty notification)"
        blocks.append({
                          "type": "section",
                          "text": {
                                     "type": type,
                                     "text": "(empty notification)"
                                  }
                      })

    if update:
        response = _try_multiple_times(client.chat_update,
                   channel=update_dict["channel"],
                   text=text,
                   blocks=blocks,
                   ts=update_dict["ts"])
    elif len(file)>0:
        initial_comment = ""
        if len(title)>0:
            initial_comment += "*" + title + "*\n"
        if len(message)>0:
            initial_comment += message +"\n"
        if footer:
            initial_comment += footer_text

        # v2 will be the only supported way in Feb 2025
        # https://api.slack.com/changelog/2024-04-a-better-way-to-upload-files-is-here-to-stay

        try:
            _=client.files_upload_v2
            v2=True
        except AttributeError:
            v2=False

        if v2:
            if reply:
                response = _try_multiple_times(client.files_upload_v2,
                                               file=file,
                                               channel=reply_dict["channel"],
                                               title=file,
                                               thread_ts=reply_dict["ts"])
            else:
                response = _try_multiple_times(client.files_upload_v2,
                                               file=file,
                                               title=file,
                                               channel=channel,
                                               initial_comment=initial_comment)

            if len(list(response["files"][0]["shares"].keys()))>0:
                k=list(response["files"][0]["shares"].keys())[0] # empirically, pick the first one. There should be only one!
                channel=list(response["files"][0]["shares"][k].keys())[0] # empirically, pick the first one. There should be only one!
                ts=response["files"][0]["shares"][k][channel][0]["ts"]
            else:
                file_id=response["files"][0]["id"]
                max_attempts=10
                num_attempts=0
                num_attempts_delay=3
                jittering=0.2
                while True:
                    num_attempts+=1
                    response = _try_multiple_times(client.files_info, file=file_id)
                    if len(list(response["file"]["shares"].keys()))>0:
                      k=list(response["file"]["shares"].keys())[0] # empirically, pick the first one. There should be only one!
                      channel=list(response["file"]["shares"][k].keys())[0] # empirically, pick the first one. There should be only one!
                      ts=response["file"]["shares"][k][channel][0]["ts"]
                      break
                    if num_attempts>=max_attempts:
                      raise RuntimeError("Cannot obtain shares info for file ID "+str(file_id))
                    wait=2.0
                    if num_attempts>num_attempts_delay:
                      wait*=2**(num_attempts-num_attempts_delay)
                    wait*=random.uniform(1,1+jittering)
                    warnings.warn("Slack API, missing shares for file ID " + file_id  +", retry after "
                                  +str(wait)
                                  +" seconds"+
                                  " ["+str(num_attempts)+"/"+str(max_attempts)+"]",
                                  UserWarning)
                    time.sleep(wait)
        else:
            if reply:
                response = _try_multiple_times(client.files_upload,
                                               file=file,
                                               channels=reply_dict["channel"],
                                               title=file,
                                               thread_ts=reply_dict["ts"])
            else:
                response = _try_multiple_times(client.files_upload,
                                               file=file,
                                               title=file,
                                               channels=channel,
                                               initial_comment=initial_comment)
            k=list(response["file"]["shares"].keys())[0] # empirically, pick the first one. There should be only one!
            channel=list(response["file"]["shares"][k].keys())[0] # empirically, pick the first one. There should be only one!
            ts=response["file"]["shares"][k][channel][0]["ts"]

    elif reply:
        response = _try_multiple_times(client.chat_postMessage,
                   blocks=blocks,
                   text=text,
                   channel=reply_dict["channel"],
                   thread_ts=reply_dict["ts"])
    elif reply_broadcast:
        response = _try_multiple_times(client.chat_postMessage,
                   blocks=blocks,
                   text=text,
                   channel=reply_dict["channel"],
                   thread_ts=reply_dict["ts"],
                   reply_broadcast=True)
    else:
        response = _try_multiple_times(client.chat_postMessage,
                   blocks=blocks,
                   text=text,
                   channel=channel)

    response = cast(SlackResponse, response)

    if len(organization)==0:
        base_url=_try_multiple_times(client.auth_test)["url"]
    else:
        base_url="https://" + organization + ".slack.com/"

    if len(file)==0:
        url=base_url + "archives/" + response["channel"] + "/p" + response["ts"][:-7] + response["ts"][-6:]
    else:
        url=base_url + "archives/" + channel + "/p" + ts[:-7] + ts[-6:]
        url+="," + base_url + "files/" + response["file"]["user"] + "/" + response["file"]["id"]

    return url
