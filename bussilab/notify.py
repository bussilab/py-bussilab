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

Alternatively, you can upload a file:
```python
from bussilab.notify import notify
notify("text here",file="/path/to/file")
```

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

The commands above will return the URL of the message. This URL can be used
later to update or delete them:
```bash
url=$(bussilab notify --message "text here")
bussilab notify --update $url --message "revised message"
bussilab notify --delete $url
```
or
```python
from bussilab.notify import notify
url=notify("text here")
notify("text here", update=url)
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

def _parse_url(url: str):
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

           The URL of a message to be deleted.
       
       reply: None or str
       
           The URL of a message to be replied
       
       reply_broadcast: None or str
       
           The URL of a message to be broadcast-replied
           
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
        raise TypeError("")

    config = None
    if token is None:
        config = coretools.config()
        token=config["notify"]["token"]

    client = WebClient(token=token)

    if delete:
        delete_multi=delete.split(",")
        if len(delete_multi)>1:
            for d in delete_multi:
                notify(message,channel,delete=d,token=token)
            return ""
        delete_dict=_parse_url(delete)
        if not delete_dict:
           raise TypeError("")
        if delete_dict["type"]=="message":
            client.chat_delete(channel=delete_dict["channel"], ts=delete_dict["ts"])
            return ""
        elif delete_dict["type"]=="file":
            client.files_delete(file=delete_dict["id"])
            return ""
        raise RuntimeError("unknown type")
    
    if react:
        react_dict=_parse_url(react.split(",")[0])

        response = client.reactions_add(
          name=react.split(",")[1],
          timestamp=react_dict["ts"],
          channel=react_dict["channel"])
        return react.split(",")[0]


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
        blocks.append(
           {
               "type": "section",
               "text": {"type": "mrkdwn", "text": "*" + title + "*"},
           }
           )

    if len(message) > 0:
        blocks.append(
           {
               "type": "section",
               "text": {
                         "type": type,
                         "text": message
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
        blocks.append({
                          "type": "section",
                          "text": {
                                     "type": type,
                                     "text": "(empty notification)"
                                  }
                      })

    if len(title) > 0:
        text=title+"\n"
    elif len(message) > 0:
        text=message+"\n"
    else:
        text="(empty message)"

    if update:
        response = client.chat_update(
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

        attempts=5
        

        v2=False
# # Note:
# # UserWarning: client.files_upload() may cause some issues like timeouts for
# # relatively large files. Our latest recommendation is to use client.files_upload_v2(),
# # which is mostly compatible and much stabler, instead.
# # However, to me v2 does not work (the file is not accessible and not posted to the channel)
#         try:
#             x=client.files_upload_v2
#             v2=True
#         except AttributeError:
#             pass


        for i in range(attempts):
            try:
                if v2:
                    response = client.files_upload_v2(file=file,title=file,channel=channel)
                else:
                    if reply:
                        response = client.files_upload(file=file,channels=reply_dict["channel"],
                                                       title=file,thread_ts=reply_dict["ts"])
                    else:
                        response = client.files_upload(file=file,title=file,channels=channel,
                                                      initial_comment=initial_comment)
                break
            except SlackApiError:
                if(i+1==attempts):
                    raise
                print("retrying ...",i+1)
    elif reply:
        response = client.chat_postMessage(
                   blocks=blocks,
                   text=text,
                   channel=reply_dict["channel"],
                   thread_ts=reply_dict["ts"])
    elif reply_broadcast:
        response = client.chat_postMessage(
                   blocks=blocks,
                   text=text,
                   channel=reply_dict["channel"],
                   thread_ts=reply_dict["ts"],
                   reply_broadcast=True)
    else:
        response = client.chat_postMessage(
                   blocks=blocks,
                   text=text,
                   channel=channel)

    response = cast(SlackResponse, response)

    if len(organization)==0:
        base_url=client.auth_test()["url"]
    else:
        base_url="https://" + organization + ".slack.com/"

    if len(file)==0:
        url=base_url + "archives/" + response["channel"] + "/p" + response["ts"][:-7] + response["ts"][-6:]
    else:
        
        k=list(response["file"]["shares"].keys())[0] # empirically, pick the first one. There should be only one!
        channel=list(response["file"]["shares"][k].keys())[0] # empirically, pick the first one. There should be only one!
        ts=response["file"]["shares"][k][channel][0]["ts"]
        url=base_url + "archives/" + channel + "/p" + ts[:-7] + ts[-6:]
        url+="," + base_url + "files/" + response["file"]["user"] + "/" + response["file"]["id"]

    return url
