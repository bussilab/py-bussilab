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

Link and media previews can be disabled using the `unfurl` option:
```bash
bussilab notify --message "https://example.com" --no-unfurl
```
or from python:
```python
notify("https://example.com", unfurl=False)
```

A file written in standard Markdown can be sent without adding a title or
footer:
```bash
bussilab notify --markdown-file report.md
```
or from python:
```python
notify(markdown_file="report.md")
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
from urllib.error import URLError
from urllib.parse import urlsplit

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

_MARKDOWN_BLOCK_LIMIT = 12000
_TRUNCATION_MARKER = " [truncated]"
_CODE_BLOCK_PREFIX = "```\n"
_CODE_BLOCK_SUFFIX = "\n```\n"


def _try_multiple_times(func,*args,**kwargs):
    max_attempts=5
    max_wait=30.0
    max_total_wait=300.0
    jittering=0.2
    transient_attempts=0
    rate_limit_attempts=0
    total_wait=0.0
    while True:
        try:
            return func(*args,**kwargs)
        except SlackApiError as e:
            response = e.response
            error = response.get("error") if hasattr(response, "get") else None
            status_code = getattr(response, "status_code", None)
            headers = getattr(response, "headers", {}) or {}
            retry_after = next(
                (value for key, value in headers.items()
                 if str(key).lower() == "retry-after"),
                None
            )

            if error == "ratelimited" or status_code == 429:
                rate_limit_attempts += 1
                if retry_after is None:
                    wait = min(2.0 ** rate_limit_attempts, max_wait)
                else:
                    try:
                        wait = float(retry_after)
                    except (TypeError, ValueError):
                        raise e from None
                    if wait < 0:
                        raise
                    # Avoid a busy retry loop if Slack returns zero.
                    wait = max(wait, 1.0)
                problem = "rate limited"
            elif (isinstance(status_code, int) and
                  500 <= status_code < 600):
                transient_attempts += 1
                if transient_attempts >= max_attempts:
                    raise
                wait = min(2.0 ** transient_attempts, max_wait)
                problem = "server-side problem"
            else:
                raise

            wait *= random.uniform(1,1+jittering)
            if total_wait + wait > max_total_wait:
                raise
            if problem == "rate limited":
                retry_progress = " [rate-limit retry " + \
                                 str(rate_limit_attempts) + "]"
            else:
                retry_progress = " [" + str(transient_attempts) + "/" + \
                                 str(max_attempts) + "]"
            warnings.warn("Slack API, " + problem + "; retrying after "
                          +str(wait)
                          +" seconds"+
                          retry_progress,
                          UserWarning)
            time.sleep(wait)
            total_wait += wait
        except (URLError, TimeoutError, ConnectionError) as e:
            transient_attempts += 1
            if transient_attempts >= max_attempts:
                raise
            wait = min(2.0 ** transient_attempts, max_wait)
            wait *= random.uniform(1,1+jittering)
            if total_wait + wait > max_total_wait:
                raise
            warnings.warn("Slack API, transport problem: " + str(e) + "; "
                          "retrying after " + str(wait) + " seconds"+
                          " ["+str(transient_attempts)+"/"+
                          str(max_attempts)+"]",
                          UserWarning)
            time.sleep(wait)
            total_wait += wait

def _parse_url(url: str):
    if not isinstance(url, str):
        return {}

    reaction = None
    reaction_match = re.match(r"^(.*):([A-Za-z0-9_+-]+)$", url)
    if reaction_match:
        url = reaction_match.group(1)
        reaction = reaction_match.group(2)

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return {}

    hostname = parsed.hostname
    suffix = ".slack.com"
    if (parsed.scheme != "https" or hostname is None or port is not None or
            parsed.username is not None or parsed.password is not None or
            not hostname.endswith(suffix)):
        return {}

    organization = hostname[:-len(suffix)]
    organization_pattern = r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?"
    if not re.fullmatch(organization_pattern, organization):
        return {}

    path = parsed.path.split("/")
    if len(path) == 4 and path[0] == "" and path[1] == "archives":
        channel = path[2]
        timestamp_match = re.fullmatch(r"p(\d{7,})", path[3])
        if not re.fullmatch(r"[A-Z0-9]+", channel) or not timestamp_match:
            return {}
        compact_timestamp = timestamp_match.group(1)
        result = {
            "type": "message",
            "ts": compact_timestamp[:-6] + "." + compact_timestamp[-6:],
            "channel": channel,
            "organization": organization
        }
        if reaction is not None:
            result["type"] = "reaction"
            result["reaction"] = reaction
        return result

    if (reaction is None and len(path) >= 4 and path[0] == "" and
            path[1] == "files" and
            re.fullmatch(r"[A-Z0-9]+", path[2]) and
            re.fullmatch(r"[A-Z0-9]+", path[3])):
        return {
            "type": "file",
            "id": path[3],
            "user": path[2],
            "organization": organization
        }
    return {}


def _require_url(url: str, operation: str, *allowed_types: str):
    parsed = _parse_url(url)
    if parsed.get("type") not in allowed_types:
        raise TypeError("cannot parse " + operation + " URL")
    return parsed

def notify(message: str = "",
           channel: str = None,
           *,
           markdown_file: str = "",
           react: str = None,
           update: str = None,
           delete: str = None,
           reply: str = None,
           reply_broadcast: str = None,
           title: str = "",
           screenlog: str = "",
           screenlog_maxlines: int = 0,
           footer: bool = True,
           unfurl: bool = True,
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

       markdown_file: str

           Read a standard Markdown message from this UTF-8 text file. This
           implies `type="markdown"` and cannot be combined with a non-empty
           `message` argument.

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

       screenlog: str

           The path of a GNU Screen log file. Its contents are displayed in
           a fenced native Markdown block.

       screenlog_maxlines: int

           If positive, include only this many lines from the end of the
           Screen log.

       footer: bool

           If True, a footer is added with current user, machine, and
           directory.

       unfurl: bool

           If False, link and media previews are disabled when posting a
           message or a reply. The option does not affect file-upload
           comments or message updates.

       type: str

           The type of message. Can be "mrkdwn", "plain_text", or
           "markdown". Standard Markdown uses a native Markdown block;
           `title`, `screenlog`, file uploads, and footers are not supported.

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

    if type not in ("mrkdwn", "plain_text", "markdown"):
        raise TypeError("type should be 'mrkdwn', 'plain_text', or 'markdown'")

    if message and markdown_file:
        raise TypeError("message and markdown_file are mutually incompatible")

    if markdown_file:
        if type == "plain_text":
            raise TypeError("markdown_file is incompatible with plain_text")
        type="markdown"

    if type == "markdown":
        if title:
            raise TypeError("title is not supported with standard Markdown")
        if screenlog:
            raise TypeError("screenlog is not supported with standard Markdown")
        if file:
            raise TypeError("file uploads are not supported with standard Markdown")
        footer=False

    if [bool(channel),
        bool(update),
        bool(react),
        bool(delete),
        bool(reply),
        bool(reply_broadcast)
       ].count(True)>1:
        raise TypeError("channel/update/react/delete/reply/reply_broadcast are mutually incompatible")

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
        delete_dict=_require_url(delete, "delete", "message", "file", "reaction")
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
        react_dict=_require_url(react, "reaction", "reaction")
        response = _try_multiple_times(client.reactions_add,
          name=react_dict["reaction"],
          timestamp=react_dict["ts"],
          channel=react_dict["channel"])
        return react

    if markdown_file:
        with open(markdown_file, encoding="utf-8") as handler:
            message=handler.read()

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

    screenlog_limit = (_MARKDOWN_BLOCK_LIMIT - len(_CODE_BLOCK_PREFIX)
                       - len(_CODE_BLOCK_SUFFIX))
    if len(screenlog_message)>screenlog_limit:
        screenlog_message = (screenlog_message[
            :screenlog_limit-len(_TRUNCATION_MARKER)
        ] + _TRUNCATION_MARKER)
        
    if type == "markdown":
        if len(message)>_MARKDOWN_BLOCK_LIMIT:
            message = message[
                :_MARKDOWN_BLOCK_LIMIT-len(_TRUNCATION_MARKER)
            ] + _TRUNCATION_MARKER
    elif len(message)>2900:
        message=message[:2900] + " [truncated]"

    if len(title)>2900:
        title=title[:2900] + " [truncated]"

    if update:
        update_dict=_require_url(update, "update", "message")
        organization=update_dict["organization"]
    elif reply:
        reply_dict=_require_url(reply, "reply", "message")
        organization=reply_dict["organization"]
    elif reply_broadcast:
        reply_dict=_require_url(reply_broadcast, "reply_broadcast", "message")
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
        if type == "markdown":
            blocks.append({"type": "markdown", "text": message})
        else:
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
        blocks.append({
            "type": "markdown",
            "text": (_CODE_BLOCK_PREFIX + screenlog_message
                     + _CODE_BLOCK_SUFFIX)
        })

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
        if type == "markdown":
            blocks.append({"type": "markdown", "text": "(empty notification)"})
        else:
            blocks.append({
                              "type": "section",
                              "text": {
                                         "type": type,
                                         "text": "(empty notification)"
                                      }
                          })

    unfurl_options = {}
    if not unfurl:
        unfurl_options["unfurl_links"] = False
        unfurl_options["unfurl_media"] = False

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
                                               thread_ts=reply_dict["ts"],
                                               initial_comment=initial_comment)
            else:
                response = _try_multiple_times(client.files_upload_v2,
                                               file=file,
                                               title=file,
                                               channel=channel,
                                               initial_comment=initial_comment)

            uploaded_file=response["files"][0]
            if len(list(uploaded_file["shares"].keys()))>0:
                k=list(uploaded_file["shares"].keys())[0] # empirically, pick the first one. There should be only one!
                channel=list(uploaded_file["shares"][k].keys())[0] # empirically, pick the first one. There should be only one!
                ts=uploaded_file["shares"][k][channel][0]["ts"]
            else:
                file_id=uploaded_file["id"]
                max_attempts=10
                num_attempts=0
                num_attempts_delay=3
                jittering=0.2
                time.sleep(2.0) # wait before first attempt
                while True:
                    num_attempts+=1
                    response = _try_multiple_times(client.files_info, file=file_id)
                    if len(list(response["file"]["shares"].keys()))>0:
                      uploaded_file=response["file"]
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
                                               thread_ts=reply_dict["ts"],
                                               initial_comment=initial_comment)
            else:
                response = _try_multiple_times(client.files_upload,
                                               file=file,
                                               title=file,
                                               channels=channel,
                                               initial_comment=initial_comment)
            uploaded_file=response["file"]
            k=list(response["file"]["shares"].keys())[0] # empirically, pick the first one. There should be only one!
            channel=list(response["file"]["shares"][k].keys())[0] # empirically, pick the first one. There should be only one!
            ts=response["file"]["shares"][k][channel][0]["ts"]

    elif reply:
        response = _try_multiple_times(client.chat_postMessage,
                   blocks=blocks,
                   text=text,
                   channel=reply_dict["channel"],
                   thread_ts=reply_dict["ts"],
                   **unfurl_options)
    elif reply_broadcast:
        response = _try_multiple_times(client.chat_postMessage,
                   blocks=blocks,
                   text=text,
                   channel=reply_dict["channel"],
                   thread_ts=reply_dict["ts"],
                   reply_broadcast=True,
                   **unfurl_options)
    else:
        response = _try_multiple_times(client.chat_postMessage,
                   blocks=blocks,
                   text=text,
                   channel=channel,
                   **unfurl_options)

    response = cast(SlackResponse, response)

    if len(organization)==0:
        base_url=_try_multiple_times(client.auth_test)["url"]
    else:
        base_url="https://" + organization + ".slack.com/"

    if len(file)==0:
        url=base_url + "archives/" + response["channel"] + "/p" + response["ts"][:-7] + response["ts"][-6:]
    else:
        url=base_url + "archives/" + channel + "/p" + ts[:-7] + ts[-6:]
        url+="," + base_url + "files/" + uploaded_file["user"] + "/" + uploaded_file["id"]

    return url
