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

from slack import WebClient
from slack.web.base_client import SlackResponse
from . import coretools

from typing import cast

_organization="bussilab" # this is hardcoded for now

def notify(message: str = "",
           channel: str = None,
           *,
           update: str = None,
           delete: str = None,
           title: str = "",
           footer: bool = True,
           type: str = "mrkdwn",
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

    if channel and update:
        raise TypeError("")

    if channel and delete:
        raise TypeError("")

    if update and delete:
        raise TypeError("")

    config = None
    if token is None:
        config = coretools.config()
        token=config["notify"]["token"]

    if update:
        if not re.match(r"^https://.*\.slack\.com/archives/.*",update):
            raise TypeError("")
        organization = re.sub(r"^https://","", re.sub(r"\.slack\.com/archives/.*","",update))
        update=re.sub(r"^https://.*\.slack\.com/archives/","",update)
        update=update[:-6]+"."+update[-6:]
        channel=re.sub("/p.*","",update)
        ts=re.sub(".*/p","",update)
    elif delete:
        if not re.match(r"^https://.*\.slack\.com/archives/.*",delete):
            raise TypeError("")
        delete=re.sub(r"^https://.*\.slack\.com/archives/","",delete)
        delete=delete[:-6]+"."+delete[-6:]
        channel=re.sub("/p.*","",delete)
        ts=re.sub(".*/p","",delete)
    else:
        if channel is None:
            if config is None:
                config = coretools.config()
            channel=config["notify"]["channel"]
        if re.match(r"^https://.*\.slack\.com/archives/.*", channel):
            organization = re.sub("^https://","", re.sub(r"\.slack\.com/archives/.*","",channel))
            channel=re.sub(r"^https://.*\.slack\.com/archives/","",channel)
        else:
        # this is needed to set organization correctly (so as to build the
        # proper link) when passing the name of a channel
            organization = _organization

    client = WebClient(token=token)

    blocks=[]
    text=""

    if len(title) > 0:
        blocks.append(
           {
               "type": "section",
               "text": { "type": "mrkdwn", "text": "*" + title + "*"},
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
        h = ""
        if update:
            h += "Updated"
        else:
            h += "Sent"
        h += " by "+ os.environ['USER']
        h += " at " + socket.gethostname() +'\n'
        h += "pwd: " + os.getcwd() + '\n'
        h += '{}'.format(datetime.datetime.now())
        blocks.append({
                          "type": "context",
                          "elements": [
                              { # type: ignore
                                "type": "mrkdwn",
                                "text": h
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

    if update:
        response = client.chat_update(
                   channel=channel,
                   text=text,
                   blocks=blocks,
                   ts=ts)
    elif delete:
        client.chat_delete(channel=channel, ts=ts)
        return ""
    else:
        response = client.chat_postMessage(
                   blocks=blocks,
                   text=text,
                   channel=channel)

    response = cast(SlackResponse, response)

    return "https://" + organization + ".slack.com/archives/" + response["channel"] + "/p" + response["ts"][:-7] + response["ts"][-6:]
