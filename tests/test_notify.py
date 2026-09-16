import os
import unittest
from unittest.mock import Mock, patch

from bussilab.notify import notify


class TestNotifyUnit(unittest.TestCase):
    def test_file_upload_v2_with_immediate_share(self):
        client = Mock()
        client.files_upload_v2.return_value = {
            "files": [{
                "id": "F123",
                "user": "U123",
                "shares": {
                    "public": {
                        "C123": [{"ts": "1700000000.123456"}]
                    }
                }
            }]
        }
        client.auth_test.return_value = {"url": "https://acme.slack.com/"}

        with patch("bussilab.notify.WebClient", return_value=client):
            url = notify(file=__file__, token="token", channel="C123",
                         footer=False)

        self.assertEqual(
            url,
            "https://acme.slack.com/archives/C123/p1700000000123456,"
            "https://acme.slack.com/files/U123/F123"
        )

    def test_file_upload_v2_reply_includes_initial_comment(self):
        client = Mock()
        client.files_upload_v2.return_value = {
            "files": [{
                "id": "F123",
                "user": "U123",
                "shares": {
                    "public": {
                        "C123": [{"ts": "1700000000.123457"}]
                    }
                }
            }]
        }
        reply = "https://acme.slack.com/archives/C123/p1700000000123456"

        with patch("bussilab.notify.WebClient", return_value=client):
            notify("Description", file=__file__, token="token", reply=reply,
                   title="Title", footer=False)

        client.files_upload_v2.assert_called_once_with(
            file=__file__,
            channel="C123",
            title=__file__,
            thread_ts="1700000000.123456",
            initial_comment="*Title*\nDescription\n"
        )


# only run tests if env vars are configured
if 'BUSSILAB_TEST_NOTIFY_TOKEN' in os.environ:
    token=os.environ["BUSSILAB_TEST_NOTIFY_TOKEN"]
    channel=os.environ["BUSSILAB_TEST_NOTIFY_CHANNEL"]
    class TestNotify(unittest.TestCase):
        def test_notify(self):
            url=notify("unittest1", token=token, channel=channel)
            notify(react=url+":white_check_mark",token=token)
            notify(delete=url+":white_check_mark",token=token)

            url=notify("unittest2 *WRONG*", token=token, channel=channel)
            notify("unittest2", update=url, token=token)

            url=notify("unittest3 *WRONG*", token=token, channel=channel)
            url=notify("unittest3 *WRONG 2*", update=url, token=token)
            url2=notify("unittest3 reply", reply=url, token=token)
            url3=notify("unittest3 broadcast", reply_broadcast=url, token=token)
            notify(delete=url, token=token)

            with self.assertRaises(Exception):
                notify("unittest3 *WRONG 3*", update=url, token=token)

            with self.assertRaises(Exception):
                notify(delete=url, token=token)

            with self.assertRaises(TypeError):
                notify("unittest1", token=token, channel=channel, update=url)
            with self.assertRaises(TypeError):
                notify("unittest1", token=token, channel=channel, delete=url)
            with self.assertRaises(TypeError):
                notify("unittest1", token=token, update=url, delete=url)
            with self.assertRaises(TypeError):
                notify("unittest1", token=token, update="incorrect-url")
            with self.assertRaises(TypeError):
                notify("unittest1", token=token, delete="incorrect-url")

            url=notify("unittest4 *WRONG*", token=token, channel=channel)
            url2=notify("test upload",file=os.path.realpath(__file__), token=token, reply=url)
            notify(delete=url, token=token)
            notify(delete=url2, token=token)
            
            url=notify("test upload",file=os.path.realpath(__file__), token=token, channel=channel)
            notify(delete=url, token=token)


if __name__ == "__main__":
    unittest.main()
