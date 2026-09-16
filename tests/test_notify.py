import os
import unittest
from unittest.mock import Mock, call, patch
from urllib.error import URLError

from bussilab.notify import _parse_url, _try_multiple_times, notify


class _SlackResponse(dict):
    def __init__(self, error, status_code, headers=None):
        super().__init__(error=error)
        self.status_code = status_code
        self.headers = {} if headers is None else headers


class _SlackApiError(Exception):
    def __init__(self, response):
        self.response = response


class TestNotifyUnit(unittest.TestCase):
    @patch("bussilab.notify.time.sleep")
    @patch("bussilab.notify.SlackApiError", _SlackApiError)
    def test_retry_does_not_retry_permanent_api_errors(self, sleep):
        operation = Mock(side_effect=_SlackApiError(
            _SlackResponse("invalid_auth", 400)
        ))

        with self.assertRaises(_SlackApiError):
            _try_multiple_times(operation)

        self.assertEqual(operation.call_count, 1)
        sleep.assert_not_called()

    @patch("bussilab.notify.random.uniform", return_value=1.0)
    @patch("bussilab.notify.time.sleep")
    @patch("bussilab.notify.SlackApiError", _SlackApiError)
    def test_retry_uses_bounded_backoff_for_server_errors(self, sleep, uniform):
        server_error = _SlackApiError(_SlackResponse("server_error", 503))
        operation = Mock(side_effect=[server_error, server_error, "result"])

        self.assertEqual(_try_multiple_times(operation), "result")
        self.assertEqual(sleep.call_args_list, [call(2.0), call(4.0)])

    @patch("bussilab.notify.random.uniform", return_value=1.0)
    @patch("bussilab.notify.time.sleep")
    @patch("bussilab.notify.SlackApiError", _SlackApiError)
    def test_retry_honors_retry_after_without_exponential_growth(self, sleep,
                                                                  uniform):
        rate_limit = _SlackApiError(_SlackResponse(
            "ratelimited", 429, {"retry-after": "10"}
        ))
        operation = Mock(side_effect=[rate_limit] * 4 + ["result"])

        self.assertEqual(_try_multiple_times(operation), "result")
        self.assertEqual(sleep.call_args_list, [call(10.0)] * 4)

    @patch("bussilab.notify.random.uniform", return_value=1.0)
    @patch("bussilab.notify.time.sleep")
    def test_retry_retries_transport_errors(self, sleep, uniform):
        operation = Mock(side_effect=[URLError("temporary"), "result"])

        self.assertEqual(_try_multiple_times(operation), "result")
        sleep.assert_called_once_with(2.0)

    @patch("bussilab.notify.random.uniform", return_value=1.0)
    @patch("bussilab.notify.time.sleep")
    @patch("bussilab.notify.SlackApiError", _SlackApiError)
    def test_retry_stops_after_five_attempts(self, sleep, uniform):
        operation = Mock(side_effect=_SlackApiError(
            _SlackResponse("server_error", 503)
        ))

        with self.assertRaises(_SlackApiError):
            _try_multiple_times(operation)

        self.assertEqual(operation.call_count, 5)
        self.assertEqual(
            sleep.call_args_list,
            [call(2.0), call(4.0), call(8.0), call(16.0)]
        )

    @patch("bussilab.notify.random.uniform", return_value=1.0)
    @patch("bussilab.notify.time.sleep")
    @patch("bussilab.notify.SlackApiError", _SlackApiError)
    def test_retry_stops_before_exceeding_total_wait_budget(self, sleep,
                                                             uniform):
        operation = Mock(side_effect=_SlackApiError(_SlackResponse(
            "ratelimited", 429, {"Retry-After": "101"}
        )))

        with self.assertRaises(_SlackApiError):
            _try_multiple_times(operation)

        self.assertEqual(operation.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(101.0), call(101.0)])

    def test_parse_url_accepts_supported_slack_urls(self):
        message = (
            "https://acme.slack.com/archives/C123/p1700000000123456"
            "?thread_ts=1700000000.123456&cid=C123"
        )
        self.assertEqual(_parse_url(message), {
            "type": "message",
            "ts": "1700000000.123456",
            "channel": "C123",
            "organization": "acme"
        })
        self.assertEqual(_parse_url(message + ":white_check_mark"), {
            "type": "reaction",
            "ts": "1700000000.123456",
            "channel": "C123",
            "organization": "acme",
            "reaction": "white_check_mark"
        })
        self.assertEqual(
            _parse_url("https://acme.slack.com/files/U123/F123/file.txt"),
            {
                "type": "file",
                "id": "F123",
                "user": "U123",
                "organization": "acme"
            }
        )

    def test_parse_url_rejects_malformed_message_urls(self):
        malformed_urls = (
            "https://acme.slack.com/archives/foo",
            "https://acme.slack.com/archives/C123/not-a-timestamp",
            "https://.slack.com/archives/C123/p1700000000123456",
        )
        for url in malformed_urls:
            with self.subTest(url=url):
                self.assertEqual(_parse_url(url), {})

    def test_operations_require_the_correct_url_type(self):
        message = "https://acme.slack.com/archives/C123/p1700000000123456"
        file = "https://acme.slack.com/files/U123/F123"

        invalid_operations = (
            {"react": message},
            {"react": "incorrect-url"},
            {"update": message + ":heart"},
            {"update": file},
            {"reply": file},
            {"reply": "incorrect-url"},
            {"reply_broadcast": file},
        )
        for arguments in invalid_operations:
            with self.subTest(arguments=arguments):
                with self.assertRaises(TypeError):
                    notify(token="token", footer=False, **arguments)

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
