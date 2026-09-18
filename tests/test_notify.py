import os
import unittest
from io import StringIO
from unittest.mock import Mock, call, mock_open, patch
from urllib.error import URLError

from bussilab.cli import cli
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
    def test_post_message_builds_payload_and_returns_url(self):
        client = Mock()
        client.chat_postMessage.return_value = {
            "channel": "C123",
            "ts": "1700000000.123456"
        }
        client.auth_test.return_value = {"url": "https://acme.slack.com/"}

        with patch("bussilab.notify.WebClient", return_value=client):
            url = notify("Hello", channel="C123", token="token",
                         title="Title", type="plain_text", footer=False)

        self.assertEqual(
            url,
            "https://acme.slack.com/archives/C123/p1700000000123456"
        )
        client.chat_postMessage.assert_called_once_with(
            channel="C123",
            text="*Title*\nHello\n",
            blocks=[
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "*Title*"}
                },
                {
                    "type": "section",
                    "text": {"type": "plain_text", "text": "Hello"}
                }
            ]
        )

    def test_empty_notification_builds_fallback_payload(self):
        client = Mock()
        client.chat_postMessage.return_value = {
            "channel": "C123",
            "ts": "1700000000.123456"
        }
        client.auth_test.return_value = {"url": "https://acme.slack.com/"}

        with patch("bussilab.notify.WebClient", return_value=client):
            notify(channel="C123", token="token", footer=False)

        client.chat_postMessage.assert_called_once_with(
            channel="C123",
            text="(empty notification)",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "(empty notification)"}
            }]
        )

    def test_disable_unfurls_for_messages_and_replies(self):
        client = Mock()
        client.chat_postMessage.side_effect = [
            {"channel": "C123", "ts": "1700000000.123456"},
            {"channel": "C123", "ts": "1700000000.123457"},
            {"channel": "C123", "ts": "1700000000.123458"}
        ]
        client.auth_test.return_value = {"url": "https://acme.slack.com/"}
        parent = "https://acme.slack.com/archives/C123/p1700000000123456"

        with patch("bussilab.notify.WebClient", return_value=client):
            notify("https://example.com", channel="C123", token="token",
                   footer=False, unfurl=False)
            notify("https://example.com", reply=parent, token="token",
                   footer=False, unfurl=False)
            notify("https://example.com", reply_broadcast=parent,
                   token="token", footer=False, unfurl=False)

        common = {
            "text": "https://example.com\n",
            "blocks": [{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "https://example.com"}
            }],
            "unfurl_links": False,
            "unfurl_media": False
        }
        self.assertEqual(client.chat_postMessage.call_args_list, [
            call(channel="C123", **common),
            call(channel="C123", thread_ts="1700000000.123456", **common),
            call(channel="C123", thread_ts="1700000000.123456",
                 reply_broadcast=True, **common)
        ])

    def test_markdown_message_file_builds_native_markdown_block(self):
        client = Mock()
        client.chat_postMessage.return_value = {
            "channel": "C123",
            "ts": "1700000000.123456"
        }
        client.auth_test.return_value = {"url": "https://acme.slack.com/"}
        markdown = "# Report\n\n| Result | Value |\n| --- | --- |\n| A | 1 |"

        with patch("builtins.open", mock_open(read_data=markdown)) as opened, \
             patch("bussilab.notify.WebClient", return_value=client):
            notify(markdown_file="report.md", channel="C123", token="token")

        opened.assert_called_once_with("report.md", encoding="utf-8")
        client.chat_postMessage.assert_called_once_with(
            channel="C123",
            text=markdown + "\n",
            blocks=[{"type": "markdown", "text": markdown}]
        )

    def test_markdown_rejects_automatic_formatting_and_file_uploads(self):
        invalid_arguments = (
            {"message": "# Report", "title": "Report"},
            {"message": "# Report", "screenlog": "screen.log"},
            {"message": "# Report", "file": "results.dat"},
            {"message": "# Report", "markdown_file": "report.md"},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(TypeError):
                    notify(type="markdown", token="token", **arguments)

    def test_markdown_truncation_respects_block_limit(self):
        client = Mock()
        client.chat_postMessage.return_value = {
            "channel": "C123",
            "ts": "1700000000.123456"
        }
        client.auth_test.return_value = {"url": "https://acme.slack.com/"}

        with patch("bussilab.notify.WebClient", return_value=client):
            notify("x" * 12001, type="markdown", channel="C123",
                   token="token")

        arguments = client.chat_postMessage.call_args.kwargs
        markdown = arguments["blocks"][0]["text"]
        self.assertEqual(len(markdown), 12000)
        self.assertTrue(markdown.endswith(" [truncated]"))
        self.assertEqual(arguments["text"], markdown + "\n")

    def test_screenlog_uses_full_markdown_block_limit(self):
        client = Mock()
        client.chat_postMessage.return_value = {
            "channel": "C123",
            "ts": "1700000000.123456"
        }
        client.auth_test.return_value = {"url": "https://acme.slack.com/"}
        screenlog = b"x" * 12001

        with patch("builtins.open", mock_open(read_data=screenlog)), \
             patch("bussilab.notify.WebClient", return_value=client):
            notify(screenlog="screen.log", channel="C123", token="token",
                   footer=False)

        arguments = client.chat_postMessage.call_args.kwargs
        block = arguments["blocks"][0]
        self.assertEqual(block["type"], "markdown")
        self.assertEqual(len(block["text"]), 12000)
        self.assertTrue(block["text"].startswith("```\n"))
        self.assertTrue(block["text"].endswith(" [truncated]\n```\n"))
        self.assertEqual(arguments["text"], block["text"][4:-5] + "\n")

    def test_update_reply_and_broadcast_build_expected_calls(self):
        client = Mock()
        client.chat_update.return_value = {
            "channel": "C123",
            "ts": "1700000000.123456"
        }
        client.chat_postMessage.side_effect = [
            {"channel": "C123", "ts": "1700000000.123457"},
            {"channel": "C123", "ts": "1700000000.123458"}
        ]
        parent = "https://acme.slack.com/archives/C123/p1700000000123456"

        with patch("bussilab.notify.WebClient", return_value=client):
            updated = notify("Updated", update=parent, token="token",
                             footer=False)
            reply = notify("Reply", reply=parent, token="token",
                           footer=False)
            broadcast = notify("Broadcast", reply_broadcast=parent,
                               token="token", footer=False)

        self.assertEqual(updated, parent)
        self.assertEqual(
            reply,
            "https://acme.slack.com/archives/C123/p1700000000123457"
        )
        self.assertEqual(
            broadcast,
            "https://acme.slack.com/archives/C123/p1700000000123458"
        )
        client.chat_update.assert_called_once_with(
            channel="C123",
            ts="1700000000.123456",
            text="Updated\n",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Updated"}
            }]
        )
        self.assertEqual(client.chat_postMessage.call_args_list, [
            call(
                channel="C123",
                thread_ts="1700000000.123456",
                text="Reply\n",
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Reply"}
                }]
            ),
            call(
                channel="C123",
                thread_ts="1700000000.123456",
                reply_broadcast=True,
                text="Broadcast\n",
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Broadcast"}
                }]
            )
        ])

    def test_reaction_and_multi_delete_route_to_expected_methods(self):
        client = Mock()
        message = "https://acme.slack.com/archives/C123/p1700000000123456"
        reaction = message + ":white_check_mark"
        file = "https://acme.slack.com/files/U123/F123"

        with patch("bussilab.notify.WebClient", return_value=client):
            self.assertEqual(notify(react=reaction, token="token"), reaction)
            self.assertEqual(notify(delete=reaction, token="token"), "")
            self.assertEqual(notify(delete=message + "," + file,
                                    token="token"), "")

        client.reactions_add.assert_called_once_with(
            channel="C123",
            timestamp="1700000000.123456",
            name="white_check_mark"
        )
        client.reactions_remove.assert_called_once_with(
            channel="C123",
            timestamp="1700000000.123456",
            name="white_check_mark"
        )
        client.chat_delete.assert_called_once_with(
            channel="C123", ts="1700000000.123456"
        )
        client.files_delete.assert_called_once_with(file="F123")

    def test_configuration_defaults_and_footer(self):
        client = Mock()
        client.chat_postMessage.return_value = {
            "channel": "C123",
            "ts": "1700000000.123456"
        }
        client.auth_test.return_value = {"url": "https://acme.slack.com/"}
        fixed_datetime = Mock()
        fixed_datetime.now.return_value.isoformat.return_value = (
            "2026-09-16 12:34:56.789"
        )

        with patch("bussilab.notify.coretools.config", return_value={
                "notify": {"token": "configured-token", "channel": "C123"}
             }), \
             patch("bussilab.notify.WebClient", return_value=client) as web, \
             patch("bussilab.notify.datetime.datetime", fixed_datetime), \
             patch("bussilab.notify.socket.gethostname", return_value="host"), \
             patch("bussilab.notify.os.getcwd", return_value="/work"), \
             patch.dict("bussilab.notify.os.environ", {"USER": "alice"}):
            notify("Hello")

        web.assert_called_once_with(token="configured-token")
        call_arguments = client.chat_postMessage.call_args.kwargs
        self.assertEqual(call_arguments["channel"], "C123")
        self.assertEqual(
            call_arguments["text"],
            "Hello\nSent by alice at host\npwd: /work\n"
            "2026-09-16 12:34:56.789\n"
        )
        self.assertEqual(call_arguments["blocks"][-1], {
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": "Sent by alice at host\npwd: /work\n"
                        "2026-09-16 12:34:56.789"
            }]
        })

    def test_notify_cli_translates_arguments_and_prints_url(self):
        output = StringIO()
        with patch("bussilab.notify.notify", return_value="message-url") as send, \
             patch("sys.stdout", output):
            cli(["notify", "--message", "Hello", "--channel", "C123",
                 "--no-footer", "--no-unfurl", "--screenlog-maxlines", "3"],
                prog="bussilab")

        send.assert_called_once_with(
            message="Hello",
            channel="C123",
            footer=False,
            unfurl=False,
            screenlog_maxlines=3,
            type="mrkdwn"
        )
        self.assertEqual(output.getvalue(), "message-url\n")

    def test_markdown_cli_reads_message_from_file(self):
        output = StringIO()
        with patch("bussilab.notify.notify", return_value="message-url") as send, \
             patch("sys.stdout", output):
            cli(["notify", "--markdown-file", "report.md",
                 "--channel", "C123"],
                prog="bussilab")

        send.assert_called_once_with(
            markdown_file="report.md",
            channel="C123",
            screenlog_maxlines=0,
            type="markdown"
        )
        self.assertEqual(output.getvalue(), "message-url\n")

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

    @patch("bussilab.notify.warnings.warn")
    @patch("bussilab.notify.random.uniform", return_value=1.0)
    @patch("bussilab.notify.time.sleep")
    @patch("bussilab.notify.SlackApiError", _SlackApiError)
    def test_retry_uses_bounded_backoff_for_server_errors(self, sleep, uniform,
                                                           warn):
        server_error = _SlackApiError(_SlackResponse("server_error", 503))
        operation = Mock(side_effect=[server_error, server_error, "result"])

        self.assertEqual(_try_multiple_times(operation), "result")
        self.assertEqual(sleep.call_args_list, [call(2.0), call(4.0)])
        self.assertEqual(warn.call_count, 2)

    @patch("bussilab.notify.warnings.warn")
    @patch("bussilab.notify.random.uniform", return_value=1.0)
    @patch("bussilab.notify.time.sleep")
    @patch("bussilab.notify.SlackApiError", _SlackApiError)
    def test_retry_honors_retry_after_without_exponential_growth(self, sleep,
                                                                  uniform,
                                                                  warn):
        rate_limit = _SlackApiError(_SlackResponse(
            "ratelimited", 429, {"retry-after": "10"}
        ))
        operation = Mock(side_effect=[rate_limit] * 4 + ["result"])

        self.assertEqual(_try_multiple_times(operation), "result")
        self.assertEqual(sleep.call_args_list, [call(10.0)] * 4)
        self.assertEqual(warn.call_count, 4)

    @patch("bussilab.notify.warnings.warn")
    @patch("bussilab.notify.random.uniform", return_value=1.0)
    @patch("bussilab.notify.time.sleep")
    @patch("bussilab.notify.SlackApiError", _SlackApiError)
    def test_rate_limit_retries_are_not_limited_to_five_attempts(
            self, sleep, uniform, warn):
        rate_limit = _SlackApiError(_SlackResponse(
            "ratelimited", 429, {"Retry-After": "1"}
        ))
        operation = Mock(side_effect=[rate_limit] * 6 + ["result"])

        self.assertEqual(_try_multiple_times(operation), "result")
        self.assertEqual(operation.call_count, 7)
        self.assertEqual(sleep.call_args_list, [call(1.0)] * 6)
        self.assertEqual(warn.call_count, 6)

    @patch("bussilab.notify.warnings.warn")
    @patch("bussilab.notify.random.uniform", return_value=1.0)
    @patch("bussilab.notify.time.sleep")
    @patch("bussilab.notify.SlackApiError", _SlackApiError)
    def test_rate_limit_without_retry_after_uses_backoff(
            self, sleep, uniform, warn):
        rate_limit = _SlackApiError(_SlackResponse("ratelimited", 429))
        operation = Mock(side_effect=[rate_limit, rate_limit, "result"])

        self.assertEqual(_try_multiple_times(operation), "result")
        self.assertEqual(sleep.call_args_list, [call(2.0), call(4.0)])
        self.assertEqual(warn.call_count, 2)

    @patch("bussilab.notify.warnings.warn")
    @patch("bussilab.notify.random.uniform", return_value=1.0)
    @patch("bussilab.notify.time.sleep")
    def test_retry_retries_transport_errors(self, sleep, uniform, warn):
        operation = Mock(side_effect=[URLError("temporary"), "result"])

        self.assertEqual(_try_multiple_times(operation), "result")
        sleep.assert_called_once_with(2.0)
        warn.assert_called_once()

    @patch("bussilab.notify.warnings.warn")
    @patch("bussilab.notify.random.uniform", return_value=1.0)
    @patch("bussilab.notify.time.sleep")
    @patch("bussilab.notify.SlackApiError", _SlackApiError)
    def test_retry_stops_after_five_attempts(self, sleep, uniform, warn):
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
        self.assertEqual(warn.call_count, 4)

    @patch("bussilab.notify.warnings.warn")
    @patch("bussilab.notify.random.uniform", return_value=1.0)
    @patch("bussilab.notify.time.sleep")
    @patch("bussilab.notify.SlackApiError", _SlackApiError)
    def test_retry_stops_before_exceeding_total_wait_budget(self, sleep,
                                                             uniform, warn):
        operation = Mock(side_effect=_SlackApiError(_SlackResponse(
            "ratelimited", 429, {"Retry-After": "101"}
        )))

        with self.assertRaises(_SlackApiError):
            _try_multiple_times(operation)

        self.assertEqual(operation.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(101.0), call(101.0)])
        self.assertEqual(warn.call_count, 2)

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
            {"update": "incorrect-url"},
            {"reply": file},
            {"reply": "incorrect-url"},
            {"reply_broadcast": file},
            {"delete": "incorrect-url"},
        )
        for arguments in invalid_operations:
            with self.subTest(arguments=arguments):
                with self.assertRaises(TypeError):
                    notify(token="token", footer=False, **arguments)

    def test_operations_are_mutually_exclusive(self):
        message = "https://acme.slack.com/archives/C123/p1700000000123456"
        invalid_operations = (
            {"channel": "C123", "update": message},
            {"channel": "C123", "delete": message},
            {"update": message, "delete": message},
            {"update": message, "react": message + ":heart"},
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

            url=notify("https://example.com", token=token, channel=channel,
                       unfurl=False)
            notify(delete=url, token=token)

            url=notify(markdown_file=os.path.realpath(__file__), token=token,
                       channel=channel)
            notify(delete=url, token=token)

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
