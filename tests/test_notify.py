import os
import unittest

from bussilab.notify import notify

# only run tests if env vars are configured
if 'BUSSILAB_TEST_NOTIFY_TOKEN' in os.environ:
    token=os.environ["BUSSILAB_TEST_NOTIFY_TOKEN"]
    channel=os.environ["BUSSILAB_TEST_NOTIFY_CHANNEL"]
    class TestNotify(unittest.TestCase):
        def test_notify(self):
            notify("unittest1", token=token, channel=channel)

            url=notify("unittest2 *WRONG*", token=token, channel=channel)
            notify("unittest2", update=url, token=token)

            url=notify("unittest3 *WRONG*", token=token, channel=channel)
            url=notify("unittest3 *WRONG 2*", update=url, token=token)
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

            url=notify("test upload",file=os.path.realpath(__file__), token=token)
            notify(delete=url, token=token)


if __name__ == "__main__":
    unittest.main()
