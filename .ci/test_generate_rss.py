"""Run with: python -m unittest discover -s .ci -p test_generate_rss.py"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET


SPEC = importlib.util.spec_from_file_location(
    "generate_rss", Path(__file__).with_name("generate_rss.py"))
rss = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rss)


SOURCE = '''"""Example module.

## Changelog

- **v0.0.53**
  - Future change.
- **v0.0.52**
  - Fixed `x < y` & improved stability.
    Continued description.
  - Another change.
- **v0.0.51**
  - First release.

Usage instructions that must not enter the feed.
"""
raise RuntimeError("Do not import modules to extract changelogs")
'''


class TestGenerateRSS(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init")
        self.git("config", "user.name", "RSS test")
        self.git("config", "user.email", "rss@example.com")
        (self.root / "bussilab").mkdir()
        self.source = self.root / "bussilab/example.py"
        self.source.write_text(SOURCE, encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-m", "Initial", date="2026-01-01T12:00:00+00:00")
        self.git("tag", "v0.0.51")
        self.git("tag", "-a", "v0.0.52", "-m", "Release",
                 date="2026-02-01T12:00:00+00:00")
        self.output = self.root / "feeds"

    def git(self, *args, date=None):
        env = os.environ.copy()
        if date:
            env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
        return subprocess.run(["git", *args], cwd=self.root, env=env,
                              check=True, capture_output=True, text=True).stdout

    def generate(self):
        rss.generate(self.root, self.output, "https://example.com/bussilab/")
        return ET.parse(self.output / "example.xml").getroot().find("channel")

    def test_complete_feed_and_publication_filter(self):
        channel = self.generate()
        items = channel.findall("item")
        self.assertEqual([item.findtext("title") for item in items],
                         ["bussilab.example v0.0.52", "bussilab.example v0.0.51"])
        self.assertEqual(parsedate_to_datetime(items[0].findtext("pubDate")),
                         datetime(2026, 2, 1, 12, tzinfo=timezone.utc))
        self.assertEqual(parsedate_to_datetime(items[1].findtext("pubDate")),
                         datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
        self.assertEqual(items[0].findtext("link"),
                         "https://example.com/bussilab/example.html#changelog")
        self.assertEqual(items[0].findtext("guid"), "bussilab.example:v0.0.52")
        self.assertEqual(items[0].find("guid").get("isPermaLink"), "false")
        html = items[0].findtext("description")
        self.assertIn("<ul>", html)
        self.assertIn("<code>x &lt; y</code> &amp;", html)
        self.assertNotIn("Usage instructions", items[1].findtext("description"))
        before = (self.output / "example.xml").read_bytes()
        self.generate()
        self.assertEqual(before, (self.output / "example.xml").read_bytes())

        self.git("tag", "-a", "v0.0.53", "-m", "Next release",
                 date="2026-03-01T12:00:00+00:00")
        self.assertEqual(len(self.generate().findall("item")), 3)

    def test_corrections_keep_identity_and_dates(self):
        before = self.generate().find("item")
        self.source.write_text(SOURCE.replace("Another change.", "Corrected text."),
                               encoding="utf-8")
        after = self.generate().find("item")
        for field in ("guid", "pubDate"):
            self.assertEqual(before.findtext(field), after.findtext(field))
        self.assertIn("Corrected text.", after.findtext("description"))

    def test_section_boundary_and_duplicate_versions(self):
        source = SOURCE.replace("Usage instructions", "## Usage\n\nUsage instructions")
        entries = rss.changelog_entries(source)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[-1], ("v0.0.51", "- First release."))
        self.assertEqual(rss.changelog_entries('"""No changelog."""'), [])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            rss.changelog_entries(SOURCE.replace("v0.0.51", "v0.0.52"))


if __name__ == "__main__":
    unittest.main()
