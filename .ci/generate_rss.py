"""Build module changelog feeds from current docstrings and available release tags.

Requires feedgen and Markdown (documentation dependencies only).
"""

import argparse
import ast
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import textwrap

from feedgen.feed import FeedGenerator
import markdown


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://bussilab.github.io/doc-py-bussilab/bussilab"
RELEASE = re.compile(r"^- \*\*(v[0-9]+\.[0-9]+\.[0-9]+)\*\*\s*$")


class PlainText(HTMLParser):
    """Keep literal list bullets and line breaks for readers that strip HTML."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("ul", "ol"):
            self.depth += 1
        elif tag == "li":
            self.parts.append("\n" + "  " * max(0, self.depth - 1) + "• ")
        elif tag in ("p", "br", "pre"):
            self.parts.append("\n")
        elif tag == "a":
            # Retain destinations as text even when a reader removes markup.
            href = dict(attrs).get("href")
            if href:
                self.parts.append(href + " ")

    def handle_endtag(self, tag):
        if tag in ("ul", "ol"):
            self.depth -= 1
        if tag in ("li", "p", "pre"):
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(re.sub(r"\s+", " ", data))

    def text(self):
        return "\n".join(line.rstrip() for line in "".join(self.parts).splitlines()
                         if line.strip()).strip()


def tag_dates(root):
    """Use tagger dates for annotated tags, committer dates for lightweight tags."""
    result = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname:strip=2) %(creatordate:unix)",
         "refs/tags"], cwd=root, check=True, capture_output=True, text=True)
    dates = {}
    for line in result.stdout.splitlines():
        tag, timestamp = line.split()
        if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
            dates[tag] = datetime.fromtimestamp(int(timestamp), timezone.utc)
    return dates


def changelog_entries(source):
    """Extract version bullets and their indented Markdown, without importing code.

    Stop at the next heading or unindented prose: some module docstrings
    continue with usage instructions immediately after the changelog list.
    """
    doc = ast.get_docstring(ast.parse(source)) or ""
    lines = doc.splitlines()
    try:
        start = lines.index("## Changelog") + 1
    except ValueError:
        return []
    entries = []
    version = None
    body = []
    seen = set()
    for line in lines[start:] + ["# End"]:
        release = RELEASE.fullmatch(line)
        if release or (line.strip() and not line[0].isspace()):
            if version is not None:
                content = textwrap.dedent("\n".join(body)).strip()
                if not content:
                    raise ValueError("Empty changelog entry: " + version)
                entries.append((version, content))
            if not release:
                break
            version = release[1]
            if version in seen:
                raise ValueError("Duplicate changelog entry: " + version)
            seen.add(version)
            body = []
        elif version is not None:
            body.append(line)
    return entries


def generate(root, output, base_url=BASE_URL):
    """Rebuild complete feeds; entries lacking a matching tag are excluded."""
    dates = tag_dates(root)
    base_url = base_url.rstrip("/")
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted((root / "bussilab").rglob("*.py")):
        relative = path.relative_to(root / "bussilab")
        if any(part.startswith("_") for part in relative.parts[:-1]):
            continue
        if path.name == "__init__.py":
            parts = relative.parts[:-1]
            page = "/".join((*parts, "index.html"))
        elif path.stem.startswith("_"):
            continue
        else:
            parts = relative.with_suffix("").parts
            page = relative.with_suffix(".html").as_posix()
        module = ".".join(("bussilab", *parts))
        entries = changelog_entries(path.read_text(encoding="utf-8"))
        if not entries:
            continue
        published = [(tag, body) for tag, body in entries if tag in dates]
        published.sort(key=lambda entry: (dates[entry[0]], entry[0]), reverse=True)
        feed_name = ".".join(parts) if parts else "index"
        link = base_url + "/" + page + "#changelog"
        feed = FeedGenerator()
        feed.title(module + " changelog")
        feed.description("Released changes to " + module)
        feed.link(href=link, rel="alternate")
        feed.link(href=base_url + "/feeds/" + feed_name + ".xml", rel="self")
        feed.language("en")
        if published:
            feed.lastBuildDate(dates[published[0][0]])
        for tag, body in published:
            entry = feed.add_entry(order="append")
            entry.guid(module + ":" + tag, permalink=False)
            entry.title(module + " " + tag)
            entry.link(href=link)
            entry.pubDate(dates[tag])
            html = markdown.markdown(body, extensions=["fenced_code", "tables"])
            plain = PlainText()
            plain.feed(html)
            # RSS descriptions may be interpreted as HTML. Escape literal code
            # such as "x < y" while keeping bullets and newlines as actual text.
            entry.description(escape(plain.text(), quote=False))
            entry.content(html, type="CDATA")
        feed.rss_file(str(output / (feed_name + ".xml")), pretty=True)
        print("{}: {} released entries".format(module, len(published)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "doc/bussilab/feeds")
    parser.add_argument("--base-url", default=BASE_URL,
                        help="Published URL of the bussilab documentation directory")
    args = parser.parse_args()
    generate(ROOT, args.output, args.base_url)


if __name__ == "__main__":
    main()
