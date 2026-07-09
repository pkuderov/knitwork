#!/usr/bin/env python3
"""Generate shiroa/book.typ from shiroa/book.typ.tmpl + docs/_sidebar.md.

Keeps the Shiroa book TOC in sync with the docsify sidebar automatically,
so a new docs/methods/*.md entry only needs to be added to _sidebar.md —
book.typ itself is generated and not checked into git.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIDEBAR = ROOT / "docs" / "_sidebar.md"
TEMPLATE = Path(__file__).resolve().parent / "book.typ.tmpl"
OUT = Path(__file__).resolve().parent / "book.typ"

CATEGORY_RE = re.compile(r'^-\s+\*\*(.+?)\*\*\s*$')
ITEM_RE = re.compile(r'^\s*-\s+\[(.+?)\]\((.+?)\.md\)\s*$')


def build_summary() -> str:
    lines = SIDEBAR.read_text().splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        m = CATEGORY_RE.match(line)
        if m:
            if out:
                out.append("")
            out.append(f"= {m.group(1)}")
            continue
        m = ITEM_RE.match(line)
        if m:
            title, path = m.group(1), m.group(2)
            out.append(f'- #chapter("{path}.typ")[{title}]')
            continue
        print(f"warning: unrecognized sidebar line: {line!r}", file=sys.stderr)
    return "\n".join(out)


def main():
    summary = build_summary()
    book = TEMPLATE.read_text().replace("__SUMMARY__", summary)
    OUT.write_text(book)
    print(f"wrote {OUT} from {TEMPLATE} + {SIDEBAR}")


if __name__ == "__main__":
    main()
