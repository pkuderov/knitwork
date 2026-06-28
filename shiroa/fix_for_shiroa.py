#!/usr/bin/env python3
"""Post-process pandoc md→typst output for Shiroa (non-HTML, PDF/web targets).

Usage: pandoc file.md -f gfm -t typst | python3 fix_for_shiroa.py "Page Title"
"""
import re
import sys


HEADER = """\
#import "/book.typ": book-page
#show: book-page.with(title: "{title}")

"""


def fix(src: str) -> str:
    lines = src.splitlines(keepends=True)
    result = []
    in_code = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code = not in_code
            result.append(line)
            continue
        if in_code:
            result.append(line)
            continue
        result.append(_fix_line(line))

    src = ''.join(result)
    # orphan ] left over from #align wrapper removal
    src = re.sub(r'(\n\))\n\]\n', r'\1\n', src)
    return src


def _fix_line(line: str) -> str:
    # remove Typst section labels pandoc generates from heading IDs
    if re.match(r'^<[a-zA-Z][^>\n]*>\s*$', line):
        return ''

    line = line.replace('#horizontalrule', '#line(length: 100%)')
    line = line.replace('#blockquote[', '#quote(block: true)[')
    line = re.sub(r'#align\(center\)\[#table\(', '#table(', line)

    if re.search(r'align: \(col, row\) =>', line):
        return ''

    # / at start of line triggers Typst term-definition syntax — escape it
    if re.match(r'^/ ', line):
        line = '#"/" ' + line[2:]

    # [*text*]* → [*text*]
    line = re.sub(r'\[\*([^*\]]*)\]\*', r'[*\1*]', line)

    # // inside table cells causes Typst to treat it as a comment.
    # Protect :// (URLs) and fix remaining // occurrences.
    line = _fix_double_slash(line)

    return line


def _fix_double_slash(line: str) -> str:
    parts = re.split(r'(`[^`]+`)', line)
    fixed = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            fixed.append(part)
        else:
            placeholder = '\x00URL\x00'
            s = re.sub(r'://', placeholder, part)
            s = s.replace('//', '#"/"#"/"')
            s = s.replace(placeholder, '://')
            fixed.append(s)
    return ''.join(fixed)


if __name__ == '__main__':
    title = sys.argv[1] if len(sys.argv) > 1 else "Untitled"
    src = sys.stdin.read()
    out = HEADER.format(title=title) + fix(src)
    sys.stdout.write(out)
