#!/usr/bin/env python3
"""Post-process pandoc md→typst output for HTML export compatibility."""
import re
import sys

def fix(src: str) -> str:
    # Process line by line, skipping fenced code blocks
    lines = src.splitlines(keepends=True)
    result = []
    in_code = False
    for line in lines:
        if line.strip().startswith('```'):
            in_code = not in_code
            result.append(line)
            continue
        if in_code:
            result.append(line)
            continue
        result.append(_fix_line(line))

    src = ''.join(result)

    # Remove orphan ] lines that follow ) — leftover from #align wrapper removal
    src = re.sub(r'(\n\))\n\]\n', r'\1\n', src)

    return src


def _fix_line(line: str) -> str:
    # Remove Typst section labels: <heading-id> on its own line
    if re.match(r'^<[a-zA-Z][^>\n]*>\s*$', line):
        return ''

    # #horizontalrule → #line(length: 100%)
    line = line.replace('#horizontalrule', '#line(length: 100%)')

    # #blockquote[...] → #quote(block: true)[...]
    line = line.replace('#blockquote[', '#quote(block: true)[')

    # #align(center)[#table( → #table(
    line = re.sub(r'#align\(center\)\[#table\(', '#table(', line)

    # Remove align: lambda lines (not supported in HTML export)
    if re.search(r'align: \(col, row\) =>', line):
        return ''

    # [*text*] where closing * is outside [] → [*text*]
    line = re.sub(r'\[\*([^*\]]*)\]\*', r'[*\1*]', line)

    # / at start of line triggers Typst term-definition syntax — escape it
    if re.match(r'^/ ', line):
        line = '#"/" ' + line[2:]

    # Fix // inside table cells: in Typst, // inside () inside [] is a comment.
    # Replace // with unicode escape sequence for / repeated twice.
    # Only outside of backtick spans.
    line = _fix_double_slash(line)

    return line


def _fix_double_slash(line: str) -> str:
    """Replace // with #"/"#"/" outside raw spans and URLs."""
    parts = re.split(r'(`[^`]+`)', line)
    fixed = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # inside backticks — keep verbatim
            fixed.append(part)
        else:
            # protect :// (URL protocol) from replacement
            placeholder = '\x00URL\x00'
            safe = re.sub(r'://', placeholder, part)
            safe = safe.replace('//', '#"/"#"/"')
            safe = safe.replace(placeholder, '://')
            fixed.append(safe)
    return ''.join(fixed)

if __name__ == '__main__':
    for path in sys.argv[1:]:
        text = open(path).read()
        fixed = fix(text)
        open(path, 'w').write(fixed)
        print(f'  fixed: {path}')
