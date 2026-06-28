#!/usr/bin/env python3
"""Minimal Markdown → Typst converter for knitwork method docs."""
import re
import sys
from pathlib import Path


def md2typst(src: str) -> str:
    lines = src.splitlines()
    out = []
    in_code = False
    code_lang = ""
    in_table = False
    table_rows: list[list[str]] = []

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            return
        # skip separator row (only dashes/pipes)
        data = [r for r in table_rows if not all(c.strip("-: ") == "" for c in r)]
        n_cols = max(len(r) for r in data)
        col_w = ",".join(["1fr"] * n_cols)
        out.append(f"#table(columns: ({col_w}),")
        for i, row in enumerate(data):
            for cell in row:
                text = inline(cell.strip())
                if i == 0:
                    text = f"[*{cell.strip()}*]"
                else:
                    text = f"[{inline(cell.strip())}]"
                out.append(f"  {text},")
        out.append(")")
        table_rows = []
        in_table = False

    def escape_typst(s: str) -> str:
        # escape Typst special chars in plain text: # @ < >
        s = s.replace("#", r"\#")
        s = s.replace("@", r"\@")
        return s

    def inline(s: str) -> str:
        # protect inline code from escaping: collect backtick spans
        parts = re.split(r'(`[^`]+`)', s)
        result = []
        for part in parts:
            if part.startswith("`") and part.endswith("`") and len(part) > 1:
                result.append(part)  # keep verbatim
            else:
                p = part
                # bold **text** → *text*
                p = re.sub(r'\*\*(.+?)\*\*', r'*\1*', p)
                # italic *text* → _text_
                p = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'_\1_', p)
                # links [text](url) → #link("url")[text]
                p = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'#link("\2")[\1]', p)
                # escape # and @ outside code spans
                p = escape_typst(p)
                result.append(p)
        return "".join(result)

    for raw in lines:
        line = raw

        # fenced code block
        if re.match(r'^```', line):
            if not in_code:
                code_lang = line[3:].strip()
                in_code = True
                if in_table:
                    flush_table()
                lang_part = code_lang if code_lang else ""
                out.append(f"```{lang_part}")
            else:
                in_code = False
                out.append("```")
            continue

        if in_code:
            out.append(line)
            continue

        # table row
        if line.startswith("|"):
            cells = [c for c in line.split("|")]
            # strip leading/trailing empty from split
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            table_rows.append(cells)
            in_table = True
            continue
        else:
            if in_table:
                flush_table()

        # headings
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            text = inline(m.group(2))
            out.append("=" * level + " " + text)
            continue

        # horizontal rule
        if re.match(r'^---+\s*$', line):
            out.append("#line(length: 100%)")
            continue

        # unordered list
        m = re.match(r'^(\s*)[-*]\s+(.*)', line)
        if m:
            indent = m.group(1)
            text = inline(m.group(2))
            out.append(f"{indent}- {text}")
            continue

        # ordered list
        m = re.match(r'^(\s*)\d+\.\s+(.*)', line)
        if m:
            indent = m.group(1)
            text = inline(m.group(2))
            out.append(f"{indent}+ {text}")
            continue

        # blank line
        if line.strip() == "":
            out.append("")
            continue

        # regular paragraph
        out.append(inline(line))

    if in_table:
        flush_table()

    return "\n".join(out)


def convert_file(src: Path, dst: Path, title: str = ""):
    text = src.read_text(encoding="utf-8")
    body = md2typst(text)
    title = title or src.stem
    result = f'#import "_template.typ": template\n#show: template.with(title: "{title}")\n\n{body}\n'
    dst.write_text(result, encoding="utf-8")


def main():
    methods_dir = Path(__file__).parent.parent / "methods"
    out_dir = Path(__file__).parent

    files = sorted(methods_dir.glob("*.md"))
    if len(sys.argv) > 1:
        files = [Path(a) for a in sys.argv[1:]]

    for src in files:
        dst = out_dir / (src.stem + ".typ")
        convert_file(src, dst)
        print(f"  {src.name} → {dst.name}")

    print(f"Done. {len(files)} files converted.")
    print("To compile: typst compile <name>.typ")
    print("To compile all: for f in *.typ; do typst compile \"$f\"; done")


if __name__ == "__main__":
    main()
