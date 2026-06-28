#!/bin/bash
# Convert method docs Markdown → Typst, compile to HTML for browser viewing.
# Requires: pandoc >= 3.2, typst >= 0.13
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
METHODS_DIR="$SCRIPT_DIR/../methods"
HTML_DIR="$SCRIPT_DIR/html"

if ! command -v pandoc &>/dev/null; then
    echo "pandoc not found. Install with: sudo apt install pandoc"
    exit 1
fi

mkdir -p "$HTML_DIR"

for f in "$METHODS_DIR"/*.md; do
    name=$(basename "$f" .md)
    raw="$SCRIPT_DIR/${name}.raw.typ"
    typ="$SCRIPT_DIR/${name}.typ"
    html_typ="$SCRIPT_DIR/${name}_html.typ"
    html="$HTML_DIR/${name}.html"

    # Step 1: pandoc md → raw typst
    pandoc "$f" -f gfm -t typst -o "$raw"

    # Step 2: fix pandoc quirks (Python handles multiline patterns)
    python3 "$SCRIPT_DIR/fix_pandoc.py" "$raw" > /dev/null

    # Step 3: PDF variant
    { echo "#import \"_template.typ\": template"
      echo "#show: template.with(title: \"${name}\")"
      echo ""
      cat "$raw"; } > "$typ"

    # Step 4: HTML variant (no page config)
    { echo "#import \"_template_html.typ\": template"
      echo "#show: template.with(title: \"${name}\")"
      echo ""
      cat "$raw"; } > "$html_typ"

    rm -f "$raw"

    # Step 5: compile to HTML
    if typst compile --features html --format html "$html_typ" "$html" 2>/dev/null; then
        echo "  OK  $name.html"
    else
        echo "  ERR $name:"
        typst compile --features html --format html "$html_typ" "$html" 2>&1 \
            | grep "error:" | head -2 | sed 's/^/      /'
    fi
done

# Generate index page
cat > "$HTML_DIR/index.html" <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>knitwork — method docs</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
  h1 { color: #3f51b5; border-bottom: 2px solid #3f51b5; padding-bottom: 8px; }
  ul { line-height: 2.2; padding-left: 1.2em; }
  a { color: #3f51b5; text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
</head>
<body>
<h1>knitwork — method docs</h1>
<ul>
EOF

for hf in $(ls "$HTML_DIR"/*.html 2>/dev/null | grep -v index); do
    n=$(basename "$hf" .html)
    echo "  <li><a href=\"${n}.html\">${n}</a></li>" >> "$HTML_DIR/index.html"
done

cat >> "$HTML_DIR/index.html" <<'EOF'
</ul>
</body>
</html>
EOF

echo ""
echo "Serve with:"
echo "  cd docs/typst/html && python3 -m http.server 3001"
echo "  → http://localhost:3001"
