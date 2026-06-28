#!/bin/bash
# Convert docs/*.md → shiroa/methods/*.typ and shiroa/experiments/*.typ
# Run from repo root: bash shiroa/build-content.sh
# With full build + CSS injection: bash shiroa/build-content.sh --build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCS_DIR="$(dirname "$SCRIPT_DIR")/docs"
FIX="$SCRIPT_DIR/fix_for_shiroa.py"

convert() {
    local src="$1" dst="$2" title="$3"
    pandoc "$src" -f gfm -t typst | python3 "$FIX" "$title" > "$dst"
    echo "  $title"
}

inject_css() {
    local dist="$1"
    cp "$SCRIPT_DIR/templates/extra.css" "$dist/theme/css/extra.css"
    echo "Injecting extra.css..."

    for f in "$dist"/*.html; do
        [ -f "$f" ] || continue
        sed -i 's|<!-- Custom theme stylesheets -->|<!-- Custom theme stylesheets -->\n    <link rel="stylesheet" href="theme/css/extra.css">|' "$f"
    done

    for f in "$dist"/methods/*.html "$dist"/experiments/*.html; do
        [ -f "$f" ] || continue
        sed -i 's|<!-- Custom theme stylesheets -->|<!-- Custom theme stylesheets -->\n    <link rel="stylesheet" href="../theme/css/extra.css">|' "$f"
    done

    echo "  injected into $(find "$dist" -name '*.html' | wc -l) HTML files"
}

# ── Convert .md → .typ ──────────────────────────────────────────

mkdir -p "$SCRIPT_DIR/methods" "$SCRIPT_DIR/experiments"

echo "Converting methods..."
for f in "$DOCS_DIR/methods/"*.md; do
    name=$(basename "$f" .md)
    title=$(head -1 "$f" | sed 's/^# //')
    convert "$f" "$SCRIPT_DIR/methods/${name}.typ" "$title"
done

echo "Converting experiments..."
for f in "$DOCS_DIR/experiments/"*.md; do
    name=$(basename "$f" .md)
    title=$(head -1 "$f" | sed 's/^# //')
    convert "$f" "$SCRIPT_DIR/experiments/${name}.typ" "$title"
done

echo "Converting README..."
convert "$DOCS_DIR/README.md" "$SCRIPT_DIR/README.typ" "knitwork"

echo "Done converting."

# ── Optional: build Shiroa + inject CSS ─────────────────────────

if [[ "${1:-}" == "--build" ]]; then
    echo "Building shiroa..."
    shiroa build "$SCRIPT_DIR/"
    inject_css "$SCRIPT_DIR/dist"
fi
