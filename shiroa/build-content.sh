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

    # On macos sed -i doesn't work, so we use sed -i '' instead
    if [[ "$OSTYPE" == darwin* ]]; then
        SED_INPLACE=(-i '')
    else
        SED_INPLACE=(-i)
    fi

    for f in "$dist"/*.html; do
        [ -f "$f" ] || continue
        sed "${SED_INPLACE[@]}" 's|<!-- Custom theme stylesheets -->|<!-- Custom theme stylesheets -->\n    <link rel="stylesheet" href="theme/css/extra.css">|' "$f"
    done

    for f in "$dist"/methods/*.html "$dist"/experiments/*.html; do
        [ -f "$f" ] || continue
        sed "${SED_INPLACE[@]}" 's|<!-- Custom theme stylesheets -->|<!-- Custom theme stylesheets -->\n    <link rel="stylesheet" href="../theme/css/extra.css">|' "$f"
    done

    echo "  injected into $(find "$dist" -name '*.html' | wc -l) HTML files"
}

# ── Convert .md → .typ ──────────────────────────────────────────

echo "Converting pages..."
# every page the sidebar can link to: methods/, experiments/ (recursive), papers/, reviews/ and root pages
while IFS= read -r -d '' f; do
    rel=${f#"$DOCS_DIR/"}
    mkdir -p "$SCRIPT_DIR/$(dirname "$rel")"
    title=$(head -1 "$f" | sed 's/^# //')
    convert "$f" "$SCRIPT_DIR/${rel%.md}.typ" "$title"
done < <(cd "$DOCS_DIR" && find . -name '*.md' ! -name 'README.md' ! -name '_*' ! -path './README.md' -print0 \
    | sed -z 's#^\./##' | sort -z | sed -z "s#^#$DOCS_DIR/#")

echo "Copying images..."
# pages reference figures relative to their own directory; mirror them next to the .typ files
(cd "$DOCS_DIR" && find methods experiments papers reviews -type f \( -name '*.png' -o -name '*.svg' -o -name '*.jpg' \) -print0 \
    | xargs -0 -r cp --parents -t "$SCRIPT_DIR")

echo "Converting README..."
convert "$DOCS_DIR/README.md" "$SCRIPT_DIR/README.typ" "knitwork"

echo "Done converting."

# ── Optional: build Shiroa + inject CSS ─────────────────────────

if [[ "${1:-}" == "--build" ]]; then
    echo "Generating summary-gen.typ from docs/_navigation.md..."
    python3 "$SCRIPT_DIR/gen_summary.py"

    # resolve shiroa command: either from script dir or from PATH
    if [[ -x "$SCRIPT_DIR/shiroa" ]]; then
        SHIROA="$SCRIPT_DIR/shiroa"
    else
        SHIROA="shiroa"
    fi

    echo "Building shiroa..."
    "$SHIROA" build "$SCRIPT_DIR/"
    inject_css "$SCRIPT_DIR/dist"
fi
