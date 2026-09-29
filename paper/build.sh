#!/usr/bin/env bash
# Build main.pdf and supplement.pdf with latexmk.
set -euo pipefail
cd "$(dirname "$0")"
for name in main supplement; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build "$name.tex"
  cp "build/$name.pdf" "$name.pdf"
done
