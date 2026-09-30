#!/usr/bin/env bash
# Build main.pdf (AAMAS 2027 format) and supplement.pdf with latexmk.
# The appendix imports the theorem numbers of the main paper (xr), so the main paper is built first.
set -euo pipefail
cd "$(dirname "$0")"
python3 gen_tables.py
mkdir -p build/main build/supplement
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/main main.tex
TEXINPUTS="build/main:" latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/supplement supplement.tex
TEXINPUTS="build/main:" latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/supplement supplement.tex
cp build/main/main.pdf main.pdf
cp build/supplement/supplement.pdf supplement.pdf
