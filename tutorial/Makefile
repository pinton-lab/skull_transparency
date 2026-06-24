# Single source of truth: tutorial.tex
# Derived artifacts (do NOT hand-edit): tutorial.pdf, tutorial.md, figs/*.png
#
#   make          build PDF + Markdown (and figures if stale)
#   make pdf      build tutorial.pdf
#   make md       build tutorial.md  (zero-dependency tex2md.py converter)
#   make figs     regenerate figs/*.png from figs.py
#   make clean    remove LaTeX aux files and the figure stamp
PY := python3

.PHONY: all pdf md figs clean
all: pdf md

figs: figs/.stamp
figs/.stamp: figs.py
	$(PY) figs.py
	@touch figs/.stamp

tutorial.pdf: tutorial.tex figs/.stamp
	pdflatex -interaction=nonstopmode -halt-on-error tutorial.tex >tutorial.build.log 2>&1 || (cat tutorial.build.log; false)
	pdflatex -interaction=nonstopmode -halt-on-error tutorial.tex >tutorial.build.log 2>&1 || (cat tutorial.build.log; false)
	@rm -f tutorial.aux tutorial.log tutorial.out tutorial.build.log
	@echo "built tutorial.pdf"

tutorial.md: tutorial.tex tex2md.py figs/.stamp
	$(PY) tex2md.py tutorial.tex tutorial.md

pdf: tutorial.pdf
md: tutorial.md

clean:
	rm -f tutorial.aux tutorial.log tutorial.out tutorial.build.log figs/.stamp
