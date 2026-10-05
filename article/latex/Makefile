.PHONY: all evidence

all: paper.pdf

EVIDENCE_TABLES := $(wildcard aistats_*.tex)

# Offline: consumes retained JSON snapshots, with no training or network access.
evidence:
	cd ../.. && .venv/bin/python inference/aistats_revision_evidence.py

paper.pdf paper_evidence_only.pdf: %.pdf: %.tex refs.bib aistats2027.sty fancyhdr.sty $(EVIDENCE_TABLES) fig_architecture.pdf fig_aistats_historical_curves.pdf fig_aistats_mid_memory.pdf
	pdflatex -interaction=nonstopmode -halt-on-error $<
	bibtex $*
	pdflatex -interaction=nonstopmode -halt-on-error $<
	pdflatex -interaction=nonstopmode -halt-on-error $<
	pdflatex -interaction=nonstopmode -halt-on-error $<
