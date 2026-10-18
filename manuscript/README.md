# Manuscript

`main.tex` includes the section sources, bibliography and figures. The current
numerical references are [run #4](../results/run4/README.md) and the
[five-seed sensitivity grid](../results/sensitivity/README.md). Their paper-number
indices identify the saved source cells for the reported numerical results.

Build from this directory with a TeX distribution containing Latin Modern,
BibTeX and the packages used by `main.tex`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The Elsevier class and Harvard bibliography style are included with their
class sources. The figure PDFs are vector artwork; numerical sources and
rendering commands are in [results/figures/run4](../results/figures/run4/README.md)
and [results/sensitivity](../results/sensitivity/README.md).

Revisions made after run #4 remain red through `\edited{...}` or explicit red
blocks. Earlier text is black. The AIL description remains “To be filled.”
for its authors. Author metadata and submission declarations still require
coauthor confirmation; this is a review draft.
