NOTEBOOKS = $(basename $(wildcard *.tex))

.PHONY: all
all: # None for building the thesis

.PHONY: notebooks
notebooks: $(addsuffix .pdf, $(NOTEBOOKS))

%.pdf: %.tex
	latexmk $<

.PHONY: clean
clean:
	ls -F1 \
		| grep '.tex$$' \
		&& latexmk -c \
		|| true
	rm -fv *.tex
	rm -fv *.xdv *.bbl *.brf
	rm -rfv _minted-*

.PHONY: distclean
distclean: clean
	rm -rfv *_files
	rm -fv *.pdf
