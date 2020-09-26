ENTRIES = $(basename $(wildcard *.tex))

.PHONY: all
all: notebook

.PHONY: notebook
notebook: entries

.PHONY: entries
entries: $(addsuffix .pdf, $(ENTRIES))

%.pdf: %.tex
	latexmk $<

.PHONY: clean
clean:
	ls -F1 \
		| grep '.tex$$' \
		&& latexmk -c \
		|| true
	rm -fv *.tex
	rm -fv *.xdv
	rm -rfv _minted-*

.PHONY: distclean
distclean: clean
	rm -rfv *_files
	rm -fv *.pdf