INCFILE = subfiles.txt
TEXFILES = $(wildcard *.tex)
SUBFILES = $(basename $(TEXFILES))

all: $(INCFILE)

.PHONY: $(INCFILE)
$(INCFILE): $(TEXFILES)
	ls *.tex \
		| sort -n \
		| sed "s/\(.*\)\.tex/\\\\subfile{$(shell basename $(CURDIR))\/\1}/" \
		> $@

.PHONY: subfiles
subfiles: $(addsuffix .pdf, $(SUBFILES))

.PHONY: clean
clean:
	latexmk -c
	rm -fv *.xdv *.bbl *.brf *.aux *.run.xml xelatex*.fls
	rm -rfv _minted-*

.PHONY: distclean
distclean: clean
	rm -fv $(INCFILE)
	rm -fv *.pdf

