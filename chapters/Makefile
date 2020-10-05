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

.PHONY: HEAD.out
HEAD.out:
	echo 'Previous commit: ' > $@
	git rev-list --format=%s --max-count=1 HEAD | tail -1 >> $@
	echo '\\' >> $@
	echo "\url{https://github.com/jfjhh/reed-thesis/commit/`git rev-parse --short HEAD`}" >> $@
	echo '' >> $@
	echo 'Current diff:' >> $@
	echo '\vspace{-\baselineskip}' >> $@
	echo '\begin{minted}{text}' >> $@
	git diff --stat --stat-width=80 HEAD >> $@
	echo '\end{minted}' >> $@

%.pdf: %.tex HEAD.out
	latexmk $<

.PHONY: clean
clean:
	latexmk -c
	rm -fv *.xdv *.bbl *.brf
	rm -rfv _minted-*

.PHONY: distclean
distclean: clean
	rm -fv $(INCFILE)
	rm -fv *.pdf

