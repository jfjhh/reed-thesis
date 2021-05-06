PD = pandoc
NBC = jupyter-nbconvert
NBTEMPLATE = template.tplx
TEXDIR = tex
MINTED_FILTER = minted.py
SUBDIRS = tex
CLEANDIRS = $(SUBDIRS:%=clean-%)
DISTCLEANDIRS = $(SUBDIRS:%=distclean-%)
MDS=$(basename $(wildcard *.md))

.PHONY: all
all: $(SUBDIRS) notebook

.PHONY: edits
edits: # Convenience
	JULIA_PROJECT="." JULIA_NUM_THREADS=`nproc` jupyter notebook

.PHONY: subdirs $(SUBDIRS)
$(SUBDIRS): notebook
	$(MAKE) -C $@

.PHONY: notebook
notebook: gentex

.PHONY: gentex
gentex: $(MDS:%=tex/%.tex)

%.ipynb: %.md
	jupytext --to notebook $<

tex/%.tex: %.ipynb $(NBTEMPLATE)
	$(NBC) \
		--to latex \
		--output-dir=$(TEXDIR) \
		--TagRemovePreprocessor.remove_cell_tags "hide" \
		$<

.PHONY: subdirs $(CLEANDIRS)
.PHONY: clean
clean: $(CLEANDIRS)
$(CLEANDIRS):
	$(MAKE) -C $(@:clean-%=%) clean

.PHONY: subdirs $(DISTCLEANDIRS)
.PHONY: distclean
distclean: clean $(DISTCLEANDIRS)
	rm -fv *.ipynb

$(DISTCLEANDIRS):
	$(MAKE) -C $(@:distclean-%=%) distclean
