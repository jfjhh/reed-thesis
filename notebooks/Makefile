PD = pandoc
NBC = jupyter-nbconvert
NBTEMPLATE = template.tplx
TEXDIR = tex
MINTED_FILTER = minted.py
SUBDIRS = tex
CLEANDIRS = $(SUBDIRS:%=clean-%)
DISTCLEANDIRS = $(SUBDIRS:%=distclean-%)

NOTEBOOKS=$(basename $(wildcard *.ipynb))

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
gentex: $(NOTEBOOKS:%=tex/%.tex)

# --TagRemovePreprocessor.remove_cell_tags='{"hide"}' \

tex/%.tex: %.ipynb $(NBTEMPLATE)
	# Implicitly uses config file and template
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
$(DISTCLEANDIRS):
	$(MAKE) -C $(@:distclean-%=%) distclean
