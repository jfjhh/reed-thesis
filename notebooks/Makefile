PD = pandoc
NBC = jupyter-nbconvert
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
	jupyter notebook

.PHONY: subdirs $(SUBDIRS)
$(SUBDIRS): notebook
	$(MAKE) -C $@

.PHONY: notebook
notebook: gentex

.PHONY: gentex
gentex: $(NOTEBOOKS:%=tex/%.tex)

tex/%.tex: %.ipynb
	# Implicity uses config file and template
	$(NBC) \
		--to latex \
		--output-dir=$(TEXDIR) \
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
