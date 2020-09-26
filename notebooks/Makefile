PD = pandoc
NBC = jupyter-nbconvert
PYDIR = src
TEXDIR = tex
MINTED_FILTER = minted.py
TEMPLATE = template.tex
SUBDIRS = src tex
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
notebook: gensrc gentex

.PHONY: gensrc
gensrc: $(NOTEBOOKS:%=src/%.py)

src/%.py: %.ipynb
	$(NBC) \
		--to python \
		--no-prompt \
		--output-dir=$(PYDIR) \
		$<

.PHONY: gentex
gentex: $(NOTEBOOKS:%=tex/%.tex)

tex/%.tex: %.ipynb
	$(NBC) \
		--to markdown \
		--output-dir=$(TEXDIR) \
		$<
	$(PD) \
		$(TEXDIR)/$*.md \
		-o $@ \
		--template=$(TEMPLATE) \
		-f markdown-implicit_figures \
		-t latex \
		--no-highlight \
		--filter $(MINTED_FILTER) \
		--verbose
	rm -fv $(TEXDIR)/$*.md

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