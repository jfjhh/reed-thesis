MAIN = thesis
SUBDIRS = chapters notebooks
CLEANDIRS = $(SUBDIRS:%=clean-%)
DISTCLEANDIRS = $(SUBDIRS:%=distclean-%)

all: $(MAIN).pdf

$(MAIN).pdf: $(SUBDIRS) $(MAIN).tex
	latexmk $(MAIN).tex

.PHONY: subdirs $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@

.PHONY: subdirs $(CLEANDIRS)
$(CLEANDIRS):
	$(MAKE) -C $(@:clean-%=%) clean

.PHONY: clean
clean: $(CLEANDIRS)
	latexmk -c
	rm -fv *.xdv *.bbl *.brf
	rm -rfv _minted-*

.PHONY: distclean
distclean: clean $(DISTCLEANDIRS)

.PHONY: subdirs $(DISTCLEANDIRS)
$(DISTCLEANDIRS):
	$(MAKE) -C $(@:distclean-%=%) distclean