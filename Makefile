MAIN = thesis
SUBDIRS = chapters notebooks
SUBDOCS = mini-oral oral questions seminar
CLEANDIRS = $(SUBDIRS:%=clean-%)
DISTCLEANDIRS = $(SUBDIRS:%=distclean-%)
CLEANDOCS = $(SUBDOCS:%=clean-%)
DISTCLEANDOCS = $(SUBDOCS:%=distclean-%)

all: $(MAIN).pdf $(SUBDOCS)

$(MAIN).pdf: $(SUBDIRS) $(MAIN).tex
	latexmk $(MAIN).tex

.PHONY: subdirs $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@

.PHONY: subdirs $(CLEANDIRS)
$(CLEANDIRS):
	$(MAKE) -C $(@:clean-%=%) clean

.PHONY: subdocs $(SUBDOCS)
$(SUBDOCS):
	$(MAKE) -C $@

.PHONY: subdocs $(CLEANDOCS)
$(CLEANDOCS):
	$(MAKE) -C $(@:clean-%=%) clean

.PHONY: clean
clean: $(CLEANDIRS)
	latexmk -c
	rm -rfv _minted-*

.PHONY: distclean
distclean: clean $(DISTCLEANDIRS) $(DISTCLEANDOCS)
	latexmk -C

.PHONY: subdirs $(DISTCLEANDIRS)
$(DISTCLEANDIRS):
	$(MAKE) -C $(@:distclean-%=%) distclean

.PHONY: subdocs $(DISTCLEANDOCS)
$(DISTCLEANDOCS):
	$(MAKE) -C $(@:distclean-%=%) distclean

