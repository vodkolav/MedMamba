
.PHONY: help all article article-docx article-pdf article-tex presentation presentation-html presentation-pdf presentation-pptx presentation-tex thesis thesis-docx thesis-epub thesis-pdf thesis-tex
.DEFAULT_GOAL := help

help:
	head -2 Makefile

all:
	make article presentation thesis

article-docx:
	(pandoc --defaults=defaults.yaml --defaults=docx.yaml)

article-pdf:
	(pandoc --defaults=defaults.yaml --defaults=pdf.yaml)

article-tex:
	(pandoc --defaults=defaults.yaml --defaults=tex.yaml)

article:
	make article-docx article-pdf article-tex

presentation-html:
	(pandoc --defaults=defaults.yaml --defaults=html.yaml)

presentation-pdf:
	(pandoc --defaults=defaults.yaml --defaults=pdf.yaml)

presentation-pptx:
	(pandoc --defaults=defaults.yaml --defaults=pptx.yaml)

presentation-tex:
	(pandoc --defaults=defaults.yaml --defaults=tex.yaml)

presentation:
	make presentation-html presentation-pdf presentation-pptx presentation-tex

thesis-docx:
	(pandoc --defaults=defaults.yaml --defaults=docx.yaml)

thesis-epub:
	(pandoc --defaults=defaults.yaml --defaults=epub.yaml)

thesis-pdf:
	(pandoc --defaults=defaults.yaml --defaults=pdf.yaml)

thesis-tex:
	(pandoc --defaults=defaults.yaml --defaults=tex.yaml)

thesis:
	make thesis-docx thesis-epub thesis-pdf thesis-tex
