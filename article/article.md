---
abstract: 'In this project, we bla bla bla Our demo is avaliable on Github on this link: [https://github.com/vodkolav/MedMamba](https://github.com/vodkolav/MedMamba)'
author:
 - '[Shany Herskovits](mailto:shanyh@gmail.com)'
 - '[Ofir Nahshon](mailto:ofiranava@gmail.com)'
 - '[Michael Berger](mailto:michael.berger.e@gmail.com)'
bibliography: references.bib # bibliography to use for resolving references
csl: https://www.zotero.org/styles/chicago-note-bibliography
date: 18 March 2025
keywords: # list of keywords to be included in HTML, PDF, ODT, pptx, docx and AsciiDoc metadata; repeat as for author, above
lang: en-US

references:
  - id: yue2024medmamba
    author: 'Yue et. al`'
    citation-key: yue2024medmamba
    title: 'MedMamba: Vision Mamba for Medical Image Classification'
    type: article
    year: '2024'

title: 'MedMamba: Toward Efficient and Accurate Medical Image Classification'
subtitle: 'With Rejection'
toc: true

# LaTeX
abstract-title: 'Abstract'
beamerarticle: # produce an article from Beamer slides
classoption: # option for document class, e.g. oneside (a list).
documentclass: scrartcl # document class: usually one of the standard classes, article, book, and report; the KOMA-Script equivalents, scrartcl, scrbook, and scrreprt, which default to smaller margins; or memoir
geometry: # option for geometry package, e.g. margin=1in; repeat for multiple options:
header-includes: # contents specified by -H/--include-in-header (may have multiple values)
  - |
    ```{=latex}
    \raggedbottom % or \flushbottom
    ```
  - |
    ```{=latex}
    % keep figures where there are in the text
    \usepackage{float} 
    \floatplacement{figure}{H}
    ```
  - |
    ```{=latex}
    % add custom hyphentation rules
    \hyphenation
    {%
      Hyphenate-me-like-this
      Dontyoueverhyphenateme
    }%
    ```
hyperrefoptions: # option for hyperref package, e.g. linktoc=all; repeat for multiple options:
include-before: # contents specified by -B/--include-before-body (may have multiple values)
include-after: # contents specified by -A/--include-after-body (may have multiple values)
indent: # if true, pandoc will use document class settings for indentation (the default LaTeX template otherwise removes indentation and adds space between paragraphs)
linestretch: # adjusts line spacing using the setspace package, e.g. 1.25, 1.5
#lof: true
#lot: true
pagestyle: # control \pagestyle{}: the default article class supports plain (default), empty (no running heads or page numbers), and headings (section titles in running heads)
papersize: # paper size, e.g. letter, a4
secnumdepth: # numbering depth for sections (with --number-sections option or numbersections variable)
toc-depth: 1
toc-title: 'Contents'

# Fonts
fontenc: # allows font encoding to be specified through fontenc package (with pdflatex); default is T1 (see LaTeX font encodings guide)
fontfamily: # font package for use with pdflatex: TeX Live includes many options, documented in the LaTeX Font Catalogue. The default is Latin Modern.
fontfamilyoptions: # options for package used as fontfamily; repeat for multiple options.
fontsize: # font size for body text. The standard classes allow 10pt, 11pt, and 12pt. To use another size, set documentclass to one of the KOMA-Script classes, such as scrartcl or scrbook.
mainfont:
sansfont:
monofont:
mathfont:
mainfontoptions:
sansfontoptions:
monofontoptions:
mathfontoptions:

# Word
category: # document category, included in docx and pptx metadata
description: # document description, included in ODT, docx and pptx metadata. Some applications show this as Comments metadata.
subject: # document subject, included in ODT, PDF, docx, EPUB, and pptx metadata
---



# Introduction

Content  
this is how you make citations: MedMamba[@yue2024medmamba]  
the yue2024medmamba part must be an id in references in frontmatter block above 

# Problem description 
Content

# Data Description
Content

# Preprocessing
Content

# Base model description 
Content   
then inline formula:   $l/2 Hz$ 

Embedding images:  
![Embedded image description](../Equation_plot3.png)


big formula, will be centered on page:

$$\arg\min_{G_\theta} \|v_k - G(z_k, k, \mathcal{F}_{\text{enc}}(X_l); \theta)\|_2^2$$

another formula example:

$$v_t = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1 - \bar{\alpha}_t} x_0$$

To put greek letters in text, just use inline formula : $\epsilon$ and $x_0$. 

# Rejection model description 
Content

# Model parameters and how it's trained
Content

# Experiments
Content

# Results
Content  
a markdown table: 

| Model      |     LSD    | 
|------------|------------|
| GT-Mel     |  0.61      | 
| Unprocessed| 1.99-4.25  | 
| NVSR-DNN   | 1.13- 1.67 | 
| NVSR-ResUNet | 1.7- 0.95| 
| AUDIOSR    | 0.99- 0.73|
| AUDIOSR + Noise (our model) | 2 - 1.2| 


# Conclusion
Content

# References

::: {#refs}
:::