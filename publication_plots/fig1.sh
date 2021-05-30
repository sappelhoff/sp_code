#!/bin/sh

# Tested under Ubuntu 18.04
# Needs pdfjam and pdfcrop, available from texlive distribution
# Needs also pdftk, qpdf, and exiftool to strip metadata from the resulting pdf, see: https://gist.github.com/hubgit/6078384
# Needs also cpdf, available here: https://community.coherentpdf.com/ --> free academic use is covered by the license
# Needs also "convert" from imagemagick

set -eu # if subsequent commands fail, set -eu will cause the shell script to exit immediately

# make sure there is a fig1a.pdf in this directory, as exported with Libre Office Draw from fig1a.odg

# crop and join pdfs
pdfcrop fig1a.pdf fig1a.pdf

pdfjam --a4paper fig1a.pdf -o fig1a.pdf
pdfjam --a4paper fig1bcd.pdf -o fig1bcd.pdf

# trim: left, bottom, right, top
# put "--frame true" for testing
pdfjam fig1a.pdf fig1bcd.pdf --nup 1x2 -o fig1.pdf --trim '0mm 111mm 0mm 100mm'

# write panel letters into figure
# ... exact positions could be calculated --> use pdfinfo to get page size in pts
cpdf -add-text "a" -pos-left '023 650' -font "Helvetica-Bold" -font-size 17 fig1.pdf -o fig1.pdf

pdfcrop fig1.pdf fig1.pdf


# strip all metadata from the pdf
pdftk fig1.pdf dump_data | sed -e 's/\(InfoValue:\)\s.*/\1\ /g' | pdftk fig1.pdf update_info - output fig1-clean.pdf && mv fig1-clean.pdf fig1.pdf
exiftool -all:all= fig1.pdf
rm fig1.pdf_original
qpdf --linearize fig1.pdf fig1-linearized.pdf && mv fig1-linearized.pdf fig1.pdf

# also provide a PNG version in 300 dpi size
convert -density 300 -units PixelsPerInch fig1.pdf fig1.png

# done!
echo Done!
