set -eu
pdfcrop fig1-tikz.pdf fig1-tikz.pdf
convert -flatten -density 300 fig1-tikz.pdf fig1-tikz.png
convert -flatten -density 600 -compress lzw -units pixelsperinch -depth 8 fig1-tikz.pdf fig1-tikz.tif
