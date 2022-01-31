set -eu
pdfcrop fig1-tikz.pdf fig1-tikz.pdf
convert -flatten -density 300 fig1-tikz.pdf fig1-tikz.png
