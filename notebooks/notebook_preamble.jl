using Plots, LaTeXStrings
using Unitful, Measurements
using LinearAlgebra, Arpack, QuantumOptics

import PGFPlots: pushPGFPlotsPreamble, popPGFPlotsPreamble
popPGFPlotsPreamble() # If reevaluating, so no duplicates
pushPGFPlotsPreamble("
    \\usepackage{siunitx}
    \\usepackage[semibold,osf]{libertinus}
    \\usepackage[scr=boondoxo,cal=esstix]{mathalfa}
    \\usepackage{bm}
    \\input{latexdefs}
    ")
pgfplots();

using PlotThemes
theme(:vibrant,
    size=(400, 300),
    dpi=300,
    titlefontsize=12,
    tickfontsize=12,
    legendfontsize=12,
)

