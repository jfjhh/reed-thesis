using Plots, LaTeXStrings
using Unitful, Measurements
using LinearAlgebra, Arpack, QuantumOptics

# import PGFPlots: pushPGFPlotsPreamble, popPGFPlotsPreamble
# popPGFPlotsPreamble() # If reevaluating, so no duplicates
# pushPGFPlotsPreamble("
#     \\usepackage{amsmath}
#     \\usepackage{physics}
#     \\usepackage{siunitx}
#     \\usepackage[full]{textcomp} % to get the right copyright, etc.
#     \\usepackage{libertinus-otf}
#     \\usepackage[scaled=.95,type1]{cabin} % sans serif in style of Gill Sans
#     \\usepackage[T1]{fontenc} % LY1 also works
#     \\setmainfont[Numbers={OldStyle,Proportional}]{fbb}
#     \\usepackage[supstfm=fbb-Regular-sup-t1]{superiors}
#     \\usepackage[cal=boondoxo,bb=boondox,frak=boondox]{mathalfa}
#     \\input{latexdefs}
#     ")
# pgfplots()

import PGFPlotsX
# If reevaluating, so no duplicates
!isempty(PGFPlotsX.CUSTOM_PREAMBLE) && pop!(PGFPlotsX.CUSTOM_PREAMBLE)
push!(PGFPlotsX.CUSTOM_PREAMBLE, "
    \\usepackage{amsmath}
    \\usepackage{physics}
    \\usepackage{siunitx}
    \\usepackage[full]{textcomp} % to get the right copyright, etc.
    \\usepackage{libertinus-otf}
    \\usepackage[scaled=.95,type1]{cabin} % sans serif in style of Gill Sans
    \\usepackage[T1]{fontenc} % LY1 also works
    \\setmainfont[Numbers={OldStyle,Proportional}]{fbb}
    \\usepackage[supstfm=fbb-Regular-sup-t1]{superiors}
    \\usepackage[cal=boondoxo,bb=boondox,frak=boondox]{mathalfa}
    \\input{$(pwd())/latexdefs.tex}
    ");
pgfplotsx()

using PlotThemes
theme(:vibrant,
    size=(400, 300),
    dpi=300,
    titlefontsize=12,
    tickfontsize=11,
    legendfontsize=11,
)
