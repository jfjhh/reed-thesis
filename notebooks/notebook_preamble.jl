using Plots, LaTeXStrings
using Unitful, Measurements
using LinearAlgebra, Arpack, QuantumOptics
using LsqFit, Roots
using ThreadsX

import PGFPlotsX
# If reevaluating, so no duplicates
!isempty(PGFPlotsX.CUSTOM_PREAMBLE) && pop!(PGFPlotsX.CUSTOM_PREAMBLE)
push!(PGFPlotsX.CUSTOM_PREAMBLE, "
    \\usepackage{amsmath}
    \\usepackage{physics}
    \\usepackage{siunitx}
    \\usepackage[full]{textcomp} % to get the right copyright, etc.
    \\usepackage[semibold]{libertinus-otf}
    \\usepackage[T1]{fontenc} % LY1 also works
    \\setmainfont[Numbers={OldStyle,Proportional}]{Libertinus Serif}
    \\usepackage[supstfm=libertinesups,supscaled=1.2,raised=-.13em]{superiors}
    \\setmonofont[Scale=MatchLowercase]{JuliaMono} % We need lots of unicode, like ⊗
    \\usepackage[cal=cm,bb=boondox,frak=boondox]{mathalfa}
    \\input{$(pwd())/latexdefs.tex}
    ");
pgfplotsx()

using PlotThemes
_fs = 12
theme(:vibrant,
    size=(400, 300),
    dpi=300,
    titlefontsize = _fs,
    tickfontsize = _fs,
    legendfontsize = _fs,
    guidefontsize = _fs,
    legendtitlefontsize = _fs
)

rubric = RGB(0.7, 0.05, 0.0); # The red color used in the thesis document.

# Pauli matrices
const σ0 = [1 0; 0 1]
const σx = [0 1; 1 0]
const σy = [0 -im; im 0]
const σz = [1 0; 0 -1]
const σp = [0 1; 0 0]
const σm = [0 0; 1 0]

⊗ₖ(a, b) = kron(b, a);
function siteop(A, i, n)
    i = i > 0 ? 1 + ((i - 1) % n) : throw(ArgumentError("Site index must be positive."))
    ops = repeat([one(A)], n)
    ops[i] = A
    reduce(⊗ₖ, ops)
end
