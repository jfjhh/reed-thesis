---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Julia 1.6.0
    language: julia
    name: julia-1.6
---

```julia tags=["hide"]
include("notebook_preamble.jl");
```

```julia caption="Elementary excitation spectra of the transverse-field Ising model across $g$." label="fig:ising-modes"
isings = [1.5, 1.0, 0.5]'
plot([k -> √(1 + g^2 + 2g*cos(π - k)) for g in isings], xlim=(-1.0π, 1.0π),
    legendtitle=L"g", label=[L"%$g" for g in isings], legend=:bottomright,
    xlabel=L"\pi - k", ylabel=L"E_k")
```

\begin{figure}[H]
\begin{center}
\includegraphics[width=0.75\linewidth]{../../figs/ising-modes}
\caption{%
Elementary excitation spectra of the transverse-field Ising model across $g$.
}\label{fig:ising-modes}
\end{center}
\end{figure}
