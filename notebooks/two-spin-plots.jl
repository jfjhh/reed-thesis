module TwoSpinPlots

using Reexport

@reexport using GLMakie
@reexport using Colors
@reexport using LinearAlgebra

include("two-spin-jumps.jl")
@reexport using .TwoSpinJumps
TSJ = TwoSpinJumps;
export TSJ

function jumpplot!(f, g0)
    ψ = rand(4) + im*rand(4)
    ψ /= norm(ψ)
    b0 = blochballpoint(ψ)

    TSJ.plotblochsphere!(f)
    TSJ.plotblochpoint!(f, [1, 0])
    TSJ.plotblochpoint!(f, [0, 1])
    for Js in values(Jωs)
        for J in Js
            head(i) = lift(g0) do g
                b1 = jumppoint(J, ψ, g)
                [(b1 - b0)[i]]
            end
            arrows!(f, (x -> [x]).(b0)..., head(1), head(2), head(3), arrowsize=0.1);
        end
    end
end

function jumpfigure()
    f = Figure()
    ls = labelslider!(f.scene, "g", 0.01:0.01:10; format=x -> "$(round(x, digits=3))")
    set_close_to!(ls.slider, 1.0)
    f[2,1] = ls.layout
    jumpplot!(f[1,1], ls.slider.value)
    f
end

export jumpfigure

end

