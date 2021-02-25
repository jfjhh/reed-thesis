module TwoSpinJumps

using LaTeXStrings
using LinearAlgebra, QuantumOptics

include("TransverseIsingModels.jl")
using .TransverseIsingModels
TIM = TransverseIsingModels;

using SymPy

⊗ₖ = kron;
export ⊗ₖ

const σ0 = [1 0; 0 1];
const σx = [0 1; 1 0];
const σy = [0 -im; im 0];
const σz = [1 0; 0 -1];
const σp = [0 1; 0 0];
const σm = [0 0; 1 0];

export σ0, σx, σy, σz, σp, σm

function symeigen(H)
    symeig = H.eigenvects()
    vals, vecs = eltype(H)[], []
    for (λ, _, vs) in symeig
        for v in vs
            push!(vals, λ)
            push!(vecs, vec(v))
        end
    end
    vals, vecs
end;

function addentry!(dict, key, value; isequal=isequal)
    for k in keys(dict)
        if isequal(k, key)
            push!(dict[k], value)
            return dict
        end
    end
    dict[key] = [value]
    dict
end;

firstvalue(i, (x, y)) = x
lastvalue(i,  (x, y)) = y
function dictby(A; isequal=isequal, keyof=firstvalue, valof=lastvalue)
    i0, x0 = 1, first(A)
    k0, v0 = keyof(i0, x0), valof(i0, x0)
    dict = Dict(k0 => typeof(v0)[])
    dict = Dict()
    for (i, x) in enumerate(A)
        k, v = keyof(i, x), valof(i, x)
        addentry!(dict, k, v, isequal=isequal)
    end
    dict
end;

import QuantumOpticsBase.projector
projector(ψ::Vector{Sym}) = ψ * ψ';

function symbolicjumps(vals, vecs; combine=false)
    eigendict = dictby(zip(vals, vecs))
    ωs = dictby(((E2 - E1, (E1, E2)) for E1 in keys(eigendict) for E2 in keys(eigendict)))
    As = [σx ⊗ₖ σ0,  σy ⊗ₖ σ0,  σz ⊗ₖ σ0, σ0 ⊗ₖ σx,  σ0 ⊗ₖ σy,  σ0 ⊗ₖ σz]
    Πs = TIM.projectors(eigendict)
    Jωs = dictmap(ΔEs -> filter(!iszero, [simplify.(sum(Πs[E1]*A*Πs[E2] for (E1, E2) in ΔEs)) for A in As]), ωs)
    combine ? dictmap(combinejumps, Jωs) : Jωs
end;

isnegation(s) = (isreal(s) && s < 0) || (s.func.__name__ == "Mul" && isreal(s.args[1]) && s.args[1] < 0);

function spinop(l, r, i)
    if l == 0 && r == 0
        L"\sigma_%$i^+ \sigma_%$i^-"
    elseif l == 0 && r == 1
        L"\sigma_%$i^+"
    elseif l == 1 && r == 0
        L"\sigma_%$i^-"
    elseif l == 1 && r == 1
        L"\sigma_%$i^- \sigma_%$i^+"
    end
end;

function jumplatex(J)
    _, s = mapreduce(((nx, x), (ny, y)) -> (nx, latexstring(x, ny ? "" : " + ", y)),
        CartesianIndices(J)[J .!= 0]) do I
        x = J[I]
        lx = sympy.latex(x)
        i, j = Tuple(I - CartesianIndex(1, 1))
        c = x.func.__name__ == "Add" ? "\\left(" * lx * "\\right)" : lx
        term = latexstring(L"%$c \; ", spinop(i÷2, j÷2, 1), spinop(i%2, j%2, 2))
        isnegation(x), term
    end
    s
end;

@vars s1p=>"σ₁⁺" s1m=>"σ₁⁻" commutative=false
@vars s2p=>"σ₂⁺" s2m=>"σ₂⁻" commutative=false
@vars s1x=>"σ₁ˣ" s2x=>"σ₂ˣ" commutative=false
@vars s1y=>"σ₁ʸ" s2y=>"σ₂ʸ" commutative=false
@vars s1z=>"σ₁ᶻ" s2z=>"σ₂ᶻ" commutative=false
@vars n1=>"n₁" N1=>"ñ₁" commutative=false
@vars n2=>"n₂" N2=>"ñ₂" commutative=false

export s1p, s1m, s2p, s2m, s1x, s2x, s1y, s2y, s1z, s2z
export n1, N1, n2, N2
const spinops = [s1p, s1m, s2p, s2m, s1x, s2x, s1y, s2y, s1z, s2z, n1, N1, n2, N2]

# function symspinop(l, r, i)
#     if l == 0 && r == 0
#         _symspinop(i, :+) * _symspinop(i, :-)
#     elseif l == 0 && r == 1
#         _symspinop(i, :+)
#     elseif l == 1 && r == 0
#         _symspinop(i, :-)
#     elseif l == 1 && r == 1
#         1 - _symspinop(i, :+) * _symspinop(i, :-)
#     end
# end;

import PyCall
PyCall.pyimport_conda("sympy.physics.paulialgebra", "sympy")
import_from(sympy.physics.paulialgebra.Pauli)
const Pauli = sympy.physics.paulialgebra.Pauli

# _symspinop(i, pm) = [s1p s1m; s2p s2m][i, pm == :+ ? 1 : 2]

# function symspinop(l, r, i)
#     if l == 0 && r == 0
#         (1 + Pauli(3, "σ_$(i)^")) / 2
#     elseif l == 0 && r == 1
#         _symspinop(i, :+)
#         (Pauli(1, "σ_$(i)^") + im*Pauli(2, "σ_$(i)^")) / 2
#     elseif l == 1 && r == 0
#         (Pauli(1, "σ_$(i)^") - im*Pauli(2, "σ_$(i)^")) / 2
#     elseif l == 1 && r == 1
#         (1 - Pauli(3, "σ_$(i)^")) / 2
#     end
# end;

const _symspinop = cat([n1 s1p; s1m N1], [n2 s2p; s2m N2], dims=3)
const _dummy_spinop = Dict(s => sympy.Dummy(s.name) for s in spinops)

symspinop(l, r, i) = _dummy_spinop[_symspinop[l+1, r+1, i]]

function polynomial_coefficients(s, vars)
    if s.func.__name__ != "Add"
        throw(ArgumentError("Input expression is not a polynomial"))
    end
    c = map(s.args) do term
        if term.func.__name__ == "Mul"
            prod(filter(!in(vars), term.args))
        elseif term.func.__name__ == "Symbol"
            term ∈ vars ? Sym(1) : term
        else
            throw(ArgumentError("Input expression is not a polynomial"))
        end
    end
    unique(c)
end

const _collect_ops = [_dummy_spinop[a]*_dummy_spinop[b]
                      for a in [s1p, s1m, n1, N1]
                      for b in [s2p, s2m, n2, N2]]

function jumpsimplify(J)
    s = mapreduce(+, CartesianIndices(J)[J .!= 0]) do I
        x = J[I]
        i, j = Tuple(I - CartesianIndex(1, 1))
        x * symspinop(i÷2, j÷2, 1) * symspinop(i%2, j%2, 2)
    end
    s = subs(expand(s), √(g^2 + 1) => d)
    for op in _collect_ops
        s = s.collect(op)
    end
    s.simplify()
end;

commutative_operators(J) = J.xreplace(Dict(s => sympy.Dummy(s.name)
                                           for s in free_symbols(J))).simplify()

function polynomial_collect(s, xs)
    for x in xs
        s = s.collect(x)
    end
    s
end

displayjump(J) = (display(J); display(jumplatex(J)));

function incentry!(dict, key; isequal=isequal)
    for k in keys(dict)
        if isequal(k, key)
            dict[k] += 1
            return dict
        end
    end
    dict[key] = 1
    dict
end;

function combinejumps(Js)
    d = Dict()
    for J in Js
        incentry!(d, J)
    end
    [√(Sym(N))*J for (J, N) in d]
end;

@vars g d real=true
export g, d

H = -2*(σx ⊗ₖ σx) - g*(σz ⊗ₖ σ0 + σ0 ⊗ₖ σz)

vals, vecs = symeigen(H);

Jωs = symbolicjumps(vals, vecs; combine=true)
export Jωs

using PyCall

qu = pyimport("qutip");
b = qu.Bloch();
export qu, b

function add_jump!(b, J, ψ, g0)
    ϕ = map(SymPy.N ∘ subs(g => g0), J * ψ)
    ρ1 = ptrace(qoket(ϕ), 2)
    B = [qu.ket([0,0]), qu.ket([0,1]), qu.ket([1,0]), qu.ket([1,1])]
    s = sum(x*y for (x, y) in zip(B, ϕ))
    b.add_states(s, "points")
    b
end;

using QuantumOptics
qoket(ψ) = Ket(SpinBasis(1//2) ⊗ SpinBasis(1//2), ψ[[1,3,2,4]])
plainket(ψ) = ψ[[1,3,2,4]];

using GLMakie
using CoordinateTransformations

function blochpoint(ψ)
    ψ1, ψ2 = ψ[1], ψ[2]
    α = angle(ψ1)
    phase = exp(-im*α)
    ψ1 *= phase
    ψ2 *= phase
    ϕ = real(angle(ψ2))
    θ = real(2*acos(ψ1))
    CartesianFromSpherical()(Spherical(1, θ, ϕ))
end

plotblochpoint!(f, ψ; kwargs...) = mesh!(f, Sphere(Point3f0(blochpoint(ψ)...), 0.05f0), shading = true; kwargs...);
export plotblochpoint!

plotblochsphere!(f; kwargs...) = mesh(f, Sphere(Point3f0(0), 1f0), color=RGBA(1.,1.,1.,0.5), shading=true, transparency=true; kwargs...)
export plotblochsphere

function blochballpoint(ρ::Operator)
    vals, vecs = eigen(ρ.data)
    sum(p * blochpoint(vecs[:,i]) for (i, p) in enumerate(vals))
end;

blochballpoint(ψ::Vector) = blochballpoint(ptrace(qoket(ψ), 2));
export blochballpoint

function jumppoint(J, ψ, g0)
    ϕ = map(SymPy.N ∘ subs(g => g0), J * ψ)
    ϕ /= norm(ϕ)
    blochballpoint(ϕ)
end;
export jumppoint

arrow!(tail, head; kwargs...) = arrows!((x -> [x]).(tail)..., (x -> [x]).(head - tail)..., arrowsize=0.1, kwargs...);
export arrow!

# function jump_function(J)
#     entries = convert.(Expr, J)
# end

end

