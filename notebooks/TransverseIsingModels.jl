module TransverseIsingModels

using LinearAlgebra, Arpack, QuantumOptics

export TransverseIsingModel, ThermalOhmicBath, jumpoperators


# Systems
#
# TODO: System functions like `hamiltonian`, `sx`, etc. are not type-stable. A
# solution would be to have the number of spins as a type parameter and make
# `tpow` type-stable.
abstract type System end
hamiltonian(s::System) = undef
basis(s::System) = undef


# Eigenbases

struct EigenBasis <: Basis
    shape::Vector{Int}
    system::System
    EigenBasis(system) = new([prod(basis(system).shape)], system)
end

eigenbasis(s) = EigenBasis(s)

Base.:(==)(b1::EigenBasis, b2::EigenBasis) = (b1.shape == b2.shape) && (b1.system == b2.system)

basiseigen(s) = basiseigen(s, eigenbasis(s))

function basiseigen(s, b)
    H = hamiltonian(s)
    bl, br = H.basis_l, H.basis_r
    vals, vecs = eigen(dense(H).data)
    _basiseigen(vals, vecs, bl, br, b)
end

function _basiseigen(vals, vecs, bl, br, b::Basis)
    P, Pinv = identityoperator(b, br), identityoperator(bl, b)
    kets = [Ket(b, vecs[:, i]) for i in eachindex(vals)]
    vals, kets, P, Pinv
end

function _basiseigen(vals, vecs, bl, br, b::EigenBasis)
    P, Pinv = Operator(b, br, inv(vecs)), Operator(bl, b, vecs)
    kets = [basisstate(b, i, sparse=true) for i in eachindex(vals)]
    vals, kets, P, Pinv
end


# Utilities

# This is slow.
sparsify(A, atol=sqrt(eps(1.0))) = sparse([abs(x) > atol ? x : zero(x) for x in A])

function sparsify(op::Operator)
    bl, br = op.basis_l, op.basis_r
    Operator(bl, br, sparsify(op.data))
end

trnorm(op) = tracenorm(dense(op))
comm(x, y) = x*y - y*x
acomm(x, y) = x*y + y*x

function acomm_table(a, b, N)
    A = Matrix{Float64}(undef, N, N)
    Threads.@threads for i in 1:N
        for j in i:N
            A[j, i] = trnorm(acomm(a(i), b(j))) / 2^N
        end
    end
    LowerTriangular(A)
end

function acomm_tables(f, N=Int(log2(size(f(1))[1])); show=true, atol=1e-12)
    ft = dagger ∘ f
    acomms = [
        acomm_table(ft, f, N),
        acomm_table(f, f, N),
        acomm_table(ft, ft, N)
    ]
    if show
        for acomm in acomms
            display(acomm)
        end
    end
    fermion_acomms = [
        one(acomms[1]),
        zero(acomms[2]),
        zero(acomms[3])
    ]
    all(@. norm(acomms - fermion_acomms) < atol)
end

acomm_tables(s::System, f; kwargs...) = acomm_tables(i -> f(s, i), s.N; kwargs...)
fermionic(args...; kwargs...) = acomm_tables(args...; kwargs..., show=false)


# Spins

const sb = SpinBasis(1//2)
const σx, σy, σz = sigmax(sb), sigmay(sb), sigmaz(sb)
const σp, σm = sigmap(sb), sigmam(sb)

"`tpow(x, n)` takes the tensor product of `x` with itself `n` times."
tpow(x, n) = ⊗(repeat([x], n)...)

abstract type SpinSystem <: System end

spinbasis(s::SpinSystem) = tpow(sb, s.N)
Ispin(s::SpinSystem) = identityoperator(spinbasis(s))
basis(s::SpinSystem) = spinbasis(s)

struct TransverseIsingModel <: SpinSystem
    N::Int
    λ::Number # Currently λ_Striff. Should change to g = -λ_Striff in the future
end

# Define the Pauli operators at each site
for (opi, op) in [(:sx,:σx), (:sy,:σy), (:sz,:σz), (:sp,:σp), (:sm,:σm)]
    eval(:(export $opi))
    eval(:($opi()  = $op))
    eval(:($opi(s) = $op))
    eval(:($opi(s, i) = embed(spinbasis(s), (i-1)%s.N + 1, $op)))
end

Hspin(s) = -sum(-s.λ*sz(s, i) + sx(s, i)*sx(s, i+1) for i in 1:s.N)

hamiltonian(s::TransverseIsingModel) = Hspin(s)

# Lower bound. Could be refined or made certain.
ΔEbound(s::TransverseIsingModel) = (1/3) * 3.0^-s.N
isapproxΔE(s::TransverseIsingModel) = (x, y) -> isapprox(x, y, atol=ΔEbound(s))

k(s, m) = 2π*(m-1)/s.N - π*(s.N - (s.N%2))/s.N # for m in 1:s.N

function c(s, i) # for i in 1:s.N
    i = (i-1)%s.N + 1 
    if i == 1
        -sm(s, i)
    else
        prod(-sz(s, j) for j in 1:(i-1)) * -sm(s, i)
    end
end
ct = dagger ∘ c

Lend(s) = sum(ct(s, i)*c(s, i) for i in 1:s.N)
Hend(s) = (ct(s, s.N) - c(s, s.N))*(ct(s, 1) + c(s, 1))*((sparse ∘ exp ∘ dense)(im*π*Lend(s)) + Ispin(s))
Htrans(s) = Hspin(s) - Hend(s)

Hc(s) = Ispin(s)*(s.N * -s.λ) + Hend(s) - sum(
    (2 * -s.λ)*ct(s, i)*c(s, i) + (ct(s, i) - c(s, i))*(ct(s, i+1) + c(s, i+1))
    for i in 1:s.N)

C(s, m) = sum(exp(-im*k(s, m)*i) * c(s, i) for i in 1:s.N) / √s.N
Ct = dagger ∘ C

Hk(s, m) = s.λ ≈ -1 && k(s, m) ≈ -π ? zeros(2, 2) : [
   s.λ - cos(k(s, m))  -im*sin(k(s, m))
   im*sin(k(s, m))     cos(k(s, m)) - s.λ
]
vk(s, m) = [C(s, m); Ct(s, minusk(s, m))]
vkt(s, m) = [Ct(s, m) C(s, minusk(s, m))]
HC(s) = sum((vkt(s, m)*Hk(s, m)*vk(s, m))[1] for m in 1:s.N)

Heigs(s, m) = eigen(Hk(s, m))
E(s, m) = Heigs(s, m).values[2] # Positive energy
η(s, m) = Heigs(s, m).vectors'[2,1] * C(s, m) + Heigs(s, m).vectors'[2,2] * Ct(s, minusk(s, m))
ηt = dagger ∘ η
ηm(s, m) = Heigs(s, m).vectors'[1,1] * C(s, m) + Heigs(s, m).vectors'[1,2] * Ct(s, minusk(s, m))
ηmt = dagger ∘ ηm

E0(s) = -sum(E(s, m) for m in 1:s.N)
Hη(s) = sum(2E(s, m)*ηt(s, m)*η(s, m) for m in 1:s.N) + E0(s)*Ispin(s)

# For tests:
# all(fermionic(sys, f) for f in [c, C, η])


# Many-body basis

import QuantumOpticsBase: ManyBodyBasis, SparseOperator, isnonzero

function destroyfermion(b::ManyBodyBasis, index::Int)
    c = SparseOperator(b)
    # <{m}_j| c |{m}_i>
    for i in 1:length(b)
        occ_i = b.occupations[i]
        if occ_i[index] == 0
            continue
        end
        sign = sum(occ_i[1:(index-1)]) % 2 == 0 ? 1 : -1
        for j in 1:length(b)
            if isnonzero(occ_i, b.occupations[j], index)
                c.data[j, i] = sign * sqrt(occ_i[index])
            end
        end
    end
    c
end

minusk(s, m) = s.N%2 == 0 ? (m == 1 ? 1 : s.N - (m - 2)) : (s.N - (m - 1)) # π ≡ -π

fb(s) = NLevelBasis(s.N) # "Levels" 1 to N are indices of k's
states(s) = fermionstates(fb(s), [0:s.N...])
mbb(s) = ManyBodyBasis(fb(s), states(s))
Imb(s) = identityoperator(mbb(s))
ηmb(s, m) = destroyfermion(mbb(s), m)
ηtmb = dagger ∘ ηmb
Hf(s) = diagonaloperator(fb(s), @. 2E(k(s, 1:s.N)) + E0(s))
Hmb(s) = manybodyoperator(mbb(s), Hf(s))

Cmbs(s, m) = (Heigs(s, m).vectors * [ηtmb(s, minusk(s, m)), ηmb(s, m)])[1]

cmb(s, i) = sum(exp(im*k(s, m)*(i-1)) * Cmbs(s, m) for m in 1:s.N) / √s.N
cmbt = dagger ∘ cmb
sxmb(s, i) = -(i == 1 ? Imb(s) : prod(Imb(s) - 2*cmbt(s, j)*cmb(s, j) for j in 1:(i-1))) * (cmbt(s, i) + cmb(s, i))


# Spectral correlations for an ohmic bath in a thermal state

# TODO: Bath type hierarchy?
# TODO: Composition of interactions, so that the spin-boson implementation may
# be used for σx, σy, σz, and then combined to form one bath related type?

abstract type Bath end

struct ThermalOhmicBath{T} <: Bath
    β::T
    cutoff::T
    rate::T
end

# TODO: Verify damping rate proportionality and other constants
function spectraldensity(bath::ThermalOhmicBath, ω)
    (2*bath.rate / π) * ω / (1 + (ω / bath.cutoff)^2)
end

# TODO: Verify constants, polarization, etc. for magnetic 3D spin interaction
# TODO: Check limit ω->0 case
function γ(bath::ThermalOhmicBath, ω)
    # g = V * (ω^2 / (2π*c)^3) * 8π/3
    g = (ω^2 / (2π)^3) * 8π/3
    if ω == zero(ω)
        g / bath.β
    else
        nB = 1 / (exp(bath.β * ω) - 1)
        g * spectraldensity(bath, ω) * nB
    end
end


# Construction of jump operators

firstvalue(i, (x, y)) = x
lastvalue(i,  (x, y)) = y

dictmap(f, dict) = Dict(key => f(value) for (key, value) in dict)

function dictby(A; isequal=isequal, keyof=firstvalue, valof=lastvalue)
    i0, x0 = 1, first(A)
    k0, v0 = keyof(i0, x0), valof(i0, x0)
    dict = Dict(k0 => typeof(v0)[])
    for (i, x) in enumerate(A)
        k, v = keyof(i, x), valof(i, x)
        if isequal(k, k0)
            push!(dict[k0], v)
        else
            k0 = k
            dict[k0] = [v]
        end
    end
    dict
end

function dictbysort(itr; keyof=firstvalue, valof=lastvalue,
        isequal=isequal, by=x -> x[1], kwargs...)
    A = sort([(keyof(i, x), valof(i, x)) for (i, x) in enumerate(itr)];
        by=by, kwargs...)
    dictby(A, isequal=isequal)
end

# `Es` are guaranteed to be sorted.
eigendict(Es, kets; isequal=isapprox) = dictby(zip(Es, kets), isequal=isequal)

function energydiffs(eigdict; isequal=isapprox)
    Es = keys(eigdict)
    dictbysort(((E2 - E1, (E1, E2)) for E1 in Es for E2 in Es), isequal=isequal)
end

sumprojector(A) = sum(projector(a) for a in A)
projectors(eigdict) = dictmap(sumprojector, eigdict)

function changebasis(As, P, Pinv; sparse=true)
    if sparse
        [sparsify(P * A * Pinv) for A in As]
    else
        [P * A * Pinv for A in As]
    end
end

# This recalculates the eigensystem, so a better way of only calculating this
# once may be necessary to consider larger N.
function changebasis(As, s::System; basis=eigenbasis(s), kwargs...)
    _, _, P, Pinv = basiseigen(s, basis)
    changebasis(As, P, Pinv; kwargs...)
end

interactions(s::TransverseIsingModel) = [op(s, i) for i in 1:s.N, op in [sx, sy, sz]]

# TODO: Make jumpoperators somehow into generators, so that only the full computation
# resulting in just numbers is stored.

jumpoperator(ΔEs, A, Πs) = sum(Πs[E1] * A * Πs[E2] for (E1, E2) in ΔEs)
ωjumpoperators(ΔEs, As, Πs) = [jumpoperator(ΔEs, A, Πs) for A in As]

function jumpoperators(s::SpinSystem, interactions; basis=eigenbasis(s))
    Es, kets, P, Pinv = basiseigen(s, basis)
    approxΔE = isapproxΔE(s)
    eigdict = eigendict(Es, kets, isequal=approxΔE)
    As = changebasis(interactions, P, Pinv)
    Πs = projectors(eigdict)
    ωs = energydiffs(eigdict, isequal=approxΔE)
    # From here on, everything can be lazily computed.
    dictmap(ΔEs -> ωjumpoperators(ΔEs, As, Πs), ωs)
end

jumpoperators(s::SpinSystem; kwargs...) = jumpoperators(s, interactions(s); kwargs...)

function sitejumps(s::SpinSystem; basis=eigenbasis(s))
    Qs = [op(s, i) for i in 1:s.N, op in [sx, sy, sz]]
    changebasis(Qs, s, basis=basis)
end

# These can likely be made more efficient.
opip(A, B) = tr(dagger(A) * B)
opip(A) = tr(dagger(A) * A)
opnorm(A) = √opip(A)
opnormalize(A) = A / opnorm(A)
opcos(A, B) = real(abs(opip(A, B)) / (opnorm(A) * opnorm(B)))

# TODO: Handle zero jump operators differently?

project(P, J) = J == zero(J) ? zero(eltype(J)) : opip(P / opip(P), J)
jumpprojections(Js, Ps) = [project(P, J) for J in Js, P in Ps]
jumpprojections(Jωs::Dict, Ps) = dictmap(Js -> jumpprojections(Js, Ps), Jωs)

jumpcos(P, J) = J == zero(J) ? real(one(eltype(J))) : opcos(P, J)
jumpcosines(Js, Ps) = [jumpcos(P, J) for J in Js, P in Ps]
jumpcosines(Jωs::Dict, Ps) = dictmap(Js -> jumpcosines(Js, Ps), Jωs)

end

