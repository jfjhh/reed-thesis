module TransverseIsingModels

using LinearAlgebra, Arpack, QuantumOptics

export TransverseIsingModel, ThermalOhmicBath, jumpoperators

# Systems

abstract type System end
hamiltonian(s::System) = undef
eigen(s::System) = eigenstates(dense(hamiltonian(s)))

# Utilities

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

struct TransverseIsingModel <: System
    λ::Number # Currently λ_Striff. Should change to g = -λ_Striff in the future
    N::Int
end

spinbasis(s) = tpow(sb, s.N)
Ispin(s) = identityoperator(spinbasis(s))

# Define the Pauli operators at each site
for (opi, op) in [(:sx,:σx), (:sy,:σy), (:sz,:σz), (:sp,:σp), (:sm,:σm)]
    eval(:(export $opi))
    eval(:($opi()  = $op))
    eval(:($opi(s) = $op))
    eval(:($opi(s, i) = embed(spinbasis(s), (i-1)%s.N + 1, $op)))
end

Hspin(s) = -sum(-s.λ*sz(s, i) + sx(s, i)*sx(s, i+1) for i in 1:s.N)

hamiltonian(s::TransverseIsingModel) = Hspin(s)

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

# TODO: Verify constants, polariztion, etc. for magnetic 3D spin interaction
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

function addentry!(dict, key, value, isequal=isequal)
    for k in keys(dict)
        if isequal(k, key)
            push!(dict[k], value)
            return dict
        end
    end
    dict[key] = [value]
    dict
end

function eigendict(s::System)
    d = Dict()
    for (energy, state) in zip(eigen(s)...)
        addentry!(d, energy, state, isapprox)
    end
    d
end

function energydiffs(eigdict)
    d = Dict()
    for E1 in keys(eigdict), E2 in keys(eigdict)
        addentry!(d, E2 - E1, (E1, E2), ≈)
    end
    d
end

function projectors(eigdict)
    Dict(energy => sum(projector(state) for state in eigstates)
         for (energy, eigstates) in eigdict)
end

struct SystemInteraction{SO <: AbstractOperator}
    sys::System
    As::AbstractVector{SO}
end

# TODO: Working in the spin basis takes too much memory, so figure out how to
# automatically do sparse transformations to the energy eigenbasis.
function jumpoperators(si::SystemInteraction, bath::Bath)
    opdict = Dict()
    rates = []
    eigdict = eigendict(si.sys)
    Πs = projectors(eigdict)
    ωs = energydiffs(eigdict)

    for A in si.As, (ω, Ediffs) in ωs
        Aω = sum(Πs[E1] * A * Πs[E2] for (E1, E2) in Ediffs)
        if tracenorm(Aω) > √(eps(1.0))
            addentry!(opdict, ω, Aω, isapprox)
            push!(rates, γ(bath, ω))
        end
    end
    opdict, rates
end

function jumpoperators(s::TransverseIsingModel, b::Bath)
    As = Iterators.flatten([sx(s, i), sy(s, i), sz(s, i)] for i in 1:s.N)
    jumpoperators(SystemInteraction(s, collect(As)), b)
end

end

