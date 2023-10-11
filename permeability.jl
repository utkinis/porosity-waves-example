abstract type Permeability end

struct ConstantPermeability{T} <: Permeability
    k0::T
end

(k::ConstantPermeability)(ϕ) = k.k0

struct KarmanCozeny{I,T} <: Permeability
    npow::I
    k0::T
end

(k::KarmanCozeny)(ϕ) = k.k0 * ϕ^k.npow

struct KarmanCozenyLimited{I1,I2,T} <: Permeability
    npow::I1
    mpow::I2
    k0::T
end

(k::KarmanCozenyLimited)(ϕ) = k.k0 * ϕ^k.npow / (1 - ϕ)^k.mpow
