using KernelAbstractions
using Printf
using Plots
default(; lw=4)

include("permeability.jl")

@kernel function update_qD!(qx, Pe, ϕ, k_m, Δρg, dx)
    ix = @index(Global, Linear)
    @inbounds begin
        k_ηf = k_m(0.5 * (ϕ[ix] + ϕ[ix + 1]))
        qx[ix] = k_ηf * ((Pe[ix + 1] - Pe[ix]) / dx + Δρg)
    end
end

@kernel function update_Pf!(Pe, qx, k_m, ϕ, ϕ_bg, η_ϕ0, dx)
    ix = @index(Global, Linear)
    @inbounds begin
        η_ϕ = η_ϕ0 * ϕ_bg / ϕ[ix + 1]
        dτ_β = dx^2 / max(k_m(ϕ[ix]), k_m(ϕ[ix + 1]), k_m(ϕ[ix + 2])) / 3.1
        Pe[ix + 1] += dτ_β * ((qx[ix + 1] - qx[ix]) / dx - Pe[ix + 1] / η_ϕ)
    end
end

@kernel function update_residual!(r_Pe, Pe, qx, ϕ, ϕ_bg, η_ϕ0, dx)
    ix = @index(Global, Linear)
    @inbounds begin
        η_ϕ = η_ϕ0 * ϕ_bg / ϕ[ix + 1]
        r_Pe[ix] = (qx[ix + 1] - qx[ix]) / dx - Pe[ix + 1] / η_ϕ
    end
end

@kernel function update_ϕ!(ϕ, Pe, η_ϕ0, ϕ_bg, dt)
    ix = @index(Global, Linear)
    @inbounds begin
        η_ϕ = η_ϕ0 * ϕ_bg / ϕ[ix]
        ϕ[ix] -= dt * (1 - ϕ[ix]) * Pe[ix] / η_ϕ
    end
end

function porosity_wave_1D(backend=CPU())
    # physics
    lc    = 1.0
    lx    = 100lc
    lw    = 4lc
    ϕ_bg  = 0.01
    ϕA    = 0.1
    Δρg   = 1.0
    η_ϕ0  = 1.0
    tsc   = η_ϕ0 / (Δρg * lc)
    dt    = 1e-3tsc
    k_ηf0 = lc^2 / η_ϕ0
    # switch between different formulations for permeability
    # k_m   = ConstantPermeability(k_ηf0)
    # k_m   = KarmanCozeny(3, k_ηf0 / ϕ_bg^3)
    # k_m   = KarmanCozeny(2, k_ηf0 / ϕ_bg^2)
    # k_m   = KarmanCozeny(1, k_ηf0 / ϕ_bg)
    # k_m   = KarmanCozenyLimited(3, 3, k_ηf0 / ϕ_bg^3)
    k_m(ϕ) = k_ηf0 * (ϕ / ϕ_bg)^3 # Karman-Cozeny model use closure
    # numerics
    nx     = 256
    nt     = 100
    niter  = 50nx^2
    ncheck = ceil(Int, 0.1nx^2)
    nvis   = 5
    ϵtol   = 1e-6
    # preprocessing
    dx = lx / nx
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    # allocate arrays
    ϕ = KernelAbstractions.zeros(backend, Float64, nx)
    Pe = KernelAbstractions.zeros(backend, Float64, nx)
    qx = KernelAbstractions.zeros(backend, Float64, nx - 1)
    r_Pe = KernelAbstractions.zeros(backend, Float64, nx - 2)
    # init
    KernelAbstractions.copyto!(backend, ϕ, @. ϕ_bg + ϕA * exp(-((xc + 8lw) / lw)^2))
    ϕ_ini = copy(ϕ)
    # time loop
    for it in 1:nt
        println("it = $it")
        # iter loop
        for iter in 1:niter
            update_qD!(backend, 256, length(qx))(qx, Pe, ϕ, k_m, Δρg, dx)
            update_Pf!(backend, 256, length(Pe) - 2)(Pe, qx, k_m, ϕ, ϕ_bg, η_ϕ0, dx)
            if iter % ncheck == 0
                update_residual!(backend, 256, length(r_Pe))(r_Pe, Pe, qx, ϕ, ϕ_bg, η_ϕ0, dx)
                err_Pe = maximum(abs.(r_Pe))
                @printf("  iter/nx² = %.1f, err_Pe = %1.3e\n", iter / (nx^2), err_Pe)
                if err_Pe <= ϵtol
                    break
                end
            end
        end
        update_ϕ!(backend, 256, length(ϕ))(ϕ, Pe, η_ϕ0, ϕ_bg, dt)
        if it % nvis == 0
            KernelAbstractions.synchronize(backend)
            p1 = plot([Array(ϕ_ini), Array(ϕ)], xc; title="ϕ", label=false)
            p2 = plot(Array(Pe), xc; title="Pₑ", label=false)
            display(plot(p1, p2; layout=(1, 2), size=(400, 600)))
        end
    end
    return
end

porosity_wave_1D()