using KernelAbstractions
using Printf
using Plots
default(; aspect_ratio=1, c=:turbo)

include("permeability.jl")

@kernel function init_ϕ!(ϕ, ϕ_bg, ϕA, xc, yc, lw)
    ix, iy = @index(Global, NTuple)
    @inbounds ϕ[ix, iy] = ϕ_bg + ϕA * exp(-(xc[ix] / lw)^2 - ((yc[iy] + 8lw) / lw)^2)
end

@kernel function update_qD!(qx, qy, Pe, ϕ, k_m, Δρgy, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds if checkbounds(Bool, qx, ix, iy)
        k_ηf = k_m(0.5 * (ϕ[ix, iy] + ϕ[ix + 1, iy]))
        qx[ix, iy] = k_ηf * ((Pe[ix + 1, iy] - Pe[ix, iy]) / dx)
    end
    @inbounds if checkbounds(Bool, qy, ix, iy)
        k_ηf = k_m(0.5 * (ϕ[ix, iy] + ϕ[ix, iy + 1]))
        qy[ix, iy] = k_ηf * ((Pe[ix, iy + 1] - Pe[ix, iy]) / dy + Δρgy)
    end
end

@kernel function update_Pf!(Pe, qx, qy, k_m, ϕ, ϕ_bg, η_ϕ0, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        η_ϕ = η_ϕ0 * ϕ_bg / ϕ[ix + 1, iy + 1]
        dτ_β = min(dx, dy)^2 / max(k_m(ϕ[ix, iy + 0]), k_m(ϕ[ix + 1, iy + 0]), k_m(ϕ[ix + 2, iy + 0]),
                                   k_m(ϕ[ix, iy + 1]), k_m(ϕ[ix + 1, iy + 1]), k_m(ϕ[ix + 2, iy + 1]),
                                   k_m(ϕ[ix, iy + 2]), k_m(ϕ[ix + 1, iy + 2]), k_m(ϕ[ix + 2, iy + 2])) / 6.1
        ∇q = (qx[ix + 1, iy + 1] - qx[ix, iy + 1]) / dx +
             (qy[ix + 1, iy + 1] - qy[ix + 1, iy]) / dy
        Pe[ix + 1, iy + 1] += dτ_β * (∇q - Pe[ix + 1, iy + 1] / η_ϕ)
    end
end

@kernel function update_residual!(r_Pe, Pe, qx, qy, ϕ, ϕ_bg, η_ϕ0, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        η_ϕ = η_ϕ0 * (ϕ_bg / ϕ[ix + 1, iy + 1])
        ∇q = (qx[ix + 1, iy + 1] - qx[ix, iy + 1]) / dx +
             (qy[ix + 1, iy + 1] - qy[ix + 1, iy]) / dy
        r_Pe[ix, iy] = ∇q - Pe[ix + 1, iy + 1] / η_ϕ
    end
end

@kernel function update_ϕ!(ϕ, Pe, η_ϕ0, ϕ_bg, dt)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        η_ϕ = η_ϕ0 * ϕ_bg / ϕ[ix, iy]
        ϕ[ix, iy] -= dt * (1 - ϕ[ix, iy]) * Pe[ix, iy] / η_ϕ
    end
end

function porosity_wave_2D(backend=CPU())
    # physics
    lc     = 1.0
    lx, ly = 40lc, 100lc
    lw     = 4lc
    ϕ_bg   = 0.01
    ϕA     = 0.1
    Δρgy   = 1.0
    η_ϕ0   = 1.0
    tsc    = η_ϕ0 / (Δρgy * lc)
    dt     = 1e-3tsc
    k_ηf0  = lc^2 / η_ϕ0
    # switch between different formulations for permeability
    # k_m   = ConstantPermeability(k_ηf0)
    # k_m   = KarmanCozeny(3, k_ηf0 / ϕ_bg^3)
    # k_m   = KarmanCozeny(2, k_ηf0 / ϕ_bg^2)
    # k_m   = KarmanCozenyLimited(3, 3, k_ηf0 / ϕ_bg^3)
    k_m(ϕ) = k_ηf0 * (ϕ / ϕ_bg)^3 # Karman-Cozeny model use closure
    # numerics
    nx     = 50
    ny     = ceil(Int, ly / lx * nx)
    nt     = 100
    niter  = 2max(nx, ny)^2
    ncheck = ceil(Int, 0.1max(nx, ny)^2)
    nvis   = 1
    ϵtol   = 1e-6
    # preprocessing
    dx, dy = lx / nx, ly / ny
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
    # allocate arrays
    ϕ = KernelAbstractions.zeros(backend, Float64, nx, ny)
    Pe = KernelAbstractions.zeros(backend, Float64, nx, ny)
    qx = KernelAbstractions.zeros(backend, Float64, nx - 1, ny)
    qy = KernelAbstractions.zeros(backend, Float64, nx, ny - 1)
    r_Pe = KernelAbstractions.zeros(backend, Float64, nx - 2, ny - 2)
    # init
    init_ϕ!(backend, 256, size(ϕ))(ϕ, ϕ_bg, ϕA, xc, yc, lw)
    # time loop
    for it in 1:nt
        println("it = $it")
        # iter loop
        for iter in 1:niter
            update_qD!(backend, 256, (nx, ny))(qx, qy, Pe, ϕ, k_m, Δρgy, dx, dy)
            update_Pf!(backend, 256, size(Pe) .- 2)(Pe, qx, qy, k_m, ϕ, ϕ_bg, η_ϕ0, dx, dy)
            if iter % ncheck == 0
                update_residual!(backend, 256, size(r_Pe))(r_Pe, Pe, qx, qy, ϕ, ϕ_bg, η_ϕ0, dx, dy)
                err_Pe = maximum(abs.(r_Pe))
                @printf("  iter/ny² = %.1f, err_Pe = %1.3e\n", iter / (ny^2), err_Pe)
                if err_Pe <= ϵtol
                    break
                end
            end
        end
        update_ϕ!(backend, 256, size(ϕ))(ϕ, Pe, η_ϕ0, ϕ_bg, dt)
        if it % nvis == 0
            KernelAbstractions.synchronize(backend)
            p1 = heatmap(xc, yc, Array(ϕ)'; xlims=(-lx / 2, lx / 2), ylims=(-ly / 2, ly / 2), title="ϕ")
            p2 = heatmap(xc, yc, Array(Pe)'; xlims=(-lx / 2, lx / 2), ylims=(-ly / 2, ly / 2), title="Pₑ")
            display(plot(p1, p2; layout=(1, 2), size=(800, 600)))
        end
    end
    return
end

porosity_wave_2D()