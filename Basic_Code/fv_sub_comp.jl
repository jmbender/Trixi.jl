using Trixi, OrdinaryDiffEq, Plots

include("./lin_ad_manually_trixi.jl")
import .disc

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

coordinates_min = 0.0 # minimum coordinate
coordinates_max = 2.0 # maximum coordinate

init = 4
N = 2^init
dx = abs(coordinates_max-coordinates_min)/N
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=init, # number of elements = 2^init
                n_cells_max=30_000)
dt = 0.8*dx
T = 0.5
tspan = (0.0,T)
timesteps = Int(round(last(tspan)/dt))

function flux_lax_wendroff(u_ll,u_rr,orientation::Int,equation::LinearScalarAdvectionEquation1D)
    u_l = u_ll[1]
    u_r = u_rr[1]
    return SVector(0.5*(u_l+u_r)-dt/(2*dx)*(u_r-u_l))
end

initial_condition_sine_wave(x, t, equations) = SVector(1.0 + 0.5 * sin(pi * sum(x - equations.advection_velocity * t)))

function initial_condition_block(x,t,equations)
    if 0.8 < x[1] < 1.2
        return SVector(1.0)
    else
        return SVector(0.01)
    end
end
#flux = flux_lax_friedrichs
flux = flux_lax_wendroff
solver = DGSEM(polydeg=1, surface_flux=flux,
               volume_integral=VolumeIntegralPureLGLFiniteVolume(flux))

semid = SemidiscretizationHyperbolic(mesh, equations, initial_condition_block, solver)


ode = semidiscretize(semid, tspan)

sol = solve(ode, Heun(),save_everystep=false)

pl = PlotData1D(sol)
plot(pl)


surface_flux = flux_lax_wendroff
#surface_flux = flux_lax_friedrichs
dg = DGSEM(polydeg=1)

semi = SemidiscretizationHyperbolic(mesh,equations,initial_condition_block,dg)
cache = semi.cache


u = compute_coefficients(first(tspan),semi)

for i in 1:(T/dt)
    disc.MOL_SSPRK_step(u,dt,dx,surface_flux,equations,dg,mesh,cache)
    #disc.MP_Heun(u,dt,dx,surface_flux,equations,dg,mesh,cache)
end

pd = PlotData1D(u,semi)
plot!(pd, label="out",legend=:bottomleft)

function exact_block(x,t,equations)
    if 1.3 < x[1] < 1.7
        return SVector(1.0)
    else
        return SVector(0.01)
    end
end

semi = SemidiscretizationHyperbolic(mesh,equations,exact_block,dg)
cache = semi.cache

ex = compute_coefficients(last(tspan),semi)
pd_ex = PlotData1D(ex,semi)
plot!(pd_ex,label="exact",legend=:bottomleft)
#plot!(getmesh(pl))