using Trixi, OrdinaryDiffEq, Plots

include("./lin_ad_manually_trixi.jl")
import .disc

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

coordinates_min = 0.0 # minimum coordinate
coordinates_max = 2.0 # maximum coordinate

init = 5
N = 2^init
dx = abs(coordinates_max-coordinates_min)/N
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=init, # number of elements = 2^init
                n_cells_max=30_000)
dt = 0.9*dx

tspan = (0.0,2.0)
timesteps = Int(round(last(tspan)/dt))

function flux_lax_wendroff(u_ll,u_rr,orientation::Int,equation::LinearScalarAdvectionEquation1D)
    u_l = u_ll[1]
    u_r = u_rr[1]
    return SVector(0.5*(u_l+u_r)-dt/(2*dx)*(u_r-u_l))
end

#surface_flux = flux_lax_friedrichs
surface_flux = flux_lax_wendroff
dg = DGSEM(polydeg=1)

initial_condition_sine_wave(x, t, equations) = SVector(1.0 + 0.5 * sin(pi * sum(x - equations.advection_velocity * t)))
function initial_condition_block(x,t,equations)
    if 0.8 < x[1] < 1.2
        return SVector(1.0)
    else
        return SVector(0.01)
    end
end

#semi = SemidiscretizationHyperbolic(mesh,equations,initial_condition_sine_wave,dg)
semi = SemidiscretizationHyperbolic(mesh,equations,initial_condition_block,dg)
cache = semi.cache


u = compute_coefficients(first(tspan),semi)
pd_0 = PlotData1D(u,semi) #store initial_condtion because it gets overwritten

for i in 1:5
    disc.MOL_SSPRK_step(u,dt,dx,surface_flux,equations,dg,mesh,cache)
    #disc.FV_classic_step(u,dt,dx,surface_flux,equations,dg,mesh,cache)
    #disc.MP_Heun(u,dt,dx,surface_flux,equations,dg,mesh,cache)
end


#disc.FV_classic_step(u,dt,dx,surface_flux,equations,dg,mesh,cache)
pd = PlotData1D(u,semi)

plot(pd_0, label="init")
plot!(pd, label="out",legend=:bottomleft)

#wrap = Trixi.wrap_array(u,mesh,equations,dg,cache)

#disc.prod(32,wrap,surface_flux,dx,equations,dg,mesh,cache)