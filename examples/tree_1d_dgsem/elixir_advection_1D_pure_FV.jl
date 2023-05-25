using OrdinaryDiffEq
using Trixi

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)
initial_condition = initial_condition_gauss

solver = DGSEM(polydeg=1, surface_flux=flux_lax_friedrichs,
               volume_integral=VolumeIntegralPureLGLFiniteVolume(flux_lax_friedrichs))

coordinates_min = 0.0
coordinates_max = 2.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

sol = solve(ode,Heun(),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);