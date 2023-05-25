using Trixi, OrdinaryDiffEq, Plots, PolynomialBases

# equation with a advection_velocity of `1`.
advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

coordinates_min = 0.0 # minimum coordinate
coordinates_max = 2.0 # maximum coordinate
init = 4
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=init, # number of elements = 2^4
                n_cells_max=30_000)
tspan = (0.0, 2.0)
surface_flux = flux_lax_friedrichs

dg = DGSEM(polydeg=1, surface_flux=flux_lax_friedrichs)
initial_condition_sine_wave(x, t, equations) = SVector(1.0 + 0.5 * sin(pi * sum(x - equations.advection_velocity * t)))

semi = SemidiscretizationHyperbolic(mesh,equations,initial_condition_sine_wave,dg)
cache = semi.cache

#u0 = compute_coefficients(initial_condition_sine_wave,first(tspan),semi)
u0 = compute_coefficients(first(tspan),semi)
u0_wrap = Trixi.wrap_array(u0,mesh,equations,dg,cache)

u_knoten = zeros(nelements(dg,cache),nnodes(dg))
knoten = zeros(nelements(dg,cache),nnodes(dg))


for element in eachelement(dg,cache)
    for i in eachnode(dg)
        #knoten[element,i] = Trixi.get_node_coords(cache.elements.node_coordinates,equations,dg,element,i)[1]
        u_knoten[element,i] = Trixi.get_node_vars(u0_wrap,equations,dg,i,element)[1]
        u_ll, u_rr = Trixi.get_surface_node_vars(u0_wrap,equations,dg,i,element)
        u_knoten_l = u_ll[1]
        u_knoten_r = u_rr[1]
    end
end

u_knoten_line = zeros(2*nelements(dg,cache))

for element in eachelement(dg,cache)
        u_knoten_line[2*element-1] = Trixi.get_node_vars(u0_wrap,equations,dg,1,element)[1]
        u_knoten_line[2*element] = Trixi.get_node_vars(u0_wrap,equations,dg,2,element)[1]
end
