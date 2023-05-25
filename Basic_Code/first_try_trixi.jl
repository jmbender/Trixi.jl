using Trixi, OrdinaryDiffEq, Plots, PolynomialBases

#include("./lin_ad_manually.jl")
#import .disc

# equation with a advection_velocity of `1`.
advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

coordinates_min = 0.0 # minimum coordinate
coordinates_max = 2.0 # maximum coordinate
init = 4
N = 2^init
dx = abs(coordinates_max-coordinates_min)/N
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=init, # number of elements = 2^4
                n_cells_max=30_000)

dt = 0.9*dx
timesteps = 5
grid = collect(coordinates_min:dx:coordinates_max);
x_values = grid[1:(end-1)].+(dx/2);
tspan = (0.0, timesteps*dt)
surface_flux = flux_lax_friedrichs
LF(u_l,u_r) = 0.5*(u_l+u_r)-dx/(2*dt)*(u_r-u_l) #test, eigentlich nicht benötigt

dg = DGSEM(polydeg=1)
initial_condition_sine_wave(x, t, equations) = SVector(1.0 + 0.5 * sin(pi * sum(x - equations.advection_velocity * t)))

#basis = LobattoLegendreBasis(1)

#RealT=real(dg)
#cache = Trixi.create_cache(mesh, equations,dg,real(dg), RealT)


semi = SemidiscretizationHyperbolic(mesh,equations,initial_condition_sine_wave,dg)
cache = semi.cache


u0 = compute_coefficients(first(tspan),semi)
pd_0 = PlotData1D(u0,semi) #generate plot data for initial condition

u0_wrap = Trixi.wrap_array(u0,mesh,equations,dg,cache)


u_knoten = zeros(nelements(dg,cache),nnodes(dg))
knoten = zeros(nelements(dg,cache),nnodes(dg))


for element in eachelement(dg,cache)
    for i in eachnode(dg)
        u_knoten[element,i] = Trixi.get_node_vars(u0_wrap,equations,dg,element,i)[1]
    end
end



function FV_flux_diff_trixi(dt,timesteps,dx, u0_wrap,surface_flux,equations)
    #timesteps = Int(round(T/dt))
    flux_numerical = zeros(nelements(dg,cache))
    u = u0_wrap
    new = zeros(nelements(dg,cache))
    for n in 1:(timesteps)
        u_knoten = zeros(nelements(dg,cache),nnodes(dg))
        for element in eachelement(dg,cache)
            for i in eachnode(dg)
                u_knoten[element,i] = Trixi.get_node_vars(u,equations,dg,i,element)[1]
            end
        end
        flux_numerical = zeros(nelements(dg,cache))
        new = zeros(nelements(dg,cache))
        for j in 1:(nelements(dg,cache)-1)
            f_l = LF(u_knoten[j,1],u_knoten[j,2])
            f_r = LF(u_knoten[j+1,1],u_knoten[j+1,2])
            #f_l = surface_flux(u_knoten[j,1],u_knoten[j,2],1,equations)
            #f_r = surface_flux(u_knoten[j+1,1],u_knoten[j+1,2],1,equations)
            flux_numerical[j+1] = (f_r-f_l)
            new[j+1] = u_knoten[j,2] -(dt/dx)*(f_r-f_l)
        end
        f_l = LF(u_knoten[end,1],u_knoten[end,2])
        f_r = LF(u_knoten[end,2],u_knoten[1,2])
        #f_l = surface_flux(u_knoten[end,1],u_knoten[end,2],1,equations)
        #f_r = surface_flux(u_knoten[end,2],u_knoten[1,2],1,equations)
        flux_numerical[1] = (-dt/dx)*(f_r-f_l)
        new[1] = u_knoten[1,2] -(dt/dx)*(f_r-f_l)
        
        u_knoten = zeros(nelements(dg,cache),nnodes(dg))
        for element in eachelement(dg,cache)
            for i in eachnode(dg)
                #Trixi.multiply_add_to_node_vars!(u, -dt/dx, flux_numerical, equations, dg, i,element)
                #Trixi.add_to_node_vars!(u,flux_numerical,equations,dg,i,element)
                Trixi.set_node_vars!(u,new,equations,dg,i,element)
            end
        end
    end
    return new, flux_numerical
end
    

#u, num_flux = FV_flux_diff_trixi(dt,timesteps,dx,u0_wrap,surface_flux,equations)


u_out = zeros(2*nelements(dg,cache))

for element in eachelement(dg,cache)
    for i in eachnode(dg)
        u_out[2*element-1] = Trixi.get_node_vars(u,equations,dg,1,element)[1]
        u_out[2*element] = Trixi.get_node_vars(u,equations,dg,2,element)[1]
    end
end


u_out_short = zeros(nelements(dg,cache))
for element in eachelement(dg,cache)
    for i in eachnode(dg)
        u_out_short[element] = Trixi.get_node_vars(u,equations,dg,2,element)[1]
    end
end

u0 = compute_coefficients(first(tspan),semi)
u0_short = zeros(nelements(dg,cache))
u0_wrap = Trixi.wrap_array(u0,mesh,equations,dg,cache)
for element in eachelement(dg,cache)
    for i in eachnode(dg)
        u0_short[element] = Trixi.get_node_vars(u0_wrap,equations,dg,2,element)[1]
    end
end

#classic = disc.FV_flux_diff(dt,last(tspan),dx,N,u0_short[:,1],LF)

#pd = PlotData1D(u_out,semi)

#=
u_out_line = zeros(2*nelements(dg,cache))
for i in eachelement(dg,cache)
    u_out_line[2*i-1] = u_out[i,1]
    u_out_line[2*i] = u_out[i,2]
end
=#

#u_ll, u_rr = Trixi.get_surface_node_vars(u0_wrap,equations,dg)
#u_bound = [u_ll[1],u_rr[1]]


#nach Vorbild von dg_1d.jl
begin function FV_classic_trixi(u0, surface_flux, equations,dg, cache)
    for interface in 2:(Trixi.ninterfaces(dg,cache)-1)
        u_ll, u_rr = Trixi.get_surface_node_vars(u0, equations, dg, interface-1)
        #flux1 = surface_flux(u_ll,u_rr,1,equations)
        flux1 = LF(u_ll,u_rr)
        u_ll_2, u_rr_2 = Trixi.get_surface_node_vars(u0, equations,dg,interface)
        #flux2 = surface_flux(u_ll_2,u_rr_2,1,equations)
        flux2 = LF(u_ll_2,u_rr_2)
        numflux = flux2-flux1
        Trixi.multiply_add_to_node_vars!(u0,-dt/dx,numflux,equations,dg,2,interface-1)
        Trixi.multiply_add_to_node_vars!(u0,-dt/dx,numflux,equations,dg,1,interface)
    end
    u_ll, u_rr = Trixi.get_surface_node_vars(u0, equations, dg, Trixi.ninterfaces(dg,cache))
    #flux1 = surface_flux(u_ll,u_rr,1,equations)
    flux1 = LF(u_ll,u_rr)
    u_ll_2, u_rr_2 = Trixi.get_surface_node_vars(u0, equations,dg,1)
    #flux2 = surface_flux(u_ll_2,u_rr_2,1,equations)
    flux2 = LF(u_ll_2,u_rr_2)
    numflux = flux2-flux1
    Trixi.multiply_add_to_node_vars!(u0,-dt/dx,numflux,equations,dg,1,1)
    Trixi.multiply_add_to_node_vars!(u0,-dt/dx,numflux,equations,dg,2,Trixi.ninterfaces(dg,cache))
end

end

begin function test(u0, surface_flux, equations,dg, interface)
    u_ll, u_rr = Trixi.get_surface_node_vars(u0, equations, dg, interface-1)
    #flux1 = surface_flux(u_ll,u_rr,1,equations)
    flux1 = LF(u_ll,u_rr)
    u_ll_2, u_rr_2 = Trixi.get_surface_node_vars(u0, equations,dg,interface)
    #flux2 = surface_flux(u_ll_2,u_rr_2,1,equations)
    flux2 = LF(u_ll_2,u_rr_2)
    return flux2, flux1
end
end

#polydeg=2: bekomme Werte in der Mitte der Zellen
begin function FV_cT_2(u0, surface_flux, equations,dg,cache,dt,dx)
    for element in 2:(Trixi.nelements(dg,cache)-1)
        u_l = Trixi.get_node_vars(u0,equations,dg,2,element-1)
        u_mid = Trixi.get_node_vars(u0,equations,dg,2,element)
        u_r = Trixi.get_node_vars(u0,equations,dg,2,element+1)
        numflux = LF(u_mid,u_r)-LF(u_l,u_mid)
        #numflux = surface_flux(u_mid,u_r,1,equations)-surface_flux(u_l,u_mid,1,equations)
        Trixi.multiply_add_to_node_vars!(u0,-dt/dx,numflux,equations,dg,2,element)
    end
    
    u_l = Trixi.get_node_vars(u0,equations,dg,2,Trixi.nelements(dg,cache))
    u_mid = Trixi.get_node_vars(u0,equations,dg,2,1)
    u_r = Trixi.get_node_vars(u0,equations,dg,2,2)
    numflux = LF(u_mid,u_r)-LF(u_l,u_mid)
    #numflux = surface_flux(u_mid,u_r,1,equations)-surface_flux(u_l,u_mid,1,equations)
    
    Trixi.multiply_add_to_node_vars!(u0,-dt/dx,numflux,equations,dg,2,1)
    u_l = Trixi.get_node_vars(u0,equations,dg,2,Trixi.nelements(dg,cache)-1)
    u_mid = Trixi.get_node_vars(u0,equations,dg,2,Trixi.nelements(dg,cache))
    u_r = Trixi.get_node_vars(u0,equations,dg,2,1)
    numflux = LF(u_mid,u_r)-LF(u_l,u_mid)
    #numflux = surface_flux(u_mid,u_r,1,equations)-surface_flux(u_l,u_mid,1,equations)
    Trixi.multiply_add_to_node_vars!(u0,-dt/dx,numflux,equations,dg,2,Trixi.nelements(dg,cache))
    #Trixi.multiply_add_to_node_vars!(u0,-dt/dx,numflux,equations,dg,2,1)
end

function getvector(u0,dg,cache)
    u_res = zeros(Trixi.nelements(dg,cache))
    for element in Trixi.eachelement(dg,cache)
        u_res[element] = Trixi.get_node_vars(u0,equations,dg,2,element)[1]
    end
    return u_res
end

end