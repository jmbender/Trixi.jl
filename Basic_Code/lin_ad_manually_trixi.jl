module disc

using LinearAlgebra
using LinearSolve
using Trixi

## reformulation classic finite volume 1D to use trixi inbuild functions
# for linear advection with a=1 the same as one MOL Euler Step
function FV_classic_step(u0, dt, dx, surface_flux, equations,dg,mesh, cache)
    c = deepcopy(u0)
    copy = Trixi.wrap_array(c,mesh,equations,dg,cache)
    u0_wrap = Trixi.wrap_array(u0,mesh,equations,dg,cache)
    for interface in 2:(Trixi.ninterfaces(dg,cache))
        u_ll, u_rr = Trixi.get_surface_node_vars(copy, equations, dg, interface-1)
        flux1 = surface_flux(u_ll,u_rr,1,equations)
        u_ll_2, u_rr_2 = Trixi.get_surface_node_vars(copy, equations,dg,interface)
        flux2 = surface_flux(u_ll_2,u_rr_2,1,equations)
        Trixi.multiply_add_to_node_vars!(u0_wrap,-dt/dx,flux2-flux1,equations,dg,2,interface-1)
        Trixi.multiply_add_to_node_vars!(u0_wrap,-dt/dx,flux2-flux1,equations,dg,1,interface)
    end
    u_ll, u_rr = Trixi.get_surface_node_vars(copy, equations, dg, Trixi.ninterfaces(dg,cache))
    flux1 = surface_flux(u_ll,u_rr,1,equations)
    u_ll_2, u_rr_2 = Trixi.get_surface_node_vars(copy, equations,dg,1)
    flux2 = surface_flux(u_ll_2,u_rr_2,1,equations)
    Trixi.multiply_add_to_node_vars!(u0_wrap,-dt/dx,flux2-flux1,equations,dg,1,1)
    Trixi.multiply_add_to_node_vars!(u0_wrap,-dt/dx,flux2-flux1,equations,dg,2,Trixi.ninterfaces(dg,cache))
    return u0
end

#end

#MOL with Heun

#method of line semidiscretization with SSPRK based on two forward Euler steps
function MOL_SSPRK_step(u0,dt::Float64,dx::Float64,surface_flux,equations,dg,mesh,cache)
    u_help = deepcopy(u0)
    u0_wrap = Trixi.wrap_array(u0,mesh,equations,dg,cache)
    #first Euler step
    FV_classic_step(u_help,dt,dx,surface_flux,equations,dg,mesh,cache)
    #second Euler step
    FV_classic_step(u_help,dt,dx,surface_flux,equations,dg,mesh,cache)
    u_help_wrap = Trixi.wrap_array(u_help,mesh,equations,dg,cache)
    #set the next value
    for element in eachelement(dg, cache)
        for i in eachnode(dg)
            Trixi.set_node_vars!(u0_wrap,0.5*(u0_wrap[1,i,element]+u_help_wrap[1,i,element]),equations,dg,i,element)
        end
    end
    return u0
end



#MP Heun
function MP_Heun(u,dt,dx,surface_flux,equations,dg,mesh,cache)
    N = Trixi.nelements(mesh,dg,cache)
    c = deepcopy(u)
    wrap_c = Trixi.wrap_array(c,mesh,equations,dg,cache)
    red_c = red(wrap_c,dg,cache)
    u_wrap = Trixi.wrap_array(u,mesh,equations,dg,cache)
    #######compute production/destruction matrices based on the numerical flux
    P = prod(N,red_c,surface_flux,dx,equations)
    #solve equation system for the first stage
    first_stage = LinearProblem(A_h(c,P,N,dt),red_c)
    sol_c_1 = solve(first_stage)
    c_1 = sol_c_1.u
    #c_1_wrap = Trixi.wrap_array(c_1,mesh,equations,dg,cache)
    #solve main problem 
    P_c_1 = prod_c_1(N,P,c_1,surface_flux,dx,equations)
    main = LinearProblem(A_h(c_1,P_c_1,N,dt),red_c)
    sol_main = solve(main)
    #wrap_sol = Trixi.wrap_array(sol_main.u,mesh,equations,dg,cache)
    for element in 1:nelements(dg,cache)-1
        Trixi.set_node_vars!(u_wrap,sol_main.u[element],equations,dg,1,element)
        Trixi.set_node_vars!(u_wrap,sol_main.u[element+1],equations,dg,2,element)
    end
    Trixi.set_node_vars!(u_wrap,sol_main.u[nelements(dg,cache)],equations,dg,1,nelements(dg,cache))
    Trixi.set_node_vars!(u_wrap,sol_main.u[1],equations,dg,2,nelements(dg,cache))
    return u
end



#build matrix A_h based on numerical fluxfunction g
#main part
function A_h(c_1,P::Matrix{Float64},N::Int64,dt::Float64)
    A = zeros(N,N)
    for i in 1:N
        for j in 1:N
            if i == j
                A[i,j] = 1+dt*sum(P[k,i]/c_1[i] for k in 1:N)
            else
                A[i,j] = -dt*P[i,j]/c_1[j]
            end
        end
    end
    return A
end

#precomputation
function A_h(c,P::Matrix{Float64},N::Int64,dt::Float64)
    A = zeros(N,N)
    for i in 1:N
        for j in 1:N
            if i == j
                A[i,j] = 1+dt*sum(P[k,i]/c[i] for k in 1:N)
            else
                A[i,j] = -dt*P[i,j]/c[j]
            end
        end
    end
    return A
end

function prod(N,c,surface_flux,dx,equations)
    P = zeros(N,N)
    l_vals = zeros(N)
    r_vals = zeros(N)
    Flux_r = 1/dx*(surface_flux(c[1],c[2],1,equations))[1]
    r_vals[1] = -min(Flux_r,0)
    for i in 2:(N-1)
        Flux_l = 1/dx*(surface_flux(c[i-1],c[i],1,equations))[1] #inner element left to right
        l_vals[i] = max(Flux_l,0)
        Flux_r = 1/dx*(surface_flux(c[i],c[i+1],1,equations))[1] #inner element right to left
        r_vals[i] = -min(Flux_r,0)
        l_vals[i] = max(Flux_l,0)
    end
    Flux_l = 1/dx*(surface_flux(c[N-1],c[N],1,equations))[1]
    l_vals[N] = max(Flux_l,0)
    Flux_l = 1/dx*(surface_flux(c[end],c[1],1,equations))[1]
    first = max(Flux_l,0)
    Flux_r = 1/dx*(surface_flux(c[end],c[1],1,equations))[1]
    last = -min(Flux_r,0)
    popat!(r_vals,lastindex(r_vals))
    popfirst!(l_vals)
    P = diagm(-1 => l_vals,1 => r_vals)
    P[1,N] = first
    P[N,1] = last
    return P
end

#=
#compute production terms and save into matrix, transposed production matrix delievers destruction
function prod(N,c,surface_flux,dx,equations,dg,mesh,cache)
    P = zeros(N,N)
    l_vals = zeros(2*Trixi.nelements(dg,cache))
    r_vals = zeros(2*Trixi.nelements(dg,cache))
    it = 1
    for i in 1:(Trixi.nelements(dg,cache))
        u_ll, u_rr = Trixi.get_surface_node_vars(c, equations, dg, i)
        Flux_l = 1/dx*surface_flux(u_ll,u_rr,1,equations)[1] #inner element left to right
        l_vals[it] = max(Flux_l,0)
        Flux_r = 1/dx*surface_flux(u_rr,u_ll,1,equations)[1] #inner element right to left
        r_vals[it] = -min(Flux_r,0)
        it += 1
        l_vals[it] = max(Flux_l,0)
        r_vals[it] = -min(Flux_r,0)
        it += 1
    end
    first = popfirst!(l_vals)
    last = popat!(r_vals,lastindex(r_vals))
    P = diagm(-1 => l_vals,1 => r_vals)
    P[1,N] = first
    P[N,1] = last
    return P
end
=#

#=
function prod(N,c,surface_flux,dx,equations,dg,mesh,cache)
    P = zeros(N,N)
    l_vals = zeros(Trixi.nelements(dg,cache))
    r_vals = zeros(Trixi.nelements(dg,cache))
    it = 1
    for i in 1:(Trixi.nelements(dg,cache))
        u_ll, u_rr = Trixi.get_surface_node_vars(c, equations, dg, i)
        Flux_l = 1/dx*surface_flux(u_ll,u_rr,1,equations)[1] #inner element left to right
        l_vals[it] = max(Flux_l,0)
        Flux_r = 1/dx*surface_flux(u_rr,u_ll,1,equations)[1] #inner element right to left
        r_vals[it] = -min(Flux_r,0)
    end
    first = popfirst!(l_vals)
    last = popat!(r_vals,lastindex(r_vals))
    P = diagm(-1 => l_vals,1 => r_vals)
    P[1,N] = first
    P[N,1] = last
    return P
end
=#
#=
function prod_c_1(N,P,g,c_1,dx,equations,dg,mesh,cache)
    P_1 = prod(N,c_1,g,dx,equations,dg,mesh,cache)
    return 0.5*(P.+ P_1)
end
=#


function prod_c_1(N,P,c,surface_flux,dx,equations)
    P_1 = prod(N,c,surface_flux,dx,equations)
    return 0.5*(P.+ P_1)
end



function red(u,dg,cache)
    red_u = zeros(nelements(dg,cache))
    for i in 1:nelements(dg,cache)
        red_u[i] = u[1,2,i]
    end
    return red_u
end


end