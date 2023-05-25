
module disc
   
using LinearAlgebra
using LinearSolve

##############################################################################################################
#= MP Heun with explicit production and destruction terms
Input: dt   timestepsize
        T   endtime
        N   number of gridcells
        p   production terms (as function of: timepstep n ; cosidered cells i,j ; solution vector c)
        d   destruction terms (as function of: timepstep n ; cosidered cells i,j ; solution vector c)
        c0  intial value
=#


function MP_Heun(dt::Float64,T::Float64,N::Int64,p::Function,d::Function,c0::Vector{Float64})
    timesteps = Int(T/dt)
    c = zeros(timesteps,N)
    c_1 = zeros(timesteps,N)
    c[1,:] = c0
    for n in 1:timesteps-1
        #solve equation system for the first stage
        first_stage = LinearProblem(A_h(n,c,d,p,N,dt),c[n,:])
        sol_c_1 = solve(first_stage)
        c_1[n,:] = sol_c_1.u
        #solve main problem 
        main = LinearProblem(A_h(n,c,c_1,d,p,N,dt),c[n,:])
        sol_main = solve(main)
        c[n+1,:] = sol_main.u
    end
    return c
end



#=compute coefficients of solution matrix A_h and the precomputation system

build matrix A_h
main part
=#

function A_h(n::Int64,c::Matrix{Float64},c_1::Matrix{Float64},d::Function,p::Function,N::Int64,dt::Float64)
    A = zeros(N,N)
    for i in 1:N
        for j in 1:N
            if i == j
                A[i,j] = 1+dt*sum(d(n,i,k,c,c_1)/c_1[n,i] for k in 1:N)
            else
                A[i,j] = -dt*p(n,i,j,c,c_1)/c_1[n,j]
            end
        end
    end
    return A
end

#precomputation
function A_h(n::Int64,c::Matrix{Float64},d::Function,p::Function,N::Int64,dt::Float64)
    A = zeros(N,N)
    for i in 1:N
        for j in 1:N
            if i == j
                A[i,j] = 1+dt*sum(d(n,i,k,c)/c[n,i] for k in 1:N)
            else
                A[i,j] = -dt*p(n,i,j,c)/c[n,j]
            end
        end
    end
    return A
end

#modification for precomputation
p(n,i,j,c,c_1) = 0.5*(p(n,i,j,c).+p(n,i,j,c_1))
d(n,i,j,c,c_1) = 0.5*(d(n,i,j,c).+d(n,i,j,c_1))


#########################################################################################################

#extension for when only the numerical flux, but no concrete production/destruction terms are given

#= MP Heun based on numerical flux
Input:  dt  timestepsize
        dx  gridwidth
        T   endtime
        N   number of gridcells
        g   numerical flux function (as function of left and right value)
        c0  intial value
=#

function MP_Heun(dt::Float64,dx::Float64,T::Float64,N::Int64,g::Function,c0::Vector{Float64})
    timesteps = Int(round(T/dt))
    c = zeros(timesteps,N)
    c_1 = zeros(timesteps,N)
    c[1,:] = c0
    for n in 1:timesteps-1
        #compute production/destruction matrices based on the numerical flux
        P = prod(N,n,c,g,dx)
        #solve equation system for the first stage
        first_stage = LinearProblem(A_h(n,c,P,N,dt),c[n,:])
        sol_c_1 = solve(first_stage)
        c_1[n,:] = sol_c_1.u
        #solve main problem 
        P_c_1 = prod_c_1(N,n,P,g,c_1,dx)
        main = LinearProblem(A_h(n,c_1,P_c_1,N,dt),c[n,:])
        sol_main = solve(main)
        c[n+1,:] = sol_main.u
    end
    return c
end

#build matrix A_h based on numerical fluxfunction g
#main part
function A_h(n::Int64,c_1::Matrix{Float64},P::Matrix{Float64},N::Int64,dt::Float64)
    A = zeros(N,N)
    for i in 1:N
        for j in 1:N
            if i == j
                A[i,j] = 1+dt*sum(P[k,i]/c_1[n,i] for k in 1:N)
            else
                A[i,j] = -dt*P[i,j]/c_1[n,j]
            end
        end
    end
    return A
end

#precomputation
function A_h(n::Int64,c::Matrix{Float64},P::Matrix{Float64},N::Int64,dt::Float64)
    A = zeros(N,N)
    for i in 1:N
        for j in 1:N
            if i == j
                A[i,j] = 1+dt*sum(P[k,i]/c[n,i] for k in 1:N)
            else
                A[i,j] = -dt*P[i,j]/c[n,j]
            end
        end
    end
    return A
end

#=
#compute production terms and save into matrix, transposed production matrix delievers destruction
function prod(N,n,c,g,dx)
    P = zeros(N,N)
        for i in 1:N
            for j in 1:N
                if j == i-1
                    Flux = 1/dx*(g(c[n,j],c[n,i]))
                    P[i,j] = max(Flux,0)
                elseif j == i+1
                    Flux = 1/dx*(g(c[n,i],c[n,j]))
                    P[i,j] = -min(Flux,0)
                elseif i == 1 && j == N  #left boundary flux
                    Flux = 1/dx*(g(c[n,j],c[n,i]))
                    P[i,j] =  max(Flux,0) #right boundary flux
                elseif i == N && j == 1
                    Flux = 1/dx*(g(c[n,i],c[n,j]))
                    P[i,j] = -min(Flux,0)
                else 
                    P[i,j] = 0
                end
            end
        end
    return P
end
=#



function prod(N,n,c,g,dx)
    P = zeros(N,N)
    l_vals = zeros(N)
    r_vals = zeros(N)
    Flux_r = 1/dx*(g(c[n,1],c[n,2]))
    r_vals[1] = -min(Flux_r,0)
    for i in 2:(N-1)
        Flux_l = 1/dx*(g(c[n,i-1],c[n,i])) #inner element left to right
        l_vals[i] = max(Flux_l,0)
        Flux_r = 1/dx*(g(c[n,i],c[n,i+1])) #inner element right to left
        r_vals[i] = -min(Flux_r,0)
        l_vals[i] = max(Flux_l,0)
    end
    Flux_l = 1/dx*(g(c[n,N-1],c[n,N]))
    l_vals[N] = max(Flux_l,0)
    Flux_l = 1/dx*(g(c[n,end],c[n,1]))
    first = max(Flux_l,0)
    Flux_r = 1/dx*(g(c[n,end],c[n,1]))
    last = -min(Flux_r,0)
    popat!(r_vals,lastindex(r_vals))
    popfirst!(l_vals)
    P = diagm(-1 => l_vals,1 => r_vals)
    P[1,N] = first
    P[N,1] = last
    return P
end


function prod_c_1(N,n,P,g,c_1,dx)
    P_1 = prod(N,n,c_1,g,dx)
    return 0.5*(P.+ P_1)
end



#########################################################################################################

#=fully discrete finite volume flux differencing scheme with flux function g
 Input: dt  timestep
        T   endtime
        dx  gridwidth
        N   number of gridcells
        u0  initial condition
        g   numerical flux function
=#
function FV_flux_diff(dt::Float64,T::Float64,dx::Float64,N::Int64,u0::Vector{Float64},g::Function)
    timesteps = Int(round(T/dt))
    U = zeros(timesteps,N)
    U[1,:] = u0
    for n in 1:(timesteps-1)
        for i in 2:N-1
            U[n+1,i] = U[n,i] .- dt/dx*(g(U[n,i],U[n,i+1]).-g(U[n,i-1],U[n,i]))
        end
        U[n+1,1] = U[n,1] .- dt/dx*(g(U[n,1],U[n,2]).-g(U[n,(end-1)],U[n,1])) #periodic boundary conditions
        U[n+1,end] = U[n,end] .- dt/dx*(g(U[n,end],U[n,2]).-g(U[n,(end-1)],U[n,end]))
    end
    return U
end

###########################################################################################################

#im Prinzip einfach Heun

#method of line semidiscretization with SSPRK based on two forward Euler steps
function MOL_SSPRK(dt::Float64,T::Float64,dx::Float64,N::Int64,u0::Vector{Float64},g::Function)
    timesteps = Int(round(T/dt))
    U = zeros(timesteps,N)
    U_help = zeros(timesteps,N)
    U_helpp = zeros(timesteps,N)
    U[1,:] = u0
    for n in 1:(timesteps-1)
        #first Euler step
        for i in 2:N-1
            U_help[n,i] = U[n,i] .- dt/dx*(g(U[n,i],U[n,i+1]).-g(U[n,i-1],U[n,i]))
        end
        #periodic boundary conditions
        U_help[n,1] = U[n,1] .- dt/dx*(g(U[n,1],U[n,2]).-g(U[n,end-1],U[n,1]))
        U_help[n,end] = U[n,end] .- dt/dx*(g(U[n,end],U[n,2]).-g(U[n,(end-1)],U[n,end]))
        #second Euler step
        for i in 2:N-1
            U_helpp[n,i] = U_help[n,i] .- dt/dx*(g(U_help[n,i],U_help[n,i+1]).-g(U_help[n,i-1],U_help[n,i]))
        end
        #periodic boundary conditions
        U_helpp[n,1] = U_help[n,1] .- dt/dx*(g(U_help[n,1],U_help[n,2]).-g(U_help[n,end-1],U_help[n,1]))
        U_helpp[n,end] = U_help[n,end] .- dt/dx*(g(U_help[n,end],U[n,2]).-g(U_help[n,(end-1)],U_help[n,end]))
        #compute the next value
        U[n+1,:] = 0.5*(U[n,:].+ U_helpp[n,:])
    end
    return U
end



end

