using LinearAlgebra
using Plots
using LaTeXStrings
using LinearSolve

include("./lin_ad_manually.jl")
import .disc

#discretization in space (equidistant grid)
coordinates_min = 0
coordinates_max = 2
n_elements =  2^5
dx = abs(coordinates_max-coordinates_min)/n_elements
grid = collect(coordinates_min:dx:coordinates_max);
#x_values = grid[1:(end-1)]
x_values = grid[1:(end-1)].+(dx/2); #midpoints of the cells

#in time
T = 1.0
dt = 0.9*dx
timesteps = Int(round(T/dt))


#initialization
initial_condition_sin_wave(x) = 1.0 + 0.5*sin(pi*x) #initial condition

function initial_condition_block(x)
    if 0.75 < x < 1.25
        return 1.0
    else
        return 0.01
    end
    
end


function initial_condition_block(x,c,mid)
    if mid == 0.0
        if x < 0.25 
            return 1.0
        elseif x > 1.75
            return 1.0
        else return c
        end
    elseif 0.25 > mid > 1.75
        if x < mid+0.25
            return 1.0
        elseif x > mid-0.25
            return 1.0
        else return c
        end
    elseif abs(mod(mid-0.25,2)) < x < abs(mod(mid+0.25,2))
        return 1.0
    else
        return c
    end
end


    
#u0 = initial_condition_sin_wave.(x_values);
u0 = initial_condition_block.(x_values,0.01,1.0)
#u0 = initial_condition_block.(x_values)

#flux function
LF(u_l,u_r) = 0.5*(u_l+u_r)-dx/(2*dt)*(u_r-u_l) #Lax-Friedrichs
LW(u_l,u_r) = 0.5*(u_l+u_r)-dt/(2*dx)*(u_r-u_l) #Lax-Wendroff

#different solution methods

#Lax-Friedrichs
U_classic_LF = disc.FV_flux_diff(dt,T,dx,n_elements,u0,LF)
c_MPRK_LF = disc.MP_Heun(dt,dx,T,n_elements,LF,u0)
MOL_SSPRK_LF = disc.MOL_SSPRK(dt,T,dx,n_elements,u0,LF)

#Lax-Wendroff
U_classic_LW = disc.FV_flux_diff(dt,T,dx,n_elements,u0,LW)
c_MPRK_LW = disc.MP_Heun(dt,dx,T,n_elements,LW,u0)
MOL_SSPRK_LW = disc.MOL_SSPRK(dt,T,dx,n_elements,u0,LW)


#visualization

t1 =0.5 #plotted time
t1_values = Int(round(t1/dt))

t2 =2.0 #plotted time
t2_values = Int(round(t2/dt))


exakt_t1 = initial_condition_block.(x_values,0.01,mod(t1+1.0,2.0))
exakt_t2 = initial_condition_block.(x_values,0.01,mod(t2+1.0,2.0))

l = @layout([[a b;c d] S{.1w}])
p1_l = plot(x_values,[U_classic_LF[1,:] U_classic_LF[t1_values,:] c_MPRK_LF[t1_values,:] MOL_SSPRK_LF[t1_values,:] exakt_t1], label=["t=0" "flux differencing" "MP Heun" "MOL SSPRK" "exact"],title="Lax-Friedrichs at t=$(t1)")
p2_l = plot(x_values,[U_classic_LW[1,:] U_classic_LW[t1_values,:] c_MPRK_LW[t1_values,:] MOL_SSPRK_LW[t1_values,:] exakt_t1], label=["t=0" "flux differencing" "MP Heun" "MOL SSPRK" "exact"],title="Lax-Wendroff at t=$(t1)")
p3_l = plot(x_values,[U_classic_LF[1,:] U_classic_LF[t2_values,:] c_MPRK_LF[t2_values,:] MOL_SSPRK_LF[t2_values,:] exakt_t2], label=["t=0" "flux differencing" "MP Heun" "MOL SSPRK" "exact"],title="Lax-Friedrichs at t=$(t2)")
p4_l = plot(x_values,[U_classic_LW[1,:] U_classic_LW[t2_values,:] c_MPRK_LW[t2_values,:] MOL_SSPRK_LW[t2_values,:] exakt_t2], label=["t=0" "flux differencing" "MP Heun" "MOL SSPRK" "exact"],title="Lax-Wendroff at t=$(t2)")
p1 = plot(x_values,[U_classic_LF[1,:] U_classic_LF[t1_values,:] c_MPRK_LF[t1_values,:] MOL_SSPRK_LF[t1_values,:] exakt_t1],title="Lax-Friedrichs at t=$(t1)", legend=false)
p2 = plot(x_values,[U_classic_LW[1,:] U_classic_LW[t1_values,:] c_MPRK_LW[t1_values,:] MOL_SSPRK_LW[t1_values,:] exakt_t1],title="Lax-Wendroff at t=$(t1)",legend=false)
p3 = plot(x_values,[U_classic_LF[1,:] U_classic_LF[t2_values,:] c_MPRK_LF[t2_values,:] MOL_SSPRK_LF[t2_values,:] exakt_t2],title="Lax-Friedrichs at t=$(t2)",legend=false)
p4 = plot(x_values,[U_classic_LW[1,:] U_classic_LW[t2_values,:] c_MPRK_LW[t2_values,:] MOL_SSPRK_LW[t2_values,:] exakt_t2],title="Lax-Wendroff at t=$(t2)",legend=false)
legend = plot!([0 0 0 0], showaxis = false, grid = false, label=["t=0" "flux differencing" "MP Heun" "MOL SSPRK" "exact"])

#plot(p1,p3,p2,p4,legend,layout=l)
#plot(p2_l)
#plot(x_values,c_MPRK_LF[t1_values,:])
#plot!(x_values,MOL_SSPRK_LF[t1_values,:])
#plot!(x_values,U_classic_LF[t1_values,:])
#plot!(x_values,exakt_t1)
