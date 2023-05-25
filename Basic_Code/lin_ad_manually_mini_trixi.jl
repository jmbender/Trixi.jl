using Trixi, OrdinaryDiffEq, Plots, PolynomialBases

include("./lin_ad_manually.jl")
import .disc

# equation with a advection_velocity of `1`.
advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

coordinates_min = 0.0 # minimum coordinate
coordinates_max = 2.0 # maximum coordinate
init = 6
N = 2^init
dx = abs(coordinates_max-coordinates_min)/N
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=init, # number of elements = 2^4
                n_cells_max=30_000)

dt = 0.9*dx
tspan = (0.0, 2.0)
timesteps = Int(round(last(tspan)/dt))

LF(u_l,u_r) = 0.5*(u_l+u_r)-dx/(2*dt)*(u_r-u_l) 
LW(u_l,u_r) = 0.5*(u_l+u_r)-dt/(2*dx)*(u_r-u_l) #Lax-Wendroff


initial_condition_sine_wave(x, t, equations) = SVector(1.0 + 0.5 * sin(pi * sum(x - equations.advection_velocity * t)))
function initial_condition_block(x,t,equations)
    c = 0.001
    mid = 1.0
    if mid == 0.0
        if x[1] < 0.25 
            return SVector(1.0)
        elseif x[1] > 1.75
            return SVector(1.0)
        else return SVector(c)
        end
    elseif 0.25 > mid > 1.75
        if x[1] < mid+0.25
            return SVector(1.0)
        elseif x[1] > mid-0.25
            return SVector(1.0)
        else return SVector(c)
        end
    elseif abs(mod(mid-0.25,2)) < x[1] < abs(mod(mid+0.25,2))
        return SVector(1.0)
    else
        return SVector(c)
    end
end

#only used for computation of coefficients
dg = DGSEM(polydeg=1)
#semi = SemidiscretizationHyperbolic(mesh,equations,initial_condition_sine_wave,dg)
semi = SemidiscretizationHyperbolic(mesh,equations,initial_condition_block,dg)
cache = semi.cache

u0 = compute_coefficients(first(tspan),semi)

#u0 double saves grid values; u0_short reduces them 
# => same vector size as in direct implementation
u0_short = zeros(nelements(dg,cache))
u0_wrap = Trixi.wrap_array(u0,mesh,equations,dg,cache)
for element in eachelement(dg,cache)
    for i in eachnode(dg)
        u0_short[element] = Trixi.get_node_vars(u0_wrap,equations,dg,2,element)[1]
    end
end

###################################################################
#functions from lin_ad_manually file, as in first try

#return matrices including all timesteps
FV_LF = disc.FV_flux_diff(dt,last(tspan),dx,N,u0_short[:,1],LF); 
FV_LW = disc.FV_flux_diff(dt,last(tspan),dx,N,u0_short[:,1],LW);

c_MPRK_LF = disc.MP_Heun(dt,dx,last(tspan),N,LF,u0_short[:,1]);
c_MPRK_LW = disc.MP_Heun(dt,dx,last(tspan),N,LW,u0_short[:,1]);

MOL_SSPRK_LF = disc.MOL_SSPRK(dt,last(tspan),dx,N,u0_short[:,1],LF);
MOL_SSPRK_LW = disc.MOL_SSPRK(dt,last(tspan),dx,N,u0_short[:,1],LW);

###################################################################
#general setup visualization
t1 = 0.1 #plotted time
T = Int(round(t1/dt))

###################################################################
#plot with Trixi !!!does not work!!!
###################################################################

#Trixi always needs size of variables*dimensions*nodes*elements, while every gridcell
# is one element with two nodes (case polydeg=1, doubles cell interface values)
function blow(v)
    sz = length(v)
    v_long = zeros(2*sz)
    v_long[1] = v[1]
    for i in 1:sz
        v_long[2*i-1] = v[i]
        v_long[2*i] = v[i]
    end
    return v_long
end

FV_LF_long = blow(FV_LF[T,:])
FV_LW_long = blow(FV_LW[T,:])

c_MPRK_LF_long = blow(c_MPRK_LF[T,:])
c_MPRK_LW_long = blow(c_MPRK_LW[T,:])

MOL_SSPRK_LF_long = blow(MOL_SSPRK_LF[T,:])
MOL_SSPRK_LW_long = blow(MOL_SSPRK_LW[T,:])

#pd = PlotData1D(FV_LF_long,semi)
#pd = PlotData1D(FV_LW_long,semi)
plot!(pd,label="Trixi LW")

#pd = PlotData1D(c_MPRK_LF_long,semi)
pd = PlotData1D(c_MPRK_LW_long,semi)
plot!(pd,label="Trixi MPRK")

#pd = PlotData1D(MOL_SSPRK_LF_long,semi)
#pd = PlotData1D(MOL_SSPRK_LW_long,semi)
#plot!(pd,label="Trixi MOL")
#plot!(getmesh(pd))

####################################################################
#plot with simply Plots.jl
####################################################################

grid = collect(coordinates_min:dx:coordinates_max);
x_values = grid[1:(end-1)].+(dx/2);
#plot!(x_values,FV_LF[T,:],label="simple")
plot(x_values,FV_LW[T,:],label="simple LW")
#scatter(x_values,FV_LW[T,:],label="simple LW")

#plot!(x_values,c_MPRK_LF[T,:],label="simple")
plot!(x_values,c_MPRK_LW[T,:],label="simple MPRK")
#scatter!(x_values,c_MPRK_LW[T,:],label="simple MPRK")


#plot!(x_values,MOL_SSPRK_LF[T,:],label="simple")
#plot!(x_values,MOL_SSPRK_LW[T,:],label="simple MOL")
#scatter!(x_values,MOL_SSPRK_LW[T,:],label="simple MOL")

plot!(legend=:outerbottom)