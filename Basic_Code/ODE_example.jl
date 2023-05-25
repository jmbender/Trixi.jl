using OrdinaryDiffEq
f(u,p,t) = 1.01*u
u0=1/2
tspan = (0.0,1.0)
#dt = 0.1
prob = ODEProblem(f,u0,tspan)
sol = solve(prob,Euler(),dt=0.1,reltol=1e-8,abstol=1e-8)
using Plots
plot(sol,linewidth=5,title="Solution to the linear ODE with a thick line",
     xaxis="Time (t)",yaxis="u(t) (in μm)",label="My Thick Line!") # legend=false
plot!(sol.t, t->0.5*exp(1.01t),lw=3,ls=:dash,label="True Solution!")