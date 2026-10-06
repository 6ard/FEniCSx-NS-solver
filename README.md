# FEniCSx Navier-Stokes (NS) Solver
---
Technical solver details:
- Taylor-Hood elements for velocity and pressure
- BDF2 time-stepping scheme

Goals for this solver:
- 2D Exact Taylor-Green vortex
    - Compare norms
- 3D Exact solution Steinmann vortex problem
    - Compare norms
- 2D DFG benchmark problem
    - Does solver preform as expected?
- Simple anulus spinal geometry
---
Current status:
- 2D Taylor exact solution is implemented 

---

## Taylor-Green Vortex Problem [[1]](https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex)

The original work describes a 3 dimensional flow. It's defined by the three velocity components $\boldsymbol v = (u,v,w)$ at time $t = 0$ governed bt the three equations:

$$
\begin{align*}
u = A \cos(ax)\sin(by)\sin(cz), \\
v = B \sin(ax)\cos(by)\sin(cz), \\
w = C \sin(ax)\sin(by)\cos(cz). 
\end{align*}
$$

The continuity eq. $\nabla \cdot \boldsymbol v = 0$ determines that $Aa + Bb + Cc = 0$.

An exact solution is known in two spatial dimensions, and it's what we will try to replicate with the FEniCSx NS solver.

### The Incompressible NS eq.
The incompressible Navier-Stokes equations are given by:

$$
\begin{align*}
&\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0, \\
&\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = - \frac 1 \rho \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right), \\
&\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = - \frac 1 \rho \frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right).
\end{align*}
$$

The first equation is the continuity equation. The two under are the momentum equations.


### Taylor-Green Vortex Exact Solution

In an infinite domain, the doubly periodic solution is given by 

$$
\begin{align*}
u = U_0 \sin(kx)\cos(ky)F(t), & \quad v = -U_0 \cos(kx)\sin(ky)F(t), \\
F(t)&=e^{-\nu k^2t}
\end{align*}
$$

Here $U_0$ is the maximum velocity in the flow field, $k$ is the inverse length scale, $\nu$ is the kinematic viscosity. From Taylor and Green we know that with $A=a=b=1$ we get this exact solution. Further we can expand the exponential as a Taylor series, $F(t) = 1 - 2\nu k^2t + \mathcal O(t^2)$. The pressure field $p$ is obtained by substituting the velocity solution in to the momentum equations, 

$$
\begin{align*}
p = \frac{\rho U_0^2}{4}(\cos(2k x)+ \cos (2 k y)) F(t)^2
\end{align*}
$$

With $\rho$ being the fluid density.

The stream function of the Taylor-Green vortex is given by, that satisfies $\boldsymbol v = \nabla \times \psi$,

$$
\begin{align*}
\psi = U_0 \sin(kx)\sin(ky)F(t) \boldsymbol{\hat z}.
\end{align*}
$$

With the similar vorticity given by, that also satisfies $\boldsymbol \omega = \nabla \times \boldsymbol v$,

$$
\begin{align*}
\omega = 2U_0\sin(kx)\sin(ky)F(t)\boldsymbol{\hat z}.   
\end{align*}
$$

---

Taylor-Green exact solution gif:
<img width="1024" height="768" alt="taylor_green" src="https://github.com/user-attachments/assets/9aa2e92b-2ee6-4d79-be34-33a0f6554ffc" />


