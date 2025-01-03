# Running this easily
Recommended to use Spyder, as it would contain scipy and numpy. https://www.spyder-ide.org/

# What is it solving
It is solving the following mechanics problem, where we apply Newtonian gravity
```math
\begin{gather}
m \mathbf{a} = - \frac{G M m}{|\mathbf{r}|^2}
\\
\end{gather}
```
```math
\begin{align*}
\text{where : } \mathbf{r} &= x \mathbf{e}_x +  y \mathbf{e}_y + z \mathbf{e}_z
\\
\mathbf{a} &= \frac{\mathrm{d} \mathbf{v}}{\mathrm{d} t} = \frac{\mathrm{d}^2 \mathbf{r}}{\mathrm{d} t^2}
\end{align*}
```
Our initial conditions are that of a perfectly rigid vertical ladder, and a ball being released at the top. This leads to the following initial conditions
[TBC]
```math
\begin{align*}
x(t_0) &= r \cos{(\theta)}
\end{align*}
```
This ordinary differential equation (ODE) is solved numerically using `SciPy`'s `solve_ivp` function.
