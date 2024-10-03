In this file I'm going to put all needed formulae for time and space discretization.

So we want to use a staggered grid to recover the coupling between even and odd nodes.

For ease of notation I will refer to $\mathbf{u}$ as the  velocity in vector components $\vec{u} = (u, v, w)^T$ to avoid useless repetition i will call u the $u_{x}$ component, v the $u_{y}$, and w the $u_{z}$ component.

Also another usefull tipical notation is to indicate $$ \frac{\partial{ K }}{\partial{ L }} as \partial_{L}{ K }$$

We want to use a 2 step second order method such that:

- I step: <br>
$\mathbf{u}^{*}  = \mathbf{u}^{n} + \Delta t \cdot (  - \mathbf{ \mathbf{u}^{**}} \cdot \nabla \mathbf{u}^{**}+ \frac{ 1}{R\text{e}} \cdot \nabla^2 \mathbf{u}^{**} - \nabla p^{n} )$
<br>
NOTE:
<br> We need to calculate all the terms above for $u^{*}|_{i +0.5, j, k}$, $w^{*}|_{i, j, k+0.5}$ and $ v^{*}|_{i, j+0.5, k} $
<br> $\mathbf{u}^{**}$ has to be provided as function of $\mathbf{u}^{n}$, the prof didn't give use the formula yet !!

- II step:
<br> $$\nabla ^2( p^{n+1} - p^{n}) = \frac{\nabla \cdot \mathbf{u}^{*} }{\delta t} 
\text{  That comes with the necessary BCs to have a well defined system } \partial_{n}{(p^{n+1} - p^{n}) } = 0 \text{   on  } \partial \Omega $$ 
<br> $$ \mathbf{u}^{n+1} = \mathbf{u}^{*} - \nabla (p^{n+1} - p^{n} ) $$

OK so I start as in the photo with the discretization of $\nabla P$ needed in the momentum equations. We want to evaluate it in all point where we evaluate $\mathbf{u}$

$$\partial_{x }{ P } |_{i+0.5, j, k} \simeq  \frac{ P_{i +1, j, k} - P_ {i , j, k} }{ \Delta x} $$
$$\partial_{y }{ P } |_{i, j+0.5, k} \simeq  \frac{ P_{i , j+1, k} - P_ {i , j, k} }{ \Delta y} $$
$$\partial_{z }{ P } |_{i, j, k+0.5} \simeq  \frac{ P_{i , j, k+1} - P_ {i , j, k} }{ \Delta z} $$

At this point the biggest problem of the system: the non linear part $\mathbf{u} \cdot \nabla \mathbf{u} $ in Einstein's notation becomes : $ u_{i}\partial_{ x_{i} }{ u_{j} } $, in explicit form is a vector with components:
(I'm a noob in \LaTeX with very slow internet at home so I just did the vector in column, I couldn't figure out how to do the damn braces)
$\mathbf{u}^{**} \cdot \nabla \mathbf{u}^{**} =  \begin{matrix}  u^{**} \cdot \partial_{ x}{ u }^{**} + v^{**} \cdot \partial_{ y}{u}^{**} + w^{**}\cdot\partial_{z }{u }^{**}  \\ u^{**} \cdot\partial_{ x}{ v }^{**} + v^{**}\cdot \partial_{ y}{v}^{**} + w^{**}\cdot\partial_{z }{v }^{**} \\  u^{**} \cdot\partial_{ x}{ w }^{**} + v^{**} \cdot\partial_{ y}{w}^{**} + w^{**}\cdot\partial_{z }{w }^{**} \end{matrix} $

The discretized form should be like this: 

$\mathbf{u}^{**} \cdot \nabla \mathbf{u}^{**} |_{i, j, k} \simeq  \begin{matrix}
u^{**}|_{i, j, k} \cdot \frac{ { u }^{**}|_{i+0.5, j, k} - { u }^{**}|_{i-0.5, j, k} }{\Delta x}+ v^{**}|_{i, j, k} \cdot \frac{ { u }^{**}|_{i, j+0.5, k} - { u }^{**}|_{i, j-0.5, k} }{\Delta y}+w^{**}|_{i, j, k} \cdot \frac{ { u }^{**}|_{i, j, k+0.5} - { u }^{**}|_{i, j, k-0.5} }{\Delta z}
\\
u^{**}|_{i, j, k} \cdot \frac{ { v }^{**}|_{i+0.5, j, k} - { v }^{**}|_{i-0.5, j, k} }{\Delta x}+ v^{**}|_{i, j, k} \cdot \frac{ { v }^{**}|_{i, j+0.5, k} - { v }^{**}|_{i, j-0.5, k} }{\Delta y}+w^{**}|_{i, j, k} \cdot \frac{ { v }^{**}|_{i, j, k+0.5} - { v }^{**}|_{i, j, k-0.5} }{\Delta z}
\\
u^{**}|_{i, j, k} \cdot \frac{ { w }^{**}|_{i+0.5, j, k} - { w }^{**}|_{i-0.5, j, k} }{\Delta x}+
v^{**}|_{i, j, k} \cdot \frac{ { w }^{**}|_{i, j+0.5, k} - { w }^{**}|_{i, j-0.5, k} }{\Delta y}+
w^{**}|_{i, j, k} \cdot \frac{ { w }^{**}|_{i, j, k+0.5} - { w }^{**}|_{i, j, k-0.5} }{\Delta z}
 \end{matrix} $



We need to evaluate it in the points where we look for the velocity (In the first step)


> The discretization of the laplacian was not provided, but it would assume it can be done as we did in the second lecture:
Here is the discretizzation made for the 3 components:

$$  (\frac{ 1}{R\text{e}} \cdot \nabla^2 u^{**} )|_{i, j, k} \simeq \frac{ 1}{R\text{e}} \cdot ( \frac{{u}_{i+0.5, j, k}^{**} -2 \cdot {u}_{i, j, k}^{**} + {u}_{i -0.5, j, k}^{**} }{\Delta x^2} +  \frac{{u}_{i, j+0.5, k}^{**} -2 \cdot {u}_{i, j, k}^{**} + {u}_{i, j+0.5, k}^{**} }{\Delta y^2} +  \frac{{u}_{i, j, k+0.5}^{**} -2 \cdot {u}_{i, j, k}^{**} + {u}_{i, j, k-0.5}^{**} }{\Delta z^2} ) $$ 
$$  (\frac{ 1}{R\text{e}} \cdot \nabla^2 v^{**} )|_{i, j, k} \simeq \frac{ 1}{R\text{e}} \cdot ( \frac{{v}_{i+0.5, j, k}^{**} -2 \cdot {v}_{i, j, k}^{**} + {v}_{i -0.5, j, k}^{**} }{\Delta x^2} + \frac{{v}_{i, j+0.5, k}^{**} -2 \cdot {v}_{i, j, k}^{**} + {v}_{i, j+0.5, k}^{**} }{\Delta y^2} + \frac{{v}_{i, j, k+0.5}^{**} -2 \cdot {v}_{i, j, k}^{**} + {v}_{i, j, k-0.5}^{**} }{\Delta z^2} ) $$ 
$$  (\frac{ 1}{R\text{e}} \cdot \nabla^2 w^{**} )|_{i, j, k} \simeq \frac{ 1}{R\text{e}} \cdot ( \frac{{w}_{i+0.5, j, k}^{**} -2 \cdot {w}_{i, j, k}^{**} + {w}_{i -0.5, j, k}^{**} }{\Delta x^2} +  \frac{{w}_{i, j+0.5, k}^{**} -2 \cdot {w}_{i, j, k}^{**} + {w}_{i, j+0.5, k}^{**} }{\Delta y^2} + \frac{{w}_{i, j, k+0.5}^{**} -2 \cdot {w}_{i, j, k}^{**} + {w}_{i, j, k-0.5}^{**} }{\Delta z^2} ) $$ 


AT THIS POINT THE FIRST EQ IS TOTLLY DISCRETIZED, I MISS THE SECOND STEP.



