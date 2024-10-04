In this file I'm going to put all needed formulae for time and space discretization.

So we want to use a staggered grid to recover the coupling between even and odd nodes.

For ease of notation I will refer to $\mathbf{u}$ as the  velocity in vector components $\vec{u} = (u, v, w)^T$ to avoid useless repetition i will call u the $u_{x}$ component, v the $u_{y}$, and w the $u_{z}$ component.

Also another usefull tipical notation is to indicate $$ \frac{\partial{ \Phi }}{\partial{ \psi }} = \partial_{ \psi}{ \Phi }$$

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


AT THIS POINT THE FIRST EQ IS TOTALLY DISCRETIZED, LET'S GO FOR THE SECOND STEP.


<br> $$\nabla ^2( p^{n+1} - p^{n}) = \frac{\nabla \cdot \mathbf{u}^{*} }{\Delta t} \text{in} \Omega$$ (1) 
<br> $$\partial_{n}{(p^{n+1} - p^{n}) } = 0 \text{   on  } \partial \Omega $$ (2) 
<br> $$ \mathbf{u}^{n+1} = \mathbf{u}^{*} - \nabla (p^{n+1} - p^{n} ) \text{in} \Omega $$ (3)

Starting with (2), playing a bit with the equations (You'll see the image I made on paper or later I try to include them in this file) we can end up with the system below where A in general is a point on $\partial \Omega$ and C is the first point along the normal direction (But opposite verse since n point outwards). Let's suppose we're on the face of the inlet and A is the point (0, 1, 1) then C is (1, 1, 1) (We skip the 0,5 since is just for the velocity!)
$$
\frac{p_{C}^{n+1} -p_{A}^{n+1} }{2*\Delta x} = \frac{p_{C}^{n} -p_{A}^{n} }{2*\Delta x} <br>
\text{Simplifying the denominators since they both are equal and remembering that we have} p_{A}^{n+1} \text{from the BC}^s \text{we can then recover the value for} p_{C}^{n+1} = p_{A}^{n+1} + p_{C}^{n} -p_{A}^{n}
$$

At this point we have initialized the value for the internal points of the domain ( $\Omega$ besides   $\partial \Omega$ to be heavy)

$$\nabla^2 p^{n+1} = \underbrace{ \nabla ^2 p^{n} + \frac{1}{\Delta t} \cdot \nabla \cdot \mathbf{u}^{*} }_{\text{  This whole part is know, I will just call it } f \text{from now on}} $$
$$
f|_{i, j, k} = \frac{ p_{i+1, j, k}^{n} -2 p_{i, j, k}^{n} + p_{i-1, j, k}^{n}  }{\Delta x} +  \frac{ p_{i, j, k+1}^{n} -2 p_{i, j, k}^{n} + p_{i, j-1, k}^{n}  }{\Delta y} +  \frac{ p_{i, j, k+1}^{n} -2 p_{i, j, k}^{n} + p_{i, j, k-1}^{n}  }{\Delta z} +  \frac{1}{\Delta t} \cdot (\frac{u_{i-0.5, j, k}^{*} - u_{i+0.5, j, k}^{*}    }{\Delta x}  + \frac{v_{i, j-0.5, k}^{*} - v_{i, j+0.5, k}^{*}    }{\Delta y} +  \frac{w_{i, j, k-0.5}^{*} - w_{i, j, k+0.5}^{*}    }{\Delta z}  )
$$
Now the _not so much fun_ part, cause here we may encounter a problem since we have 3 unknowns ($p_{i+1, j, k}^{n+1}$  , $p_{i, j+1, k}^{n+1}$ and $p_{i, j, k+1}^{n+1}$) I have no clue on how to avoid this problem but to use a decentered scheme (Obv we need to keep it still second order) or just solve the linear system. 
Blindly applying the second order scheme studied earlier we obtain: 
$$
\frac{ p_{i+1, j, k}^{n+1} -2 p_{i, j, k}^{n+1} + p_{i-1, j, k}^{n+1}  }{\Delta x} +  \frac{ p_{i, j, k+1}^{n+1} -2 p_{i, j, k}^{n+1} + p_{i, j-1, k}^{n+1}  }{\Delta y} +  \frac{ p_{i, j, k+1}^{n+1} -2 p_{i, j, k}^{n+1} + p_{i, j, k-1}^{n+1}  }{\Delta z} = f|_{i, j, k}
$$

Once we get how to sort out this problem the next step is trivial. Ok actually thinking about it it can be build like a pyramid but has to be studied in a optimal way.

$$
u_{i+0.5, j, k}^{n+1} = u_{i+0.5, j, k} ^{*}- \Delta t \cdot \frac{ p^{n+1}_{i, j, k} - p^{n+1}_{i-1, j, k}  }{\Delta x} 
$$
$$
v_{i, j+0.5, k}^{n+1} = v_{i, j+0.5, k} ^{*}- \Delta t \cdot \frac{ p^{n+1}_{i, j, k} - p^{n+1}_{i, j-1, k}  }{\Delta y} 
$$
$$
w_{i+0.5, j, k}^{n+1} = w_{i, j, k+0.5}^{*} - \Delta t \cdot \frac{ p^{n+1}_{i, j, k} - p^{n+1}_{i, j, k-1}  }{\Delta z} 
$$
