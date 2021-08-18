Title: Notes on Adjoint Methods
Date: 2021-08-08 10:40
Tags: Math
Author: Cody Fernandez
Summary: Collecting the notes I took while reading adjoint method primers

# MIT Notes
- output $\vec{x}$
- set of $M$ equations parameterized by $P$ variables $\vec{p}$. These $\vec{p}$ could be:
    - design parameters
    - control variables
    - decision parameters
- compute $g(\vec{x}, \vec{p})$. $g$ might be loss funciton of a neural network comparing prediction $\vec{x}$ to data
- Also want gradient $\frac{dg}{d\vec{p}}$. *Adjoint methods* give efficient way to evaluate $\frac{dg}{d\vec{p}}$, with cost independent of $P$ and comparable to the cost of solving for $\vec{x}$ once. 
- In neural network, adjoint methods are called backpropagation. In automatic differentiation they are reverse-mode differentiation.
- So $g(\vec{x}, \vec{p})$
    - measures sensitivity of answers to $\vec{p}$
    - indicates a useful search direction to optimize $g$, picking the $\vec{p}$ that produce some result
- shape or topology optimization: $\vec{p}$ controls the shape and placement of blobs, a.k.a. *inverse design*. $P$ can be hundreds, thousands, millions.
- Take $A\vec{x}=\vec{b}$. $A$ and $\vec{b}$ are real and depend in some way on $\vec{p}$. Evaluate the gradient directly, where subscripts indicate partial derivatives:
$$
\frac{dg}{d\vec{p}}=g_{\vec{p}}+g_{\vec{x}}\vec{x}_{\vec{p}}
$$
