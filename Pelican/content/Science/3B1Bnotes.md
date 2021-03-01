Title: Notes on 3B1B ML Videos
Date: 2021-02-23 11:26
Tags: Neural Networks, AI, ML
Author: Cody Fernandez
Summary: Collecting the notes I took while watching the 3Blue1Brown video series on neural networks.

# 3Blue1Brown video notes
I watch the four-episode playlist on Youtube. Here's what I felt compelled to write down.

## Episode 1
- Neuron - a thing that holds a number
- 28x28 pixels = 784 neurons
- value of neuron is its "activation"
- these 784 neurons make up hte first layer of the network
- last layer is 10 neurons, one for each digit choice
  - activation of these neuorons is 0 -> 1, likelihood digit is that number
- "Hidden layers" -> the black box
- 2 layers of 16 neurons - arbitrary, lots of room to experiment with structure
- it's all edge detection (not acutally true)
- what parameters should the network have?
- assign weights to each connection from one layer to the next, then compute the weighted sum of activations per neuron
- activations should be between zero and one

Sigmoid or logistic curve:
$$
\sigma(x)=\frac{1}{1+e^{x}}
$$

- so the activation is basically a measure of how positive the relevant weighted sum is
- bias against inactivity, so add bias to weighted sum before applying the sigmoid function
- learning -> finding the right weights and biases
- "What are the weights and biases doing?"
$$
a^{(1)}=\sigma(Wa^{(0)}+b)
$$

- neuron - actually, a function: inputs are the previous neurons, output is one number
- sigmoid -> old school. ReLU -> easier to train. $ReLU(a)=max(0,a)$. Rectified Linear Unit.

## Episode 2
- cost function - taken set of weights and biases as input, averages to single number for output
- find the minimum of the cost function
- no guarantee on local minimum
- gradient - direction of steepest descent
- length of gradient also reveals steepness
- take $-\nabla C(\vec{W})$ 
- learning is just minimizing a cost function
  - also cost function should be smooth
- gradient dscent -> sign of each weigt tells direction of change, value tells magnitude of change
- old tech -> multilayer peceptron

## Episode 3
- backpropagation - computes the gradient
- change weights in proportion to activations
- change activations in proporiton to weights
- starting from outputs, sum desired changes on the preceding netowek layer for each neuron of the output layer
- Recursively apply across layers, average over all data 
- stochastic gradient descent - gd in mini-batches

## Episode 4
- For one example (e.g. image):
$$
\begin{array}{rcl} 
Cost C_{0}(\ldots) & = & {(a^{(L)}-y)}^2 \\
z^{L} & = & w^{L}a^{l-1}+b^{L}\\
a^{L} & = & \sigma(z^{L})
\end{array} 
$$

```
{
    w  a  b
    |__|__|
       |
       z
       |
    y__a
       |
       C
}
```

- y is desired output
- how sensitive is the cost function to small changes in weights?

$$
\begin{array}{rcl}
\frac{\partial C_0}{\partial w^{(L)}} & = &  \frac{\partial z^{(L)}}{\partial w^{(L)}} \frac{\partial a^{(L)}}{\partial z^{(L)}} \frac{\partial C_0}{\partial a^{(L)}}\\
\frac{\partial C_0}{\partial w^{(L)}} & = & 2(a^{(L)}-y)\\
\frac{\partial a^{(L)}}{\partial z^{(L)}} & = & \sigma^{\prime} (z^{(L)})\\
\frac{\partial z^{(L)}}{\partial w^{(L)}} & = & a^{(L-1)}
\end{array}
$$

- That last one depends on the strength of the previous neuron

$$
\frac{\partial C}{\partial w^{(L)}} = \frac{1}{n} \sum_{k = 1}^{n-1} \frac{\partial C_k}{\partial w^{(L)}}\\
\begin{align}
   \nabla C  &= \begin{bmatrix}
                \frac{\partial C}{\partial w^{(1)}} \\
                \vdots \\
                \frac{\partial C}{\partial w^{(L)}}
                \end{bmatrix}
\end{align}
$$