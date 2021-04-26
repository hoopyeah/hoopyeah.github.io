Title: Notes on Chapter 10
Date: 2021-04-26 20:40
Tags: Neural Networks, AI, ML
Author: Cody Fernandez
Summary: My notes on Chapter 10 of __Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow__ by Aurelien Geron


# Chapter 10
- Artificial neuron
    - 1 or more binary inputs
    - 1 binary output
    - activates output when more than a certain number of its inputs are active
    - a neuron is activated when at least 2 inputs are active
- A network of artificial neurons can compute any logical proposition
- Perceptron based on
    - threshold logic unit (TLU) or
    - linear threshold unit (LTU)
- TLU
    - inputs and outputs are numbers
        - each input connection has an associated weight
    - computes a weighted sum of its inputs
        - $z=w_1x_1 + w_2x_2+ \cdots + x_nw_n = \vec{x}^T\vec{w}$
    - then applies a step function and outputs result
        - $h_w(\vec{x})=step(z)$
- Step functions
$$heaviside(z) = \begin{cases} 0, & \mbox{if } z>0 \\ 1, & \mbox{if } z>1\end{cases}$$
$$sgn(z) = \begin{cases} -1, & \mbox{if }z<0 \\ 0, & \mbox{if } z=0 \\ 1 & \mbox{if } z>0\end{cases}$$
- Perceptron is composed of a single layer of TLUs, with each TLU connected to all the inputs
- Compute the outputs of a fully connected layer
    - $h_{\vec{w},\vec{b}}(\vec{X})=\phi(\vec{X}\vec{w}+\vec{b})$
        - $\vec{X}$: input features
        - $\vec{w}$: weight matrix (of all connection weights)
        - $\vec{b}$: bias vector (one bias term per neuron)
        - $\phi$: activation function
- Hebb's rule: ``Cells that fire together, wire together'', or the connection weight between two neurons tends to increase when they fire simultaneously.
- $w_{i,y}^{(next step)}=w_{i,j}+\eta(y_j-\hat{y}_j)x_i$ - Perceptron learning rule (weight update)
    - $w_{i,j}$: connection weight between $i^{th}$ input neuron and $j^{th}$ output neuron
    - $x_i$: $i^{th}$ input value of current training instance
    - $\hat{y}_j$: output of $j^{th}$ output neuron for current training instance
    - $y_i$: target output of $j^{th}$ output neuron for current training instance
    - $\eta$: learning rate
- Perceptrons kinda suck - they can't even XOR
- Multilayered Perceptrons (MLPs)
    - stacking multiple perceptrons in layers
    - can totally XOR
    - Structure:
        - 1 input layer (passthrough)
        - 1 or more hidden layers (of TLUs) [lower layers $\rightarrow$ upper layers]
        - 1 output layer (of TLUs)
    - every layer except output has a bias neuron and is fully connected to the next layer
    - feedforward neural network: signal flows in one direction, from input to output
    - deep neural net: ANN with a "deep" stack of hidden layers
        - how deep? 2 or dozens (lol)
- Backpropagation: Gradient decent with one forward pass and one backward pass. It computes the gradient of the network's error with regard to every single model parameter. With these gradients, it performs another gradient descent, and repeats until it converges to the soluion.
    - autodiff - automatic differentiation $\rightarrow$ automatically computing gradients. Backpropagation uses "reverse-mode autodiff" $\rightarrow$ see Appendix D
- In detail:
    1. One minibatch at a time. Go through the full training set multiple times. Each pass is an "epoch"
    2. Pass a minibatch to the input layer. Compute the output of all neurons. Pass to the next layer. This is the "forward pass". We have preserved all intermediate results along the way
    3. Measure the network's output error (use a loss function)
    4. Compute each output connection's contribution to the error using the chain rule
    5. Measure the layer below's error contributions to the current layer using the chain rule. This measures the error gradient across all connection weights as we propagate backward through the network (backpropagation, the "backward pass").
    6. Perform Gradient Descent using these error gradients to adjust all the network's connection weights.
    Summary: FOR each training instance, make a prediction (forward pass) and measure error, go through each layer in reverse to measure error contribution of each connection (reverse pass), tweak connection weights to reduce error (Gradient Descent).
- **Initialize all hidden layers' connection weights randomly**
- Change MLP activation function to something with a well-defined non-zero derivative everywhere. Choices include:
    - logistic (sigmoid):
        - $\sigma(z)=\frac{1}{1+e^{-z}}$    $[0 \rightarrow 1]$
    - hyperbolic tangent:
        - $\tanh(z)=2\sigma(2z)-1$    $[-1 \rightarrow 1]$
    - Rectified Linear Unit (ReLU):
        - $ReLU(z)=max(0,z)$    $[0 \rightarrow z]$
        - fast to compute, current default
- We need activation functions to introduce nonlinearities 
    - Any large-enough DNN can approximate any continuous function
- Regression MLP - output continuous
    - \# input neurons $\rightarrow$ one per input feature
    - \# hidden layers $\rightarrow$ 1-5 typically
    - \# neurons per hidden layer $\rightarrow$ 10-100 typically
    - \# output neurons $\rightarrow$ 1 per prediction dimension
    - Hidden activation $\rightarrow$ ReLU (or SELU)
    - Output activation $\rightarrow$ None, ReLU/Softmax (positive outputs), logistic/tanh (bounded ouptputs)
    - Loss function $\rightarrow$ MSE or MAE/Huber (if outliers)
- Softplus: (smooth variant of ReLU)
    - $softplus(z)=\log(1+e^z)$    $[0 \rightarrow z]$

Classification MLPs

|Hyperparameter          |Binary        |Multilabel Binary |Multiclass|
|:---                    |---:          |---:              |---:|
|\# output neurons       |1             |1 per label       |1 per class|
|Output layer activation |Logistic      |Logistic          |Softmax|
|Loss function           |cross entropy |cross entropy     |cross entropy|


- binary classification: single output neuron, logistic activation, 0 or 1
- multilabel binary classification: X output neurons, logistic activation, 0 or 1, any combination of output 0's and 1's possible
- multiclass classification: one output neuron per class, softmax activation: ensures all probabilities use $0\rightarrow1$ and add up to 1 (required if classes are exclusive)
- Fine-tuning NN Hyperparameters
    - perform K-fold cross-validation to find best learning rate, number of hidden layers, and number of neurons per hidden layer by testing all combinations.
    - Or use a hyperparameter optimizing toolkit
    - Perhaps evolutionary algorithms are replacing gradient descent? (Deep Neuro evolution by Uber).
- Number of Hidden Layers
    - deep networks have much higher "parameter efficiency": modeling complex functions using exponentially fewer neurons, thus better performance using exponentialy fewer neurons, thus better performance with same training data 
    - real-world data is structured. lower-level hidden layers model low-level structures, and increasingly higher layers model increasingly higher-level structures (increasingly abstract structures as well).
    - Transfer learning - taking your pre-trained lower layers and applying them to a new problem.
    - Start with 1 or 2 hidden layers. Or just ramp up the hidden layers until you start overfitting the training data. Or do transfer learning with parts of an existing model.
- Number of Neurons per hidden layer
    - input and output layer neuron numbers dependent on problem
    - pyramid lower $\rightarrow$ higher layers? Nope, old. and adds more hyperparameters to tune (neurons per layer)
    - Same on all layers? Yes. One hyperparameter to tune, no change in performance.
    - Okay, maybe have first hidden layer larger than others
    - Use "stretch pants" approach: Pick a model with more layers and neurons than necessary, then use early stopping and regularization techniques to avoid overfitting.
- $\star$ Almost always better to increase number of layers than to increase number of neurons.
- Learning Rate
    - the most important hyperparameter
    - optimal learning rate is half the max learning rate. (the learning rate above which the training algorithm diverges).
    - Start low and go up, training the model for a few hundred iterations: $\exp^{\frac{\log(10^6)}{500}}$ goes from $10^{-5}$ to $10$ in $500$ iterations. Plot loss as log-scale function of learning rate, see the turning point where loss starts to climb again, pick the learning rate as $\frac{1}{10}$ of that turning point. Then retrain with this good learning rate.
    - Or use a different learning rate technique (TBD)
- Optimizers
    - Use a different optimizer (TBD)
- Batch size
    - Nerd fight!
        1. Use the biggest batch possible, with GPU RAM as limiting factor
             - this could lead to training instabilities (especially early in training) and a generalization gap
        2. Use small batches (2 to 32)
             - better models in less training time
        3. Use a large batch size, learning rate warmup
             - short traiing time, no generalization gap
             - if it sucks, just use small batches instead
- Activation function
    - when in doubt, ReLU
- Number of interations
    - Don't bother, just use early stopping
- $\star$ Check out 2018 Leslie Smith "Disciplined Approach..."
