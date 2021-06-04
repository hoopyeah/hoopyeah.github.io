Title: Notes on Chapter 11
Date: 2021-04-28 20:40
Tags: Neural Networks, AI, ML
Author: Cody Fernandez
Summary: My notes on Chapter 11 of __Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow__ by Aurelien Geron

# Chapter 11
- Problems that occur when training a DNN:
    - Vanishing gradients: gradients shrink when flowing backward through the DNN
    - Exploding gradients: gradients grow when flowing backward through the DNN
    - Both make it lower layers difficult to train
- You might lack data or it might be too costly to label
- Training may be extremely slow
- Models with millions of parameters severely risk overfitting, especially if there aren't enough training instances or if they're too noisy
- Vanishing gradients: gradients get smaller and smaller as we progress to the lower layers. The hte lower layers weights don't change, thus training never converges to a good solution. 
- exploding gradients: the opposite, weights explode and Gradient Descent diverges instead
- unstable gradients: different layers of hte DNN learn at widely different speeds
- For signal to flow properly, Glorot and He argue:
    - variance of the outputs of each layer must equal the variance of its inputs
    - gradients must have equal variance before and flowing through a layer in the reverse direction
    - not actually possible unless a layer has equal numbers of inputs and neurons (fan-in and fan-out)
    - compromise: initialize randomly where $fan_{avg}=\frac{fan_{in}+fan_{out}}{2}$
        - called Xavier initialization or Glorot initialization
        - when using logistic activation function:
            - Normal distribution with zero-mean and variance $\sigma^2$: $\frac{1}{fan_{avg}}$
            - Or a uniform distribution between $-r \rightarrow +r$ with $r=\sqrt{\frac{3}{fan_{avg}}}$
    - LeCun initialization: $\sigma^2 = \frac{1}{fan_{in}}$
        - equivalent to Glorot when $fan_{in}=fan_{out}$

Different strategies for different activation functions

|Initialization |Activation Functions           |$\sigma^2$ (Normal)   |
|:---           |---:                           |---:                  |
|Glorot         |None, tanh, logisitic, softmax |$\frac{1}{fan_{avg}}$ |
|HE             |ReLU and variants              |$\frac{2}{fan_{in}}$  |
|LeCun          |SELU                           |$\frac{1}{fan_{in}}$  |

- ReLU computes fast and does not saturate for positive values
    - problem: "dying ReLUS". Given a large learning rate, neurons just output zero and never stop outputting zeroes, since the gradient of ReLU is 0 for negative input.
- Leaky ReLU: $ReLU_{\alpha}(z)=max(\alpha z, z)$
    - $\alpha$ defines "leakage". Slope of function for $z<0$, typically 0.01. Ensures leaky ReLUs never die. 
    - Leaky ReLUs always outperform strict ReLU
    - Big leakage ($\alpha =0.2)$ outperformed by small leakage ($\alpha=0.01$)
- Randomized leaky ReLU (RReLU): $\alpha$ is random within a given range during training and fit to an average value during testing. Performed well and acted as a regularizer.
- Parametric Leaky ReLU (PReLU): let $\alpha$ become a parameter modifiable by backpropagation. Strongly outperforms ReLU on large datasets, overfits small datasets.
- ELU (exponential linear unit): outperformed ReLU variants
$$
ELU_{\alpha}(z) = \begin{cases} \alpha (\exp^z-1), & \mbox{if } z<0 \\ z, & \mbox{if } z>0 \end{cases}
$$
- Differences from ReLU:
    - goes negative for $z<0$. Average output thus closer to 0, alleviating vanishing gradient problem, $\alpha$ usually equals 1.
    - nonzero gradient for $z<0$, avoids dead neurons
    - For $\alpha=1$, function is smooth everywhere, including around $z=0$. This helps speed up gradient descent.
    - Slower to compute than ReLU: Faster convergence rate only partially compensates.
- Scaled ELU (SELU)
    - build a NN of exclusively dense layers, have all hidden layers use SELU, then network will self-normalize: output of each layer will tend ot preserne a mean of 0 and standard deviation of 1 during training. Solves vanishing/exploding gradients. Thus SELU outperforms other activations in these nets.
    - Conditions for self-normalization:
        - input must be standardized: 0 mean and 1 standard deviation.
        - initialize hidden layer weights with LeCun normalization
        - network architecture must be sequential
        - (maybe) all layers must be dense
- In general, SELU > ELU > leaky ReLU > ReLU > tanh > logistic
- care about runtime latency? pick leaky ReLU
- network overfitting? RReLU
- huge training set? PReLU
- speed? ReLU
- Batch Normalization
    - vanishing/exploding gradient can still come back during training
    - add an operation before/after the activation function of each hidden layer
        - this operation: zero-centers and normalizes each input, then scales and shifts the result
        - this operation: lets the model learn the optimal scale and mean of each of the layers inputs
    - If you add a BN layer as the first layer of your NN, the BN will approximately standardize your training set.
    - algo estimates each inputs' mean and standard deviation by evaluating the mean and standard deviation of the input over the current mini-batch. In order to zero-center and normalize the inputs.
    - Algorithm
        1. $\vec{\mu}_{B} = \frac{1}{m_B} \sum_{i=1}^{m_B} \vec{x}^{(i)}$
        2. $\vec{\sigma}_{B}^2=\frac{1}{m_B}\sum_{i=1}^{m_B}(\vec{x}^{(i)}-\vec{u}_{B})^2$
        3. $\hat{\vec{x}}^{(i)}=\frac{\vec{x}^{(i)}-\vec{\mu}_B}{\sqrt{\vec{\sigma}_B^2+\epsilon}}$
        4. $\vec{z}^{(i)}=\vec{\gamma}\otimes\hat{\vec{x}}^{(i)}+\vec{\beta}$
    - $\vec{\mu}_B$: vector of input means, evaluated over whole mini-batch B
    - $\vec{\sigma}_B$: vector of input standard deviations evaluated over whole mini-batch B
    - $m_B$: number of instances of mini-batch B
    - $\hat{\vec{x}}^{(i)}$: vector of zero-centered and normalized inputs for instance $i$
    - $\vec{\gamma}$: output scale parameter vector for layer
    - $\otimes$: element-wise multiplication
    - $\vec{\beta}$: output shift (offset) parameter vector per layer. Each input is offset by its corresponding shift parameter.
    - $\epsilon$: tiny number to avoid division by zero (typically $10^{-5}$). Called a smoothing term.
    - $\vec{z}^{(i)}$: output of the BN operation. Rescaled and shifted version of the inputs.
    - During training BN standardizes its inputs, then rescales and offsets them.
    - During testing, final statistics are estimated using a moving average over the layer's input means and standard deviations. $\vec{\gamma}$ and $\vec{\beta}$ are learned during backpropagation, and $\vec{\mu}$ and $\vec{\sigma}$ are estimated using an exponential moving average. 
    - BN also acts as a regularizer.
    - Fuse BN layer with previous layer (after training) to avoid runtime penalty.
    - wall time (measuremd by clock on wall) will be shorter using BN. Though each epoch takes longer, solution will converge much faster.
    - Hyperparameter "momentum" tweaks the running average $\hat{\vec{v}}$: $\hat{\vec{v}} \leftarrow \hat{\vec{v}} \times momentum + \hat{\vec{v}} \times (1-momentum)$
        - 0.9, 0.99, 0.999, (more 9s for larger datasets and smaller mini-batches) 
    - hyperparameter "axis" determines which axis should be normalized.
        - default -1, meaning it will normalize the last axis using the weights and standard deviations of computed across other axes
- Gradient Clipping - clip gradients so they never exceed some threshold. Mainly used in RNNs.
- Transfer Learning - reuse lower layers of an existing NN that addresses a similar task. 
    - Does not work well with small dense networks. Works best with deep CNNs.
- Unsupervised Pretraining
    - use an autoencoder or GAN first on unlabeled data, then reuse the lower layers in a different network with labelled training data.
- Pretrain on auxiliary task
    - Train NN on similar labelled training data, reuse lower layers in actual task
- self-supervised learning: automatically generate labels from teh data itself, then train NN. This is technically unsupervised learning since a human didn't label.
- 4 ways to speed up training
    1. Good initilization strategy for connection weights
    2. Good activation function
    3. Batch Normalization
    4. Reuse lower layers of pretrained network
    5. Faster Optmiizer
- Faster Optimizer
    - momentum optimization - start out slow, then quickly reach terminal velocity
    - Gradient descent weight update:
    - $\vec{\theta}\leftarrow - \eta \nabla_{\theta} J(\vec{\theta})$
    - $\vec{\theta}$: weights
    - $J(\vec{\theta})$: loss function
    - $\eta$: learning rate
    - but now we'll use the gradient for acceleration, not for speed
    - Momentum algorithm
        1. $\vec{m} \leftarrow \beta\vec{m}-\eta \nabla_\theta J(\vec{\theta})$
        2. $\vec{\theta} \leftarrow \vec{\theta} + \vec{m}$
            - $\vec{m}$: momentum vector
            - $\beta$: momentum (like friction). 0 is high friction. 1 is low friciton. Typically 0.9
        - At $\beta = 0.9$ this algorithm will go 10 times faster than Gradient Descent.
        - Escapes plateaus much faster
        - Useful in deep networks that lack Batch Normalization
        - Can oscillate near minimum before stabilizing. Add friction to reduce oscillation time.
    - Nesterov Accelerated Gradient - almost always faster, measures the gradient of the cost function slightly ahead in the direction of momentum
        1. $\vec{m} \leftarrow \beta\vec{m}-\eta \nabla_\theta J(\vec{\theta} + \beta \vec{m})$
        2. $\vec{\theta} \leftarrow \vec{\theta} + \vec{m}$
        - ends up slightly closer to the optimum
    - Adagrad - scales down the gradient vector along the steepest dimensions to correct direction toward global optimum earlier
        1. $\vec{s} \leftarrow \vec{s} + \nabla_{\theta} J(\vec{\theta}) \otimes \nabla_{\theta} J(\vec{\theta})$
        2. $\vec{\theta} \leftarrow \vec{\theta} - \eta \nabla_\theta J(\vec{\theta}) \oslash \sqrt{\vec{s}+\epsilon}$
        - $\vec{s}$: vector of squares of gradients
        - $\epsilon$: typically $10^{-10}$
        - Adaptive learning rate: requires less tuning for $\eta$
        - Never use AdaGrad on NN
    - RMSProp - only accumulate the most recent gradients. 
        1. $\vec{s} \leftarrow \beta \vec{s} + (1-\beta)\nabla_{\theta} J(\vec{\theta}) \otimes \nabla_{\theta} J(\vec{\theta})$
        2. $\vec{\theta} \leftarrow \vec{theta} - \eta \nabla_\theta J(\vec{\theta}) \oslash \sqrt{\vec{s}+\epsilon}$
        - $\beta$: decay rate, usually 0.9
        - preferred algo until Adam
    - Adam: adaptive moment estimation. keeps track of an exponentially decaying average of past gradients AND exponentially decaying average of past square gradients. 
        1. $\vec{m} \leftarrow \beta_1 \vec{m} - (1-\beta_1)\nabla_{\theta} J(\vec{\theta})$
        2. $\vec{s} \leftarrow \beta_2 \vec{s} + (1-\beta_2)\nabla_{\theta} J(\vec{\theta}) \otimes \nabla_{\theta} J(\vec{\theta})$
        3. $\hat{\vec{m}} \leftarrow \frac{\hat{m}}{1-\beta_2^t}$
        4. $\hat{s} \leftarrow \frac{\hat{s}}{1-\beta_2^t}$
        5. $\vec{\theta} \leftarrow \vec{\theta} + \eta \hat{\vec{m}} \oslash \sqrt{\hat{\vec{s}}+\epsilon}$
        - $t$: iteration number (starts at 1)
        - $\epsilon$: typically $10^{-7}$
        - Steps 3 and 4 are bookkeeping. Prevent $\vec{m}$ and $\vec{s}$ from biasing toward 0 at beginning training.
        - Rarely need to tune $\eta$.
    - Adamax - uses $l_{\inf}$ norm rather than $l_2$ norm. Sometimes more stable for some datasets.
    - Nadam - Adam with Nesterov trick
    - Sometimes adaptive optimization methods generalize poorly. Switch to NAG if so.
- Jacobians: first order partial derivatives
- Hessians: second order partial derivatives. $n^2$ Hessians per output. Too big, too slow.
- Apply strong $l_1$ regularization during training to enforce sparsity. (zero out as many weights as possible)

|Class                       |Convergence Speed |Convergence Quality |
|:---                        |---:              |---:                |
|SGD                         |+                 |+++                 |
|SGD with momentum           |++                |+++                 |
|SGD with momentum, Nesterov |++                |+++                 |
|Adagrad                     |+++               |+ (stops too early) |
|RMSProp                     |+++               |++ or +++           |
|Adam                        |+++               |++ or +++           |
|Nadam                       |+++               |++ or +++           |
|AdaMax                      |+++               |++ or +++           |

- Learning Rate Scheduling: goldilock situation
    - learning schedules: to reduce learning rate during training
    - Power scheduling: Set the learning rate to a function of the iteration number $t$: $\eta(t)=\frac{\eta_0}{(1+\frac{t}{s})^c}$
        - $\eta$: initial learning rate
        - $c$: power (typically 1)
        - $s$: steps
        - Schedule drops quickly, then more slowly
        - $\eta_0$, $s$, maybe $c$ must be tuned
    - Exponential scheduling
        - faster than power: $\eta(t)=\eta_0 0.1^{\frac{t}{s}}$
    - Piecewise Constant Scheduling
        - Fiddle with different learning rates for different epochs: $\eta_0=0.1$ for 5 epochs, then $\eta_1=0.001$ for 50 epochs
    - Performance scheduling: measure validation rate every N steps (like early stopping) and reduce learning rate by a factor of $\lambda$ when error stops dropping. 
    - I-cycle scheduling: Increase $\eta_0$ linearly to $\eta_1$ halfway through training. Decrease back to $\eta_0$ linearly, finish last few epochs dropping learning rate by orders of magnitude (still linearly). 
        - find $\eta_1$ using the optimal learning rate approach, then set $\eta_0=\frac{1}{10}\eta_1$
        - If momentum, start high (0.95), drop low (0.85) linearly for first half, back up to high (0.95) for second half, finish last few epochs at max
    - I-cycle > exponential > performance
- Regularization to prevent overfitting: Batch Normalization and early stopping are two good techniques
- Use $l_1$ or $l_2$ regularization to constrain a networks connection weights, computing a regularization loss at each training step, which is added to the final loss. 
- $l_1$ for a sparse model (many weights equal to zero)
- typically apply the same regularizer to all layers in a network.
- Dropout is a regularization technique, adds almost 1-2% accuracy.
    - At almost every training step, every neuron has probability $p$ of being temporarily "dropped out", ignored for this training step.
    - $p$ is also called the "dropout rate", set 10-50%, 20-30% for RNNs, 40-50% for CNNs. Dropout is zero after training 
    - Network with dropout is more robust and generalizes better.
    - In practice, you only add dropout to the top 1-3 layers (excluding the output layer).
    - We must multiply each input connection weight by the "keep probability" $(1-p)$ after training. Or divide each neuron's output by the keep probability during training. This is becasue dropout artificially increases input connection weights.
    - Make sure to evaluate the training loss without dropout (after training), since a model can overfit the training set and have similar training and validation loss. 
    - Model overfitting, $p \uparrow$
    - Model underfitting, $p \downarrow$
    - large layers, $p \uparrow$
    - small layer, $p \downarrow$
    - state-of-the-art architectures only use dropout after last hidden layer.
    - Dropout slows convergence, but results in much better model
    - use alpha dropout for SELU
    - Training a dropout network is mathematically equivalent to approximate Bayesian inference in a Deep Gaussian Process
- Monte Carlo (MC) Dropout - boosts performance, provides better measure of uncertainty, simple to implement.
    - Averaging over multiple with dropout on gives us a MC estimate that is more reliable.
    - MC samples is a tunable hyperparameter, increasing it increases prediction accuracies, uncertainity estimates, and inference time.
- Max-norm regularization: for each neuron, constrain the weights $\vec{w}$ of incoming connections such that $\left\lVert \vec{w} \right\rVert_2 \leq r$, where $r$ is the max-norm hyperparameter, $\left\lVert \right\rVert_2$ is the $l_2$ norm. 
- Typically, compute $\left\lVert \vec{w} \right\rVert_2$ after each training step and rescale $\vec{w}$ if needed: $\vec{w} \leftarrow \frac{\vec{w}r}{\left\lVert \vec{w} \right\rVert_2}$
- Reduce $r$; increase regularization, reduce overfitting
- Max-norm regularization can help alleviate unstable gradients

Default DNN configuration

|Hyperparameter         |Default Value                                                 |Self-Normalizing                            |
|:---                   |---:                                                          |---:                                        |
|Kernal initializer     |He initialization                                             |LeCun initialization                        |
|Activation Function    |ELU                                                           |SELU                                        |
|Normalization          |None if shallow; Batch Norm if deep                           |None (self-normalization)                   |
|Regularization         |Early stopping (plus $l_2$ reg. if needed|+ (stops too early) |Alpha Dropout (if needed)                   |
|Optimizer              |Momentum optimization (or RMSProp or Nadam)                   |Momentum optimization (or RMSProp or Nadam) |
|Learning rate schedule |1-cycle                                                       |1-cycle                                     |

- Normalize input features
- for sparse model, use $l_1$ reg.
- for low-latency model (fast predictions), use fewer layers, fold in BN layers, use RELU, sparse model, drop precision 32 -> 16 -> 8 bits.
- For risk-sensitive or irrelevent inference latency application, use MC Dropout to boost performance, improve reliavility of probability estimates, improve uncertainty estimates.