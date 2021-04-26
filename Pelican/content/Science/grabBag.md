Title: Grab Bag
Date: 2021-03-21 11:26
Tags: Neural Networks, AI, ML, Transformers
Author: Cody Fernandez
Summary: Collecting the notes I took from __Neural Networks and Deep Learning__ by Charu C. Aggarwal, an article by Jason Brownlee, an article by Sumit Saha, and a YouTube video by Yannic Kilcher

# Neural Networks and Deep Learning
- The weight implies the strength of the connection across neurons
- A neural networks can be viewed as a computational graph of elementary units, (such as least squares regression of logistic regression) where the connection of these units are combined in a structured way to enable a more complex function of the input data.
- Two key advantages of NNs:
    - Provides a higher-level abstraction of expressing semantic insights about data domains by architectural design choices in the computational graph.
    - Adding or removing neurons from the achitecture provides a simple way to adjust the complexity of a model.

# [Brownlee](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)
- A convolution is the simple application of a filter to an input that results in an activation.
- CNNs apply a filter to an input to create a feature map that summarizes the presence of detected features in the input.
- Features are abstracted ot higher orders as the depth of hte network is increased. 

# [Saha](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) 
- Conventionally, the first ConvLayer is responsible for capturing the low-level features such as edges, color, and gradient orientation.
- Max pooling performs better than average pooling. It de-noises and reduces dimensionality. Average pooling only reduces dimensionality, maintainin noise.
- The Convolutional Layer and Pooling Layer compine to form the *i*-th layer of a CNN.
- Add a fully-connected layer (multi-level perceptron), flatten the image to a column, feed it into a feed-forward NN, and apply backpropagation to every iteration of training. Softmax classification layer at end.

# [Kilcher](https://www.youtube.com/watch?v=TrdevFK_am4)
### [Paper](https://arxiv.org/abs/2010.11929)
- Transformers operate on sets
- Attention is the pairwise inner product between each pair of tokens (quadratic operation)
    - Limitation of transformers ($500$ tokens, $500^2$ products) 
- Patch + positional embedding 
    - More sophisticated embeddings unnecessary
- You can already pay attention to far away thing at lower layers with transformers
- First, LTSMs with attention, then transformers replace those for NLP, now transformers will replace CNNs?
- A feed-forward NN can learn any function
- CNN has good inductive priors
- LSTM good inductive prior for NLP
    - Like reading
- Inductive priors (or inductive biases)
    - To bias the estimator
- but with enough data, an unbiased model is better
- Transformers are more general than MLPs
    - learns the filters without input bias
- There are inductive biases in transformers
    - Skip connectors are the next to go.