Title: Answers to Chapter 1
Date: 2021-03-15 20:40
Tags: Neural Networks, AI, ML
Author: Cody Fernandez
Summary: My answers to Chapter 1 of __Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow__ by Aurelien Geron


# Chapter 1
1. **How would you define Machine Learning?** Machine Learning is the science and art of programming computers so they can learn from data. Alternatively, it is the field of study that gives computers the ability to learn without being explicitly programmed. 
1. **Can you name four types of problems where it shines?**
       1.	Problems for which existing solutions require a lot of fine-tuning or long lists of rules. 
       1.	Complex problems for which using a traditional approach yields no good solution.
       1.	Fluctuating environments where the algorithm must adapt to new data
       1.	Getting insights about complex problems and large amounts of data (data mining).
1.	**What is a labeled training set?** A collection of data to be fed into an ML algorithm where instances have labels (desired solutions), which are used by the algorithm to classify new objects.
1.	**What are the two most common supervised tasks?** Classification and target prediction. In classification, the algorithm learns the class of instances of the training set and must correctly classify new objects in the test set. In target prediction, the algorithm must predict a target value based on the set of features of a training instance (called predictors). Also regression.
1.	**Can you name four common unsupervised tasks?** Clustering, visualization, anomaly detection, and association rule learning. In clustering, you try to detect groups of similar instances, letting the algorithm determine their connections. Visualization algorithms output a 2D or 3D representation of data while preserving as much structure as possible, allowing the user to see data organization and determine unsuspected patterns. Related to visualization is dimensionality reduction, which involves simplifying the data without losing too much information, often by merging correlated features together in a process called feature extraction. Anomaly detection involves identifying new instances after being trained on a clean set of standard instances. Novelty detection is similar, where the system identifies new types of instances rather than defects. In association rule learning, one digs into large amounts of data and discovers interesting relationships between attributes. 
1.	**What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?** Due to resource constraints, you will want to use an online learning algorithm so that the robot can learn incrementally with mini-batches of data. Actually Reinforcement Learning would be best.
1.	**What type of algorithm would you use to segment your customers into multiple groups?** A clustering algorithm would find similarities between customer segments in an unsupervised manner.
1.	**Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?** Supervised, as its main goal is classification of emails as ham or spam, which is a supervised learning task.
1.	**What is an online learning system?** A system that trains incrementally on incoming data. The speed at which it changes how fast it should adapt to new data is called the learning rate.
1.	**What is out-of-core learning?** When the dataset is so large that it can’t fit into one machine’s main memory, the algorithm runs training steps on parts of the data in sequence in an online learning fashion.
1.	**What type of learning algorithm relies on a similarity measure to make predictions?** Instance-based learning algorithm.
1.	**What is the difference between a model parameter and a learning algorithm’s hyperparameter?** Model parameters are degrees of freedom that are adjusted during learning by a fitness function (positive) or by a cost function (negative). A regularization hyperparameter is a parameter of the learning algorithm that constrains the possible values of the model parameters to prevent overfitting.
1.	**What do model-based learning algorithms search for?** What is the most common strategy they use to succeed? How do they make predictions? They search for the optimal model parameter values given a cost/fitness function applied to a set of training data. Training a model based on the data. They take in new input, apply the model, and produce output.
1.	**Can you name four of the main challenges in Machine Learning?** Challenges can be chalked up to “bad algorithm” or “bad data”. Bad data includes too little training data, nonrepresentative training data, poor quality training data, and irrelevant features in training data. Bad algorithm includes overfitting training data and underfitting training data.
1.	**If your model performs great on the training data but generalizes poorly to new instances, what is happening?** Can you name three possible solutions? It is overfitting the training data. One can simplify the model by selecting one with fewer parameters, reducing the number of attributes in the training data, or constraining the model. One can also gather more training data or reduce noise in the existing training data.
1.	**What is a test set, and why would you want to use it?** The test set is a portion of your training data that you set aside for testing the learning algorithm. Applying your algorithm to the test set after training it on the training set allows you to evaluate the generalization error of your model.
1.	**What is the purpose of a validation set?** The validation set is a further subset of the training set (and independent of the test set) that is used to evaluate multiple models with various hyperparameters to determine the best model in a process called holdout validation.
1.	**What is the train-dev set, when do you need it, and how do you use it?** The train-dev set is a further subset of the training data that is used to evaluate models. You need it when the training data you have isn’t very representative of the inputs the model will receive. Process: train model on training set. Evaluate model on train-dev set. Failure means the model has overfit, and you can take the actions in Question 15 to address it. Success means the model is not overfitting the training data, so evaluate the model on the validation set. Failure means the model is struggling with the data mismatch, so try to reconcile the training and validation data and retrain the model. Success at this point is final success: you have a good model making good predictions.
1.	**What can go wrong if you tune hyperparameters using the test set?** You end up measuring the generalization error multiple times on just the test set. This means the hyperparameters are overtuned to fit just this test set, and will perform poorly on a different test set.

# Bonus Outline
1. Types of machine learning systems
     1. Whether or not they are trained with human supervision
          1. Supervised
          1.	Unsupervised
          1.	Semisupervised
          1.	Reinforcement Learning
      1. Whether or not they can learn incrementally or on the fly
          1. Online learning
          1.	Batch learning
      1. Whether they compare new data to known data or detect patterns in training data and build a predictive model
          1. Instance-based learning
          1.	Model-based learning
      1. These are not exclusive. A system can be an online, model-based, supervised learning system.
1. Sources of Error
      1. Sampling noise – the data sample is too small (nonrepresentative data is included as a result of chance)
      1. Sampling bias – Lots of data, but it is nonrepresentative due to a flawed sampling method.

