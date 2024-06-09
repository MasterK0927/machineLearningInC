# Machine Learning in C

This repository explores the challenges and possibilities of implementing machine learning algorithms in the C programming language. The goal is to understand the intricacies of machine learning in a low-level language and explore the performance trade-offs.

### 1: [Logic Gates modelling (AND,OR,NAND,XOR) and Number Prediction]

# P1 

Single neuron and single input based number predictor.

# P2: 

In the context of machine learning, logic gates can be modeled as binary classification problems. A logic gate takes one or more binary inputs and produces a binary output based on a predefined logic function.

Linearly separable logic gates, such as AND and OR gates, can be easily modeled using a single-layer perceptron. The perceptron learns the weights and biases that allow it to separate the input space into two distinct regions, corresponding to the two possible output values. The training process involves adjusting the weights and biases based on the error between the predicted output and the desired output.

Non-linearly separable logic gates, such as NAND and XOR gates, require more complex models like multi-layer perceptrons (neural networks) to accurately represent their behavior. Neural networks consist of multiple layers of interconnected neurons, each performing a weighted sum of inputs followed by a non-linear activation function. The network learns the optimal weights and biases through a process called backpropagation, which adjusts the parameters to minimize the prediction error.

The main problem with the single layer XOR gate model is that it will not reduce the cost function effectively and will becoeme stable around the cost function of 2.24, which is high. So to remove that complexity, we model it using multiple Neurons.


### 2: [Franework for Neural Networks in C]

# P1

Coded a header file aka framework in C for implementing the neural network easily and effectively, removing the need to hard coding various memory allocation stuffs.

# P2

Porting the whole XOR model (which was the complicated 9 parameters model) into this framework, for testing the framework.