For optimising our costfunction, we were using a hack or say a very vague method called Finite Differences, so we will be migrating to the back propagation for the real world use cases.


===================Tools================
---------
Pdflatex|
---------
For installing it
```
sudo apt-get install texlive-latex-base
sudo apt-get install texlive-fonts-recommended
sudo apt-get install texlive-fonts-extra
sudo apt-get install texlive-latex-extra
```
For running it
```
pdflatex latex_source_name.tex
```

========================================

-> The value of the gradient descent needs to be infinitesmally small, not zero. Its similar to the definition of a derivative.It gives the velocity of the function at a point.

In gradient descent, we uses the exact definition of the derivative, which ultimately tells us about the velocity of the function at a point, whether it grows or not. And we have to move in the opposite direction, in order to make sure to have the cost function reduced.

Its very very fast for the rate = 1e-1; compared to finite diff its very very fast and efficient. This method will work like a butter on a bread for the small parameter models,  but for the models having tons of parameters, we use something called Back Propagation.

Back Propagation is a technique of calculating derivatives of complex nested functions. And it is the standard cost function used in the neural network.


================================================================

## Arbitrary Neurons Model with Single Input

To demonstrate the concept of backpropagation, we'll start with a simple neural network with a single input and multiple neurons in a single layer. Let's assume we have \( n \) neurons in our network.

### Step-by-Step Explanation:

1. **Input and Weights**:
    - Let \( x \) be the input to the network.
    - Each neuron \( i \) has a weight \( w_i \) and a bias \( b_i \).

2. **Activation Function**:
    - Each neuron's output is passed through an activation function \( \sigma \). Common activation functions include:
        - **Sigmoid**: \( \sigma(z) = \frac{1}{1 + e^{-z}} \)
        - **ReLU**: \( \sigma(z) = \max(0, z) \)
        - **Tanh**: \( \sigma(z) = \tanh(z) \)

3. **Output of Each Neuron**:
    - The output of each neuron \( i \) can be represented as:
      \[
      a_i = \sigma(z_i)
      \]
    - where \( z_i \) is the linear combination of the input and weights:
      \[
      z_i = w_i x + b_i
      \]

4. **Cost Function**:
    - Let \( y \) be the true output (target).
    - The cost function \( J \) measures the difference between the network's output and the true output. For simplicity, we use the Mean Squared Error (MSE):
      \[
      J = \frac{1}{2n} \sum_{i=1}^n (a_i - y_i)^2
      \]

### Backpropagation

Backpropagation is the method used to calculate the gradients of the cost function with respect to each parameter (weights and biases) in the network. It consists of two main steps: forward pass and backward pass.

1. **Forward Pass**:
    - Compute the input to each neuron:
      \[
      z_i = w_i x + b_i
      \]
    - Compute the activated output:
      \[
      a_i = \sigma(z_i)
      \]

2. **Backward Pass**:
    - Calculate the derivative of the cost function with respect to each neuron's output:
      \[
      \frac{\partial J}{\partial a_i} = a_i - y_i
      \]
    - Calculate the derivative with respect to the input to each neuron (using the chain rule):
      \[
      \frac{\partial J}{\partial z_i} = \frac{\partial J}{\partial a_i} \cdot \sigma'(z_i)
      \]
      - For the sigmoid activation function:
        \[
        \sigma'(z_i) = \sigma(z_i) \cdot (1 - \sigma(z_i))
        \]
      - For the ReLU activation function:
        \[
        \sigma'(z_i) = 
        \begin{cases} 
        1 & \text{if } z_i > 0 \\
        0 & \text{if } z_i \leq 0 
        \end{cases}
        \]
      - For the tanh activation function:
        \[
        \sigma'(z_i) = 1 - \tanh^2(z_i)
        \]
    - Calculate the gradients with respect to the weights and biases:
      \[
      \frac{\partial J}{\partial w_i} = \frac{\partial J}{\partial z_i} \cdot x
      \]
      \[
      \frac{\partial J}{\partial b_i} = \frac{\partial J}{\partial z_i}
      \]

### Updating Parameters

Using the calculated gradients, we update the weights and biases using gradient descent:
\[
w_i \leftarrow w_i - \eta \frac{\partial J}{\partial w_i}
\]
\[
b_i \leftarrow b_i - \eta \frac{\partial J}{\partial b_i}
\]
where \( \eta \) is the learning rate.

### Example

Let's consider a neural network with 3 neurons and a single input \( x \):

- Neuron 1: \( w_1, b_1 \)
- Neuron 2: \( w_2, b_2 \)
- Neuron 3: \( w_3, b_3 \)

For an input \( x \) and true outputs \( y_1, y_2, y_3 \):

1. **Forward Pass**:
    - Compute the inputs to each neuron:
      \[
      z_1 = w_1 x + b_1
      \]
      \[
      z_2 = w_2 x + b_2
      \]
      \[
      z_3 = w_3 x + b_3
      \]
    - Compute the activated outputs:
      \[
      a_1 = \sigma(z_1)
      \]
      \[
      a_2 = \sigma(z_2)
      \]
      \[
      a_3 = \sigma(z_3)
      \]

2. **Backward Pass**:
    - Calculate the derivatives of the cost function with respect to each neuron's output:
      \[
      \frac{\partial J}{\partial a_1} = a_1 - y_1
      \]
      \[
      \frac{\partial J}{\partial a_2} = a_2 - y_2
      \]
      \[
      \frac{\partial J}{\partial a_3} = a_3 - y_3
      \]
    - Calculate the derivatives with respect to the inputs to each neuron:
      \[
      \frac{\partial J}{\partial z_1} = (a_1 - y_1) \cdot \sigma'(z_1)
      \]
      \[
      \frac{\partial J}{\partial z_2} = (a_2 - y_2) \cdot \sigma'(z_2)
      \]
      \[
      \frac{\partial J}{\partial z_3} = (a_3 - y_3) \cdot \sigma'(z_3)
      \]
    - Calculate the gradients with respect to the weights and biases:
      \[
      \frac{\partial J}{\partial w_1} = \frac{\partial J}{\partial z_1} \cdot x
      \]
      \[
      \frac{\partial J}{\partial w_2} = \frac{\partial J}{\partial z_2} \cdot x
      \]
      \[
      \frac{\partial J}{\partial w_3} = \frac{\partial J}{\partial z_3} \cdot x
      \]
      \[
      \frac{\partial J}{\partial b_1} = \frac{\partial J}{\partial z_1}
      \]
      \[
      \frac{\partial J}{\partial b_2} = \frac{\partial J}{\partial z_2}
      \]
      \[
      \frac{\partial J}{\partial b_3} = \frac{\partial J}{\partial z_3}
      \]

3. **Update Parameters**:
    - Update the weights and biases for each neuron:
      \[
      w_1 \leftarrow w_1 - \eta \frac{\partial J}{\partial w_1}
      \]
      \[
      b_1 \leftarrow b_1 - \eta \frac{\partial J}{\partial b_1}
      \]
      \[
      w_2 \leftarrow w_2 - \eta \frac{\partial J}{\partial w_2}
      \]
      \[
      b_2 \leftarrow b_2 - \eta \frac{\partial J}{\partial b_2}
      \]
      \[
      w_3 \leftarrow w_3 - \eta \frac{\partial J}{\partial w_3}
      \]
      \[
      b_3 \leftarrow b_3 - \eta \frac{\partial J}{\partial b_3}
      \]

## Algorithm intuition

-> For performing the back propagation algoritmically, we will be having 4 nested loops.

    ->The first outer loop will be the loop for the samples from the training data .

    We will be basically, fast forwarding the data from the backward layer to the front layer, until we get the output at the front layer, then we propagate backwards calculating the differences in the activation layers

    ->Then we have another inner loop for each layer.

    ->As we need to propagate back from all the possible paths, we will be having a loop running through the current activation layer.

    ->For each activation layer, we will be needed to iterate over all the individual activations in the activation layer. While iterating over the individual activations, we need to check the activation of the every individual activation of the previous layer too. For that, we will be requiring another nested loop.

    By doing this, we need to keep propagating the differences and the partial derivatives, and it makes the back propagation algorithm.

### Conclusion

Backpropagation, combined with gradient descent, enables efficient optimization of neural networks by computing exact gradients, as opposed to finite differences, which is computationally expensive and less accurate. This method scales well with the complexity and size of neural networks, making it essential for training deep learning models.





