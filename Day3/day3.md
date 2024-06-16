The main problem with this (xor.c) kind of implementation is that,we have to manually add the neurons and keep track of its weight and all, instead it will be nice, if we try to implement something where we can define weights and other parameters as array.

As of now, our xor implementation is a fixed kind of matrix implementation, but what if, we want to create a different type of Neural Network?

So we can create a model/architecture where we have array of weigtht matrices and array of bias matrices.So, we need to have a trackof number of layers. By accepting weights as the array of weights and biases as the array of biases, we can now declare a neural network easily. 

Now instead of having xor xor_alloc, we will be having nn nn_alloc, where we have to define the architecture of the neural network.

First thing we will need to provide the nn_alloc is "count", for keeping a track of number of neurons.

Suppose we have an input as {input layers,hidden layers,output layers}, then it will be very very useful, if we can allot a / declare a neural network by just doing 

NN nn = nn_alloc(2,2,1);

So, we want to allocate the matrices, based on the inputted architecture. Hence, we have to write the alloc function in such a way,that it remains error less.

One possible way of doing it is, accept the number of inputs, count and the layers as parameters for the alloc.

size_t layers[]={2,2,1}
NN nn_alloc(size_t *arch, len(layers));

in the implementation of the nn_alloc

-> The arch_count-1 indicates the count of neural network.
-> Also we need to allocate array of matrices.




