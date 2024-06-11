/*
Modelling lineary separable gates.

At the end of the neurons, neuron have an activation function which is a step function,
which is used to classify the output of the neuron, i.e. if the output is greater than 0.5, then the output is 1, else 0.
It is called a step function, as it is a step from 0 to 1.

Mostly Signmoid function is used as an activation function, as it is a smooth function, and it is derivable at every point.
Mathematically it looks like,
f(x) = 1/(1+e^(-x)) = 1/(1+exp(-x)) = 1/(1+e^(-wx)) = 1/(1+exp(-wx))
where x is the input to the neuron and w is the weight of the neuron.

There is also a ReLU function, which is used as an activation function, it is a piecewise function, which is 0 for x<0 and x for x>=0.
Mathematically it looks like,
f(x) = 0 for x<0, x for x>=0
Its really important to use the ReLU when we have a deep neural network along with back propagation, as it helps in preventing the vanishing gradient problem.

ML on the surface is just a bunch of matrix multiplications and additions, but the real magic is in the back propagation, which is used to train the model.
And also, when we start talking about optimising the model, we have to talk about the activation functions, as they play a crucial role in the training of the model.

In the modelling of the linerally separable gates, the bias plays a crucial role, as it helps in shifting the line, which is used to separate the data points.
As, without adding the bias, it was having a shift of 1/2 in the line, which is used to separate the data points, so to shift the line to the origin, we have to add the bias.

UNderstNDING IN DETAIL

In the modelling of the OR gate, we have to find the weights and bias, which are used to separate the data points, i.e. the input data points, which are used to train the model.
So, we use the above model, which is a single neuron model, which is used to model the OR gate, and we have to find the weights and bias. And we can see that, when model was trained without bias, it was failing to separate the data points, as it was having a shift of 1/2 in the line, which is used to separate the data points, so to shift the line to the origin, we have to add the bias.

We are using gnuplot-qt for plotting the cost function, which is used to find the minimum point, where the cost function is minimum, i.e. the model is perfect.
>>>>> plot "cost.txt" with lines

If we don't have a bias, then model will only modify the o/p based on the paramaeters.
But asap bias is added, the model becomes able to shift it left and right without depending on the input.
*/

//------------------------------Code------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float sigmoidf(float x){
    return 1.0f/(1.0f + (expf(-x)));
}

//OR gate training dataser
float train_or[][3] = {
    {0, 0, 0},
    {0, 1, 1},   //first 2 are i/p and third is o/p
    {1, 0, 1},
    {1, 1, 1},
};

#define TRAIN_COUNT (sizeof(train_or)/sizeof(train_or[0]))

float cost(float a, float b, float bias){
    float result = 0.0f;
    for(size_t i=0; i<TRAIN_COUNT; i++){
        float x1 = train_or[i][0];
        float x2 = train_or[i][1];
        float y = sigmoidf(x1*a + x2*b +bias);
        float error = y - train_or[i][2];
        result += error*error;
    }
    result /= (float)TRAIN_COUNT;
    return result;
}

float rand_float(void){
    return (float) rand() / (float) RAND_MAX;
}


int main(){

    // for(float x = -10; x<=10; x+=1.0f){
    //     printf("%f => %f\n", x, sigmoidf(x));
    // }

    // return 0;
    srand(time(0));
    srand(60);
    float w1 = rand_float();
    float w2 = rand_float();
    float bias = rand_float();
    float eps = 1e-3;
    float rate = 1e-1;
    for(size_t i=0; i<100*1000; i++){
        float cf = cost(w1, w2, bias);
        // printf("%f\n",cf);
        printf("w1 = %f, w2 = %f, bias = %f, cost = %f\n", w1, w2, bias, cf);
        float dcost1 = (cost(w1+eps, w2, bias) - cf) / eps;
        float dcost2 = (cost(w1, w2+eps, bias) - cf) / eps;
        float dbias = (cost(w1, w2, bias+eps) - cf) / eps;
        w1 -= rate * dcost1;
        w2 -= rate * dcost2;
        bias -= rate * dbias;
    }
    printf("w1 = %f, w2 = %f, c = %f, bias = %f\n", w1, w2, cost(w1,w2,bias), bias);
    
    //forwarding the trained model to predict the output of the OR gate for the given inputs and comparing it with the actual output
    
    for(size_t i=0; i<2; ++i){
        for(size_t j=0; j<2; ++j){
            printf("%zu | %zu = %f\n", i, j, sigmoidf(w1*i + w2*j + bias));
        }
    }
}