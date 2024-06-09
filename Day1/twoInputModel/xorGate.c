/*
For modelling XOR gate, we require 9 parameters, 6 weights and 3 biases, which are used to separate the data points, i.e. the input data points, which are used to train the model.
This can do all the tasks of the OR, AND, and NAND gates, and can also do the XOR gate.
*/

//------------------------------Code------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//for squishing the output between 0 and 1, we need the sigmoid function
float sigmoidf(float x)
{
    return 1.0f/(1.0f + expf(-x));
}
//constructing the structure for the XOR Model;
typedef struct {
    float or_w1;
    float or_w2;
    float or_b;

    float nand_w1;
    float nand_w2;
    float nand_b;

    float and_w1;
    float and_w2;
    float and_b;
} Xor;

typedef float sample[3];

//XOR gate training dataset
sample train_xor[] = {
    {0, 0, 0},
    {0, 1, 1},   //first 2 are i/p and third is o/p
    {1, 0, 1},
    {1, 1, 0},
};

//OR gate training dataset
sample train_or[] = {
    {0, 0, 0},
    {0, 1, 1},   //first 2 are i/p and third is o/p
    {1, 0, 1},
    {1, 1, 1},
};

//AND gate training dataset
sample train_and[] = {
    {0, 0, 0},
    {0, 1, 0},   //first 2 are i/p and third is o/p
    {1, 0, 0},
    {1, 1, 1},
};

//NAND gate training dataset
sample train_nand[] = {
    {0, 0, 1},
    {0, 1, 1},   //first 2 are i/p and third is o/p
    {1, 0, 1},
    {1, 1, 0},
};


sample *train = train_xor;
size_t TRAIN_COUNT = 4;

//we need a function for feeding the input {x,y} in the neural network
//that function is feed_forward, process is called forward propagation.

//intakes the model, and inputs as arguments;
float forward(Xor m, float x1, float x2){

    //first layer

    //feeding data into the OR layer
    float a = sigmoidf(m.or_w1*x1 + m.or_w2*x2 + m.or_b);
    //feeding data into the NAND layer
    float b = sigmoidf(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b);
    
    //then we utilise the values a and b to feed into the AND layer
    //as the output of the XOR gate is the output of the AND gate of the OR and NAND gate\

    //last layer
    return sigmoidf(m.and_w1*a + m.and_w2*b + m.and_b);

}

//cost function for the XOR gate
//cost function doesnt know what the model is, it just knows that it has to take the model and calculate the cost
float cost(Xor m){
    float result = 0.0f;
    for(size_t i=0; i<TRAIN_COUNT; i++){
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m, x1, x2);
        float error = y - train[i][2];
        result += error*error;
    }
    result /= TRAIN_COUNT;
    return result;
}

//random float generator

float rand_float(void){
    return (float) rand() / (float) RAND_MAX;
}

Xor rand_xor(void){
    srand(time(0));
    Xor m;
    m.or_w1 = rand_float();
    m.or_w2 = rand_float();
    m.or_b = rand_float();
    m.nand_w1 = rand_float();
    m.nand_w2 = rand_float();
    m.nand_b = rand_float();
    m.and_w1 = rand_float();
    m.and_w2 = rand_float();
    m.and_b = rand_float();
    return m;
}

Xor learn(Xor m, Xor g, float rate){
    m.or_w1 -= rate * g.or_w1;
    m.or_w2 -= rate * g.or_w2;
    m.or_b -= rate * g.or_b;
    m.nand_w1 -= rate * g.nand_w1;
    m.nand_w2 -= rate * g.nand_w2;
    m.nand_b -= rate * g.nand_b;
    m.and_w1 -= rate * g.and_w1;
    m.and_w2 -= rate * g.and_w2;
    m.and_b -= rate * g.and_b;
    return m;
}

Xor finite_diff(Xor m, float eps){
    Xor g;
    float c = cost(m);
    float saved;
    
    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c) / eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c) / eps;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c) / eps;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c) / eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c) / eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c) / eps;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c) / eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c) / eps;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c) / eps;
    m.and_b = saved;

    return g;
}

void print_xor(Xor m){
    printf("OR:   w1 = %f, w2 = %f, b = %f\n", m.or_w1, m.or_w2, m.or_b);
    printf("NAND: w1 = %f, w2 = %f, b = %f\n", m.nand_w1, m.nand_w2, m.nand_b);
    printf("AND:  w1 = %f, w2 = %f, b = %f\n", m.and_w1, m.and_w2, m.and_b);
}

int main() {
    Xor m = rand_xor();
    float eps = 1e-3;
    float rate = 1e-1;

    for(size_t i=0; i<1000*1000; i++){
        Xor g = finite_diff(m, eps);
        m = learn(m, g, rate);
        printf("Final cost = %f\n", cost(m));
    }
    printf("Final cost = %f\n", cost(m));

    printf("------------------------------------------\n");
    printf("\"XOR\" neuron:\n");
    for(size_t i =0 ; i<2; i++){
        for(size_t j=0; j<2; j++){
            printf("%zu ^ %zu = %f\n", i, j, forward(m, i, j)); //%zu is for size_t
        }
    }
    printf("------------------------------------------\n");
    printf("\"OR\" neuron:\n");
    for (size_t i = 0; i < 2; i++){
        for (size_t j = 0; j < 2; j++){
            printf("%zu | %zu = %f\n", i, j, sigmoidf(m.or_w1*i + m.or_w2*j + m.or_b));
        }
    }
    printf("------------------------------------------\n");
    printf("\"NAND\" neuron:\n");
    for (size_t i = 0; i < 2; i++){
        for (size_t j = 0; j < 2; j++){
            printf("%zu | %zu = %f\n", i, j, sigmoidf(m.nand_w1*i + m.nand_w2*j + m.nand_b));
        }
    }
    printf("------------------------------------------\n");
    printf("\"AND\" neuron:\n");
    for (size_t i = 0; i < 2; i++){
        for (size_t j = 0; j < 2; j++){
            printf("%zu | %zu = %f\n", i, j, sigmoidf(m.and_w1*i + m.and_w2*j + m.and_b));
        }
    }

    return 0;
}
