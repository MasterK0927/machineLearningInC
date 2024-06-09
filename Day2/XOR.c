/*
It actually like a two layer neural network
    1. Input layer
    2. Hidden layer
    3. Output layer
Where the first layer has 4 weights, which is basically a 2x2 matrix
And there are 2 biases, which is of the form 1x2 matrix
2nd layer has 2 weights, which is a 2x1 matrix
And there is 1 bias, which is a 1x1 matrix
activation matrix is the intermediate matrix which stores the values after passing through the first layer
*/

#define NEURALNETWORK_IMPLEMENTATION
#include "neuralNetwork.h"

typedef struct{
    Mat a0;
    Mat w1,b1,a1;
    Mat w2,b2,a2;
} Xor;


int main(void){
    //we are implementing xor gate
    srand(time(0));

    Xor xor;
    xor.a0 = mat_alloc(1,2);
    xor.w1 = mat_alloc(2,2);
    xor.b1 = mat_alloc(1,2);
    xor.a1 = mat_alloc(1,2);
    xor.w2 = mat_alloc(2,1);
    xor.b2 = mat_alloc(1,1);
    xor.a2 = mat_alloc(1,1);


    mat_rand(xor.w1,0,1); //randimizing the weights of 1st layer
    mat_rand(xor.b1,0,1); //randimizing the biases of 1st layer
    mat_rand(xor.w2,0,1); //randimizing the weights of 2nd layer
    mat_rand(xor.b2,0,1); //randimizing the biases of 2nd layer

 
    //printing the weights and biases
    MAT_PRINT(xor.w1);
    MAT_PRINT(xor.b1);
    MAT_PRINT(xor.w2);
    MAT_PRINT(xor.b2);

    return 0;
}
