/*
All the linearly derivable gates in single neuron model can be modelled using the above code.
*/

//------------------------------Code------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float sigmoidf(float x){
    return 1.0f/(1.0f + (expf(-x)));
}

//typedef float sample[3]; means that sample is an array of 3 float
typedef float sample[3]; 

//OR gate training dataset
//sample train_or[][3] means that train_or is a 2D array of 3 columns and unknown number of rows
//where sample is an array of 3 floats and train_or is an array of sample arrays
sample train_or[] = {
    {0, 0, 0},
    {0, 1, 1},   //first 2 are i/p and third is o/p
    {1, 0, 1},
    {1, 1, 1},
};

sample train_and[] = {
    {0, 0, 0},
    {0, 1, 0},   //first 2 are i/p and third is o/p
    {1, 0, 0},
    {1, 1, 1},
};

sample train_nand[] = {
    {0, 0, 1},
    {0, 1, 1},   //first 2 are i/p and third is o/p
    {1, 0, 1},
    {1, 1, 0},
};

//sample *train = train_or; is equivalent to sample *train = &train_or[0];
//and means that train is a pointer to the first element of the train_or array
sample *train = train_and;
//TRAIN_COUNT is the number of rows in the train dataset
size_t TRAIN_COUNT = 4;

float cost(float a, float b, float bias){
    float result = 0.0f;
    for(size_t i=0; i<TRAIN_COUNT; i++){
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(x1*a + x2*b +bias);
        float error = y - train[i][2];
        result += error*error;
    }
    result /= TRAIN_COUNT;
    return result;
}

float rand_float(void){
    return (float) rand() / (float) RAND_MAX;
}


int main(){
    for (size_t x = 0; x<2; x++){
        for (size_t y = 0; y<2; y++){
            printf("%zu | %zu = %zu\n", x, y, (x|y) & ~(x&y));
        }
    }
    return 0;
}