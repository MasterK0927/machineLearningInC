#include <stdio.h>
#include <stdlib.h>

#define TRAIN_COUNT (sizeof(train)/sizeof(train[0]))

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

float rand_float (void) {
    return (float) rand() / (float) RAND_MAX;
}

float cost(float w, float b){
    float result = 0.0f;
    for(size_t i = 0; i<TRAIN_COUNT; i++){
        float x = train[i][0];
        float y = x*w + b;
        float error = y - train[i][1];;
        result += error*error;
    }
    result /= TRAIN_COUNT;
    return result;
}

int main(){
    srand(60);
    float w = rand_float()*10.0f;
    float b = rand_float()*5.0f;
    float eps = 1e-3;
    float rate=1e-3;
    for (size_t i=0; i<500; i++){
        float cf = cost(w, b);
        float dcost = (cost(w+eps, b) - cf) / eps;
        float db = (cost(w, b+eps) - cf) / eps;
        w -= rate * dcost;
        b -= rate * db;
        printf("cost = %f, w = %f, b = %f\n", cost(w,b), w, b);
    }
    printf("---------------------------\n");
    printf("w = %f, b = %f\n",w, b);
    return 0;
}