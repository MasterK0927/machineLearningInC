#include <stdio.h>
#include <stdlib.h>
#include <time.h>


float train[][2]={
    {0,0},     
    {1,2},     
    {2,4},     
    {3,6},
    {4,8},
};

#define TRAIN_COUNT (sizeof(train)/sizeof(train[0]))

float rand_float(void){
    return (float) rand() / (float) RAND_MAX;
}

float cost(float w){
    
    float result = 0.0f;
    for(size_t i = 0; i<TRAIN_COUNT; i++){
        float x = train[i][0];
        float y = x*w;
        
        float error = y - train[i][1];
      
        result += error*error;  
    }
    
    result /= TRAIN_COUNT;
    return result;
}

float dcost(float w){
    float result = 0.0f;
    size_t n = TRAIN_COUNT;
    for(size_t i=0; i<n; i++){
        float x = train[i][0];
        float y = train[i][1];
        result += 2*(x*w - y)*x;
    }
    result/=n;
    return result;
}

int main(){
    
    srand(60);
     
    float w = rand_float()*10.0f;
    
    /*printf("=============== WITHOUT gradientDescent ==================\n");
    printf("initial cost: %f, ", cost(w));
    printf("w: %f\n",w); 
    printf("----------------------------------------------------\n");
    float rate = 1e-1;
    for (size_t i =0; i<10; i++){
        float eps = 1e-3;
        float c = cost(w);
        float dw = (cost(w+eps) - c)/eps ;
        w-=rate*dw;
        printf("w/o gradientDescent cost: %f, ", cost(w));
        printf("w: %f\n",w);
    }
    printf("---------------------------------------------------\n");
    printf("final 'w' w/o gradientDescent: %f\n",w);
    */

    printf("=============== WITH gradientDescent ==================\n");
    printf("initial cost: %f, ", cost(w));
    printf("----------------------------------------------------\n");
    printf("w: %f\n",w); 
    float rate = 1e-1;
    for(size_t i=0; i<10; i++){
        float dw = dcost(w);
        w-=rate*dw;
        printf("w/t gradientDescent cost: %f, ", cost(w));
        printf("w: %f\n",w);
    }
    printf("---------------------------------------------------\n");
    printf("final 'w' w/t gradientDescent: %f\n",w);
    return 0; 
}