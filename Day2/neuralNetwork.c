#include <time.h>
#define NEURALNETWORK_IMPLEMENTATION
#include "neuralNetwork.h"

int main(void){
    srand(time(NULL));
    Mat m = mat_alloc(5,5);
    mat_rand(m,0,10);
    mat_print(m);
}