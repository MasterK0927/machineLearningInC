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
#include <time.h>
typedef struct{
    //layers of the neural network
    Mat a0,a1,a2;
    Mat w1,b1;
    Mat w2,b2;
} Xor;

Xor xor_alloc(void){
    
    //allocating the memory to the matrices.
    Xor xor;
    xor.a0 = mat_alloc(1,2);
    xor.w1 = mat_alloc(2,2);
    xor.b1 = mat_alloc(1,2);
    xor.a1 = mat_alloc(1,2);
    xor.w2 = mat_alloc(2,1);
    xor.b2 = mat_alloc(1,1);
    xor.a2 = mat_alloc(1,1);
    return xor;
}

void forward_xor(Xor xor){
    //multiplication or forward passing through the network
    //for multiple layers we can loop over the layers, while maintaining the weights as arrays
    mat_dot(xor.a1,xor.a0,xor.w1);
    mat_sum(xor.a1, xor.b1);
    mat_sig(xor.a1);
    //feed through the second layer
    mat_dot(xor.a2, xor.a1, xor.w2);
    mat_sum(xor.a2,xor.b2);
    mat_sig(xor.a2);    
}

float cost(Xor xor, Mat ti, Mat to){
    assert(ti.rows==to.rows);
    assert(to.cols==xor.a2.cols);
    size_t n = ti.rows;

    float c = 0;
    for(size_t i=0;i<n;i++){
        Mat x = mat_row(ti,i);
        Mat y = mat_row(to,i);
        mat_copy(xor.a0, x);
        forward_xor(xor);

        size_t q = to.cols;
        for(size_t j=0; j<q; j++){
            float d = MAT_AT(xor.a2,0,j) - MAT_AT(y,0,j);
            c+=d*d;
        }
    }
    return c/n;
};

void finite_diff(Xor xor,Xor g, float eps, Mat ti, Mat to){
    float saved;
    float c = cost(xor,ti,to);
   for(size_t i=0; i<xor.w1.rows; i++){
        for(size_t j=0; j<xor.w1.cols; j++){
            saved = MAT_AT(xor.w1, i, j);
            MAT_AT(xor.w1,i,j) += eps;
            MAT_AT(g.w1, i,j)=(cost(xor,ti,to)-c)/eps;
            MAT_AT(xor.b1,i,j)=saved;
        }

    }
    for(size_t i=0; i<xor.b1.rows; i++){
        for(size_t j=0; j<xor.b1.cols; j++){
            saved = MAT_AT(xor.b1,i,j);
            MAT_AT(xor.b1,i,j) += eps;
            MAT_AT(g.b1, i,j)=(cost(xor,ti,to)-c)/eps;
            MAT_AT(xor.b1,i,j) = saved;
        }
    }
    for(size_t i=0; i<xor.w2.rows; i++){
        for(size_t j=0; j<xor.w2.cols; j++){
            saved = MAT_AT(xor.w2,i,j);
            MAT_AT(xor.w2,i,j) += eps;
            MAT_AT(g.w2, i,j)=(cost(xor,ti,to)-c)/eps;
            MAT_AT(xor.w2,i,j) = saved;
        }
    }
    for(size_t i=0; i<xor.b2.rows; i++){
        for(size_t j=0; j<xor.b2.cols; j++){
            saved = MAT_AT(xor.b2,i,j);
            MAT_AT(xor.b2,i,j) += eps;
            MAT_AT(g.b2, i,j)=(cost(xor,ti,to)-c)/eps;
            MAT_AT(xor.b2,i,j) = saved;
        }
    }  
}

void xor_learn(Xor xor, Xor g, float rate){
    for(size_t i=0; i<xor.w1.rows; i++){
        for(size_t j=0; j<xor.w1.cols; j++){
            MAT_AT(xor.w1,i,j) -= rate*MAT_AT(g.w1, i, j);
        }

    }
    for(size_t i=0; i<xor.b1.rows; i++){
        for(size_t j=0; j<xor.b1.cols; j++){
            MAT_AT(xor.b1,i,j) -= rate*MAT_AT(g.b1, i, j);
        }
    }
    for(size_t i=0; i<xor.w2.rows; i++){
        for(size_t j=0; j<xor.w2.cols; j++){
            MAT_AT(xor.w2,i,j) -= rate*MAT_AT(g.w2, i, j);
        }
    }
    for(size_t i=0; i<xor.b2.rows; i++){
        for(size_t j=0; j<xor.b2.cols; j++){
            MAT_AT(xor.b2,i,j) -= rate*MAT_AT(g.b2, i, j);
        }
    }

}

float td[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};

int main(void){
    //we are implementing xor gate
    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Mat to = {
        .rows = n, 
        .cols = 1, 
        .stride = stride,
        .es = td + 2
    };

    MAT_PRINT(ti);
    MAT_PRINT(to);

    Xor xor = xor_alloc();
    Xor g = xor_alloc();

    mat_rand(xor.w1,0,1); //randimizing the weights of 1st layer
    mat_rand(xor.b1,0,1); //randimizing the biases of 1st layer
    mat_rand(xor.w2,0,1); //randimizing the weights of 2nd layer
    mat_rand(xor.b2,0,1); //randimizing the biases of 2nd layer

    float eps = 1e-3;
    float rate = 1e-1;

    printf("cost = %f\n",cost(xor,ti,to));
    for(size_t i=0; i<10*1000; i++){
        finite_diff(xor,g,eps,ti,to);
        xor_learn(xor,g,rate);
        printf("%zu: cost = %f\n",i , cost(xor,ti,to));
    }

#if 0
    //printing the weights and biases
    MAT_PRINT(xor.w1);
    MAT_PRINT(xor.b1);
    MAT_PRINT(xor.w2);
    MAT_PRINT(xor.b2);
    return 0;
#endif
}
