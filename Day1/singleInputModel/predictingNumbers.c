//----------------------------NOTES-----------------------------------
/*
Model for predicting some number based on the input number
Our goal is to minimise the cost function, eventually getting it close to 0
The minimum the cost function is, the max the accuracy of the model is
If we try to add a very small value to the parameter to see, how it changes the performance of the model
epsilon is the small value that we are adding to the parameter
so we are adding epsilon to the parameter and checking the cost function
if the cost function is decreasing, then we are moving in the right direction
if the cost function is increasing, then we are moving in the wrong direction
if the cost function is not changing, then we are at the minimum point
an BANG!! It actually performed well, and reduces the error
So we will keep subtracting the epsilon from the parameter, until we reach the minimum point
And we will automate this stuff.

Simulating the cost function

We know that, the cost function is a mathematical fucntion after all,
and all the mathematical functions can be represented graphically,
Say, we assume that the graphical representation of the cost function is a parabola,
and we are at the top of the parabola, and we want to reach the bottom of the parabola,
So taking the derivative of the cost function, we can find the slope of the cost function at that point,
which tell us, in which direction the function grows and in which direction we have to move to reach the minimum point,
so upon moving in the opposite direction of the slope, we can reach the minimum point,

simulating derivative in the program

By definition, the derivative of a function at a point is the slope of the function at that point,
which is also the rate of change of the function at that point, which is mathematically represented as,
f'(x) = lim(h->0) (f(x+h) - f(x))/h
where h is the small value that we are adding to the x to find the slope of the function at that point,
So for simulating the derivative in the program, we can add a small value to the parameter and check the change in the cost function,
Mathematically, if the cost function is increasing, then the slope is positive, and if the cost function is decreasing, then the slope is negative,
So, programmatically, if the cost function is increasing, then we are moving in the wrong direction, and if the cost function is decreasing, then we are moving in the right direction,
So, we can keep adding the small value to the parameter, until we reach the minimum point, where the cost function is not changing, i.e. the slope is 0,
So, we can automate this stuff, by adding the small value to the parameter, until the cost function is not changing, i.e. the slope is 0,
And we can reach the minimum point, where the cost function is minimum, i.e. the model is perfect

so we can directly code it like,
float dcost = (cost(w-eps)-cost(w))/eps;
It is called "Finite Difference Method", but it is not directly used in ML
But for learning purpose, we can use it
It is something similar to the approximation of the derivative of the function at a point

Sometimes the dcost can be very large, so we can multiply it with a small value, say 0.01, to make it small
That value is called rate, or just training rate, or learning rate,
So, we can multiply the dcost with the rate, and subtract it from the parameter, to reach the minimum point

"JUST STIR THE PILE UNTIL, THEY START LOOKING LIKE"

Now repeat the process until the cost function is minimum, i.e. the model is perfect, so we will start iterating until the cost function is minimum
So, now the model is trained for giving values both backward and forward, i.e. the model is trained for predicting the output for the unseen data also
For example, if we give 5 to the model, it should predict 10, as the model is trained for predicting the output for the unseen data also

THEREFORE, WE ARE SUCCESSFUL IN TRAINIG MODEL USING A SINGLE NEURON
WITHOUT USING ANY LIBRARY AND FANCY LIBRARIES
*/



//-----------------------------NOTES-----------------------------------
//-----------------------------CODE-----------------------------------

#include <stdio.h>
#include <stdlib.h> //for using rand() function
#include <time.h>  //for using time() function

//creating a trainging dataset
float train[][2]={
    {0,0},     //on supplying 0, the output is 0
    {1,2},     //on supplying 1, the output is 2
    {2,4},     //on supplying 2, the output is 4
    {3,6},
    {4,8},
};

//TRAIN_COUNT is the number of rows in the train dataset
//sizeof(train) gives the size of the train dataset in bytes
//sizeof(train[0]) gives the size of the first row in the train dataset in bytes
//sizeof(train)/sizeof(train[0]) gives the number of rows in the train dataset
//this is done to make the code more dynamic, so that if we add more rows in the train dataset, we don't have to change the TRAIN_COUNT
//we figured out that, on adding the eps to w, the cf increseas, so we have to subtract it.
#define TRAIN_COUNT (sizeof(train)/sizeof(train[0]))

float rand_float(void){
    return (float) rand() / (float) RAND_MAX;
}

//cost function of the model, we can give the parameters to the function and it will return the cost
//parameters can be the dataset, the model, the weights, etc.
//as our entire model is just one parameter, i.e. w, so we are passing only one parameter to the function
float cost(float w){
    //going through the data for mapping the best value of w, that fits in our model.
    //iterating through the dataset
    float result = 0.0f;
    for(size_t i = 0; i<TRAIN_COUNT; i++){
        float x = train[i][0];
        float y = x*w;
        //finding the distance between the actual value and the predicted value
        float error = y - train[i][1];
        //accumating the square of the errors
        //this is done to make the error positive, as the error can be negative
        //also it amplifies the error, so that we can see the error clearly
        result += error*error;  
    }
    //finding the average of the error
    //doing this to make the error independent of the number of rows in the dataset and to see the error clearly
    //0 means the model is perfect
    // >0 means the model is not perfect
    // <0 means the model is overfitting, i.e. the model is too perfect, that it is fitting the noise in the data also,
    //which is not good for the model, as the model should be able to predict the output for the unseen data also and not just the training data 
    result /= TRAIN_COUNT;
    return result;
}

int main(){
    //initialising the randomm number generator with the current time
    //this is done so that the random number generated is different every time
    //time(0) returns the current time
    //time(60) returns the time after 60 seconds
    //srand(60) is used to seed the random number generator with the value 60
    //fixate the value for 60s
    srand(60);
    //our model is something like, y=x*w, where w is some parameter
    //a function for generating the random float btw 0 to 1
    // float w = rand_float()*10.0f; 
    float w = 1.0f;
    //rand_float()*10.0f, here .0f is used to specify that the number is a float
    //%f is a format specifier for float
    printf("%f\n",w);
    float eps = 1e-3;
    float rate = 1e-3;
    //calling the cost function to calculate the error
    for (size_t i =0; i<500; i++){
        printf("w/o eps: %f, ", cost(w));
        float dcost = (cost(w+eps)-cost(w))/eps; //calculating the derivative of the cost function at the point w
        w-=rate*dcost;
        printf("w/t eps: %f, ", cost(w));
        printf("w: %f\n",w);
    }

    printf("------------------------\n");
    printf("w: %f\n",w);
    return 0; 
}