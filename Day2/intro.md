#DAY1

Previously, we had implemented a singleNeuron based model, which was predicting the numbers.
There we were calculating the cost function for seeing the performance of our model, and eventually it was just too bad.
So, we then used the concept of finite differences for reducing the cost function. For whcih we were addig a eps value to the cf
which was decreasing the cost function, eventually deriving us to the minmum loss, i.e. the minimum point in the graph.
We also used bias, for seeing how it favours the model.

Then we switched to the modelling of logic gates, which was basically a 2 i/p and 1 bias, single neuron based neural network.
And we saw that its not really possible to model the xor gate using a single neuron. And to model a XOR gate, we needed to create a more complicated
architecture, so we came up with the architecture of 3 neurons, which was connected like:

Refer to the image attached in the folder, there are in total 6 i/p and 3 bias. 2 inputs per neuron and 2 bias per neuron.
We had made out that model by listing out all the parameters inside the struct, which is a very bad approach, as if we have millions of
parameters, then we have to list all the million parameters manually, which is not at all feasibe and feasible.

If we have the architecture of neural network linear instead of triangular, then we have to change the whole architectucode of the model.
So we need some new framework for organising the neural networks effectively.


#DAY2

There is a good techinique of defining the neural network in the memory, and we will be building the framework around that.
Refer to the image, we can see that we have the equation:

a1 = x1.w11 + x2.w12 + b1
a2 = x1.w21 + x2.w22 + b2

 sigmoid ( [x1 x2].[{w11 w12}  + [b1 b2] ) =  [a1 a2]
                    {w21 w22}]

sigmoid(input * connections + biases) = neurons with informations


again we will have the final result from the op layer as

 activation([a1 a2].[w1, + [b] ) = g
                     w2]
This is for the single layer, we keep on doing this until we cover all of the architecture.

So, we will try to develop a simple framework in C, that will allow us to define, train and feed forward the neural networks in C.
Will develop a header only neural network library and then port our models to our framework, and test the working of our model in that framework.



---------------COMMENTS FOR NEURALNETWORK.H-------------------------------------------------------------------------------------------------------------

In C, a library is typically created by writing a
set of source files that provide functions or 
variables to be used by other programs. These 
source files are compiled separately and linked 
together with the main program to create an 
executable file.

The basic syntax for creating a library in
C:

1. Header file: Creating a header file that contains the function prototypes, macro definations and variable declarations. This file is used by other programs to include the library written by me.

for example mylib.h

#ifndef MYLIB_H
#define MYLIB_H

#endif // MYLIB_H


2. **Source File (`.c` file)**: Create a source 
file (`*.c`) that contains the implementation of 
the functions or variables declared in your 
header file.
```c
// mylib.c

#include "mylib.h"

void my_function(void) {
    // function implementation
}

int my_variable = 0;
```
3. **Library Creation**: Compile the source file 
into an object file using a compiler like `gcc`.
```bash
gcc -c mylib.c -o mylib.o
```
This creates an object file (`mylib.o`) that 
contains the compiled code for your library.
4. **Library Linking**: Use the `ar` (archive) 
utility to create a shared library or static 
library from the object file.
```bash
ar rcs mylib.a mylib.o
```
For a shared library:
```bash
gcc -shared -o libmylib.so mylib.o
```
5. **Library Installation**: Install your library
in a location where it can be found by other 
programs (e.g., `/usr/local/lib` or `~/lib`).

**Example Library Name**: The name of the library
is typically in the format `lib<library_name>.so`
for a shared library or `<library_name>.a` for a 
static library.

For example:
```bash
// Create a shared library named libmylib.so
gcc -shared -o libmylib.so mylib.o

// Install the library in /usr/local/lib
sudo cp libmylib.so /usr/local/lib/

// Load the library using the ldconfig command 
(on Linux/macOS)
sudo ldconfig

// Now, you can use your library in other C 
programs by including the header file and linking
against the library.
#include "mylib.h"
int main() {
    my_function();
    return 0;
}
```
**Important Notes**:

* The `#ifndef` directive is used to prevent 
multiple inclusion of the header file.
* The `#define` directive defines a macro or 
constant that can be used in your library.
* The `ar` utility creates an archive file 
containing the object files, which can then be 
linked together with other programs.

For more about the std library, read at https://github.com/nothings/std


-> Since all the 


