#include <time.h> 
#define NEURALNETWORK_IMPLEMENTATION
#include "../Day2/neuralNetwork.h"

// Define the number of bits for the binary operations
#define BITS 2 

/**
 * main function: Entry point of the program
 * - Initializes random number generator with current time
 * - Allocates matrices for input (ti) and output (to)
 * - Populates the matrices with all possible input and output pairs
 * - Defines the architecture and allocates memory for the neural network (nn) and gradient network (g)
 * - Initializes the neural network with random weights
 * - Trains the neural network using backpropagation for 10 million iterations
 * - Tests the trained neural network with all possible inputs
 * - Prints the results and any errors found
 * - Returns 0 to indicate successful execution
 */
int main(void) {
    // Seed the random number generator with the current time for randomness in weight initialization
    srand(time(0)); 

    // Calculate 2^BITS, which is the number of possible values for BITS-bit numbers
    size_t n = (1 << BITS); 

    // Calculate the total number of input combinations (for all pairs of BITS-bit numbers)
    size_t rows = n * n;

    // Allocate memory for the input matrix `ti` with `rows` number of rows and `2 * BITS` columns
    Mat ti = mat_alloc(rows, 2 * BITS); 

    // Allocate memory for the output matrix `to` with `rows` number of rows and `BITS + 1` columns
    Mat to = mat_alloc(rows, BITS + 1); 

    /**
     * Populate input (ti) and output (to) matrices
     * - For each row (i):
     *   - Calculate x as the integer division of i by n (x = i / n)
     *   - Calculate y as the remainder of i divided by n (y = i % n)
     *   - Calculate z as the sum of x and y (z = x + y)
     *   - Determine overflow if z is greater than or equal to n (overflow = z >= n)
     *   - Fill input matrix (ti) with binary representations of x and y
     *   - Fill output matrix (to) with binary representation of z and the overflow bit
     * 
     * Approach:
     * - We iterate over all possible combinations of BITS-bit numbers (x and y)
     * - For each combination, we determine the sum and whether there is an overflow
     * - These values are encoded in binary and stored in the respective matrices
     * 
     * Detailed explanation of complicated operations:
     * - `(x >> j) & 1`: This shifts the bits of x to the right by j positions and extracts the least significant bit.
     * - `MAT_AT(ti, i, j)`: This macro accesses the element at row i, column j of the matrix ti.
     */
    for (size_t i = 0; i < ti.rows; i++) {
        // Calculate the first BITS-bit number
        size_t x = i / n;

        // Calculate the second BITS-bit number
        size_t y = i % n;

        // Calculate the sum of x and y
        size_t z = x + y;

        // Determine if there is an overflow (i.e., if the sum exceeds the maximum BITS-bit number)
        size_t overflow = z >= n;

        // Fill the input and output matrices with the binary representations
        for (size_t j = 0; j < BITS; j++) {
            // Fill the input matrix with the binary representation of x
            MAT_AT(ti, i, j) = (x >> j) & 1;

            // Fill the input matrix with the binary representation of y
            MAT_AT(ti, i, j + BITS) = (y >> j) & 1;

            // Fill the output matrix with the binary representation of z
            MAT_AT(to, i, j) = (z >> j) & 1;
        }

        // Fill the output matrix with the overflow bit
        MAT_AT(to, i, BITS) = overflow;
    }

    // Define the architecture of the neural network
    // - The network has 4 layers: input layer, two hidden layers, and output layer
    // - The input layer has 2 * BITS neurons
    // - The hidden layers have BITS neurons each
    // - The output layer has BITS + 1 neurons (to include the overflow bit)
    size_t arch[] = {2 * BITS, BITS, BITS, BITS + 1}; 

    // Allocate memory for the neural network with the specified architecture
    NN nn = nn_alloc(arch, ARRAY_LEN(arch)); 

    // Allocate memory for the gradient network, which is used in the training process
    NN g = nn_alloc(arch, ARRAY_LEN(arch)); 

    // Initialize the neural network with random weights between 0 and 1
    nn_rand(nn, 0, 1);

    // Print the structure of the neural network
    NN_PRINT(nn); 

    // Set the learning rate for training
    float rate = 1e-1; 

    /**
     * Train the neural network
     * - Perform backpropagation to compute gradients
     * - Update the neural network weights using the gradients
     * - Print the cost (error) to monitor training progress
     * 
     * Approach:
     * - We train the neural network for a large number of iterations (10 million) to ensure convergence
     * - In each iteration, we perform backpropagation to compute the gradient of the cost function
     * - We then update the weights of the network using the computed gradients
     * - We print the cost to monitor how well the network is learning
     * 
     * Detailed explanation of complicated operations:
     * - `nn_backprop(nn, g, ti, to)`: Computes the gradients for the neural network using backpropagation and stores them in `g`.
     * - `nn_learn(nn, g, rate)`: Updates the weights of the neural network using the gradients in `g` and the learning rate `rate`.
     */
    printf("c=%f\n", nn_cost(nn, ti, to)); // Print initial cost
    for (size_t i = 0; i < 10000 * 1000; i++) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        printf("%zu: c=%f\n", i, nn_cost(nn, ti, to));
    }

    /**
     * Test the trained neural network
     * - For each possible pair of x and y:
     *   - Set the inputs for the neural network
     *   - Perform a forward pass through the network
     *   - Check the predicted overflow bit
     *   - Decode the predicted sum from the network output
     *   - Print the results and any errors found
     * 
     * Approach:
     * - We test the neural network on all possible pairs of BITS-bit numbers
     * - For each pair, we set the inputs of the network and perform a forward pass to get the output
     * - We then check if the network's prediction matches the expected result (both sum and overflow)
     * - We print any discrepancies found to identify errors
     * 
     * Detailed explanation of complicated operations:
     * - `MAT_AT(NN_INPUT(nn), 0, z) = (x >> z) & 1`: Sets the input of the neural network with binary representation of x.
     * - `nn_forward(nn)`: Performs a forward pass through the neural network.
     * - `MAT_AT(NN_OUTPUT(nn), 0, BITS)`: Accesses the overflow bit from the output of the neural network.
     */
    size_t fails = 0;
    for (size_t x = 0; x < n; x++) {
        for (size_t y = 0; y < n; y++) {
            size_t zi = x + y;
            printf("%zu + %zu = ", x, y);

            // Set the inputs for the neural network
            for (size_t z = 0; z < BITS; z++) {
                MAT_AT(NN_INPUT(nn), 0, z) = (x >> z) & 1;
                MAT_AT(NN_INPUT(nn), 0, z + BITS) = (y >> z) & 1;
            }

            // Perform a forward pass through the network
            nn_forward(nn);

            // Check if the network predicts an overflow
            if (MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5f) {
                // If there is an overflow but the sum is less than the maximum BITS-bit number, it's an error
                if (zi < n) {
                    printf("%zu + %zu = (OVERFLOW<>%zu)\n", x, y, zi);
                    fails += 1;
                }
            } else {
                // Decode the predicted sum from the network output
                size_t a = 0;
                for (size_t j = 0; j < BITS; j++) {
                    size_t bit = MAT_AT(NN_OUTPUT(nn), 0, j) > 0.5f;
                    a |= bit << j;
                }

                // If the predicted sum does not match the actual sum, it's an error
                if (zi != a) {
                    printf("%zu + %zu = (%zu<>%zu)\n", x, y, zi, a);
                    fails += 1;
                }
            }
        }
    }
    
    // Print "OK" if all tests pass without errors
    if (fails == 0) printf("OK\n"); 
    
    // Return 0 to indicate successful execution
    return 0; 
}
