For optimising our costfunction, we were using a hack or say a very vague method called Finite Differences, so we will be migrating to the back propagation for the real world use cases.


===================Tools================
---------
Pdflatex|
---------
For installing it
```
sudo apt-get install texlive-latex-base
sudo apt-get install texlive-fonts-recommended
sudo apt-get install texlive-fonts-extra
sudo apt-get install texlive-latex-extra
```
For running it
```
pdflatex latex_source_name.tex
```

========================================

-> The value of the gradient descent needs to be infinitesmally small, not zero. Its similar to the definition of a derivative.It gives the velocity of the function at a point.

In gradient descent, we uses the exact definition of the derivative, which ultimately tells us about the velocity of the function at a point, whether it grows or not. And we have to move in the opposite direction, in order to make sure to have the cost function reduced.

Its very very fast for the rate = 1e-1; compared to finite diff its very very fast and efficient. This method will work like a butter on a bread for the small parameter models,  but for the models having tons of parameters, we use something called Back Propagation.

Back Propagation is a technique of calculating derivatives of complex nested functions. And it is the standard cost function used in the neural network.


================================================================



