I made a single-layer-perceptron using only NumPy. I used 3 methods for weight adjustment:

1. Weight adjustment-normal version
2. Weight adjustment-modified version
3. Widrow-Hoff Delta rule

When evaluating each training method, I found that all 3 of them achieved 100% accuracy on the test data, as well as a minimum
error of less than 0.2. However, the normal weight adjustment method took more number of epochs to achieve a minimum error,
and oscillated around the minimum error point a bit, due to the lack of a learning rate. 

Widrow-Hoff Delta rule and modified version of weight adjustment performed equally well.
