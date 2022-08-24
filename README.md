# Segment-Display-Recognition-Neural-Network
Created a neural netowork from scratch by following along with Linkedin Learning's "Training Neural Networks in C++".

Used a feedforward neural network with 1 input layer, 1 hidden layer, and, 1 output layer. There are 7 inputs and 10 outputs. The 7 inputs being the brightness of each display segment and the 10 outputs being the confidence of classification in each number 0-9

<img width= "350" height = "175" src= "https://user-images.githubusercontent.com/106715980/186539394-43368700-1a66-45b2-a283-70d367158202.png">


A sigmoid function is used for the activation function, this provides nonlinearity to the neuron, and enables training by backpropogation.

<img width= "300" height = "100" src= "https://user-images.githubusercontent.com/106715980/186538404-e0c7777e-f950-4bc7-acd0-39392a74d6be.png">


Although a segment display can be easily classified using hard code, the advantage of using a neural network is generability espeically when the GUI allows for dimmable sections.

## Training by Back Propogation

### Step 1: Feed a Sample to the network
We run a sample 'x' thorugh the network and assign the result to a new vector "output"

```C++
vector<double> outputs = run(x);
```

### Step 2: Calculate the Mean Squared Error
MSE is used to asses the performance of the network. The goal is to reduce the output of this function

<img width= "300" height = "200" src= "https://user-images.githubusercontent.com/106715980/186539866-4e92f39b-af92-430b-b1dd-e51937724639.png">

``` C++
 vector<double> error;
    double MSE = 0.0;
    for (int i = 0; i < y.size(); i++){ // y is the label value
        error.push_back(y[i] - outputs[i]);
        MSE += error[i] * error[i];
    }
    MSE /= layers.back();
 ```
