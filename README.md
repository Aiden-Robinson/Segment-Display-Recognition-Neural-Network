# Segment-Display-Recognition-Neural-Network
Created a neural netowork from scratch by following along with Linkedin Learning's "Training Neural Networks in C++". In the past I did my first neural network using Tensorflow, but this project goes into much more depth without the use of external libraries for the network. 

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
### Step 3: Calculate the Output Error Terms
This is an intermediate error calculation to determine the performance of a neuron. We pay attention to the output layer. We will later use these error terms to calculate error terms in the hidden layers moving backwards, hence the name back propogation

<img width= "300" height = "75" src= "https://user-images.githubusercontent.com/106715980/186541985-e65eeee3-6937-40f5-b78b-fb28e9a340bf.png">

``` C++
for (int i = 0; i < outputs.size(); i++)
        d.back()[i] = outputs[i] * (1 - outputs[i]) * (error[i]);// d is a vector storing error terms
```
### Step 4: Calculating Hidden Layer Error Terms
We iterate through the last hidden layer all the way to the first hidden layer to find an error term per neuron. In the hidden layers, error is unkown because we don't know what to expect from the intermediate neurons. We use a sum of a product that includes the error terms forward neurons connected to the current neurons output.

<img width= "400" height = "100" src= "https://user-images.githubusercontent.com/106715980/186553151-9e8abfb3-adeb-4993-a842-f115f5b6a253.png">

```C++
for (int i = network.size()-2; i > 0; i--)//iterates layers
        for (int h = 0; h < network[i].size(); h++){//iterates neurons
            double fwd_error = 0.0;
            for (int k = 0; k < layers[i+1]; k++)
                fwd_error += network[i+1][k].weights[h] * d[i+1][k]; //summation
            d[i][h] = values[i][h] * (1-values[i][h]) * fwd_error;
        }
```
### Step 5: Apply the Delta Rule

Since we now have all of the error terms, we can proceed to calculate the weight adjustments. The learning rate is constant for all neurons, it affects the rate of learning, higher values result in larger leaps for the weights.

<img width= "450" height = "100" src= "https://user-images.githubusercontent.com/106715980/186554259-66a3cb8a-00b7-422c-bbbd-cf43552e4c38.png">
