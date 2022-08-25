# Segment-Display-Recognition-Neural-Network
Created a neural netowork from scratch by following along with Linkedin Learning's "Training Neural Networks in C++". 

Used a feedforward neural network with 1 input layer, 1 hidden layer, and, 1 output layer. There are 7 inputs and 10 outputs. The 7 inputs being the brightness of each display segment and the 10 outputs being the confidence of classification in each number 0-9

`Skills:` Neural Networks, OOP


| |  |
|:---:|:---:|
|<img width= "430" height = "400" src= "https://user-images.githubusercontent.com/106715980/186570376-db5c907e-7aa2-4c60-8954-9d2588c45167.png"> |<img width= "430" height = "400" src= "https://user-images.githubusercontent.com/106715980/186570442-6f1f6783-9f64-4518-90ae-06eba608c978.png">|
|<img width= "430" height = "400" src= "https://user-images.githubusercontent.com/106715980/186570452-6f3defd3-cafd-475c-ad45-33e65827cfde.png">| <img width= "430" height = "400" src= "https://user-images.githubusercontent.com/106715980/186570461-71f5f7b7-b86e-4d34-86b6-e54064dbcb63.png">|




### Network Structure

<img width= "350" height = "175" src= "https://user-images.githubusercontent.com/106715980/186539394-43368700-1a66-45b2-a283-70d367158202.png">


A `sigmoid function` is used for the activation function, this provides nonlinearity to the neuron, and enables training by backpropogation.

<img width= "300" height = "100" src= "https://user-images.githubusercontent.com/106715980/186538404-e0c7777e-f950-4bc7-acd0-39392a74d6be.png">


Although a segment display can be easily classified using hard code, the advantage of using a neural network is generability espeically when the GUI allows for dimmable sections. Graphical and user interface was done with a template from the JUCE framework.

<img width= "300" height = "100" src= "https://user-images.githubusercontent.com/106715980/186569291-c11ed917-fe7d-4e3b-81d2-fe60b0b58adf.png">

Without the use of libraries, perceptrons, weights,  activation function, back propogation training, were hard coded with the use of these classes:

```C++
class Perceptron {
	public: 
		vector<double> weights;
		double bias; //
		Perceptron(int inputs, double bias=1.0); //stating all of the functions that will be declared in this class
        double run(vector<double> x);
		void set_weights(vector<double> w_init);
		double sigmoid(double x); //sigmoid function
};

class MultiLayerPerceptron {
	public: 
		MultiLayerPerceptron(vector<int> layers, double bias=1.0, double eta = 0.5); //stating all of the functions that will be declared in this class
		void set_weights(vector<vector<vector<double> > > w_init);
		void print_weights();
		vector<double> run(vector<double> x);
		double bp(vector<double> x, vector<double> y); // back propogation
		
		vector<int> layers;
		double bias;
		double eta;// learning rate
		vector<vector<Perceptron> > network;// vector of vector of perceptrons
		vector<vector<double> > values;// vector with the same dimensions as the network to hold the output values of the neurons. It will be used to propogate the results forward throught the network.
		vector<vector<double> > d;// vector of vectors containing error terms for the neurons
};
```

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
### Step 5/6: Apply the Delta Rule and Adjust Weights

Since we now have all of the error terms, we can proceed to calculate the weight adjustments of the neuron. The learning rate is constant for all neurons, it affects the rate of learning, higher values result in larger leaps for the weights.

Adjusting the weights will result will change the MSE of the output. The goal of gradient descent is to find the weight that minimized MSE as illustrated by this diagram. The learning rate is essentially how large the steps we are taking across this function to find that point. 

**Gradient Descent**

<img width= "500" height = "200" src= "https://user-images.githubusercontent.com/106715980/186555929-c1910a6c-004b-4e82-b86f-8ca73b7825af.png">


**Delta Rule**

<img width= "450" height = "100" src= "https://user-images.githubusercontent.com/106715980/186554259-66a3cb8a-00b7-422c-bbbd-cf43552e4c38.png">

**Weight Adjustent**

<img width= "300" height = "50" src= "https://user-images.githubusercontent.com/106715980/186556699-f59dbdf8-6237-4fa1-b9e5-31da1ac0bf9c.png">


```C++
for (int i = 1; i < network.size(); i++)//iterates layers
        for (int j = 0; j < layers[i]; j++)//iterates neurons
            for (int k = 0; k < layers[i-1]+1; k++){//iterates inputs, the +1 is for the bias weight
                double delta;
                if (k==layers[i-1]) //fo the following if its the last input
                    delta = eta * d[i][j] * bias;
                else
                    delta = eta * d[i][j] * values[i-1][k];// delta formula
                network[i][j].weights[k] += delta;//adjusting the weights
            }
    return MSE;
```
