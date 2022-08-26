#pragma once 
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>

using namespace std;

class Perceptron {
	public: 
		vector<double> weights;
		double bias; //
		Perceptron(int inputs, double bias=1.0); //stating all of the functions that will be declared in this class
        double run(vector<double> x);
		void set_weights(vector<double> w_init);
		double sigmoid(double x);
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

