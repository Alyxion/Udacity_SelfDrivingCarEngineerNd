#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "Dense"

using namespace std;
using Eigen::ArrayXd;

class GNB {
public:
    static const int actionCount = 3;

	vector<string> possible_labels = {"left","keep","right"};
	
	ArrayXd means[actionCount],    ///< Contains the means of the three actions
	        sds[actionCount];      ///< Contains the standard deviations of the three actions
    double priors[actionCount];      ///< The count of prior label relation


	/**
  	* Constructor
  	*/
 	GNB();

	/**
 	* Destructor
 	*/
 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string>  labels);

  	string predict(vector<double>);

};

#endif