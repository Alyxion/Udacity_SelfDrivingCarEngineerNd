#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"


using namespace std;
using Eigen::ArrayXd;

/**
 * Initializes GNB
 */
GNB::GNB() 
{
    for(int actionIndex = 0; actionIndex<actionCount; ++actionIndex)
    {
        (means[actionIndex] = ArrayXd(4)) << 0,0,0,0;
        (sds[actionIndex] = ArrayXd(4)) << 0,0,0,0;
        priors[actionIndex] = 0.0;
    }
}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{

	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/
	
	//For each label, compute ArrayXd of means, one for each data class (s, d, s_dot, d_dot).
	//These will be used later to provide distributions for conditional probabilites.
	//Means are stored in an ArrayXd of size 4.
	
	float sizes[actionCount];
	for(int i=0; i<actionCount; ++i)
	{
	    sizes[i] = 0;
	}
	
	std::vector<int> actions;
	
    //For each label, compute the numerators of the means for each class
    //and the total number of data points given with that label.
	for (int i=0; i<labels.size(); i++)
	{
    	int action = -1;
    	int index = 0;
    	
    	for(auto &cur: possible_labels)
    	{
    	    if(cur==labels[i])
    	    {
    	        action = index;
    	        break;
    	    }
    	    index++;
    	}
    	
    	actions.push_back(action);
	    
        means[action] += ArrayXd::Map(data[i].data(), data[i].size()); //conversion of data[i] to ArrayXd
        sizes[action] += 1;
	}

	
	//Compute the means. Each result is a ArrayXd of means (4 means, one for each class)..
	for(int index=0; index<actionCount; ++index)
	{
        if(sizes[index]!=0)	    
        {
	        means[index] = means[index]/sizes[index];
        }
	}

	//Begin computation of standard deviations for each class/label combination.
	ArrayXd data_point;
	
	//Compute numerators of the standard deviations.
	for (int i=0; i<labels.size(); i++)
	{
	    auto action = actions[i];
	    
	    data_point = ArrayXd::Map(data[i].data(), data[i].size());
        sds[action] += (data_point - means[action])*(data_point - means[action]);
	}
	
	//compute standard deviations
	for(int index=0; index<actionCount; ++index)
	{
    	sds[index] = (sds[index]/sizes[index]).sqrt();
    	priors[index] = sizes[index]/labels.size();
    }
}

string GNB::predict(vector<double> sample)
{
	/*
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
	*/
	
	//Calculate product of conditional probabilities for each label.
	
	double probabilities[actionCount];
	for(int action=0; action<actionCount; ++action)
	{
	    probabilities[action] = 1.0;
    	for (int i=0; i<4; i++)
    	{
    	    probabilities[action] *= (1.0/sqrt(2.0 * M_PI * pow(sds[action][i], 2))) * exp(-0.5*pow(sample[i] - means[action][i], 2)/pow(sds[action][i], 2));
    	}
    	
    	probabilities[action] *= priors[action];
	}
	
    double max = probabilities[0];
    double max_index = 0;
    for (int i=1; i<actionCount; i++){
        if (probabilities[i] > max) {
            max = probabilities[i];
            max_index = i;
        }
    }
	
	return this -> possible_labels[max_index];

}