#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    /**
    TODO:
    * Calculate the RMSE here.
    */

    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    // validity check

    if(estimations.size() == 0)
    {
        std::cout << "Zero sized estimations error" << std::endl;
        return rmse;
    }
    else if(estimations.size() != ground_truth.size())
    {
        std::cout << "Estimation size mismatches ground truth size" << std::endl;
        return rmse;
    }

    for(unsigned int i=0; i<estimations.size(); ++i)
    {
        VectorXd residuals = estimations[i] - ground_truth[i];
        residuals = residuals.array() * residuals.array();
        rmse += residuals;
    }

    // calculate the mean
    rmse = rmse / estimations.size();

    // calculate square root
    rmse = rmse.array().sqrt();

    return rmse;
}