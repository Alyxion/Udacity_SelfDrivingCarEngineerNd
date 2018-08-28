#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd)
{
    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;
    this->lastError = 0.0;
    this->totalError = 0.0;
}

void PID::UpdateError(double cte)
{
    this->lastError = cte;
    this->totalError += cte;
}

double PID::TotalError()
{
    return totalError;
}

