#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
  int xl;
  float yl = 10.0;
  xl = yl;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  //predict the state
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  //Quiz: the difference between Qx and Qv
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  
  //Error y
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  //Kalman Gain
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;

  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;


  //update
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);

  P_ = (I - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

 // find z_predict 
  
  float x = x_[0];
  float y = x_[1];
  float vx = x_[2];
  float vy = x_[3];

  float ro = sqrt(x * x + y * y);
  float theta = atan2(y, x);
  float ro_dot = (x * vx + y * vy) / ro;

  VectorXd z_pred(3);
  z_pred << ro, theta, ro_dot;

  VectorXd y_ = z - z_pred;

  //angle normalization
  while (y_(1)> M_PI) y_(1)-=2.*M_PI;
  while (y_(1)<-M_PI) y_(1)+=2.*M_PI;

  //Kalman Gain
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;

  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //update
  x_ = x_ + (K * y_);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);

  P_ = (I - K * H_) * P_;
 

}
