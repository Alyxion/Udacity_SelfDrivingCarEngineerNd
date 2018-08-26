#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.8;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3; // MI ADJUSTED
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false;

  time_us_ = 0;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Clear x
  x_ << 0, 0, 0, 0, 0;

  NIS_radar_ = 0.0;
  NIS_laser_ = 0.0;

  // Setup vector for weights
  weights_ = VectorXd(2*n_aug_+1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  // Initialize weights
  for(int i = 1; i<2 * n_aug_ + 1; ++i)
  {
      weights_(i) = 0.5 / (lambda_ + n_aug_);
  }
}

UKF::~UKF() {}

// ---------------------------------------------------------------------------------------------------------------------

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if(!is_initialized_)
  {
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;

    if(meas_package.sensor_type_==MeasurementPackage::RADAR)
    {
      float rho = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      float rhodot = meas_package.raw_measurements_(2);
      x_(0) = rho * cos(phi);
      x_(1) = rho * sin(phi);
      x_(2) = rhodot;

      P_ << 0.1, 0, 0, 0,0,
              0, 0.01, 0, 0,0,
              0, 0, 1, 0,0,
              0, 0, 0, 0.1,0,
              0, 0, 0, 0,0.1;
    }
    else if(meas_package.sensor_type_==MeasurementPackage::LASER)
    {
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);

      P_ << 0.1, 0, 0, 0,0,
              0, 0.1, 0, 0,0,
              0, 0, 1, 0,0,
              0, 0, 0, 0.1,0,
              0, 0, 0, 0,0.1;
    }
  }
  else // initialized
  {
    // Calculate delta t
    double dt = (meas_package.timestamp_-time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    // Predict
    Prediction(dt);

    // Update
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      UpdateRadar(meas_package);
    }
    else
    {
      UpdateLidar(meas_package);
    }
  }

}

// ---------------------------------------------------------------------------------------------------------------------

/**
 * Calculates the sigma points
 * @param delta_t Time since last measurement
 */
void UKF::CalculateSigmaPoints(double delta_t)
{
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  // ----------------- predict sigma points --------------------

  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001)
    {
      px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else
    {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

// ---------------------------------------------------------------------------------------------------------------------

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // predict sigma points (update Xsig_pred_)
  CalculateSigmaPoints(delta_t);

  // create predicted state (5x1)
  VectorXd x = VectorXd(n_x_);

  // create predicted covariance matrix (5x5)
  MatrixXd P = MatrixXd(n_x_, n_x_);

  x.fill(0.0);

  for(int i = 0; i<2 * n_aug_ + 1; ++i)
  {
    //predict state mean
    x += weights_(i) * Xsig_pred_.col(i);
  }

  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x;

    // get angles into range from -M_PI to +M_PI
    if(x_diff(3) > M_PI)
    {
      x_diff(3) -= 2. * M_PI;
    }
    else if(x_diff(3) < -M_PI)
    {
      x_diff(3) += 2. * M_PI;
    }
    P += weights_(i) * x_diff * x_diff.transpose();
  }

  x_ = x;
  P_ = P;
}

// ---------------------------------------------------------------------------------------------------------------------

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // lidar measures px and py
  int n_z = 2;

  // create sigma matrix (2x15)
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);

  for(int i = 0; i<2 * n_aug_ + 1; ++i)
  {
    // transform sigma points
    VectorXd state_vec = Xsig_pred_.col(i);
    double p_x = state_vec(0),
           p_y = state_vec(1);

    Zsig.col(i) << p_x, p_y;
  }

  for(int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // measurement covariance matrix S
  for(int i = 0; i<2 * n_aug_ + 1; ++i)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // normalize angles
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // Calculate R
  MatrixXd R = MatrixXd(n_z,n_z);

  R << std_laspx_*std_laspx_, 0, 0, std_laspy_*std_laspy_;
  S += R;

  // --- Update ---

  // get  measurement

  VectorXd z = VectorXd(n_z);

  double mpx = meas_package.raw_measurements_(0),
  mpy = meas_package.raw_measurements_(1);

  z << mpx, mpy;

  // create cross correlation matrix Tc

  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  // calculate NIS
  NIS_laser_ = (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() * (meas_package.raw_measurements_ - z_pred);
}

// ---------------------------------------------------------------------------------------------------------------------

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */


  // radar measures r, phi and phi_dot
  int n_z = 3;

  // create sigma matrix, 3 x 15
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

  double rho = 0.0;
  double phi = 0.0;
  double rho_d = 0.0;

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    VectorXd state_vec = Xsig_pred_.col(i);

    double px = state_vec(0);
    double py = state_vec(1);
    double v = state_vec(2);
    double yaw = state_vec(3);

    rho = sqrt(px*px+py*py);
    phi = atan2(py,px);
    rho_d = (px*cos(yaw)*v+py*sin(yaw)*v) / rho;

    Zsig.col(i) << rho, phi, rho_d;
  }

  // calc mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);

  // calculate measurement covariance matrix
  for(int i=0; i<2 * n_aug_+1; ++i)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // Calc R
  MatrixXd R = MatrixXd(3,3);
  R << std_radr_*std_radr_, 0, 0,
  0, std_radphi_*std_radphi_, 0,
  0, 0, std_radrd_*std_radrd_;
  S += R;

  // --- Update ---

  VectorXd z = VectorXd(n_z);

  // incoming rho, phi and rhod
  z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), meas_package.raw_measurements_(2);

  // create correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; ++i)
  {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // update state mean and covariance matrix
  x_ += K*z_diff;
  P_ -= K*S*K.transpose();

  // calculate NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
