#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225f, 0.f,
        0.f, 0.0225f;

  //measurement covariance matrix - radar
  R_radar_ << 0.09f, 0.f, 0.f,
        0.f, 0.0009f, 0.f,
        0.f, 0.f, 0.09f;

  /**
  TODO:
    * Finish initializing the FusionEKF.
  */

  H_laser_ << 1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f;

  ekf_.F_ = MatrixXd(4, 4);

  //for state state transition matrix F, delta_t is initialized with 1 (1 sec)
  //it is going to be updated later in the ProcessMeasurement based on measurement timestamps
  ekf_.F_ << 1.0, 0, 1, 0,
         0, 1, 0, 1,
         0, 0, 1, 0,
         0, 0, 0, 1;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      ekf_.x_(0) = measurement_pack.raw_measurements_(0)*cos(measurement_pack.raw_measurements_(1));
      ekf_.x_(1) = measurement_pack.raw_measurements_(0)*sin(measurement_pack.raw_measurements_(1));      
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_(0) = measurement_pack.raw_measurements_(0);
      ekf_.x_(1) = measurement_pack.raw_measurements_(1);      
    }
    else
    {
    }

    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
           0, 1, 0, 0,
           0, 0, 1, 0,
           0, 0, 0, 1;

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use sigma_ax = 9 and sigma_ay = 9 for your Q matrix.
   */

  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.f; //dt - expressed in seconds
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3*dt;
  previous_timestamp_ = measurement_pack.timestamp_;

  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //set the process noise covariance matrix
  float sigma_ax = 9.f;
  float sigma_ay = 9.f;
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4/4.f*sigma_ax, 0.f, dt_3/2.f*sigma_ax, 0.f,
      0.f, dt_4/4.f*sigma_ay, 0.f, dt_3/2.f*sigma_ay,
      dt_3/2.f*sigma_ax, 0.f, dt_2*sigma_ax, 0.f,
      0.f, dt_3/2.f*sigma_ay, 0.f, dt_2*sigma_ay;


  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    //recover state parameters
    const double x = ekf_.x_[0];
    const double y = ekf_.x_[1];
    
    const float minValue = 0.000000001;

    if ((fabs(x)<minValue) && (fabs(y)<minValue)) // avoid NaN values, just predict the current state
    {
      return;
    }

    MatrixXd Hj = tools.CalculateJacobian(ekf_.x_);

    ekf_.H_ = Hj;
    ekf_.R_ = R_radar_;

    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);    
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
