#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2((map_y-y),(map_x-x));

	double angle = fabs(theta-heading);
  angle = min(2*pi() - angle, angle);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  if (closestWaypoint == maps_x.size())
  {
    closestWaypoint = 0;
  }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

// ---------------------------------------------------------------------------------------------------------------------

enum SensorFusionProperties
{
	VehicleX = 3,
	VehicleY = 4,
	VehicleD = 6,
	VehicleS = 5
};

class PathPlanner
{
protected:
	vector<double> *mWayPointsX = nullptr;	///< Holds the track's x coordinates
	vector<double> *mWayPointsY = nullptr;	///< Holds the track's y coordinates
	vector<double> *mWayPointsS = nullptr;	///< Holds the track's s cooordinates (an increasing number actually)
	vector<double> *mWayPointsDx = nullptr;	///< Holds the track's normal vectors at given point
	vector<double> *mWayPointsDy = nullptr;	///< Holds the track's normal vectors at given point

	int miLaneCount = 3;			///< Defines the count of lanes defined
	double mLaneWidth = 4.0;		///< Defines the full lane width in meters
	double mHalfLaneWidth = 2.0;	///< Defines the half lane width in meters
	double mMaxLaneOffset = 0.25;	///< Maximum offset of lane center

	double mOptimalSpeed = 49.5;	///< Defines the current road's optimal speed
	double mMaxBrake = 0.224;		///< Defines the maximum deceleration per time step
	double mMaxAccelerate = 0.224;	///< Defines the maximum acceleration per time step
	double mSafetyBuffer = 20.0;		///< Defines the safety buffer in meters we require to change a lane

	int mPlanningPointCount = 50; 	///< Defines the count of points in the trajectory
	double mFrequency = 0.02;		///< The frequency (in ms) at which the simulator is triggered and the nodes handled
	double mTargetOffset = 30.0;	///< Defines the target offset in vehicle direction in meters
	int miTargetLane = -1;			///< The current target lane. -1 = No lane switch planned
	int miLeftLanePenalty = 0;		///< A penalty generated when staying for too long in the left lane for no reason

	int mBaseNodeCount = 3;				///< Defines the count of nodes to be passed to the spline function
	double mBaseNodeStepping = 30.0;	///< Defines the distance in meters between each node passed to the spline function

	double mCarX = 0.0;			///< Cars x position
	double mCarY = 0.0;			///< Cars y position
	double mCarS = 0.0;			///< Car's position on the track (along the track) in meters
	double mCarD = 0.0;			///< Car's lane position, center of most left lane = 2.0 (lane width = 4 meters)
	double mCarYaw = 0.0;		///< Car's yaw angle (viewing direction)
	double mCarSpeed = 0.0;		///< Car's current speed in miles/hour
	double mEndPathS = 0.0;		///< The path's end s
	double mEndPathD = 0.0;		///< The path's end d

	vector<double> mPreviousPathX;	///< Contains the previous path's x points
	vector<double> mPreviousPathY;	///< Contains the previous path's y points
	int mPrevSize = 0;

	vector<vector<double> >  mSensorFusionData;	///< Contains the sensor fusion data, so of all currently known vehicles on our road side

	double mTargetSpeed = 0.0;		///< Holds the current target speed


public:
	//! Constructor
	PathPlanner()
	{

	}

	//! Sets the current data. For details about the variable's content see member definitions above
	void SetFrameData(vector<double> *WayPointsX, vector<double> *WayPointsY, vector<double> *WayPointsS, vector<double> *WayPointsDx, vector<double> *WayPointsDy,
			double CarX, double CarY, double CarS, double CarD, double CarYaw, double CarSpeed, double EndPathS, double EndPathD,
			vector<double> &PreviousPathX, vector<double> &PreviousPathY, vector<vector<double> > &SensorFusionData
			)
	{
		// assign data elements 1:1
		mWayPointsX = WayPointsX;
		mWayPointsY = WayPointsY;
		mWayPointsS = WayPointsS;
		mWayPointsDx = WayPointsDx;
		mWayPointsDy = WayPointsDy;

		mCarX = CarX;
		mCarY = CarY;
		mCarS = CarS;
		mCarD = CarD;
		mCarYaw = CarYaw;
		mCarSpeed = CarSpeed;

		mEndPathS = EndPathS;
		mEndPathD = EndPathD;
		mPreviousPathX = PreviousPathX;
		mPreviousPathY = PreviousPathY;
		mSensorFusionData = SensorFusionData;

		mPrevSize = mPreviousPathX.size();

		// update car's if required
		if(mPrevSize>0)
		{
			mCarS = mEndPathS;
		}
	}

	//! Calculates the costs of each lane to decide where to continue driving
	vector<double> CalculateLaneCosts()
	{
		double safetyDistance = CalculateSafetyDistance()*2;

		int currentLane = GetCurrentLane();

		vector<double> laneCosts;
		vector<bool> blocked;

		for(int laneIndex=0; laneIndex<miLaneCount; ++laneIndex)
		{
			double costs = 0.0;

			double laneCenter = laneIndex*mLaneWidth+mHalfLaneWidth;
			double laneDiff = mCarD - laneCenter;

			// the longer the car is on the left lanes, the more attractive right ones do become
			int miMaxPenaltyBonus = 1000;
			int penaltyBonus = miLeftLanePenalty;
			if(penaltyBonus>miMaxPenaltyBonus)
			{
				penaltyBonus = miMaxPenaltyBonus;
			}
			costs += fabs(laneDiff*laneDiff/4)*((miMaxPenaltyBonus-penaltyBonus)/(double)miMaxPenaltyBonus);
			costs += (miLaneCount-1-laneIndex)*2;	// make right lanes preferred by default

			if(abs(laneIndex-currentLane)>1)	// don't allow two lanes in one step
			{
				costs += 400;
			}

			double blockedByCarInFront = 0.0;
			bool blockedByCarAtSide = false;

			for(int i=0; i<mSensorFusionData.size(); ++i)
			{
				float d = mSensorFusionData[i][SensorFusionProperties::VehicleD];
				if(d < (mHalfLaneWidth+mLaneWidth*laneIndex+mHalfLaneWidth) && d > (mHalfLaneWidth+mLaneWidth*laneIndex-mHalfLaneWidth) )
				{
					double vx = mSensorFusionData[i][SensorFusionProperties::VehicleX];
					double vy = mSensorFusionData[i][SensorFusionProperties::VehicleY];

					double check_speed = sqrt(vx*vx+vy*vy);

					// check vehicles in front
					double futureCarS = mSensorFusionData[i][SensorFusionProperties::VehicleS] + ((double)mPrevSize * mFrequency*check_speed);
					if(futureCarS>mCarS && (futureCarS-mCarS)<safetyDistance)
					{
						double bbf = safetyDistance-(futureCarS-mCarS);
						if(blockedByCarInFront<bbf)
						{
							blockedByCarInFront = bbf;
						}
					}

					// check vehicles at direct side
					double futureDist = 2.0 * (check_speed - mCarSpeed);	// we need to take our speed vs other vehicle's speed always into account
					if(futureDist<0)
					{
						futureDist = 0.0;
					}
					double safetyRegionStart = mCarS-mSafetyBuffer;
					double safetyRegionEnd = mCarS+mSafetyBuffer;

					if(futureCarS+futureDist>=safetyRegionStart && futureCarS<=safetyRegionEnd)
					{
						blockedByCarAtSide = true;
					}
				}
			}

			if(blockedByCarAtSide)
			{
				costs += 100;	// don't use the lane if there is a car blocking it
			}
			costs += blockedByCarInFront; // if there is a car in front of any lane a lane switch should be considered

			laneCosts.push_back(costs);
		}

		return laneCosts;
	}

	//! Returns the current lane index
	int GetCurrentLane()
	{
		return mCarD/mLaneWidth;
	}

	//! Chooses the target lane
	int ChooseTargetLane()
	{
		int targetLane = 0;
		auto costs = CalculateLaneCosts();
		double bestCosts = costs[0];
		for(int i=1; i<costs.size(); ++i)
		{
			if(costs[i]<bestCosts)
			{
				bestCosts = costs[i];
				targetLane = i;
			}
		}

		// don't change lances at all if situation is chaotic atm
		double maxCosts = 50;
		if(bestCosts>maxCosts)
		{
			targetLane = GetCurrentLane();
		}

		return targetLane;
	}

	//! Calculates the safety distance at the current speed
	double CalculateSafetyDistance()
	{
		double msp = mCarSpeed / 10;
		return msp*msp + msp*3;
	}

	//! Checks the distance to the vehicles in front of ourselfs. If the car is transitioning it also checks the distance to the other cars
	void CheckDistances()
	{
		double safetyDistance = CalculateSafetyDistance();

		int currentLane = GetCurrentLane();
		vector<int> lanes = {currentLane};
		double off = mCarD-(currentLane*mLaneWidth+mHalfLaneWidth);

		// if the car is transitioning to left or right also consider the lanes it already intersects
		if(off<-mHalfLaneWidth/2 && currentLane>0)
		{
			lanes.push_back(currentLane-1);
		}
		if(off>mHalfLaneWidth/2 && currentLane<miLaneCount-1)
		{
			lanes.push_back(currentLane+1);
		}

		bool brake = false; // braking not required by default

		for(int i=0; i<mSensorFusionData.size(); ++i)
		{
			for(auto lane:lanes)
			{
				float d = mSensorFusionData[i][SensorFusionProperties::VehicleD];
				if(d < (mHalfLaneWidth+mLaneWidth*lane+mHalfLaneWidth) && d > (mHalfLaneWidth+mLaneWidth*lane-mHalfLaneWidth) )
				{
					double vx = mSensorFusionData[i][SensorFusionProperties::VehicleX];
					double vy = mSensorFusionData[i][SensorFusionProperties::VehicleY];
					double check_speed = sqrt(vx*vx+vy*vy);
					double futureCarS = mSensorFusionData[i][SensorFusionProperties::VehicleS];

					futureCarS += ((double)mPrevSize * mFrequency *check_speed);

					if(futureCarS>mCarS && (futureCarS-mCarS)<safetyDistance)
					{
						brake = true;
					}
				}
			}
		}

		// accelerate if possible, brake otherwise
		if(brake)
		{
			mTargetSpeed -= mMaxBrake;
			if(mTargetSpeed<0.0)
			{
				mTargetSpeed = 0.0;
			}
		}
		else if(mTargetSpeed<mOptimalSpeed)
		{
			mTargetSpeed += mMaxAccelerate;
		}
	}

	//! Calculates the trajectory and how to behave in general in the current situation. Evaluates if a lane shift
	//! makes sense, brakes and accelerates as appropriate.
	void CalculateTrajectory(vector<double> &OutXVals, vector<double> &OutYVals)
	{
		int currentLane = mCarD/mLaneWidth;
		double offset = mCarD-(currentLane*mLaneWidth+mHalfLaneWidth);

		miLeftLanePenalty += (miLaneCount-1-currentLane); // remember for how long we are in the left lane

		if(miTargetLane==-1 && mCarSpeed>=mOptimalSpeed*2/3) // if there is no lane change in progress and we are fast enough so it's safe
		{
			miTargetLane = ChooseTargetLane();

			if(miTargetLane>currentLane)	// reset left lane penalty when ever moving to a right lane
			{
				miLeftLanePenalty = 0;
			}
		}
		else
		{
			if(currentLane==miTargetLane && fabs(offset)<mMaxLaneOffset)
			{
				miTargetLane = -1;
			}
		}

		int targetLane = miTargetLane!=-1 ? miTargetLane : currentLane;

		CheckDistances();	// accelerate or brake as required

		double referenceX = mCarX;
		double referenceY = mCarY;
		double referenceYaw = deg2rad(mCarYaw);

		vector<double> 	ptsx;
		vector<double> 	ptsy;
		if(mPrevSize<2)
		{
			double prev_car_x = mCarX - cos(mCarYaw);
			double prev_car_y = mCarY - sin(mCarYaw);

			ptsx.push_back(prev_car_x);
			ptsy.push_back(prev_car_y);

			ptsx.push_back(mCarX);
			ptsy.push_back(mCarY);
		}
		else
		{
			referenceX = mPreviousPathX[mPrevSize-1];
			referenceY = mPreviousPathY[mPrevSize-1];

			double ref_x_prev = mPreviousPathX[mPrevSize-2];
			double ref_y_prev = mPreviousPathY[mPrevSize-2];
			referenceYaw = atan2(referenceY-ref_y_prev, referenceX-ref_x_prev);

			ptsx.push_back(ref_x_prev);
			ptsy.push_back(ref_y_prev);

			ptsx.push_back(referenceX);
			ptsy.push_back(referenceY);
		}

		for(int i=0; i<mBaseNodeCount; ++i)
		{
			vector<double> next_wp = getXY(mCarS + (i+1) * mBaseNodeStepping, (2+4*targetLane), *mWayPointsS, *mWayPointsX, *mWayPointsY);
			ptsx.push_back(next_wp[0]);
			ptsy.push_back(next_wp[1]);
		}

		for(int i=0; i<ptsx.size(); ++i)
		{
			double rel_x =ptsx[i]-referenceX;
			double rel_y =ptsy[i]-referenceY;

			ptsx[i] = (rel_x * cos(0-referenceYaw)-rel_y*sin(0-referenceYaw));
			ptsy[i] = (rel_x * sin(0-referenceYaw)+rel_y*cos(0-referenceYaw));
		}

		tk::spline s;

		s.set_points(ptsx, ptsy);

		for(int i=0; i<mPreviousPathX.size(); ++i)
		{
			OutXVals.push_back(mPreviousPathX[i]);
			OutYVals.push_back(mPreviousPathY[i]);
		}

		double target_y = s(mTargetOffset);
		double target_dist = sqrt(mTargetOffset*mTargetOffset+target_y*target_y);

		double x_add_on = 0;

		for(int i=1; i<= mPlanningPointCount - mPreviousPathX.size(); ++i)
		{
			double N = (target_dist/(mFrequency*mTargetSpeed/2.24));
			double x_point = x_add_on + (mTargetOffset)/N;
			double y_point = s(x_point);

			x_add_on = x_point;

			double x_ref = x_point;
			double y_ref = y_point;

			x_point = (x_ref * cos(referenceYaw)-y_ref*sin(referenceYaw));
			y_point = (x_ref * sin(referenceYaw)+y_ref*cos(referenceYaw));

			x_point += referenceX;
			y_point += referenceY;

			OutXVals.push_back(x_point);
			OutYVals.push_back(y_point);
		}
	}

};

// ---------------------------------------------------------------------------------------------------------------------

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  	PathPlanner pathPlanner;

	h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy, &pathPlanner](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {

        	// extract data from json object
			double carX = j[1]["x"];
			double carY = j[1]["y"];
			double carS = j[1]["s"];
			double carD = j[1]["d"];
			double carYaw = j[1]["yaw"];
			double carSpeed = j[1]["speed"];
			vector<double> previousPathX = j[1]["previous_path_x"];
			vector<double> previousPathY = j[1]["previous_path_y"];
			double endPathS = j[1]["end_path_s"];
			double endPathD = j[1]["end_path_d"];
			vector<vector<double> >  sensorFusionData = j[1]["sensor_fusion"];

			// pass data to planner
			pathPlanner.SetFrameData(&map_waypoints_x, &map_waypoints_y, &map_waypoints_s, &map_waypoints_dx, &map_waypoints_dy,
					carX, carY, carS, carD, carYaw, carSpeed,
					endPathS, endPathD, previousPathX, previousPathY, sensorFusionData);

			vector<double> 	next_x_vals,
							next_y_vals;

			// calculate next trajectory
			pathPlanner.CalculateTrajectory(next_x_vals, next_y_vals);

			// forward trajectory to simulator
			json msgJson;

			// TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds
          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
