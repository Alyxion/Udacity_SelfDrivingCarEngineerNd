/**
 * Project 4 of the Udacity Self-Driving Car Engineer Nanodegree
 *
 * PID controller to steer a car around a race track.
 *
 * Copyright (C) 2018 by Michael Ikemann
 *
 * https://github.com/alyxion
 * */

#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != std::string::npos) {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main()
{
  uWS::Hub h;

  PID pid, speedPid;
  // TODO: Initialize the pid variable.

  // ----------------------------- Description of my PID controller approach -----------------------------
  /*
   * PID stands for Proportional Integral Derivative
   *
   * https://youtu.be/EKQCDRgnQMo
   *
   * In this project we use a PID controller to keep the vehicle as close to the optimal line as possible.
   *
   * The more far on the left of the lane the vehicle is, the lower (negative) the cte (cross track error) value in the
   * logic block below will become, the more to the right of the lane the higher (positive).
   *
   * The values defined above in pid.Init define the proportional, the integral and the derivative coefficients.
   *
   * The proportional component actually scales the negative absolute error and converts it into a steering action.
   * As the absolute error can become very huge we need to keep this value quite small (below 1 in our case) as
   * otherwise temporary dangerous situations would end in massive counter steering action and as a result oversteering
   * behavior what is not desired.
   *
   * The integral component reacts to the accumulated error. It's important to note at this point that due to the
   * matter that this error can also be negative this value will not increase endlessly but just become huge (or
   * very far below zero) if the  car for example slides very very far off the centre. In this case this value will
   * grow and grow and the more it grows the harder the integral component correction will "kick in" to get the vehicle
   * back to the middle of the street. When the vehicle is back to the middle of the street it will (in case of a
   * very high accumulated error) overshoot due to the integral component, but with every second it overshoots into
   * the opposite direction the greater a negative or the lower a positive accumulated error will become until it reaches
   * 0.0 again. It's important to don't set this value too high as otherwise it won't be able to recover from it's
   * own overshooting. A value in a region of E-3 is proposed such as 0.001 or 0.002 in our scenario.
   *
   * The derivative component defines the difference between the current error and the recent, so the gradient actually.
   * In situations where the oversteering becomes worser and worser so like into a oscillating mode the higher this
   * value becomes. And the higher this value becomes the more actively it counters this oversteering to normalize the
   * driving behavior again.
   *
   * How I found the values above? (0.15, 0.002, 9.0)
   *
   * I found them by experimentation and by starting with a value of 0.2, 0 and 0 which let the vehicle drive quite
   * normally until it comes into special situation with led to oscillating. The oscillation I then countered with
   * the derivative part.
   * The integral part came last and as described aboves majorly helped in extremely dangerous situations (like when
   * sliding with 80 miles through a curve). So it has very less effect in normal situations but in such special ones
   * the accumulated error becomes very quickly very fast and then this integral component then could really be
   * described as "Vin Diesel Drift Mode" ;-).
   *
   * Next to the steering controller I also integrated a speed controller which though just uses the proportional and
   * derivative component. It uses the steerings cte and if this value is below 0.7 accelerated and otherwise
   * brakes the vehicle proportionally, so it softly brakes in long curves and brakes strongly if it detects through
   * a strongly growing cte that it entered a curve too fast.
   */

  // setup steering and speed controller PID/PD controller.
  pid.Init(0.15, 0.002, 11.0);
  speedPid.Init(3.0, 0.000, 10.0);
  const double maxError = 0.7;

  h.onMessage([&pid, &speedPid, maxError](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(std::string(data).substr(0, length));
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<std::string>());
          double speed = std::stod(j[1]["speed"].get<std::string>());
          double angle = std::stod(j[1]["steering_angle"].get<std::string>());
          double steer_value;
          /*
          * TODO: Calcuate steering value here, remember the steering value is
          * [-1, 1].
          * NOTE: Feel free to play around with the throttle and speed. Maybe use
          * another PID controller to control the speed!
          */

          // calculate steering angle by combing proportional, derivative and integral part
          steer_value = -pid.Kp * cte - pid.Kd*(cte-pid.lastError) - pid.Ki * pid.TotalError();

          double speedCte = fabs(cte)-maxError;
          if(speed>40)
          {
            speedCte *= speed/40.0;
          }
          double speedValue = -speedPid.Kp * speedCte - speedPid.Kd*(speedCte-speedPid.lastError) - speedPid.Ki * speedPid.TotalError();

          // UpdateError() updates the last error for the derivative part and sums the total error for the integral one.
          pid.UpdateError(cte);
          speedPid.UpdateError(speedCte);

          // Calculate speed. We prevent backwards driving and a too careful driving style by setting a minimum speed
          // of 40. Also we limit the braking and acceleration strength to prevent to abrupt behavior.
          speedValue = speed<40.0 ? (speedValue<0.5 ? 0.5 : speedValue>1.0 ? 1.0 : speedValue) :
          speedValue<-2.0 ? -2.0 : speedValue>1.0 ? 1.0 : speedValue;

          // DEBUG
          // std::cout << "CTE: " << cte << " Steering Value: " << steer_value << " " << pid.lastError << " " <<  speedPid.lastError << std::endl;

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = speedValue;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          //std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
