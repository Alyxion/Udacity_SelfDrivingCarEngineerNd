/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    default_random_engine gen;

    // set default weight of 1.0 and a particle count of 100
    const double defWeight = 1.0;
    const int particleCount = 100;

    // setup member variables and presize particle and weight vector
    this->particles.clear();
    this->weights.clear();
    this->num_particles = particleCount;
    this->particles.resize(particleCount);
    this->weights.resize(particleCount);

    // prepare gaussian distributions with the provided deviations and the provided GPS x and y and theta as center.
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < particleCount; ++i)
    {
        // sample random values in range of the gaussian
        double sample_x = dist_x(gen);
        double sample_y = dist_y(gen);
        double sample_theta = dist_theta(gen);
        // create new "random" particle
        this->particles[i] = Particle(i, sample_x, sample_y, sample_theta, defWeight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;

    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];

    // create gaussians with a center of zero and the deviations provided for x, y and theta.
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    // add the noise to every particle and move / rotate it a bit by simulating a movement and rotation of delta_t seconds.
    for(auto &particle: this->particles)
    {
        const auto ptheta = particle.theta;

        if (abs(yaw_rate) != 0)
        {
            particle.x += (velocity/yaw_rate) * (sin(ptheta + (yaw_rate*delta_t)) - sin(ptheta)) + dist_x(gen);
            particle.y += (velocity/yaw_rate) * (cos(ptheta) - cos(ptheta + (yaw_rate*delta_t))) + dist_y(gen);
        }
        else
        {
            particle.x += velocity * delta_t * cos(particle.theta) + dist_x(gen);
            particle.y += velocity * delta_t * sin(particle.theta) + dist_y(gen);
        }
        particle.theta += yaw_rate * delta_t + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// find nearest prediction for every observation...
	for(auto &observation: observations)
	{
	    int closest = -1;
        double closestDist = std::numeric_limits<double>::max();

	    for(const auto &prediction: predicted)
	    {
	        // calculate the squared distance, as we just compare it we don't need to calculate the square root
	        double  diffX = prediction.x-observation.x,
	                diffY = prediction.y-observation.y;
	        double sqrDist = diffX*diffX + diffY*diffY;

	        // if current distance is better than the last one select the new element
            if(sqrDist<closestDist)
            {
                closest = prediction.id;
                closestDist = sqrDist;
            }
        }

        observation.id = closest; // associate with closest element
	}

}

/**
 * Computes the potential observations within sensor range of the vehicle
 */
void ParticleFilter::getInRange(const Particle &particle, const double sensor_range, const double *std_landmark,
                                const std::vector<LandmarkObs> &observations,
                                const Map &map_landmarks, std::vector<LandmarkObs> &outPredictions)
{
    outPredictions.clear();

    double  px = particle.x,
            py = particle.y;

    // find all landmarks in range of our sensor...
    for(auto &landmark: map_landmarks.landmark_list)
    {
        if(dist(landmark.x_f, landmark.y_f, px, py)<sensor_range)
        {
            outPredictions.push_back(LandmarkObs({landmark.id_i, landmark.x_f, landmark.y_f}));
        }
    }
}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// prepare multiplier and denominators
	double weightMultiplier = (1.0 / (2.0 * M_PI * std_landmark[0]*std_landmark[1]));
	const double xNorm = 2 * std_landmark[0] * std_landmark[0];
    const double yNorm = 2 * std_landmark[1] * std_landmark[1];

    // for all particles...
	int pIndex = 0;
    for(auto &particle: this->particles)
    {
        auto ptheta = particle.theta;
        auto cos_pt = cos(ptheta),
            sin_pt = sin(ptheta);

        // check which particles are in range
        vector<LandmarkObs> inRange;
        getInRange(particle, sensor_range, std_landmark, observations, map_landmarks, inRange);

        // transform all observed locations
        vector<LandmarkObs> mapSpaceObservations(observations.size());
        int index = 0;
        for (const auto &observation: observations)
        {
            const double transX = cos_pt*observation.x - sin_pt*observation.y + particle.x;
            const double transY = sin_pt*observation.x + cos_pt*observation.y + particle.y;
            mapSpaceObservations[index++] = LandmarkObs({observation.id, transX, transY});
        }

        // associate the observations in range of the sensor with all observations
        dataAssociation(inRange, mapSpaceObservations);

        particle.weight = 1.0; // reset weight to 1.0
 	    for (const auto &observation: mapSpaceObservations)
        {
 	        LandmarkObs predicted;

 	        // receive coordinate of current observation
            for(const auto &posPredicted : inRange)
            {
                if(posPredicted.id==observation.id)
                {
                    predicted = posPredicted;
                }
            }

            // calculate weight of the predicted coordinate vs where it ought to be.
            // we do not need to calculate theta here separately as a wrong theta will automatically
            // also result in wronger predictions through the movement in the prediction step
            const double xDiff = predicted.x - observation.x;
            const double yDiff = predicted.y - observation.y;
            const double xError = (xDiff*xDiff / xNorm);
            const double yError = (yDiff*yDiff / yNorm);

            const double observationWeight = weightMultiplier * exp(-(xError + yError));

            particle.weight *= observationWeight;
        }

        // assign new weight to global weights array, will be required in the resampling step to sample by probability.
        this->weights[pIndex++] = particle.weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// prepare sampling by probability
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> probDistributor(weights.begin(), weights.end());

    // sample num_particles new particles which will replace the old ones.
    // the higher theirs score (weight) is ... so the more precise they are ... the more likely they will
    // be resampled.
    vector<Particle> newParticles(this->num_particles);
    for(auto &newParticle:newParticles)
    {
        newParticle = particles[probDistributor(gen)];
    }

    particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
