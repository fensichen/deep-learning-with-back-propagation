#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP

#include <random> //std library comes first
#include <chrono>
#include <generator.hpp>


class Gaussian : public Generator {

public: 
    Gaussian( float mu, float sigma) : Generator() { 
      mu_ = mu;
      sigma_ = sigma;

      distribution_ = new std::normal_distribution<double>(mu_ ,  sigma_ );
      unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
      generator_.seed(seed1);
    }

    float DrawSample( ) ;
 private: 
    float mu_ ; 
    float sigma_;    
    std::default_random_engine generator_;
    std::normal_distribution<double> * distribution_;
};

#endif
