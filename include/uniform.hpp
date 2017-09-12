#ifndef UNIFORM_HPP
#define UNIFORM_HPP

#include <random> //std library comes first
#include <generator.hpp>

class Uniform : public Generator {

public: 
    Uniform( float down, float upper) : Generator() { 
   
        down_ = down; 
        upper_ = upper;
        distribution_ = new std::uniform_real_distribution<double>( down_ ,  upper_ );
    }

    float DrawSample( ) ;
 private: 
    float down_ ; 
    float upper_ ; 
    std::default_random_engine generator_;
    std::uniform_real_distribution<double> * distribution_;
    
};

#endif