
#include <iostream>
#include <random>
#include <gaussian.hpp>

float Gaussian:: DrawSample(){
      return (*distribution_)( generator_ );
} 

