#include <iostream>
#include <random>

#include <uniform.hpp>

float Uniform:: DrawSample(){
      return (*distribution_)( generator_ );
} 
