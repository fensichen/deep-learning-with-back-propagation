#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include <iostream>
#include <vector>

using namespace std;

#include <layer.hpp>
#include <uniform.hpp>


class Dropout : public Layer{
  public:
    Dropout( std::string name )  : Layer(name){}

    void SetUp( Datum * in) ;
    void Forward();
    void Backward();
    std::vector<float> parameter( int i ) ;

private: 
   vector<int> mask_; 
   Uniform* drawer_;

};

#endif