#ifndef FULLYCONNECTEDLAYER_HPP
#define FULLYCONNECTEDLAYER_HPP 

#include <iostream>
#include <vector>
#include <layer.hpp>
#include <generator.hpp>
using namespace std;

class FullyConnectedLayer : public Layer{ 

  public: 
      FullyConnectedLayer (std::string name) : Layer(name) {

      }
      void SetUp(Datum * in);
      void SetUp(Datum * in, int neuron_num, Generator* gen) ;
      inline void SetLR(float learning_rate) {
       lr_ = learning_rate;
      }
      void Forward();
      void Backward();
      void Update( int N );

      inline void relink( Datum * in ){
       this->input_ = in;
      }

     vector<float> parameter( int i ) ;

   private: 

      std::vector< std::vector<float> > weight_; 
      std::vector<float> bias_ ;

      //temporary memory (the gradient of both) to update the weight and bias
      std::vector< std::vector<float> > weight_diff_; 
      std::vector<float> bias_diff_;
      float lr_ = 1.0; 

      //momentum 
      std::vector< std::vector<float> > weight_diff_pre_; 
      std::vector<float> bias_diff_pre_;
      
};

#endif 