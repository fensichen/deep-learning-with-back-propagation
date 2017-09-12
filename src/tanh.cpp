#include <cmath>
#include <tanh.hpp>

float compute_sigmoid(float v){
 
    float u = 1.0/(1.0 + exp(-v));
    
    return u;
}


void TanH::SetUp(Datum * in){
    //read the input from the previous layer
    input_ = in;
    // output_ = new Datum(in->size());
    output_ = new Datum( in->shape() );
    parameterized_ = false; 
}

 void TanH::Forward(){
      auto& v = this->input_->data();
      auto& u = this->output_->data();
     for (int i = 0 ; i < v.size() ; i++) {
       u[i] = 2.0 * compute_sigmoid(v[i]) - 1.0;
    }
    
 }

  void TanH::Backward(){

    auto& d = this->output_->diff(); //gradient from the top (diff)
    auto& u = this->output_->data();
    auto& t = this->input_->diff(); //write from d -> t
    for (int i = 0 ; i < d.size() ; i++){
    
                t[i]= d[i] * (1.0 - (u[i]*u[i])) ;
    }
  }