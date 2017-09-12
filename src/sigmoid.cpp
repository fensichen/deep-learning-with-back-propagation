#include <cmath>

#include <sigmoid.hpp>

void Sigmoid::SetUp(Datum * in){
    //read the input from the previous layer
    input_ = in;
    //output_ = new Datum(in->size());
    output_ = new Datum( in->shape() );
    parameterized_ = false; 
}

/*
sigmoid(x) = 1 / ( 1+ exp (-x) )
*/
 void Sigmoid::Forward(){

    auto& v = this->input_->data();
    auto& u = this->output_->data();
    for (int i = 0 ; i < v.size() ; i++) {
       u[i] = 1.0/(1.0 + exp(-v[i]));
    }
 }

 /*back propagate from output to input*/
  void Sigmoid::Backward(){

    auto& d = this->output_->diff(); //gradient from the top (diff)
    auto& u = this->output_->data();
    auto& t = this->input_->diff(); //write from d -> t

    for (int i = 0 ; i < d.size() ; i++){                
                t[i]= d[i] * u[i] *( 1.0- u[i] );
    }
  }
