#include <cmath>
#include <iostream>
#include <crossentropy.hpp>
#include <cassert>

using namespace std;

//inplace
void softmax(Datum* in, Datum* out_d){
    auto& d = in->data();
    auto & s = out_d->data();
    float sum = 0;
    //FIXME:overflow
    for (int i = 0 ; i < d.size() ; i++){
            sum += exp(d[i]);
    }
    assert( sum );

    for (int i = 0 ; i < d.size() ; i++){
        s[i] = exp(d[i])/sum;
        assert( !std::isnan(d[i]) );
        assert( !std::isnan(s[i]) );
    } 
}

void CrossEntropy::SetUp(Datum * in){
    //read the input from the previous layer
    input_      = in;
    softmax_ = new Datum( in->size() );
    output_   = new Datum(1); //one lable for each input (image)
    parameterized_ = false; 
}

/*compute the loss */
 void CrossEntropy::Forward(){
    int label = label_->data()[0]; // get the ground truth label
    softmax( this->input_, softmax_ );
    auto& u = this->output_->data();
    
    //check if label is small than zero or out of bound
    if ( label < 0 || label > input_->size() ) {
            cout<<"Invalid"<<endl;
    }
  
    u[0] = -log(softmax_->data()[label]);
 }

/*compute the gradient to backpropagate */
  void CrossEntropy::Backward(){
    
    int label = label_->data()[0];
    auto& t = this->input_->diff(); //write from d -> t

    for (int i = 0 ; i < t.size() ; i++){
            if ( i == label){                
                t[i]= softmax_->data()[i] - 1 ;
            }else {
                t[i]= softmax_->data()[i] - 0 ;   
            }
            assert( !std::isnan( t[i] ) );
    }
  }

 void CrossEntropy::SetLabel(Datum* label){
    label_  = label;
 }


 vector<float> CrossEntropy::parameter( int i ) {
    vector<float> tmp;
    return tmp;
 }