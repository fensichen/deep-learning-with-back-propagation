#include <cmath>
#include <iostream>

#include <crossentropy.hpp>
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
    // cout << "[softmax] ";
    for (int i = 0 ; i < d.size() ; i++){
        s[i] = exp(d[i])/sum;
        // cout << s[i] << " ";
    } 
    // cout << endl;
}

void L2loss::SetUp(Datum * in){
    //read the input from the previous layer
    input_      = in;
    softmax_ = new Datum( in->size() );
    //output_ = new Datum(in->size());
    output_   = new Datum(1);
    parameterized_ = false; 
}

/*compute the loss */
 void L2loss::Forward(){
    int label = label_->data()[0]; // get the ground truth label
    softmax( this->input_, softmax_ );
    auto& u = this->output_->data();
    
    //check if label is small than zero or out of bound
    if ( label < 0 || label > input_->size() ) {
            cout<<"Invalid"<<endl;
    }
  
    float sum = 0;
    for (int i = 0; i < softmax_->size(); i++){
        if ( i == label_->data()[0])
            sum += (softmax_->data()[i] - 1)* (softmax_->data()[i] -1);
        else
            sum += (softmax_->data()[i])* (softmax_->data()[i]);
    }
    u[0] =  sum;
 }

/*compute the gradient to backpropagate */
  void L2loss::Backward(){
    
    int label = label_->data()[0];
    auto& t = this->input_->diff(); //write from d -> t

    for (int i = 0 ; i < t.size() ; i++){
            if ( i == label){                
                t[i]= (softmax_->data()[i] - 1) ;
            }else {
                t[i]= (softmax_->data()[i] - 0) ;   
            }
    }
  }

 void L2loss::SetLabel(Datum* label){
    label_  = label;
 }