#include <dropout.hpp>

  void Dropout::SetUp( Datum * in){

     input_     = in;
     output_  = new Datum(in->size());
     drawer_ = new Uniform( 0 , 1 );
     mask_.resize(in->size(), 0);
  }

  void Dropout::Forward(){
    //draw dice 
    
    for (int i = 0 ; i < mask_.size() ; i++){
      float p = drawer_->DrawSample();
      mask_[i] = p > 0.5;
    }

    auto& u   = this->input_->data();
    auto& v    = this->output_->data();

    for (int i = 0 ; i < mask_.size() ; i++){
      v[i] = mask_[i] * u[i];
    }
  }


  void Dropout::Backward(){

    auto& d   = this->input_->diff();
    auto& v    = this->output_->diff();
    for ( int i = 0 ; i < v.size(); i++){
      v[i] = d[i] * mask_[i];
    }
  }

  vector<float> Dropout::parameter( int i ) {
    vector<float> tmp;
    return tmp;

 }