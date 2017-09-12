#include <accuracy.hpp>

void Accuracy::SetUp(Datum* in){
  this->input_ = in;
}


 void Accuracy::Forward(){

  /* Check the maximun value  of the last layer is the lable ? */

  //the output of the last layer 
  auto& o = this->input_->data();
  int predicted_label = distance( o.begin() , max_element(o.begin() , o.end() )) ; 
  total_input_num_++;

  // std::cout << "[softmax] ";
  // for( int i = 0; i < 10; i++ )
  //   std::cout << o[i] << " ";
  // std::cout << std::endl;

  // std::cout << predicted_label << "/" << label_->data()[0] << std::endl;
  if ( predicted_label == label_ ->data()[0] ) {
    counter_++; 
  }

}

void Accuracy::print_acc() {

  cout<<"The accuracy is "<< setprecision(3)  << (float)counter_/(float)total_input_num_* 100.0 << '%' <<endl;
  counter_ = 0;
  total_input_num_ = 0;

}

  void Accuracy::Backward() {

  }


  vector<float> Accuracy::parameter( int i ) {
      vector<float> tmp;
      return tmp;
  }

