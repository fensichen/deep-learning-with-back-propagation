#include <fcn.hpp>
#include <cassert>
#include <cmath>

void FullyConnectedLayer::SetUp(Datum * in ){
  std::cerr << "Please use SetUp( [3] )" << std::endl;
}
void FullyConnectedLayer::SetUp(Datum * in, int neuron_num, Generator* gen){
    
    this->input_ = in;
    //this->output_ = new Datum(neuron_num);//number of neuron in the current layer
    vector<int> shape{ 1, neuron_num }; //Batch X D
    this->output_ = new Datum( shape );

    //set up weight_, weight_diff_, bias_, bias_diff_ with size of out X in 
    weight_.resize( neuron_num ); 
    weight_diff_.resize( neuron_num );
    weight_diff_pre_.resize( neuron_num );
    for (int i = 0 ; i < weight_.size() ; i++ ){
        weight_[i].resize( in->size() ); 
        weight_diff_[i].resize( in->size(), 0);
        weight_diff_pre_[i].resize( in->size(), 0);

    }
    bias_.resize( neuron_num,0  ); 
    bias_diff_.resize( neuron_num,0 );
    bias_diff_pre_.resize( neuron_num, 0);

    //initialization with random weight
    for (int i = 0 ; i < neuron_num ; i++){
      for (int j = 0 ; j  < in->size() ; j++ ){
          weight_[i][j] = gen->DrawSample();
          
      }
      bias_[i] = 0.01; 
    }

    parameterized_ = true; 

}

void FullyConnectedLayer::Forward(){

      auto& x = this->input_->data();
      auto& z = this->output_->data();

      // std::cout << "[" << this->name_ << "] ";
      // for( int j = 0; j  < x.size(); j++ )
        // std::cout << x[j] << " ";
      // std::cout << std::endl;
      //implement z = wx + b
      for (int i = 0 ; i < z.size() ; i++) {// output index
              float tmp = 0;
              for (int j = 0 ; j <  x.size() ; j++){ // input index
                     tmp +=  weight_[i][j] * x[j] ; 
              } 
             z[i] =  tmp + bias_[i]; 
      } 
}

/*1: weight update, 2: propagate down to the next layer */
void FullyConnectedLayer::Backward(){ 

    /* compute the gradient of weight and bias to update */
    //compute g_act
   auto& g_top = this->output_->diff(); //gradient from the top (diff)
   auto& x       = this->input_->data();
   auto& g_in   = this->input_->diff(); //propagate down to the bottom layer
   

   // std::cout << "[" << this->name_ << "] ";
   //collect the gradient of this item in batch
   for (int i = 0; i < this->output_->size() ; i++) {
       // std::cout << g_top[i] << " ";
      for (int j = 0 ; j < this->input_->size() ; j++) {
          weight_diff_[i][j] += g_top[i] * x[j];
      }
      bias_diff_[i] += g_top[i] * 1 ;
   }
   // std::cout << std::endl;
    // propagate the gradient to the next layer 
    /* compute the gradient regarding to input */
   for (int i = 0 ; i < this->input_->size(); i++) {
        float tmp = 0 ; 
        for (int j = 0; j < this->output_->size() ; j++){
          tmp += g_top[j]*weight_[j][i];
        }
        g_in[i] = tmp;
        assert( !std::isnan( g_in[i] ) );
   }
}


void FullyConnectedLayer::Update( int N ) {

  //apply the graident on the weight by gradient descent 

  float c = 0.9;
  for (int i = 0 ; i < weight_.size() ; i++){
    for (int j = 0 ; j < weight_[i].size() ; j++){
      //float update = lr_ * weight_diff_[i][j];
      float update = lr_ * (weight_diff_[i][j] + c * weight_diff_pre_[i][j]);
      update /= N;
      // std::cout << update << " ";
      weight_[i][j] = weight_[i][j] - update; //batch learning
      weight_diff_pre_[i][j] = weight_diff_[i][j];
      weight_diff_[i][j] = 0; //reset to zero
    }
    //bias_[i] = bias_[i] - lr_ * bias_diff_[i] ;
    float update_bias = lr_ * ( bias_diff_[i] + c * bias_diff_pre_[i]);
    update_bias /= N;
    bias_[i] = bias_[i] - update_bias;
    bias_diff_pre_[i] = bias_diff_[i];
    bias_diff_[i] = 0 ; 
  }
  // std::cout << std::endl;
}


vector<float> FullyConnectedLayer::parameter( int i ) {
  vector<float> tmp; 

  if ( i == 0 ){
    return bias_;
  }

  if ( i == 1 ){
     int n = weight_.size() * weight_[0].size();
     tmp.resize(n,0);

     //serialize the vector< vector<float> > format
     int idx = 0;
     for (int i = 0 ; i  < weight_.size(); i++){
        for (int j = 0 ; j  < weight_[i].size() ; j++ ){
            tmp[idx] = weight_[i][j];
            idx++;
        }
     }

    return tmp;
  }

}