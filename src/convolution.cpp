#include <layer.hpp>
#include <convolution.hpp>
#include <cassert>

int Convolution::compute_offset( std::vector<int> idx, std::vector<int> & shape ){
  if( idx.size() != shape.size() )
    return -1;

  std::vector<int> tmp;
  tmp.insert( tmp.begin(), shape.begin(), shape.end() );
  tmp.push_back( 1 );

  int index = 0;
  for( int i = 0; i < idx.size(); ++i ){
    index += idx[i] * tmp[i+1];
  }
  return index;
}

float Convolution::transform_coordinate( int x, int y, int w, int h, int chan) {
    if ( x < 0 || x >= w ){
        return 0;
    }
    if ( y < 0 || y >= h ){
      return 0 ;
    }
    //return this->input_->data()[ y * w + x ];
    int idx =  transform_coordinate_index(x, y,  w,  h, chan);
    return this->input_->data()[ idx ]; 
} 

int Convolution::transform_coordinate_index(int x, int y, int w, int h, int chan = 0 ){

    if ( x < 0 || x >= w ){
        return -1;
    }
    if ( y < 0 || y >= h ){
      return -1 ;
    }
    int idx = y * w + x + ( (w*h) * chan);
    return idx;
    //return  y * w + x ; 
}

void Convolution::SetUp( Datum * in ){

}

void Convolution::SetUp(Datum * in, vector<int>& kernel_shape, Generator* gen){

  kernel_shape_    = kernel_shape; 

  int c_in    = kernel_shape_ [0]; // e.g., the number of channel in the image
  int c_out = kernel_shape_ [1]; //  #filter ( number of extracted feature ) 
  int h       = kernel_shape_ [2];  // filter height
  int w      = kernel_shape_ [3];  //  filter width
  int count = c_in * c_out * h * w;

  this->input_ = in;
  //this->output_ = new Datum( c_out * in->size()/c_in ); //why?
  vector<int> output_shape{ 1, c_out, in->shape()[2], in->shape()[3] }; 
  this->output_ = new Datum(  output_shape );
  
  bias_.resize( c_out, 0);
  bias_diff_.resize( c_out, 0 );
  bias_diff_pre_.resize( c_out, 0 );

  weight_.resize( count, 0 );
  weight_diff_.resize( count, 0 );
  weight_diff_pre_.resize( count, 0 );

  for (int j = 0 ; j < count ; j++)
      weight_[j] = gen->DrawSample();


  for (int i = 0 ; i < c_out; i++) 
    bias_[i] = gen->DrawSample();
  
}

/* Apply con*/
void Convolution::Forward(){

    int h               = kernel_shape_[2];
    int w              = kernel_shape_[3];

    int axis          = 1 ;
    int img_h       = this->input_->shape()[axis+1];
    int img_w      = this->input_->shape()[axis+2]; 
    int img_chan = this->input_->shape()[axis+0];

    auto & out     = this->output_->data();//!!! 

    assert( h );
    assert( w );
    assert( img_w );
    assert( img_h );

    int idx = 0; //index of the output 

  // For each pixel in output
  for( int fout = 0; fout < kernel_shape_[1]; ++fout ){ 
    for (int i = 0 ; i < img_h ; i++) {
      for (int j = 0 ; j < img_w; j++) {
        out[idx] = 0;

        // For each relevant pixel in input
        for (int fin = 0; fin < img_chan; ++fin ){
          //traverse the kernel 
          for (int r = -h/2 ; r <= h/2 ; r++){
            for (int c = -w/2 ; c <= w/2 ; c++){
              std::vector<int> kernel_idx{ fin, fout, r+h/2, c+w/2 };
              const int offset     = compute_offset( kernel_idx , kernel_shape_ );
              const float weight = weight_[offset];

              float val                = transform_coordinate(j + c, i + r , img_w, img_h, fin );

              //float val = transform_coordinate(j + c, i + r , img_w, img_h);
              // out[idx]  += weight_[r+ h/2 ][c + w/2 ] * val ; 
              out[idx]  += weight * val ;
              //cout<<"idx: " << idx <<" , "<<out[idx]  <<"\t";
            }
          }
        }
        out[idx] += bias_[fout]; 
        idx++;
      }
    }
  }
}
void Convolution::Backward(){

   auto& g_top  = this->output_->diff(); //gradient from the top (diff)
   auto& x        = this->input_->data();
   auto& g_in    = this->input_->diff(); //propagate down to the bottom layer

   int h              = kernel_shape_[2];
   int w             = kernel_shape_[3];
   int axis          = 1;
   int img_h       = this->input_->shape()[axis+1];
   int img_w      = this->input_->shape()[axis+2]; 
   int img_chan = this->input_->shape()[axis+0];

  /* compute gradient regarding to the kernel weight and bias weight */
   /*for each g_top(a,b):
      for each  weight_diff_(i,j)
        weight_diff_(i,j) += g_top(a,b) * x(a+i, b+j)*/    

     // For each pixel in output
    //cout<<"[ Weights ]"<<endl;
    for( int fout = 0; fout < this->output_->shape()[1]; ++fout ){ 

      for (int i = 0 ; i < img_h ; i++){
        for (int j = 0; j < img_w; j++){
            for (int fin = 0 ; fin < img_chan; fin++) { 
               int idx = transform_coordinate_index(j, i, img_w, img_h, fout );       

               //traverse the kernel
               for (int r = -h/2 ; r <= h/2; r++ ){
                   for (int c = -w/2 ; c <= w/2 ; c++ ){
                         int idx_w = transform_coordinate_index(j+c, i + r, img_w, img_h, fin);          
                        
                          if (idx_w == -1){
                            continue; 
                          }else{
                             /*int rk = r + h/2;  
                             int ck = c+ w/2; */ //it is wrong 

                             //rotate by 180 degree
                             int rk  = h-1-(r + h/2) ; 
                             int ck  = w-1-(c + w/2);
                             std::vector<int> kernel_idx { fin, fout, rk, ck };
                            const int offset = compute_offset( kernel_idx , kernel_shape_ );
                             //weight_diff_[ rk ][ ck ] += g_top[idx] * x[idx_w];
                             weight_diff_[offset] += g_top[ idx ] * x[ idx_w ] ;
                            if( std::isnan( weight_diff_[offset] ) ){

                              assert( !std::isnan( weight_diff_[offset] ) );
                            }
                          }
                   }
               }
               bias_diff_[fout] += g_top[idx] * 1; 
             }            
        }
      }
    }    

    //cout<<"[ Inputs ]"<<endl;
/*
    for each g_in(a,b):
      for each weight_(i,j):
        g_in(a,b) += g_top(a+i,b+j) * w(i,j)*/
        
  // For each pixel in output
  for (int fin = 0 ; fin < img_chan; fin++) {
    for (int i = 0; i < img_h ; i++){
       for (int j = 0; j < img_w ; j++){        
         int idx = transform_coordinate_index(j, i, img_w, img_h, fin);
         g_in[idx] = 0;
         for( int fout = 0; fout < kernel_shape_[1]; ++fout ){       
            //int idx = transform_coordinate_index(j, i, img_w, img_h);     
            for (int r = -h/2; r <= h/2 ; r++){
              for (int c = -w/2; c <= w/2; c++){
                int idx_w = transform_coordinate_index( j + c, i + r , img_w, img_h, fin ); 
                int rk       = h-1-(r + h/2) ; 
                int ck       = w-1-(c + w/2);
                std::vector<int> kernel_idx { fin, fout, rk, ck };

                const int offset = compute_offset( kernel_idx , kernel_shape_ );
                
                if( offset < 0 ){
                  for( auto x : kernel_idx )
                    std::cout << x << " ";
                  std::cout << std::endl;
                  for( auto x : kernel_shape_ )
                    std::cout << x << " ";
                  std::cout << std::endl;
                  assert( offset >= 0 );
                }
                g_in[idx ] += g_top[idx_w] * weight_[offset]; 
            }
          }
        }
      }
    }   
  }  

}

/* Update the weight of kernel in batch*/
void Convolution::Update( int N ){

    float c = 0.9;
   /* for (int i = 0 ; i < weight_.size() ; i++){
      for (int j = 0 ; j < weight_[i].size() ; j++){
      
        float update = lr_ * (weight_diff_[i][j] + c * weight_diff_pre_[i][j]);
        update /= N;
      
        weight_[i][j] = weight_[i][j] - update; //batch learning
        weight_diff_pre_[i][j] = weight_diff_[i][j];
        weight_diff_[i][j] = 0; //reset to zero
      }
    
    float update_bias = lr_ * ( bias_diff_[i] + c * bias_diff_pre_[i]);
    update_bias /= N;
    bias_[i] = bias_[i] - update_bias;
    bias_diff_pre_[i] = bias_diff_[i];
    bias_diff_[i] = 0 ; 
  }*/
      for ( int offset = 0 ; offset < weight_.size() ; offset++) {
          
             //std::vector<int> kernel_idx { fin, fout, rk, ck };
             //const int offset           = compute_offset( kernel_idx , kernel_shape_ );
             float update                 = lr_ * (weight_diff_[offset] + c * weight_diff_pre_[offset] );
             update                        /= N;
             weight_[offset]              = weight_[offset] - update; //batch learning
             weight_diff_pre_[offset] = weight_diff_[offset];
             weight_diff_[offset]        = 0; //reset to zero
             //cout<<"offset: "<<weight_[offset] <<endl;
      }

      for( int fout = 0; fout < kernel_shape_[1]; ++fout ){
             //update the bias
             float update_bias    = lr_ * ( bias_diff_[fout] + c * bias_diff_pre_[fout]);
             update_bias           /= N;
             bias_[fout]              = bias_[fout] - update_bias;
             bias_diff_pre_[fout] = bias_diff_[fout];
             bias_diff_[fout]        = 0 ; 
             //cout<<"bias_ "<<bias_[fout] << endl;
      }
}


vector<float> Convolution::parameter( int i ) {

  if ( i == 0 ){
    return bias_;
  }

  if ( i == 1 ){
    return weight_;
  }

}