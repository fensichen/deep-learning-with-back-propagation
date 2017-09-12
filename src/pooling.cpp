#include <pooling.hpp>

float Pooling::transform_coordinate( int x, int y, int w, int h, int chan) {
    if ( x < 0 || x >= w ){
        return 0;
    }
    if ( y < 0 || y >= h ){
      return 0 ;
    }
    
    int idx =  transform_coordinate_index(x, y,  w,  h, chan);
    return this->input_->data()[ idx ]; 
} 

int Pooling::transform_coordinate_index(int x, int y, int w, int h, int chan = 0 ){

    if ( x < 0 || x >= w ){
        return -1;
    }
    if ( y < 0 || y >= h ){
      return -1 ;
    }
    int idx = y * w + x + ( (w*h) * chan);
    return idx;
}



void Pooling::SetUp( Datum * in ) {

} 

void Pooling::SetUp(Datum * in, vector<int>& parameter ){       
      pool_para_     = parameter; 
      int stride        = pool_para_[0];
      int ph             = pool_para_[1];  // filter height
      int pw            = pool_para_[2];  //  filter width

      int axis          = 1;
      this->input_  = in;
      int D1            = this->input_->shape()[axis+0];
      int H1            = this->input_->shape()[axis+1];
      int W1           = this->input_->shape()[axis+2]; 
      int W2           = (W1 - pw)/stride + 1;
      int H2            = (H1 - ph)/stride  + 1;
      int count       = W2 * H2 * D1;
      //this->output_  = new Datum( count );
      vector<int> shape{1, D1, H2, W2};
      this->output_  = new Datum( shape ); 
      max_map_.resize(count,-1);

      parameterized_ = false; 
}

void Pooling::Forward(){
      int stride        = pool_para_[0]; // e.g., the number of channel in the image    
      int ph             = pool_para_[1];  // filter height
      int pw            = pool_para_[2];  //  filter width
      
      int axis           = 1;
      int D1             = this->input_->shape()[axis+0];
      int H1             = this->input_->shape()[axis+1];
      int W1            = this->input_->shape()[axis+2]; 

      auto & out     = this->output_->data();
      int idx            = 0; 
      //for each output channel
      for (int fout = 0 ; fout < D1; fout++){
     
        for (int i = ph/2 ; i < H1-(ph/2) ; i = i + stride) {
          for (int j = pw/2 ; j < W1-(pw/2) ; j = j + stride) {
             //go through all mask position
             float max_val = 0;
             int max_idx    = 0;
             for (int r =  0; r <= ph/2 ; r++){
                   for (int c = 0 ; c<= pw/2 ; c++){
                       float val         = transform_coordinate( j + c , i + r , W1, H1, fout);            
                       if (   val > max_val  ) {
                          max_val      = val ;
                          max_idx      = transform_coordinate_index( j+c, i + r , W1, H1, fout) ;
                       }
                       //record the index with max_val: 
                   }
             }           
             out[idx] = max_val;
             //cout<<"max_val: "<<max_val<<endl;
             max_map_[idx] = max_idx; //keep track of the index of the max activation 
             idx++;
          
          }
        }
      }
}

void Pooling::Backward(){  
   auto& g_top   = this->output_->diff(); //gradient from the top (diff)
   auto& x          = this->input_->data();
   auto& g_in     = this->input_->diff(); //propagate down to the bottom layer
   int stride        = pool_para_[0]; // e.g., the number of channel in the image    
   int ph             = pool_para_[1];  // filter height
   int pw            = pool_para_[2];  //  filter width
   int axis           = 0;
   int D1             = this->input_->shape()[axis+0];
   int H1             = this->input_->shape()[axis+1];
   int W1            = this->input_->shape()[axis+2];
   int W2            = (W1 - pw)/stride + 1;
   int H2            = (H1 - ph)/stride  + 1;

   // For each pixel in output
   int idx = 0;
   for (int i = 0 ; i < g_in.size(); i++)
        g_in[i] = 0;

   for (int fout = 0 ; fout < D1; fout++) {
    for (int i = 0; i < H2 ; i++){
       for (int j = 0; j < W2 ; j++){

         //int idx = transform_coordinate_index(j, i, img_w, img_h, fin);
          if ( max_map_[idx] != -1  ){
                  int id = max_map_[idx] ;
                  g_in[id] += g_top[idx] ;

           }
           idx++;
      }
    }   
  }  

}

void Pooling::Update( int N ){

}

 std::vector<float> Pooling::parameter( int i ) {
    vector<float> tmp;
    return tmp;
 }