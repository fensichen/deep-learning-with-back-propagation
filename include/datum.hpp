#ifndef _datum_hpp_
#define _datum_hpp_ 

#include <vector>

class Datum {
  public:
    std::vector<float> & data() {
      return data_;
    }

    std::vector<float> & diff() {
      return diff_;
    }
    
    int size() {
      return data_.size();
    }

    //constructor
    Datum(int num){
      data_.resize(num,0);
      diff_.resize(num,0);
      shape_.resize( 1, num );
    }

    Datum( std::vector<int> shape ){
      shape_ = shape;
      int count = 1;
      for( int c : shape_ ){
        count *= c;
       }
      
      data_.resize( count, 0 );
      diff_.resize( count, 0 );
    }

    std::vector<int> shape() {
      return shape_; 
    }
 
  
  private:
    std::vector<float> data_;
    std::vector<float> diff_;
    std::vector<int> shape_ ; //e.g., channel, img_width, img_height
};

#endif