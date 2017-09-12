#include <iostream>
#include <vector>
#include <layer.hpp>
#include <datum.hpp>
#include <gaussian.hpp>

using namespace std;

class Convolution : public Layer{
  
  public: 
    Convolution (std::string name) : Layer(name) {

    }
    void SetUp( Datum * in );
    void SetUp(Datum * in, vector<int>& shape, Generator* gen) ;
    inline void SetLR(float learning_rate) {
      lr_ = learning_rate;
    }
    void Forward();
    void Backward();
    void Update( int N );

    float transform_coordinate( int x, int y, int w, int h, int c) ;
    int transform_coordinate_index( int x, int y, int w, int h, int c) ;
    int compute_offset( std::vector<int> idx, std::vector<int> & shape );
    std::vector<float> parameter( int i ) ;
  //private:  

    vector<float> weight_ ; 
    vector< float> bias_ ; //per channel
    vector<int> kernel_shape_;// c_out, c_in, h, w

    std::vector<float> weight_diff_; 
    std::vector<float> bias_diff_;

    //momentum 
    std::vector<float> weight_diff_pre_; 
    std::vector<float> bias_diff_pre_;
    float lr_ = 1.0; 
};