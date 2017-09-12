#include <iostream>
#include <vector>
#include <layer.hpp>
#include <datum.hpp>
#include <gaussian.hpp>

using namespace std;


class Pooling : public Layer{
  
  public: 
    Pooling (std::string name) : Layer(name) {

    }
    void SetUp( Datum * in );
    void SetUp(Datum * in, vector<int>& parameter );
    void Forward();
    void Backward();
    void Update( int N );
    std::vector<float> parameter( int i ) ;

    float transform_coordinate( int x, int y, int w, int h, int c) ;
    int transform_coordinate_index( int x, int y, int w, int h, int c) ;

  private: 
   //spatial extent
    vector<int> pool_para_;// stride, POOLX, POOLY

    
    vector<int> max_map_;  //keep track of the index of the max activation 
};