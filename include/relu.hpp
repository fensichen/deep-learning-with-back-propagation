#include <iostream>
#include <layer.hpp>

class ReLU : public Layer{

public:
    ReLU( std::string name )  : Layer(name){
    }
    void SetUp(Datum * in); // read the output of previous layer as input for ReLU activation layer
    void Forward();              //
    void Backward();            //
   std::vector<float> parameter( int i ) ;

};