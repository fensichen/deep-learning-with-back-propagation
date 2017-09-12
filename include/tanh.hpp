#include <iostream>
#include <layer.hpp>

class TanH : public Layer{

public:
    TanH( std::string name )  : Layer(name){
    }
    void SetUp(Datum * in);
    void Forward();
    void Backward();

};