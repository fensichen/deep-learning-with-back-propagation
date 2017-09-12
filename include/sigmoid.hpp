#include <iostream>
#include <layer.hpp>

class Sigmoid : public Layer{

public:
    Sigmoid( std::string name )  : Layer(name){
    }
    void SetUp(Datum * in);
    void Forward();
    void Backward();

};