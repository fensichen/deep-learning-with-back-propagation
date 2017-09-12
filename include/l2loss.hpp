#include <iostream>
#include <layer.hpp>

/* Loss Layer */

class L2loss : public Layer{

public:
    L2loss( std::string name )  : Layer(name){
    }
    void SetUp(Datum * in);
    void Forward();
    void Backward();

    void SetLabel(Datum* label);
    Datum* getSoftmax(){
      return softmax_;
    }

private: 
  Datum* softmax_;
  Datum * label_;

};