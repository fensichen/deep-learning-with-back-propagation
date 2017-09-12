#include <iostream>
#include <layer.hpp>

/* Loss Layer */

class CrossEntropy : public Layer{

public:
    CrossEntropy( std::string name )  : Layer(name){
    }
    void SetUp(Datum * in);
    void Forward();
    void Backward();
    std::vector<float> parameter( int i ) ;

    void SetLabel(Datum* label);
    Datum* getSoftmax(){
      return softmax_;
    }

private: 
  Datum* softmax_;
  Datum * label_;

};