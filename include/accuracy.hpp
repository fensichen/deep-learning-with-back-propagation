#ifndef ACCURACY_HPP
#define ACCURACY_HPP

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>

#include <layer.hpp>

using namespace std;

class Accuracy : public Layer {
  public: 
    Accuracy (string name): Layer(name) {}
    void SetUp(Datum * in) ;
    void Forward() ;
    void Backward() ;
    vector<float> parameter( int i ) ;

    void print_acc();
    inline void SetLabel(Datum* label){
      label_  = label;
    }

 private:  

    int counter_ = 0; 
    int total_input_num_ = 0;
    Datum* label_ ; 

};

#endif