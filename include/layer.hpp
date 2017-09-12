#ifndef _layer_hpp_
#define _layer_hpp_ 

#include <string>
#include <datum.hpp>

class Layer {
  public:
    Layer( std::string name ){
      name_ = name;
    }

    virtual void SetUp(Datum * in) = 0;
    virtual void Forward()  = 0;
    virtual void Backward()  = 0;

    Datum * output(){
      return output_;
    }

    std::string name(){
      return name_;
    }

   inline void relink( Datum * in ){
      this->input_ = in;
    }

    virtual std::vector<float> parameter( int i ) = 0;

    inline bool parameterized(){
      return parameterized_;
    }

  //protected:
    std::string name_;
    Datum * input_;
    Datum * output_;
    bool parameterized_;
};

#endif