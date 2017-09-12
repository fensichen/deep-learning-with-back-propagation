#include <relu.hpp>
#include <vector>


void ReLU::SetUp(Datum * in){
    //read the input from the previous layer
    this->input_ = in;
    //this->output_ = new Datum(in->size()); // output_ has the same size as input
    output_ = new Datum( in->shape() );

    parameterized_ = false; 
}

/* 
    ReLU(x) = x ( x >= 0)
                    0 ( x <   0)
*/
 void ReLU::Forward(){

    auto& v = this->input_->data(); //member of Layer
    auto& u = this->output_->data();
    for (int i = 0 ; i < v.size() ; i++) {
        if (v[i] >= 0){
            u[i] = v[i];
        }else{
            u[i] = 0;
        }
    }
 }

/*

*/
void ReLU::Backward(){

    auto& d = this->output_->diff(); // gradient from the top
    auto& u = this->output_->data();
    auto& t  = this->input_->diff(); //write from d -> t

    for (int i = 0 ; i < d.size() ; i++){
        if ( u[i] >= 0) {
             t[i] = d[i] * 1;
        }else{
             t[i] = d[i] * 0; 
        }
    }
}

 std::vector<float> ReLU::parameter( int i ) {

        std::vector<float> tmp;
        return tmp;

 }

