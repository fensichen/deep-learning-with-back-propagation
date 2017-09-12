#include <layer.hpp>
#include <saver.hpp>


void Saver::save( vector<Layer*> layer, std::string path){
 
 std::ofstream FILE(path, std::ios::out | std::ofstream::binary);

for (int i = 0 ; i < layer.size(); i++){

    if ( layer[i]->parameterized()== 1 ) {
      //save Layer name   

      
      vector<float> w = layer[i]->parameter( 0 ); 
      vector<float> b = layer[i]->parameter( 1 ); 

      //save layer name 
      std::copy(layer[i]->name().begin(), layer[i]->name().end(), std::ostreambuf_iterator<char>(FILE));
      //save layer weight 
      std::copy(w.begin(), w.end(), std::ostreambuf_iterator<char>(FILE));
      //save layer bias
      std::copy(b.begin(), b.end(), std::ostreambuf_iterator<char>(FILE));
      //save layer shape 
       //std::copy(layer->shape_.begin(), layer->shape_.end(), std::ostreambuf_iterator<char>(FILE));  
    }
}



}