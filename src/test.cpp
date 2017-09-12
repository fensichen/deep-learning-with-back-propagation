#include <iostream>
#include <string>
#include <convolution.hpp>
#include <datum.hpp>
#include <gaussian.hpp>
#include <data.hpp>
#include <layer.hpp>
#include <pooling.hpp>
using namespace std;


int main( int argc, char ** argv ) {  
  
  Gaussian gau(0, 0.01);
  Data data("Data Layer");
  Convolution conv("Conv Layer") ;
  Pooling pool("Pooling Layer");

  std::string filename = "monkey/train.txt";
  data.SetUp(  filename );

  vector<int> kernel{3,2,5,5}; // filter channel, #filter, filter_size_h, filter_size_w,  
  vector<int> para{2,2,2};
  conv.SetUp(data.output(),  kernel, (Generator*)&gau );
  pool.SetUp(conv.output(),  para); 

for( int z = 0; z < 1; ++z ){
  std::cout << "Iteration " << z << std::endl;
  data.Forward();
  conv.Forward();

  cout<<"Forward done"<<endl;
  
  pool.Forward();
  // Put random data in diff
  //auto & diff = conv.output()->diff();
  auto & diff = pool.output()->diff();
  for (int i = 0; i < pool.output()->size() ; i++){
      diff[i] = gau.DrawSample();
  }

  pool.Backward();
  conv.Backward(); 
  cout<<"Backward done"<<endl;
  conv.Update(1); 
}

//draw feature map
  return 0 ; 
} 