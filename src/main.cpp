#include <iostream>
#include <boost/filesystem.hpp>
#include <fstream>

#include <relu.hpp>
#include <sigmoid.hpp>
#include <tanh.hpp>
#include <data.hpp>
#include <datum.hpp>
#include <crossentropy.hpp>
#include <fcn.hpp>
#include <gaussian.hpp>
#include <accuracy.hpp>
#include <uniform.hpp>
#include <dropout.hpp>
#include <convolution.hpp>
#include <pooling.hpp>
#include <saver.hpp>

using namespace std; 

namespace fs = boost::filesystem;

/* Create object */
Data data("Data Layer");
Data test("Test Layer");
FullyConnectedLayer fc1("FC1");

ReLU relu0("ReLU0 Layer");
ReLU relu0A("ReLU0A Layer");
ReLU relu1("ReLU1 Layer");
//Sigmoid relu1("ReLU1 Layer");
FullyConnectedLayer fc2("FC2");
ReLU relu2("ReLU2 Layer");
//Sigmoid relu2("ReLU2 Layer");
FullyConnectedLayer fc3("FC3");
ReLU relu3("ReLU3 Layer");
//Sigmoid relu3("ReLU3 Layer");
FullyConnectedLayer fc4("FC4");

Accuracy acc("Accuracy Layer");
CrossEntropy loss("Loss Layer");
Dropout drop("Dropout Layer");
Convolution conv1("CONV1");
Convolution conv2("CONV2");
Pooling pool("Pooling Layer");

void Train( int, int );
void Test();

int main( int argc, char ** argv ) { 

    int N = 20;

    std::string filename = "mnist/train.txt";
    std::string testfilename = "mnist/test.txt";
    if( argc == 3 ){
      filename = std::string( argv[1] );
      testfilename = std::string( argv[2] );
    }

    Gaussian gau1(0, 0.1);
    Gaussian gau2(0, 0.01);
    Gaussian gau3(0, 0.3 );

     vector<int> kernel1{1,2,3,3}; // filter channel, #ilter (output filter), filter_size_h, filter_size_w,  
     vector<int> kernel2{2,1,3,3};
     vector<int> para{2,2,2};
  
    // Uniform gau( -0.01, 0.01 );

    // Sigmoid sig1("Sigmoid Layer");
    // TanH tanh1("TanH Layer");

    /* Set each up */
    float lr = 0.01;
    data.SetUp(  filename ); // 1 x 28 x 28
    test.SetUp( testfilename );

    conv1.SetUp( data.output(), kernel1, (Generator*)&gau3 ); //32 x 28 x 28 
    conv1.SetLR( lr );
    
    pool.SetUp( conv1.output(),   para);  // 32 x 14 x 14
    relu0.SetUp( pool.output() );

    conv2.SetUp( relu0.output(), kernel2, (Generator*)&gau2 ); 
    conv2.SetLR( lr ); 

    relu0A.SetUp( conv2.output() );

    fc1.SetUp( relu0.output(), 512, (Generator*) &gau2);
    fc1.SetLR(lr);
   
    relu1.SetUp(fc1.output()); 

    // fc2.SetUp( relu1.output(), 256, (Generator*) &gau2);
    // fc2.SetLR(lr);

    // relu2.SetUp(fc2.output()); 
    
    // fc3.SetUp( relu2.output(), 256, (Generator*) &gau2);
    // fc3.SetLR(lr);
    // drop.SetUp(fc3.output());
    // relu3.SetUp(drop.output()); 

    fc4.SetUp( relu1.output(), 10, (Generator*) &gau1);
    fc4.SetLR(lr);
    
    loss.SetUp( fc4.output());
    acc.SetUp( fc4.output());
    
    int iteration = 0;

    while( iteration < 10000 ){
      iteration++;
      Train( N, iteration%N );
      
      if( iteration % 100 == 0 ){
        cout << "Iteration: " << iteration << endl;
        Datum * error = loss.output();
        cout << "Loss: " << error->data()[0] << endl;

      }
    }

    Saver saver;
    saver.obj.push_back(&fc1);
    saver.obj.push_back(&fc4);
    saver.obj.push_back(&conv1);
    saver.obj.push_back(&conv2);
    saver.save( saver.obj, "parameter"); 

    // fc1.relink( test.output() );
    conv1.relink(test.output());

   // relu3.relink( fc3.output() ); 

  // auto w = fc1.weight_;
  // std::cout << "W: ";
  // for( int i = 0; i < w.size(); i++ )
  //   for( int j = 0; j < w[i].size(); j++ )
  //     std::cout << w[i][j] << " ";
  // std::cout << std::endl;
    


    iteration = 0;
    float avg_loss = 0;
    int test_iter = 10000;
    while( iteration < test_iter ){
      iteration++;
      Test();
      Datum * error = loss.output();
      avg_loss += error->data()[0];

      int d = 28;
      if( iteration == 1 ){
            Datum * tmp = conv1.output();
            auto vec = tmp->data();
            for( int ch = 0; ch < 2; ++ch ){
                Mat mat(d,d, 1);
                for (int r = 0; r < d; ++r)
                        {
                            uchar *pOutput = mat.ptr<uchar>(r);
                            for (int c = 0; c < d; ++c)
                            {
                                *pOutput = (uchar)(vec.at(ch*d*d+ r * d + c) * 255.0 + 128);
                                cout<<(int)*pOutput <<"\t" << vec.at(ch*d*d + r * d + c) <<"\t";
                                ++pOutput;
                            }
                            cout << endl;
                        }
                        std::stringstream ss;
                        ss << ch;

                        cv::imwrite( ss.str() +".png", mat);
                  }
       }

    }
  
    cout << "Test Loss: " << avg_loss/test_iter << endl;

    acc.print_acc();

    return 0;
}

void Train( int N, int counter ){
  
  data.Forward();
  acc.SetLabel(data.label());

  conv1.Forward();
  pool.Forward();
  relu0.Forward();

  conv2.Forward();

  relu0A.Forward();
  
  fc1.Forward();
  relu1.Forward();
  // fc2.Forward();
  // relu2.Forward();
  // fc3.Forward();
  // drop.Forward();
  // relu3.Forward();
  fc4.Forward();

  loss.SetLabel( data.label());
  loss.Forward();

  loss.Backward();
  fc4.Backward();
  // relu3.Backward();
  // drop.Backward();
  // fc3.Backward();
  // relu2.Backward();
  // fc2.Backward();
  relu1.Backward();
  fc1.Backward();
  
  relu0A.Backward();
  conv2.Backward();
  relu0.Backward();
  pool.Backward();
  conv1.Backward();
  data.Backward();

  if( counter == 0 ){
    fc1.Update( N );
    // fc2.Update( N);
    // fc3.Update( N );
    fc4.Update( N );
    conv1.Update(N);
    conv2.Update(N);
  }
}

void Test(){
  test.Forward();
  acc.SetLabel(test.label());
  conv1.Forward();
  pool.Forward();
  relu0.Forward();
 
  conv2.Forward();
  relu0A.Forward();

  fc1.Forward();
  relu1.Forward();
  // fc2.Forward();
  // relu2.Forward();
  // fc3.Forward();
  // relu3.Forward();
  fc4.Forward();

  loss.SetLabel( test.label());
  loss.Forward();
  acc.Forward();  
}