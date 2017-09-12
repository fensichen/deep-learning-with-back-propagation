#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <layer.hpp>

using namespace cv;
using namespace std;

class Data : public Layer{

public:
    Data( std::string name )  : Layer(name){
    }

    void SetUp( Datum * in) ;
    void SetUp( std::string fn ); // read image and ground truth labels from filelist ( fn )
    void Forward();
    void Backward();
    std::vector<float> parameter( int i ) ;

    //read label from file 
    Datum* label(){
      return label_;
    }
    void SetLabel( Datum * label ){
      label_ = label;
    }

private:
   Datum * label_;
   std::vector< std::pair< std::string, int > > filelist_;
   int index_ ; //to count how many training data already read
};