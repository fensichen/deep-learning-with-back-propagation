#include <data.hpp>

void Data::SetUp(Datum * in ){
}

/*Set up: read one image, and its lable, the output_ dimension equal to the number of pixels
input_ , output_ , index_, label_ , 
*/
void Data::SetUp( std::string filename ){
    
    this->input_ = nullptr;

    /* Read image from filelist */
    std::fstream filehnd( filename ) ;
    std::string imagename;
    int label;

    while ( filehnd >> imagename >> label ){
        filelist_.push_back(make_pair( imagename, label)); 
    }

   filehnd.close();
   index_    = 0;
   label_     = new Datum( 1 );
   // Read a single image to determine dimensions
   //Mat src  = imread(filelist_[0].first,0);
   Mat src  = imread(filelist_[0].first, CV_LOAD_IMAGE_UNCHANGED);
   int dim   = src.rows*src.cols;
   int chan = src.channels();
   //this->output_ = new Datum( dim );
   std::vector< int > shape { 1, chan, src.rows, src.cols};
   this->output_ = new Datum( shape );  
   parameterized_ = false; 
}

/*Read each training data as input_,  normalize it between [0,1]

store it as output_ */
 void Data::Forward(){
    int label;
    Mat src; 
    int dim    = 0 ;
    
    /*read image and label, and normlize it */
    //src  = imread(filelist_[0].first,0);
    src          = imread( filelist_[index_].first, CV_LOAD_IMAGE_UNCHANGED) ;
    dim        = src.rows * src.cols;
    int chan  = src.channels();
    label       = filelist_[index_].second;
    auto & d = this->output_->data();



    if ( chan == 1 ){
          for (int i  = 0 ; i < dim; i++){
              int r    = i / src.cols; 
              int c   = i % src.cols;
              d[i]     = (float)src.at<uchar>(r,c) / 255.0f;
          }
    }else if (chan == 3) {
           for (int i  = 0 ; i < dim; i++){
              int r                 = i / src.cols; 
              int c                 = i % src.cols;
              d[i]                   = (float)src.at<Vec3b>(r,c)[0] / 255.0f; //B
              d[i + 1 *dim ]   = (float)src.at<Vec3b>(r,c)[1] / 255.0f; //G
              d[i + 2 *dim]    = (float)src.at<Vec3b>(r,c)[2] / 255.0f; //R
          }
    }

    auto & l  = label_->data();
    l[0]         = label;

    index_++; //increase the readed input index
    if( index_ == filelist_.size() )
        index_ = 0;
}

  void Data::Backward(){

  }

  vector<float> Data::parameter( int i ){
    vector<float> tmp;
    return tmp;
  }


 


