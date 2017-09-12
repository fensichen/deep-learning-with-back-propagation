#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>


using namespace std;

class Saver {
  public: 
    Saver(){};

    void save(vector<Layer*>, std::string path);
    std::vector<Layer *> obj;

};