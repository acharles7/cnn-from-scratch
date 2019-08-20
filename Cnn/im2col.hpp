//
//  im2col.hpp
//  Cnn
//
//  Created by Charles  on 8/13/19.
//  Copyright © 2019 Charles . All rights reserved.
//

#ifndef im2col_hpp
#define im2col_hpp

#include <stdio.h>
#include<iostream>
#include<vector>
#include <png.h>

using namespace std;

class im2col{
public:
    
    vector<vector<float>> convolutionVectorized(vector<int> input, vector<int> filter, int input_size, int filter_size);
    vector<vector<float>> convolutionSimple(vector<int> input, vector<int> filter, int input_size, int filter_size);
    vector<vector<float>> featureMapConvReshape(vector<vector<float>> input, int filter_size);
    void convolutionExperiment();
    vector<vector<float>> read_image(string filename);

};


#endif /* im2col_hpp */
