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

using namespace std;

class im2col{
public:
    void im2col_cpu(const int* data_im, const int channels,
               const int height, const int width, const int kernel_h, const int kernel_w,
               const int pad_h, const int pad_w,
               const int stride_h, const int stride_w,
               const int dilation_h, const int dilation_w,
                    int* data_col);
    vector<vector<float>> convolutionSimple(vector<int> input, vector<int> filter, int input_size, int filter_size);
    void convolutionVector(vector<int> input, vector<int> filter, int input_size, int filter_size);
    vector<vector<float>> featureMapConvReshape(vector<vector<float>> input, int filter_size);
    void convolutionExperiment();
    
    
    
};


#endif /* im2col_hpp */
