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
    void convolution(vector<int> input, vector<int> filter, int input_size, int filter_size);
    
    
};


#endif /* im2col_hpp */
