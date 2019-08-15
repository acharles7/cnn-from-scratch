//
//  weights.hpp
//  Cnn
//
//  Created by Charles  on 8/13/19.
//  Copyright © 2019 Charles . All rights reserved.
//

#ifndef weights_hpp
#define weights_hpp

#include <stdio.h>
#include <vector>


extern std::vector<std::vector<std::vector<std::vector<float>>>> weights_firstConv;
extern std::vector<std::vector<std::vector<std::vector<float>>>> weights_secondConv;
extern std::vector<std::vector<std::vector<std::vector<float>>>> weights_thirdConv;
extern std::vector<std::vector<std::vector<std::vector<float>>>> weights_fourthConv;
extern std::vector<float> biases_firstConv;
extern std::vector<float> biases_secondConv;
extern std::vector<float> biases_thirdConv;
extern std::vector<float> biases_fourthConv;
extern std::vector<std::vector<float>> weights_firstDense;
extern std::vector<std::vector<float>> weights_labeller;
extern std::vector<float> biases_firstDense;
extern std::vector<float> biases_labeller;

#endif /* weights_hpp */
