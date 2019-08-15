//
//  im2col.cpp
//  Cnn
//
//  Created by Charles  on 8/13/19.
//  Copyright © 2019 Charles . All rights reserved.
//

#include "im2col.hpp"

void im2col::convolution(vector<int> input, vector<int> filter, int input_size, int filter_size){
    
    int const outm = input_size - filter_size + 1;
    int const convAw = filter_size*filter_size;
    int const convAh = input_size*input_size;
    
    vector<int> convElements(convAw*convAh);
    vector<int> ans;
    
    for (int i = 0; i < outm; i++){
        for (int j = 0; j < outm; j++){
            
            vector<int> rw(convAw);
            int wh = i * outm * convAw + j * convAw;
            
            int col1 = i * input_size + j;
            
            convElements[wh] = input[col1];
            convElements[wh + 1] = input[col1 + 1];
            convElements[wh + 2] = input[col1 + 2];

            int col2 = (i + 1) * input_size + j;
            
            convElements[wh + 3] = input[col2];
            convElements[wh + 4] = input[col2 + 1];
            convElements[wh + 5] = input[col2 + 2];

            int col3 = (i + 2) * input_size + j;
            
            convElements[wh + 6] = input[col3];
            convElements[wh + 7] = input[col3 + 1];
            convElements[wh + 8] = input[col3 + 2];

            
            rw[0] = input[col1];
            rw[1] = input[col1 + 1];
            rw[2] = input[col1 + 2];
            
            rw[3] = input[col2];
            rw[4] = input[col2 + 1];
            rw[5] = input[col2 + 2];
            
            rw[6] = input[col3];
            rw[7] = input[col3 + 1];
            rw[8] = input[col3 + 2];
    
            int sum = 0;
            for(int k = 0; k < convAw; k++){
                sum += rw[k] * filter[k];
            }
            ans.push_back(sum);

        }
    }

    
    cout << "Output Matrix:" << endl;
    for (int i = 0; i < outm; i++){
        for (int j = 0; j < outm; j++){
            cout << ans[i*outm + j] <<" ";
        }
        cout<<endl;
    }
    cout<<endl;
    
    
    vector<int> C;
    for (int i = 0; i < outm; i++){
        for (int j = 0; j < outm; j++){
            int a = 0;
            int wh = i * outm * convAw + j * convAw;
            for (int m = 0; m < convAw; m++){
                a += convElements[wh + m] * filter[m];
            }
            C.push_back(a);
        }
    }

    cout << "Convolutional output matrix:" << endl;
    for (int i = 0; i < outm; i++){
        for (int j = 0; j < outm; j++){
            cout <<C[i*outm + j]<<" ";
        }
        cout<<endl;
    }
}
