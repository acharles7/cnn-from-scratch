//
//  im2col.cpp
//  Cnn
//
//  Created by Charles  on 8/13/19.
//  Copyright © 2019 Charles . All rights reserved.
//

#include "im2col.hpp"

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}



void im2col::im2col_cpu(const int* data_im, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w,
                int* data_col) {
    const int output_h = (height + 2 * pad_h -
                          (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
                          (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    }
                    else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            }
                            else {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}



vector<vector<float>> im2col::convolutionSimple(vector<int> input, vector<int> filter, int input_size, int filter_size){
    
    int const outm = input_size - filter_size + 1;
    int const convAw = filter_size*filter_size;
    int const convAh = input_size*input_size;

    vector<vector<float>> ans;
    
    vector<float> k1 = {1, 1, 1, 1};
    vector<float> k2 = {-1, -1, -1, -1};
    vector<float> k3 = {0, 0, 0, 0};
    
    vector<vector<float>> filters= {k1, k2, k3};
    
    for (int i = 0; i < outm; i++){
        for (int j = 0; j < outm; j++){
            
            vector<float> rw(convAw);
            int col1 = i * input_size + j;
            rw[0] = input[col1];
            rw[1] = input[col1 + 1];
//            rw[2] = input[col1 + 2];

            int col2 = (i + 1) * input_size + j;
            rw[2] = input[col2];
            rw[3] = input[col2 + 1];
//            rw[5] = input[col2 + 2];
            
//            int col3 = (i + 2) * input_size + j;
//            rw[6] = input[col3];
//            rw[7] = input[col3 + 1];
//            rw[8] = input[col3 + 2];


            ans.push_back(rw);
        }
    }
    
    cout << "Convolve Matrix" << endl;
    for (int i = 0; i < ans.size(); i++){
        for (int j = 0; j < ans[0].size(); j++){
            cout << ans[i][j] <<" ";
        }
        cout<<endl;
    }
    cout<<endl;
    
    vector<vector<float>> features(filters.size(), vector<float>(ans.size()));
//
//    cout<<"Output Matrix"<<endl;
//
//    for(int row = 0; row < features.size(); ++row) {//3
//        for(int col = 0; col < features[0].size(); ++col) {//16
//            for(int k = 0; k < filters[0].size(); k++) {//4
//                features[row][col] += filters[row][k] * ans[col][k];
//            }
//        }
//    }
//
//    cout << "Convolve Matrix" << endl;
//    for (int i = 0; i < features.size(); i++){
//        for (int j = 0; j < features[0].size(); j++){
//            cout << features[i][j] <<" ";
//        }
//        cout<<endl;
//    }
//    cout<<endl;
    
    return features;
}

vector<vector<float>> im2col::featureMapConvReshape(vector<vector<float>> input, int filter_size){
    vector<vector<float>> output;
    
    for(int i = 0; i < input.size(); i++){

        int size = (input[i].size() - 1) / filter_size + 1;
        std::vector<float> vec[size];
        
        for (int k = 0; k < size; ++k)
        {

            auto start_itr = std::next(input[i].cbegin(), k*filter_size);
            auto end_itr = std::next(input[i].cbegin(), k*filter_size + filter_size);
            
            // allocate memory for the sub-vector
            vec[k].resize(filter_size);
    
            if (k*filter_size + filter_size > input[i].size()) {
                end_itr = input[i].cend();
                vec[k].resize(input[i].size() - k*filter_size);
            }
            
            // copy elements from the input range to the sub-vector
            std::copy(start_itr, end_itr, vec[k].begin());
        }

        for (int i = 0; i < size; i++) {
            vector<float> ele;
            for (auto &j: vec[i]){
//                std::cout << j <<" ";
                ele.push_back(j);
            }
            output.push_back(ele);
//            std::cout << '\n';
        }
        
        
    }
    vector<int> res;
    for(int i = 0; i < output[0].size(); i++){
        for(int j = 0; j < output.size(); j++){
            res.push_back(output[j][i]);
        }
    }
    vector<int> filter = { 1, 1, 1, 1 };
    
    vector<vector<float>> features = convolutionSimple(res, filter, 9, 2);
    
//    for(auto a: res){
//        cout<<a<<" ";
//    }
    
    
    return output;
}






void im2col::convolutionVector(vector<int> input, vector<int> filter, int input_size, int filter_size){
    
    int const outm = input_size - filter_size + 1;
    int const convAw = filter_size*filter_size;
    int const convAh = input_size*input_size;
    
    vector<int> convElements(convAw*convAh);
//    vector<int> ans;
    
    for (int i = 0; i < outm; i++){
        for (int j = 0; j < outm; j++){
            
//            vector<int> rw(convAw);
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
            
            
//            rw[0] = input[col1];
//            rw[1] = input[col1 + 1];
//            rw[2] = input[col1 + 2];
//
//            rw[3] = input[col2];
//            rw[4] = input[col2 + 1];
//            rw[5] = input[col2 + 2];
//
//            rw[6] = input[col3];
//            rw[7] = input[col3 + 1];
//            rw[8] = input[col3 + 2];
//
//            int sum = 0;
//            for(int k = 0; k < convAw; k++){
//                sum += rw[k] * filter[k];
//            }
//            ans.push_back(sum);
            
        }
    }
    
    
//    cout << "Output Matrix:" << endl;
//    for (int i = 0; i < outm; i++){
//        for (int j = 0; j < outm; j++){
//            cout << ans[i*outm + j] <<" ";
//        }
//        cout<<endl;
//    }
//    cout<<endl;
    
    
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
    
    cout<<"Vectorized Convolution"<<endl;
    for (int i = 0; i < outm; i++){
        for (int j = 0; j < outm; j++){
            cout <<C[i*outm + j]<<" ";
        }
        cout<<endl;
    }
}


void im2col::convolutionExperiment(){
//    double simple = 0;
//    double vect = 0;
//    int cnt = 0;
//    for(int i = 2; i < 22; i++){
//        auto image = read_image("/Users/lomesh/Downloads/jpg2png/"+to_string(i)+".png");
//        vector<int> inputImage;
//
//        for(auto a: image){
//            for (auto b: a) {
//                inputImage.push_back(b);
//            }
//        }
//
//        int const input_size = image.size();
//        vector<int> filter = {
//            1, 1, 1,
//            1, 1, 1,
//            1, 1, 1
//        };
//        int const filter_size = 3;
//
//        auto ts1 = std::chrono::high_resolution_clock::now();
//        convolutionSimple(inputImage, filter, input_size, filter_size);
//        auto te1 = std::chrono::high_resolution_clock::now();
//        double elaspedTimeMs1 = std::chrono::duration<double, std::milli>(te1-ts1).count();
//        simple += elaspedTimeMs1;
////        cout<<"Normal Time: "<<elaspedTimeMs1/(double)1000<<" Sec  ||";
////        cout<<endl;
//
//        auto ts2 = std::chrono::high_resolution_clock::now();
//        convolutionVector(inputImage, filter, input_size, filter_size);
//        auto te2 = std::chrono::high_resolution_clock::now();
//        double elaspedTimeMs2 = std::chrono::duration<double, std::milli>(te2-ts2).count();
//        vect += elaspedTimeMs2;
////        cout<<"Vector Time: "<<elaspedTimeMs2/(double)1000<<" Sec  ||";
////        cout<<endl;
//        cnt += 1;
//
//    }
//    cout<<cnt<<endl;
//    cout<<"Simple: "<<simple/cnt<<endl;
//    cout<<"Vector: "<<vect/cnt<<endl;
    

}
