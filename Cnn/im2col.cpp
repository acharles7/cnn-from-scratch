//
//  im2col.cpp
//  Cnn
//
//  Created by Charles  on 8/13/19.
//  Copyright © 2019 Charles . All rights reserved.
//

#include "im2col.hpp"

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
    
    vector<vector<float>> features(filters.size(), std::vector<float>(ans[0].size()));
    
    cout<<"Output Matrix"<<endl;
    
    for(int row = 0; row < features.size(); ++row) {
        for(int col = 0; col < features.at(0).size(); ++col) {
            for(int k = 0; k < ans.size(); k++) {
                features[row][col] += filters[row][k] * ans[k][col];
            }
        }
    }
    
    cout << "Convolve Matrix" << endl;
    for (int i = 0; i < features.size(); i++){
        for (int j = 0; j < features[0].size(); j++){
            cout << features[i][j] <<" ";
        }
        cout<<endl;
    }
    cout<<endl;
    
    return features;
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
