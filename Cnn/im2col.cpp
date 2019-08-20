//
//  im2col.cpp
//  Cnn
//
//  Created by Charles  on 8/13/19.
//  Copyright © 2019 Charles . All rights reserved.
//

#include "im2col.hpp"

vector<vector<float>> im2col::read_image(string filename){
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;
    
    if (!png_image_begin_read_from_file(&image, filename.c_str())) {
        throw runtime_error(string("Failed to open image file: ") + image.message);
    }
    
    //    image.format = PNG_FORMAT_RGB;
    image.format = PNG_FORMAT_GRAY;
    auto size = image.width * image.height * 1;
    vector<png_byte> buffer(size);
    
    if (!png_image_finish_read(&image, nullptr, buffer.data(), 0, nullptr)) {
        throw runtime_error
        (string("Failed to read image file: ") + image.message);
    }
    
    //    vector<vector<vector<float>>> image_tensor_color;
    vector<vector<float>> image_tensor_bw;
    auto ptr = buffer.begin();
    
    //    for (png_uint_32 i = 0; i < image.height; ++i) {
    //        image_tensor_color.push_back(vector<vector<float>>());
    //        for (png_uint_32 j = 0; j < image.width; ++j) {
    //            image_tensor_color[i].push_back(vector<float>());
    //            for (png_uint_32 k = 0; k < 3; ++k) {
    //                image_tensor_color[i][j].push_back(*ptr++);
    //            }
    //        }
    //    }
    for (png_uint_32 i = 0; i < image.height; ++i) {
        image_tensor_bw.push_back(vector<float>());
        for (png_uint_32 j = 0; j < image.width; ++j) {
            image_tensor_bw[i].push_back(*ptr++);
        }
    }
    
    return image_tensor_bw;
}



vector<vector<float>> im2col::convolutionVectorized(vector<int> input, vector<int> filter, int input_size, int filter_size){
    
    int const outm = input_size - filter_size + 1;
    int const convAw = filter_size*filter_size;


    vector<vector<float>> ans(outm*outm);
    
    vector<float> k1 = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    vector<float> k2 = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    vector<float> k3 = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    vector<vector<float>> filters= {k1, k2, k3};
    
    int cnt = 0;
    for (int i = 0; i < outm; i++){
        for (int j = 0; j < outm; j++){
            
            vector<float> rw(convAw);
            int col1 = i * input_size + j;
            rw[0] = input[col1];
            rw[1] = input[col1 + 1];
            rw[2] = input[col1 + 2];

            int col2 = (i + 1) * input_size + j;
            rw[3] = input[col2];
            rw[4] = input[col2 + 1];
            rw[5] = input[col2 + 2];
            
            int col3 = (i + 2) * input_size + j;
            rw[6] = input[col3];
            rw[7] = input[col3 + 1];
            rw[8] = input[col3 + 2];

            ans[cnt] = rw;
            cnt += 1;
        }
    }

    
    vector<vector<float>> features(filters.size(), vector<float>(ans.size()));
    
    for(int row = 0; row < features.size(); ++row) {
        for(int col = 0; col < features[0].size(); ++col) {
            for(int k = 0; k < filters[0].size(); k++) {
                features[row][col] += filters[row][k] * ans[col][k];
            }
        }
    }
    
    
//    int cnt2 = 0;
//    for (int i = 0; i < ans.size(); i++){
//        double sum = 0;
//        for (int j = 0; j < ans[0].size(); j++){
//            sum += filter[j] * ans[i][j];
//        }
//        res[cnt2] = sum;
//        cnt2 += 1;
//    }
    
//    cout<<"Vectorized"<<endl;
//    for(int i = 0; i < 2000; i++){
//        cout<<features[0][i]<<" ";
//    }
//    cout<<endl;
    
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



vector<vector<float>> im2col::convolutionSimple(vector<int> input, vector<int> filter, int input_size, int filter_size){
    
    int const outm = input_size - filter_size + 1;
    int const convAw = filter_size*filter_size;
    
    
    vector<int> ans(outm*outm);
    vector<float> k1 = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    vector<float> k2 = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    vector<float> k3 = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    vector<vector<float>> filters= {k1, k2, k3};
    vector<vector<float>> features(filters.size(), vector<float>(ans.size()));
    
    for(int row = 0; row < features.size(); ++row) {//3
        for (int i = 0; i < outm; i++){
            for (int j = 0; j < outm; j++){
                
                vector<int> rw(convAw);
                
                int col1 = i * input_size + j;
                rw[0] = input[col1];
                rw[1] = input[col1 + 1];
                rw[2] = input[col1 + 2];
                
                int col2 = (i + 1) * input_size + j;
                rw[3] = input[col2];
                rw[4] = input[col2 + 1];
                rw[5] = input[col2 + 2];
                
                int col3 = (i + 2) * input_size + j;
                rw[6] = input[col3];
                rw[7] = input[col3 + 1];
                rw[8] = input[col3 + 2];
    
                for(int k = 0; k < filters[0].size(); k++) {//4
                    features[row][i*outm + j] += filters[row][k] * rw[k];
                }
            }
        }
    }
    
//    cout<<"Simple"<<endl;
//    for(int i = 0; i < 2000; i++){
//        cout<<features[0][i]<<" ";
//    }
//    cout<<endl;
    

//    cout << "Output Matrix:" << endl;
//    for (int i = 0; i < outm; i++){
//        for (int j = 0; j < outm; j++){
//            cout << ans[i*outm + j] <<" ";
//        }
//        cout<<endl;
//    }
//    cout<<endl;
    
    
//    vector<int> C;
//    for (int i = 0; i < outm; i++){
//        for (int j = 0; j < outm; j++){
//            int a = 0;
//            int wh = i * outm * convAw + j * convAw;
//            for (int m = 0; m < convAw; m++){
//                a += convElements[wh + m] * filter[m];
//            }
//            C.push_back(a);
//        }
//    }
//    
//    cout<<"Vectorized Convolution"<<endl;
//    for (int i = 0; i < outm; i++){
//        for (int j = 0; j < outm; j++){
//            cout <<C[i*outm + j]<<" ";
//        }
//        cout<<endl;
//    }
    
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
    
//    vector<vector<float>> features = convolutionSimple(res, filter, 9, 2);
    
//    for(auto a: res){
//        cout<<a<<" ";
//    }
    
    return output;
}


void im2col::convolutionExperiment(){
    double simple = 0;
    double vect = 0;
    int cnt = 0;
    
    for(int i = 2; i < 22; i++){
        auto image = read_image("/Users/lomesh/Downloads/jpg2png/"+to_string(i)+".png");
        vector<int> inputImage;

        for(auto a: image){
            for (auto b: a) {
                inputImage.push_back(b);
            }
        }

        int const input_size = image.size();
        vector<int> filter = {
            1, 1, 1,
            1, 1, 1,
            1, 1, 1
        };
        int const filter_size = 3;

        auto ts1 = std::chrono::high_resolution_clock::now();
        auto a = convolutionVectorized(inputImage, filter, input_size, filter_size);
        auto te1 = std::chrono::high_resolution_clock::now();
        double elaspedTimeMs1 = std::chrono::duration<double, std::milli>(te1-ts1).count();
        vect += elaspedTimeMs1;
        cout<<"Vectorized Time: "<<elaspedTimeMs1/(double)1000<<" Sec  ||";


        auto ts2 = std::chrono::high_resolution_clock::now();
        auto b = convolutionSimple(inputImage, filter, input_size, filter_size);
        auto te2 = std::chrono::high_resolution_clock::now();
        double elaspedTimeMs2 = std::chrono::duration<double, std::milli>(te2-ts2).count();
        simple += elaspedTimeMs2;
        cout<<"Simple Time: "<<elaspedTimeMs2/(double)1000<<" Sec  ||";
        cout<<endl;
        cnt += 1;

    }
}
