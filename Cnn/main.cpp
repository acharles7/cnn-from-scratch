//
//  main.cpp
//  Cnn
//
//  Created by Charles  on 8/12/19.
//  Copyright © 2019 Charles . All rights reserved.
//

#include <vector>
#include <string>
#include <map>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <png.h>


#include "im2col.hpp"
#include "weights.hpp"

//#define VERBOSE 1

using namespace std;


vector<vector<vector<float>>> convolve(const vector<vector<vector<float>>> &in,
         const vector<vector<vector<vector<float>>>> &weights,
         const vector<float> &biases){
    // Weights tensor is indexed HWCK, i.e. height - width - number of
    // input channels (e.g. image depth) - number of output channels
    // (i.e. number of learned kernels)
    
    size_t kernel_height = weights.size();
    size_t kernel_width = weights[0].size();
    size_t nkernels = weights[0][0][0].size();
    
    size_t out_height = in.size();
    if (out_height < kernel_height - 1) {
        throw runtime_error("Input too small in convolve");
    }
    
    size_t out_width = in[0].size();
    if (out_width < kernel_width - 1) {
        throw runtime_error("Input too small in convolve");
    }
    
    size_t depth = in[0][0].size();
    if (depth != weights[0][0].size()) {
        cerr << "convolve: input depth is " << depth << " but we expected "
        << weights[0][0].size() << endl;
        throw runtime_error("Depth mismatch in convolve");
    }
    
    out_height -= kernel_height - 1;
    out_width -= kernel_width - 1;
    
//#ifdef VERBOSE
    cout << "convolve: " << nkernels << " kernels of size "
    << kernel_width << "x" << kernel_height << "; output size "
    << out_width << "x" << out_height << "; input depth " << depth << endl;
//#endif
    
    auto out =
    vector<vector<vector<float>>> (out_height, vector<vector<float>> (out_width, vector<float>(nkernels,0.f)));
    
    for (size_t k = 0; k < nkernels; ++k) {
        for (size_t y = 0; y < out_height; ++y) {
            for (size_t x = 0; x < out_width; ++x) {
                for (size_t c = 0; c < depth; ++c) {
                    for (size_t ky = 0; ky < kernel_height; ++ky) {
                        for (size_t kx = 0; kx < kernel_width; ++kx) {
                            out[y][x][k] +=
                            weights[ky][kx][c][k] * in[y + ky][x + kx][c];
                        }
                    }
                }
                out[y][x][k] += biases[k];
            }
        }
    }
    
    return out;
}



vector<vector<vector<float>>> maxPool(const vector<vector<vector<float>>> &in, size_t pool_y, size_t pool_x){
    
    size_t out_height = in.size() / pool_y;
    if (out_height < 1) {
        throw runtime_error("Input too small in maxPool");
    }
    
    size_t out_width = in[0].size() / pool_x;
    if (out_width < 1) {
        throw runtime_error("Input too small in maxPool");
    }
    
    size_t depth = in[0][0].size();
    
#ifdef VERBOSE
    cerr << "maxPool: input size " << in[0].size() << "x" << in.size()
    << "; pool size " << pool_x << "x" << pool_y << "; output size "
    << out_width << "x" << out_height << " and depth " << depth << endl;
#endif
    
    auto out =
    vector<vector<vector<float>>> (out_height, vector<vector<float>> (out_width, vector<float> (depth, -INFINITY)));
    
    for (size_t y = 0; y < out_height; ++y) {
        for (size_t x = 0; x < out_width; ++x) {
            for (size_t i = 0; i < pool_y; ++i) {
                for (size_t j = 0; j < pool_x; ++j) {
                    for (size_t c = 0; c < depth; ++c) {
                        float value = in[y * pool_y + i][x * pool_x + j][c];
                        out[y][x][c] = max(out[y][x][c], value);
                    }
                }
            }
        }
    }
    
    return out;
}

vector<vector<vector<float>>> zeroPad(const vector<vector<vector<float>>> &in,
        size_t pad_y,
        size_t pad_x)
{
    size_t in_height = in.size();
    if (in_height == 0) {
        throw runtime_error("Input too small in zeroPad");
    }
    
    size_t in_width = in[0].size();
    if (in_width == 0) {
        throw runtime_error("Input too small in zeroPad");
    }
    
    size_t depth = in[0][0].size();
    
#ifdef VERBOSE
    cerr << "zeroPad: input size " << in_width << "x" << in_height
    << "; padding " << pad_x << "," << pad_y << "; output size "
    << in_width + 2 * pad_x << "x" << in_height + 2 * pad_y
    << " and depth " << depth << endl;
#endif
    
    auto out =
    vector<vector<vector<float> > >
    (in_height + 2 * pad_y,
     vector<vector<float> >
     (in_width + 2 * pad_x,
      vector<float>
      (depth,
       0.f)));
    
    for (size_t y = 0; y < in_height; ++y) {
        for (size_t x = 0; x < in_width; ++x) {
            for (size_t c = 0; c < depth; ++c) {
                out[y + pad_y][x + pad_x][c] = in[y][x][c];
            }
        }
    }
    
    return out;
}

vector<float> flatten(const vector<vector<vector<float>>> &in){
    size_t height = in.size();
    if (height < 1) {
        throw runtime_error("Input too small in flatten");
    }
    
    size_t width = in[0].size();
    if (width < 1) {
        throw runtime_error("Input too small in flatten");
    }
    
    size_t depth = in[0][0].size();
    
#ifdef VERBOSE
    cerr << "flatten: input size " << in[0].size() << "x" << in.size()
    << " and depth " << depth << ", output length "
    << width * height * depth << endl;
#endif
    
    vector<float> out(width * height * depth, 0.f);
    
    size_t i = 0;
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            for (size_t c = 0; c < depth; ++c) {
                out[i++] = in[y][x][c];
            }
        }
    }
    
    return out;
}

vector<float> dense(const vector<float> &in, const vector<vector<float> > &weights, const vector<float> &biases){
    
    size_t in_size = in.size();
    if (in_size != weights.size() || in_size == 0) {
        cerr << "dense: in_size = " << in_size
        << " but we expected " << weights.size() << endl;
        throw runtime_error("Input size mismatch in dense");
    }
    
    size_t out_size = weights[0].size();
    if (out_size != biases.size() || out_size == 0) {
        cerr << "dense: out_size = " << out_size
        << " but we expected " << biases.size() << endl;
        throw runtime_error("Output size mismatch in dense");
    }
    
#ifdef VERBOSE
    cerr << "dense: input length " << in_size << ", output length "
    << out_size << endl;
#endif
    
    vector<float> out(out_size, 0.f);
    
    for (size_t i = 0; i < in_size; ++i) {
        for (size_t j = 0; j < out_size; ++j) {
            out[j] += weights[i][j] * in[i];
        }
    }
    
    for (size_t j = 0; j < out_size; ++j) {
        out[j] += biases[j];
    }
    
    return out;
}

vector<vector<vector<float>>> activation(const vector<vector<vector<float>>> &in,string type){
    auto out(in);
    
    if (type == "relu") {
        for (size_t i = 0; i < out.size(); ++i) {
            for (size_t j = 0; j < out[i].size(); ++j) {
                for (size_t k = 0; k < out[i][j].size(); ++k) {
                    if (out[i][j][k] < 0.f) {
                        out[i][j][k] = 0.f;
                    }
                }
            }
        }
    }
    else {
        throw runtime_error("Unknown activation function '" + type + "'");
    }
    
    return out;
}

vector<float> activation(const vector<float> &in, string type){
    auto out(in);
    size_t sz = out.size();
    
    if (type == "relu") {
        for (size_t i = 0; i < sz; ++i) {
            if (out[i] < 0.f) {
                out[i] = 0.f;
            }
        }
    } else if (type == "softmax") {
        float sum = 0.f;
        for (size_t i = 0; i < sz; ++i) {
            out[i] = exp(out[i]);
            sum += out[i];
        }
        if (sum != 0.f) {
            for (size_t i = 0; i < sz; ++i) {
                out[i] /= sum;
            }
        }
    } else {
        throw runtime_error("Unknown activation function '" + type + "'");
    }
    
    return out;
}

vector<float> classify(const vector<vector<vector<float>>> &image){
    vector<vector<vector<float>>> t;
    
    t = zeroPad(image, 1, 1);
    t = convolve(t, weights_firstConv, biases_firstConv);
    t = activation(t, "relu");
    t = maxPool(t, 2, 2);
    
    t = zeroPad(t, 1, 1);
    t = convolve(t, weights_secondConv, biases_secondConv);
    t = activation(t, "relu");
    t = maxPool(t, 2, 2);
    
    t = zeroPad(t, 1, 1);
    t = convolve(t, weights_thirdConv, biases_thirdConv);
    t = activation(t, "relu");
    t = maxPool(t, 2, 2);
    
    t = zeroPad(t, 1, 1);
    t = convolve(t, weights_fourthConv, biases_fourthConv);
    t = activation(t, "relu");
    t = maxPool(t, 2, 2);
    
    vector<float> flat = flatten(t);
    
    flat = dense(flat, weights_firstDense, biases_firstDense);
    flat = activation(flat, "relu");
    
    flat = dense(flat, weights_labeller, biases_labeller);
    flat = activation(flat, "softmax");
    
    return flat;
}


// Read an image from a PNG file, using libpng, into a tensor in
// format HWC (height-width-channels)



//vector<vector<float>> convolveBW(const vector<vector<float>> &in,
//                                       const vector<vector<vector<vector<float>>>> &weights,
//                                       const vector<float> &biases){
//    // Weights tensor is indexed HWCK, i.e. height - width - number of
//    // input channels (e.g. image depth) - number of output channels
//    // (i.e. number of learned kernels)
//
//    size_t kernel_height = weights.size();
//    size_t kernel_width = weights[0].size();
//    size_t nkernels = weights[0][0][0].size();
//
//    size_t out_height = in.size();
//    if (out_height < kernel_height - 1) {
//        throw runtime_error("Input too small in convolve");
//    }
//
//    size_t out_width = in[0].size();
//    if (out_width < kernel_width - 1) {
//        throw runtime_error("Input too small in convolve");
//    }
//
////    size_t depth = in[0][0].size();
//    size_t depth = 1;
//    if (depth != weights[0][0].size()) {
//        cerr << "convolve: input depth is " << depth << " but we expected "
//        << weights[0][0].size() << endl;
//        throw runtime_error("Depth mismatch in convolve");
//    }
//
//    out_height -= kernel_height - 1;
//    out_width -= kernel_width - 1;
//
//#ifdef VERBOSE
//    cerr << "convolve: " << nkernels << " kernels of size "
//    << kernel_width << "x" << kernel_height << "; output size "
//    << out_width << "x" << out_height << "; input depth " << depth << endl;
//#endif
//
//    auto out = vector<vector<vector<float>>> (out_height, vector<vector<float>> (o ut_width, vector<float>(nkernels,0.f)));
//
//    for (size_t k = 0; k < nkernels; ++k) {
//        for (size_t y = 0; y < out_height; ++y) {
//            for (size_t x = 0; x < out_width; ++x) {
//                for (size_t c = 0; c < depth; ++c) {
//                    for (size_t ky = 0; ky < kernel_height; ++ky) {
//                        for (size_t kx = 0; kx < kernel_width; ++kx) {
//                            out[y][x][k] +=
//                            weights[ky][kx][c][k] * in[y + ky][x + kx][c];
//                        }
//                    }
//                }
//                out[y][x][k] += biases[k];
//            }
//        }
//    }
//
//    return out;
//}




int main(int argc, char **argv){
    
    im2col im;
    
//    auto image = im.read_image("/Users/lomesh/Documents/Xcode/Cnn/bco.png");

//    vector<int> inputImage;
//
//    for(auto a: image){
//        for (auto b: a) {
//            inputImage.push_back(b);
//        }
//    }
    int const input_size = 5;
    vector<int> input = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    
    vector<int> filter = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    int const filter_size = 3;

    im.convolutionExperiment();
    
//    vector<vector<float>> features = im.convolutionVectorized(inputImage, filter, input_size, filter_size);
//    im.convolutionVector(inputImage, filter, input_size, filter_size);
    

    
//    vector<float> f1 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//    vector<float> f2 = {2, 3, 4, 5, 6, 7, 8, 9, 1};
//    vector<float> f3 = {3, 4, 5, 6, 7, 8, 9, 1, 2};
//    vector<vector<float>> features = {f1, f2, f3};
    
//    vector<vector<float>> featureReshape = im.featureMapConvReshape(features, 3);

//    for(int i = 0; i <  )

//    auto ts1 = std::chrono::high_resolution_clock::now();
//    im.convolutionSimple(inputImage, filter, input_size, filter_size);
//    auto te1 = std::chrono::high_resolution_clock::now();
//    double elaspedTimeMs1 = std::chrono::duration<double, std::milli>(te1-ts1).count();
//    cout<<"Normal Time: "<<elaspedTimeMs1/(double)1000<<" Sec  ||";
//    cout<<endl;
//
//    auto ts2 = std::chrono::high_resolution_clock::now();
//    im.convolutionVector(inputImage, filter, input_size, filter_size);
//    auto te2 = std::chrono::high_resolution_clock::now();
//    double elaspedTimeMs2 = std::chrono::duration<double, std::milli>(te2-ts2).count();
//    cout<<"Vector Time: "<<elaspedTimeMs2/(double)1000<<" Sec  ||";
//    cout<<endl;

//    for(int i = 30; i < 31; i++){
//        cout<<image[i].size()<<endl;
//        for(int j = 0; j < image[0].size(); j++){
//            for(int k = 0; k < image[0][0].size(); k++){
//                cout<<image[i][j].size()<<" ";
//            }
//            cout<<endl;
//        }
//        cout<<endl;
//    }
//    size_t expected_width = 128;
//    size_t expected_height = 128;
//
//    if (image.size() != expected_height ||
//        image[0].size() != expected_width) {
//        cerr << "Image has wrong size: must be exactly " << expected_width
//        << "x" << expected_height << " pixels" << endl;
//        throw runtime_error("Image has wrong size");
//    }
//
//    vector<vector<vector<float>>> out = convolve(image, weights_firstConv, biases_firstConv);
//    vector<vector<float>> out = convolveBW(image, weights_firstConv, biases_firstConv);
    

//    for(int i = 30; i < 31; i++){
//        cout<<out[i].size()<<endl;
//        for(int j = 0; j < out[0].size(); j++){
//            for(int k = 0; k < out[0][0].size(); k++){
////                cout<<out[i][j][k]<<" ";
//                cout<<out[i][j].size()<<endl;
//            }
//            cout<<endl;
//        }
//        cout<<endl;
//    }
    
    
//    auto before = chrono::high_resolution_clock::now();
//
//    auto result = classify(image);
//
//    auto after = chrono::high_resolution_clock::now();
//
//    chrono::duration<double> diff = after - before;
//
//    cerr << "Classification took " << diff.count() << " sec" << endl;
//
//    vector<string> labels {
//        "daisy", "dandelion", "roses", "sunflowers", "tulips"
//    };
//
//    map<float, string> ranking;
//
//    for (size_t i = 0; i < result.size(); ++i) {
//        if (i >= labels.size()) {
//            throw logic_error("Too many result categories!");
//        }
//        ranking[- result[i] * 100.0] = labels[i];
//    }
//
//    for (auto r = ranking.begin(); r != ranking.end(); ++r) {
//        cout << r->second << ": " << -r->first << "%" << endl;
//    }
//
    return 0;
}
