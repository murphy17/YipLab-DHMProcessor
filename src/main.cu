/*
 * main.cpp
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#include "DHMProcessor.cuh"

//#include <opencv2/opencv.hpp>
//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudafilters.hpp>

using namespace YipLab;

// TODO: add depth-map handle to callback

////////////////////////////////////////////////////////////////////////////////
// Examples of user-provided callbacks
////////////////////////////////////////////////////////////////////////////////

//void show_cb(float *d_volume, byte *d_mask, DHMParameters p)
//{
//    for (int k = 0; k < p.num_slices; k++)
//        CUDA_SHOW(d_volume + k*p.N*p.N, p.N, p.N);
//
//    // ... and set the mask
//    CUDA_CHECK( cudaMemset(d_mask, 1, p.num_slices) );
//}
//
//void id_cb(float *d_volume, byte *d_mask, DHMParameters p)
//{
//    CUDA_CHECK( cudaMemset(d_mask, 1, p.num_slices) );
//}

//void thr_cb(float *d_volume, byte *d_mask, DHMParameters p)
//{
//    // just to make the matrix sparse
//
//    for (int k = 0; k < p.num_slices; k++)
//    {
//        cv::cuda::GpuMat slice(p.N, p.N, CV_32F, d_volume + k*p.N*p.N);
//        cv::cuda::normalize(slice, slice, 0.0, 1.0, cv::NORM_MINMAX, -1);
//        cv::cuda::multiply(slice, slice, slice); // boost contrast
//        cv::cuda::threshold(slice, slice, 0.05, 0.0, cv::THRESH_TOZERO_INV); // can use Otsu too
//    }
//
//    // ... and set the mask
//    cv::cuda::GpuMat mask(p.num_slices, 1, CV_8U, d_mask);
//    mask.setTo(cv::Scalar_<unsigned char>(1));
//}

////////////////////////////////////////////////////////////////////////////////
// Example usage - folder
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    using namespace std;

    string input_dir, output_dir;
    bool save_volume;
    int num_slices, max_frames;
    float delta_z, z_init;

    // max_frames toggle not working, out queue not exiting

    if (argc < 7)
    {
        input_dir = "/mnt/image_store/Murphy_Michael/dhm_in/spheres";
        output_dir = "/mnt/image_store/Murphy_Michael/dhm_out/spheres";
        z_init = 30.0f;
        delta_z = 1.0f;
        num_slices = 100;
        max_frames = 20;
        save_volume = false;
    }
    else
    {
        input_dir = string(argv[1]);
        output_dir = string(argv[2]);
        z_init = stof(string(argv[3]));
        delta_z = stof(string(argv[4]));
        num_slices = stoi(string(argv[5]));
        max_frames = stoi(string(argv[6]));
        save_volume = stoi(string(argv[7]));
    }

    float delta_x = 0.0051992f; // (5.32f / 1024.f);
    float delta_y = 0.0051992f; // (6.66f / 1280.f);
    float lambda0 = 0.000488f;

    DHMProcessor dhm(num_slices, delta_z, z_init, delta_x, delta_y, lambda0);

    // inputs must be bitmaps, of size 1024x1024
    dhm.process_folder(input_dir, output_dir, save_volume, max_frames);
}


