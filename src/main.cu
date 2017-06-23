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

void show_cb(float *d_volume, byte *d_mask, DHMParameters p)
{
    for (int k = 0; k < p.num_slices; k++)
        CUDA_SHOW(d_volume + k*p.N*p.N, p.N, p.N);

    // ... and set the mask
    CUDA_CHECK( cudaMemset(d_mask, 1, p.num_slices) );
}

void id_cb(float *d_volume, byte *d_mask, DHMParameters p)
{
    CUDA_CHECK( cudaMemset(d_mask, 1, p.num_slices) );
}

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

    // time to make these arguments i think

//    string input_dir = "~/image_store/Murphy_Michael/dhm_in/spheres";
//    string output_dir = "~/image_store/Murphy_Michael/dhm_out/spheres";

    string input_dir = "~/image_store/Murphy_Michael/dhm_in/ruler";
    string output_dir = "~/image_store/Murphy_Michael/dhm_out/ruler";

    int num_slices = 100;
    float delta_z = 1.0f;
    float z_init = 30.0f;
    bool save_volume = true;

    DHMProcessor dhm(num_slices, delta_z, z_init);

    // TODO: allow callbacks to have state, i.e. with additional params
    // have an enum -- freq / time domain?
    // the former doesn't take the mask though...
    dhm.set_callback(DHMCallback(id_cb)); // DHM_BEFORE_FFT, DHM_AFTER_FFT

    // inputs must be bitmaps, of size 1024x1024
    // and this iterates in NOT NATURAL ORDER! (prefix filenames with zeros...)
    dhm.process_folder(input_dir, output_dir, save_volume);

//    float fps = 1;
//    int num_frames = 5;
//    dhm.process_ueye(fps, output_dir, save_volume, num_frames);
}


