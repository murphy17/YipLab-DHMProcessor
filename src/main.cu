/*
 * main.cpp
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#include "DHMProcessor.cuh"

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

// TODO: clean up file organization, it's convoluted

using namespace YipLab;

////////////////////////////////////////////////////////////////////////////////
// Examples of user-provided callbacks
////////////////////////////////////////////////////////////////////////////////

// TODO: being able to operate on the slices in Fourier domain would be nice...
// ... two callbacks would be easiest thing to do here, before/after FFT+mod
void cb_func(float *d_volume, byte *d_mask, DHMParameters p)
{
    // some example processing, in pure OpenCV. note this is *quite* slow...

    using namespace cv::cuda;
    using namespace cv;

    Ptr<Filter> gaussian = createGaussianFilter(CV_32F, CV_32F, Size(7, 7), 3);

    for (int k = 0; k < p.num_slices; k++)
    {
        GpuMat slice(p.N, p.N, CV_32F, d_volume + k*p.N*p.N);
        gaussian->apply(slice, slice);
    }

    // ... and set the mask
    GpuMat mask(p.num_slices, 1, CV_8U, d_mask);
    mask.setTo(Scalar_<unsigned char>(1));
}

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

void thr_cb(float *d_volume, byte *d_mask, DHMParameters p)
{
    // just to make the matrix sparse

    for (int k = 0; k < p.num_slices; k++)
    {
        cv::cuda::GpuMat slice(p.N, p.N, CV_32F, d_volume + k*p.N*p.N);
        cv::cuda::normalize(slice, slice, 1.0, 0.0, cv::NORM_MINMAX, -1);
        cv::cuda::threshold(slice, slice, 0.25, 0.0, cv::THRESH_TOZERO_INV);
    }

//    for (int k = 0; k < p.num_slices; k++)
//        CUDA_SHOW(d_volume + k*p.N*p.N, p.N, p.N);

    // ... and set the mask
    cv::cuda::GpuMat mask(p.num_slices, 1, CV_8U, d_mask);
    mask.setTo(cv::Scalar_<unsigned char>(1));
}

////////////////////////////////////////////////////////////////////////////////
// Example usage - folder
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    using namespace std;

    string input_dir = "../test";
    string output_dir = "~/image_store/Murphy_Michael/dhm";
    int num_slices = 100;
    float delta_z = 1.0f;
    float z_init = 30.0f;

    DHMProcessor dhm(num_slices, delta_z, z_init, DHM_UNIFIED_MEM);

    // TODO: allow callbacks to have state, i.e. with additional params
    // have an enum -- freq / time domain?
    // the former doesn't take the mask though...
    dhm.set_callback(DHMCallback(thr_cb)); // DHM_BEFORE_FFT, DHM_AFTER_FFT

    dhm.process_folder(input_dir, output_dir);

    // process camera one-at-a-time vs process camera "fully automatic"

//    dhm.process_camera(ueye, output_dir);

//    for (std::string &f_in : iter_folder(output_dir, "bin"))
//    {
//        std::cout << f_in << std::endl;
//        dhm.view_volume(f_in);
//    }
}

////////////////////////////////////////////////////////////////////////////////
// Example usage - camera
////////////////////////////////////////////////////////////////////////////////

// TODO
// micromanager would be great too

