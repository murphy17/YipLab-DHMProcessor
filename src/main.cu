/*
 * main.cpp
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#include "DHMProcessor.cuh"

#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>

#include "callback.cu"

// TODO: being able to operate on the slices in Fourier domain would be nice...
// ... two callbacks would be easiest thing to do here, before/after FFT+mod
void cb_func(float *d_volume, byte *d_mask, DHMParameters p)
{
    // some example processing, in pure OpenCV. note this is quite slow...

    using namespace cv::cuda;
    using namespace cv;

    Ptr<Filter> gaussian = createGaussianFilter(CV_32F, CV_32F, Size(7, 7), 3);

    for (int k = 0; k < p.NUM_SLICES; k++)
    {
        GpuMat volume(p.N, p.N, CV_32F, d_volume + k*p.N*p.N);
        gaussian->apply(volume, volume);
    }

    // ... and set the mask
    GpuMat mask(p.NUM_SLICES, 1, CV_8U, d_mask);
    mask.setTo(Scalar_<unsigned char>(1));
}

void show_cb(float *d_volume, byte *d_mask, DHMParameters p)
{
    for (int k = 0; k < p.NUM_SLICES; k++)
        CUDA_SHOW(d_volume + k*p.N*p.N, p.N, p.N);

    // ... and set the mask
    CUDA_CHECK( cudaMemset(d_mask, 1, p.NUM_SLICES) );
}

int main(int argc, char* argv[])
{
    using namespace std;

    string input_dir = argc == 1 ? "../test/input" : string(argv[1]);

    DHMProcessor dhm("../test/output");

    // have an enum -- freq / time domain?
    // the former doesn't take the mask though...
    dhm.set_callback(DHMCallback(show_cb)); // DHM_BEFORE_FFT, DHM_AFTER_FFT

    dhm.process_folder(input_dir);
}




