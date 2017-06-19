/*
 * main.cpp
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#include "DHMProcessor.cuh"

#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>

// TODO: move try-catch entirely inside DHMProcessor
// exceptions not working at all, error message is gibberish

void cb_func(float *d_volume, byte *d_mask, DHMParameters p)
{
    // some example processing, in pure OpenCV
    // ... and it is SLOWWWWW

//    using namespace cv::cuda;
//    using namespace cv;

    cv::Ptr<cv::cuda::Filter> gaussian = cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(7, 7), 3);

    for (int k = 0; k < p.NUM_SLICES; k++)
    {
        cv::cuda::GpuMat volume(p.N, p.N, CV_32F, d_volume + k*p.N*p.N);
        CUDA_TIMER( gaussian->apply(volume, volume) );
    }

    // ... and set the mask
    cv::cuda::GpuMat mask(p.NUM_SLICES, 1, CV_8U, d_mask);
    mask.setTo(cv::Scalar_<unsigned char>(1));
}

int main(int argc, char* argv[])
{
    std::string input_dir = argc == 1 ? "../test/input" : std::string(argv[1]);

    // TODO: move try-catch *INSIDE* DHM methods!
    try
    {
        DHMProcessor dhm("../test/output");

        DHMCallback callback(cb_func);
        dhm.set_callback(callback);

        dhm.process_folder(input_dir);
    }
    catch (DHMException &e)
    {
        std::cerr << e.what() << std::endl;
    }
}




