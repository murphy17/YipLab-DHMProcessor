/*
 * DHMProcessor.cu
 *
 *  Created on: Jun 14, 2017
 *      Author: michaelmurphy
 */

#include "DHMCommon.cuh"
#include "DHMProcessor.cuh"

//namespace YipLab {

DHMProcessor::DHMProcessor(std::string outputDir) {
    this->outputDir = outputDir;

    // reset the GPU, use proper exceptions to do this...
    CUDA_CHECK( cudaDeviceReset() );

    // camera crap would go here...

    // make sure input, output directories are fine
    using namespace boost::filesystem;
    if ( !is_directory(outputDir) ) throw DHMException("Output directory not found", __LINE__, __FILE__);

    // pack parameters
    p = { N, NUM_SLICES, NUM_FRAMES, DX, DY, DZ, Z0, LAMBDA0 };

    // allocate buffers, setup FFTs

    CUDA_CHECK( cudaStreamCreate(&math_stream) );
    CUDA_CHECK( cudaStreamCreate(&copy_stream) );

    CUDA_CHECK( cufftCreate(&fft_plan) );
    CUDA_CHECK( cufftXtMakePlanMany(fft_plan, 2, fft_dims,
                                    NULL, 1, 0, fft_type,
                                    NULL, 1, 0, fft_type,
                                    1, &fft_work_size, fft_type) );
    CUDA_CHECK( cufftSetStream(fft_plan, math_stream) );

    // only one quadrant stored on host -- point is to minimize transfer time
    CUDA_CHECK( cudaMallocHost(&h_filter, NUM_SLICES*(N/2+1)*(N/2+1)*sizeof(complex)) );
    // double buffering
    CUDA_CHECK( cudaMalloc(&d_filter[0], NUM_SLICES*N*N*sizeof(complex)) );
    CUDA_CHECK( cudaMalloc(&d_filter[1], NUM_SLICES*N*N*sizeof(complex)) );

    CUDA_CHECK( cudaMalloc(&d_frame, N*N*sizeof(byte)) );

    CUDA_CHECK( cudaMalloc(&d_image, N*N*sizeof(complex)) );

    CUDA_CHECK( cudaMallocHost(&h_volume, NUM_SLICES*N*N*sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&d_volume, NUM_SLICES*N*N*sizeof(float)) );

    CUDA_CHECK( cudaMallocHost(&h_mask, NUM_SLICES*sizeof(byte)) );
    CUDA_CHECK( cudaMalloc(&d_mask, NUM_SLICES*sizeof(byte)) );

    // construct PSF, keep on host
    gen_filter_quadrant(h_filter);

    // copy to GPU in preparation for first frame
    buffer_pos = 0;
    transfer_filter_async(h_filter, d_filter[buffer_pos]);
    CUDA_CHECK( cudaStreamSynchronize(copy_stream) );
}

DHMProcessor::~DHMProcessor() {
    CUDA_CHECK( cufftDestroy(fft_plan) );

    CUDA_CHECK( cudaStreamDestroy(math_stream) );
    CUDA_CHECK( cudaStreamDestroy(copy_stream) );

    CUDA_CHECK( cudaFreeHost(h_filter) );
    CUDA_CHECK( cudaFreeHost(h_volume) );
    CUDA_CHECK( cudaFreeHost(h_mask) );

    CUDA_CHECK( cudaFree(d_filter[0]) );
    CUDA_CHECK( cudaFree(d_filter[1]) );
    CUDA_CHECK( cudaFree(d_frame) );
    CUDA_CHECK( cudaFree(d_volume) );
    CUDA_CHECK( cudaFree(d_mask) );
    CUDA_CHECK( cudaFree(d_image) );
}

/*
void DHMProcessor::process_camera() {
    // stub

    const int codec = CV_FOURCC('F','F','V','1'); // FFMPEG lossless
    const int fps = 1;

    const std::time_t now = std::time(0);
    const std::tm *ltm = std::localtime(&now);

    std::string path = outputDir + "/" +
                       std::to_string(1900 + ltm->tm_year) + "_" +
                       std::to_string(1 + ltm->tm_mon) + "_" +
                       std::to_string(ltm->tm_mday) + "_" +
                       std::to_string(1 + ltm->tm_hour) + "_" +
                       std::to_string(1 + ltm->tm_min) + "_" +
                       std::to_string(1 + ltm->tm_sec) +
                       ".mpg";

    writer.open(path, codec, fps, cv::Size_<int>(N, N), false);
    if (!writer.isOpened()) throw DHMException("Cannot open output video", __LINE__, __FILE__);
    // async
    // also needs to save images

    // Ueye stuff...


    // save the raw video feed too
    // this could happen in another thread, don't think it'll take that long though
    cv::Mat frame_mat(N, N, CV_8U, frame);
    writer.write(frame_mat);
}
*/

void DHMProcessor::process_folder(std::string input_dir) {
    using namespace boost::filesystem;

//    float *volume = new float[NUM_SLICES*N*N];
    if ( !is_directory(input_dir) ) throw DHMException("Input directory not found", __LINE__, __FILE__);

    // loop thru bitmap files in input folder
    // ... these would probably be a video ...
    std::vector<std::string> dir;
    for (auto &i : boost::make_iterator_range(directory_iterator(input_dir), {})) {
        std::string f = i.path().string();
        if (f.substr(f.find_last_of(".") + 1) == "bmp")
            dir.push_back(f);
    }
    std::sort(dir.begin(), dir.end());

    if ( dir.size() == 0 ) throw DHMException("No bitmaps found", __LINE__, __FILE__);

    for (std::string &f : dir)
    {
        cv::Mat frame_mat = cv::imread(f, CV_LOAD_IMAGE_GRAYSCALE);

        if ( frame_mat.cols != N || frame_mat.rows != N )
            throw DHMException("Images must be of size NxN", __LINE__, __FILE__);

        h_frame = frame_mat.data;

        display_image(h_frame);

        process_frame(h_frame, h_volume, false, false); // callback!!!

        display_volume(h_volume);

        // write volume to disk... what format? HDF5?
    }
}

void DHMProcessor::process_frame(byte *h_frame, float *h_volume, bool use_camera, bool unified_mem){
    // fun part goes here
    // this would be a callback

    if (!unified_mem)
    {
        // copy from camera's frame buffer to working area on device
        CUDA_CHECK( cudaMemcpy(d_frame, h_frame, N*N*sizeof(byte), cudaMemcpyHostToDevice) );
    } else {
        d_frame = h_frame;
    }

    // start transferring filter quadrants to alternating buffer
    // ... waiting for previous ops to finish first
    CUDA_CHECK( cudaStreamSynchronize(math_stream) );
    transfer_filter_async(h_filter, d_filter[!buffer_pos]);

    // convert 8-bit image to real channel of complex float
    ops::_b2c<<<N, N, 0, math_stream>>>(d_frame, d_image);
    KERNEL_CHECK();

    // FFT image in-place
    CUDA_CHECK( cufftXtExec(fft_plan, d_image, d_image, CUFFT_FORWARD) );

    // multiply image with stored quadrant of filter stack
    ops::_quad_mul<<<N/2+1, N/2+1, 0, math_stream>>>(d_filter[buffer_pos], d_image, d_mask, p);
    KERNEL_CHECK();

    // inverse FFT the product, and take complex magnitude
    for (int i = 0; i < NUM_SLICES; i++)
    {
        if (h_mask[i])
        {
            CUDA_CHECK( cufftXtExec(fft_plan, d_filter[buffer_pos] + i*N*N, d_filter[buffer_pos] + i*N*N, CUFFT_INVERSE) );

            ops::_modulus<<<N, N, 0, math_stream>>>(d_filter[buffer_pos] + i*N*N, d_volume + i*N*N);
            KERNEL_CHECK();
        }
    }

    // construct volume from one frame's worth of slices once they're ready...
    // ...
    CUDA_CHECK( cudaStreamSynchronize(math_stream) );
    // ... which will also update the host-side slice masks
    memset(h_mask, 1, NUM_SLICES);

    // transfer volume from device to host...
    // ...if you're gonna do this, it should be async...
    CUDA_CHECK( cudaMemcpy(h_volume, d_volume, NUM_SLICES*N*N*sizeof(float), cudaMemcpyDeviceToHost) ); // temp

    // sync up the host-side and device-side masks
    CUDA_CHECK( cudaMemcpyAsync(d_mask, h_mask, NUM_SLICES*sizeof(byte), cudaMemcpyHostToDevice, math_stream) );

    // flip the buffer
    buffer_pos = !buffer_pos;
}

void DHMProcessor::display_image(byte *h_image)
{
    cv::Mat mat(N, N, CV_8U, h_image);
    cv::imshow("Display window", mat); // Show our image inside it.
    cv::waitKey(0);
}

void DHMProcessor::display_volume(float *h_volume)
{
    for (int i = 0; i< NUM_SLICES; i++)
    {
        cv::Mat mat(N, N, CV_32F, h_volume + i*N*N);
        cv::normalize(mat, mat, 1.0, 0.0, cv::NORM_MINMAX, -1);
        cv::imshow("Display window", mat); // Show our image inside it.
        cv::waitKey(0);
    }
}

//}





