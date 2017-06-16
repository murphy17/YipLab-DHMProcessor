/*
 * DHMProcessor.cu
 *
 *  Created on: Jun 14, 2017
 *      Author: michaelmurphy
 */

#include "DHMCommon.cuh"
#include "DHMProcessor.cuh"

//namespace YipLab {

// how to give the callback all the parameters?
//DHMCallback::DHMCallback(void *out, void (*func)(float *, byte *, void *), void *params) {
//    _out = out;
//    _func = func;
//    _params = params;
//}
//
//void DHMCallback::eval(float *img, byte *mask *dims) {
//    _func(img, _out, dims, _params);
//}

bool DHMProcessor::is_initialized = false;

DHMProcessor::DHMProcessor(std::string output_dir) {
    try
    {
        if (is_initialized) DHM_ERROR("Only a single instance of DHMProcessor is permitted");

        this->outputDir = output_dir;

        // reset the GPU, use proper exceptions to do this...
        CUDA_CHECK( cudaDeviceReset() );

        // camera crap would go here...

        // make sure input, output directories are fine
        using namespace boost::filesystem;
        if ( !exists(output_dir) || !is_directory(output_dir) ) DHM_ERROR("Output directory not found");

        // pack parameters
        p = { N, NUM_SLICES, NUM_FRAMES, DX, DY, DZ, Z0, LAMBDA0 };

        // allocate buffers, setup FFTs

        CUDA_CHECK( cudaStreamCreateWithFlags(&async_stream, cudaStreamNonBlocking) );

        CUDA_CHECK( cufftCreate(&fft_plan) );
        CUDA_CHECK( cufftXtMakePlanMany(fft_plan, 2, fft_dims,
                                        NULL, 1, 0, fft_type,
                                        NULL, 1, 0, fft_type,
                                        1, &fft_work_size, fft_type) );

        // only one quadrant stored on host -- point is to minimize transfer time
        CUDA_CHECK( cudaMallocHost(&h_filter, NUM_SLICES*(N/2+1)*(N/2+1)*sizeof(complex)) );
        // double buffering
        CUDA_CHECK( cudaMalloc(&d_filter[0], NUM_SLICES*N*N*sizeof(complex)) );
        CUDA_CHECK( cudaMalloc(&d_filter[1], NUM_SLICES*N*N*sizeof(complex)) );

        CUDA_CHECK( cudaMallocHost(&h_frame, N*N*sizeof(byte)) );
        CUDA_CHECK( cudaMalloc(&d_frame, N*N*sizeof(byte)) );

        CUDA_CHECK( cudaMalloc(&d_image, N*N*sizeof(complex)) );

        CUDA_CHECK( cudaMalloc(&d_volume, NUM_SLICES*N*N*sizeof(float)) );

        CUDA_CHECK( cudaMallocHost(&h_mask, NUM_SLICES*sizeof(byte)) );
        CUDA_CHECK( cudaMalloc(&d_mask, NUM_SLICES*sizeof(byte)) );

        // allow unified memory
        if (UNIFIED_MEM)
            cudaSetDeviceFlags(cudaDeviceMapHost);

        // construct filter stack, keep on host
        gen_filter_quadrant(h_filter);

        // copy to GPU in preparation for first frame
        buffer_pos = 0;
        transfer_filter_async(h_filter, d_filter[buffer_pos]);
        CUDA_CHECK( cudaStreamSynchronize(async_stream) );
        // initially query all slices
        memset(h_mask, 1, NUM_SLICES);
        CUDA_CHECK( cudaMemcpy(d_mask, h_mask, NUM_SLICES*sizeof(byte), cudaMemcpyHostToDevice) );

        is_initialized = true;
    }
    catch (DHMException &e)
    {
        std::cerr << e.what() << std::endl;
        exit(-1);
    }
}

DHMProcessor::~DHMProcessor() {
    try
    {
        CUDA_CHECK( cufftDestroy(fft_plan) );

        CUDA_CHECK( cudaStreamDestroy(async_stream) );

        CUDA_CHECK( cudaFreeHost(h_frame) );
        CUDA_CHECK( cudaFreeHost(h_filter) );
        CUDA_CHECK( cudaFreeHost(h_mask) );

        CUDA_CHECK( cudaFree(d_filter[0]) );
        CUDA_CHECK( cudaFree(d_filter[1]) );
        CUDA_CHECK( cudaFree(d_frame) );
        CUDA_CHECK( cudaFree(d_volume) );
        CUDA_CHECK( cudaFree(d_mask) );
        CUDA_CHECK( cudaFree(d_image) );

        is_initialized = false;
    }
    catch (DHMException &e)
    {
        std::cerr << e.what() << std::endl;
        exit(-1);
    }
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

    try
    {
    //    float *volume = new float[NUM_SLICES*N*N];
        if ( !exists(input_dir) || !is_directory(input_dir) ) DHM_ERROR("Input directory not found");

        // loop thru bitmap files in input folder
        // ... these would probably be a video ...
        std::vector<std::string> dir;
        for (auto &i : boost::make_iterator_range(directory_iterator(input_dir), {})) {
            std::string f = i.path().string();
            if (f.substr(f.find_last_of(".") + 1) == "bmp")
                dir.push_back(f);
        }
        std::sort(dir.begin(), dir.end());

        if ( dir.size() == 0 ) DHM_ERROR("No bitmaps found");

        for (std::string &f_in : dir)
        {
            load_image(f_in);

            process_frame(false);

            //process_volume(); // callback!!!

            std::string f_out = f_in.substr(f_in.find_last_of("/") + 1) + ".bin";
            save_volume(f_out);

            float *h_volume = new float[NUM_SLICES*N*N];
            load_volume(f_out, h_volume);
            display_volume(h_volume);
            delete[] h_volume;

            // write volume to disk... what format? HDF5?
        }
    }
    catch (DHMException &e)
    {
        std::cerr << e.what() << std::endl;
        exit(-1);
    }
}

// this would be a CALLBACK (no - but part of one, load/store ops...)
void DHMProcessor::process_frame(bool use_camera)
{
    try
    {
        // start transferring filter quadrants to alternating buffer
        // ... waiting for previous ops to finish first
        CUDA_CHECK( cudaDeviceSynchronize() );
        transfer_filter_async(h_filter, d_filter[!buffer_pos]);

        // convert 8-bit image to real channel of complex float
        ops::_b2c<<<N, N>>>(d_frame, d_image);
        KERNEL_CHECK();

        // FFT image in-place
        CUDA_CHECK( cufftXtExec(fft_plan, d_image, d_image, CUFFT_FORWARD) );

        // multiply image with stored quadrant of filter stack
        ops::_quad_mul<<<N/2+1, N/2+1>>>(d_filter[buffer_pos], d_image, d_mask, p);
        KERNEL_CHECK();

        // inverse FFT the product, and take complex magnitude
        for (int i = 0; i < NUM_SLICES; i++)
        {
            if (h_mask[i])
            {
                CUDA_CHECK( cufftXtExec(fft_plan, d_filter[buffer_pos] + i*N*N, d_filter[buffer_pos] + i*N*N, CUFFT_INVERSE) );

                ops::_modulus<<<N, N>>>(d_filter[buffer_pos] + i*N*N, d_volume + i*N*N);
                KERNEL_CHECK();

                // normalize slice to (0,1)?
            }
        }

        // construct volume from one frame's worth of slices once they're ready...
        // CUDA_CHECK( cudaDeviceSynchronize() );
        // ...
        // ... which will also update the host-side slice masks
        memset(h_mask, 1, NUM_SLICES);

        // sync up the host-side and device-side masks
        CUDA_CHECK( cudaMemcpy(d_mask, h_mask, NUM_SLICES*sizeof(byte), cudaMemcpyHostToDevice) );

        // flip the buffer
        buffer_pos = !buffer_pos;
    }
    catch (DHMException &e)
    {
        std::cerr << e.what() << std::endl;
        exit(-1);
    }
}

void DHMProcessor::display_image(byte *h_image)
{
    cv::Mat mat(N, N, CV_8U, h_image);
    cv::imshow("Display window", mat);
    cv::waitKey(0);
}

void DHMProcessor::display_volume(float *h_volume)
{
    for (int i = 0; i< NUM_SLICES; i++)
    {
        cv::Mat mat(N, N, CV_32F, h_volume + i*N*N);
        cv::normalize(mat, mat, 1.0, 0.0, cv::NORM_MINMAX, -1);
        cv::imshow("Display window", mat);
        cv::waitKey(0);
    }
}

//}





