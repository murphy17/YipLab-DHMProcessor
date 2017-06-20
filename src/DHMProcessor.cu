/*
 * DHMProcessor.cu
 *
 *  Created on: Jun 14, 2017
 *      Author: michaelmurphy
 */

#include "DHMCommon.cuh"
#include "DHMProcessor.cuh"

//namespace YipLab {

DHMCallback::DHMCallback() {
    _func = nullptr;
}

// how to give the callback all the parameters?
DHMCallback::DHMCallback(void (*func)(float *, byte *, DHMParameters)) {
    _func = func;
}

void DHMCallback::operator()(float *img, byte *mask, DHMParameters params) {
    if (_func == nullptr)
        DHM_ERROR("Callback not set");
    try {
        _func(img, mask, params);
    } catch (...) {
        DHM_ERROR("Callback failure");
    }
}

bool DHMProcessor::is_initialized = false;

DHMProcessor::DHMProcessor(std::string output_dir) {
    if (is_initialized) DHM_ERROR("Only a single instance of DHMProcessor is permitted");

    this->output_dir = output_dir;

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

    // setup sparse save - just proof of concept, this is really slow
    CUDA_CHECK( cusparseCreate(&handle) ); // this takes a long time, like 700ms
    CUDA_CHECK( cusparseCreateMatDescr(&descr) );
    CUDA_CHECK( cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL) );
    CUDA_CHECK( cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO) );

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

DHMProcessor::~DHMProcessor() {
    CUDA_CHECK( cusparseDestroyMatDescr(descr) );
    CUDA_CHECK( cusparseDestroy(handle) );

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

/*
void DHMProcessor::process_camera() {
    // stub

    const int codec = CV_FOURCC('F','F','V','1'); // FFMPEG lossless
    const int fps = 1;

    const std::time_t now = std::time(0);
    const std::tm *ltm = std::localtime(&now);

    std::string path = output_dir + "/" +
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

void DHMProcessor::process_folder(std::string input_dir)
{
    for (std::string &f_in : iter_folder(input_dir, "bmp"))
    {
        load_image(f_in);

        CUDA_TIMER( process_frame(false) ); // *this* triggers the volume processing callback
        // TODO: ... and shouldn't hand over control until it's done!

        // TODO: I *think* it's safe to move the callback outside...?
        // (sync up masks at the start, not end)
        // ... don't do that if you will have multiple callbacks at diff stages

        std::string f_out = output_dir + "/" + f_in.substr(f_in.find_last_of("/") + 1) + ".bin";
        CUDA_TIMER( save_volume(f_out) );

//        float *h_volume = new float[N*N*NUM_SLICES];
//        load_volume(f_out, h_volume);
//        display_volume(h_volume);
//        delete[] h_volume;

        // write volume to disk... what format? HDF5?
    }
}

// will I expose the callback object or no?
void DHMProcessor::set_callback(DHMCallback cb)
{
    callback = cb;
}

// this would be a CALLBACK (no - but part of one, load/store ops...)
void DHMProcessor::process_frame(bool use_camera)
{
    // start transferring filter quadrants to alternating buffer, for *next* frame
    // ... waiting for previous ops to finish first
    CUDA_CHECK( cudaDeviceSynchronize() );
    transfer_filter_async(h_filter, d_filter[!buffer_pos]);

    // convert 8-bit image to real channel of complex float
    _b2c<<<N, N>>>(d_frame, d_image);
    KERNEL_CHECK();

    // FFT image in-place
    CUDA_CHECK( cufftXtExec(fft_plan, d_image, d_image, CUFFT_FORWARD) );

    // multiply image with stored quadrant of filter stack
    _quad_mul<<<N/2+1, N/2+1>>>(d_filter[buffer_pos], d_image, d_mask, p);
    KERNEL_CHECK();

    // inverse FFT the product, and take complex magnitude
    for (int i = 0; i < NUM_SLICES; i++)
    {
        if (h_mask[i])
        {
            CUDA_CHECK( cufftXtExec(fft_plan, d_filter[buffer_pos] + i*N*N, d_filter[buffer_pos] + i*N*N, CUFFT_INVERSE) );

            _modulus<<<N, N>>>(d_filter[buffer_pos] + i*N*N, d_volume + i*N*N);
            KERNEL_CHECK();

            // normalize slice to (0,1)?
        }
    }

    // construct volume from one frame's worth of slices once they're ready...
    CUDA_CHECK( cudaStreamSynchronize(0) ); // allow the copy stream to continue in background

    // run the callback ...
    callback(d_volume, d_mask, p);
    KERNEL_CHECK();

    // sync up the host-side and device-side masks, TODO: ensure ONCE IT'S DONE!!!
    CUDA_CHECK( cudaMemcpy(h_mask, d_mask, NUM_SLICES*sizeof(byte), cudaMemcpyDeviceToHost) );

    // flip the buffer
    buffer_pos = !buffer_pos;
}

/*
// TODO: Micromanager
void DHMProcessor::process_camera() {
    // stub

    const int codec = CV_FOURCC('F','F','V','1'); // FFMPEG lossless
    const int fps = 1;

    const std::time_t now = std::time(0);
    const std::tm *ltm = std::localtime(&now);

    std::string path = output_dir + "/" +
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

//}

void DHMProcessor::view_volume(std::string f_in)
{
    float *h_volume = new float[N*N*NUM_SLICES];
    load_volume(f_in, h_volume);
    display_volume(h_volume);
    delete[] h_volume;
}



