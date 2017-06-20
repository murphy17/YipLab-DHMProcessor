/*
 * DHMProcessor.cu
 *
 *  Created on: Jun 14, 2017
 *      Author: michaelmurphy
 */

#include "DHMCommon.cuh"
#include "DHMProcessor.cuh"

namespace YipLab {

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

DHMProcessor::DHMProcessor(const int num_slices, const float delta_z, const float z_init,
                           const DHMMemoryKind memory_kind = DHM_STANDARD_MEM)
{
    if (is_initialized) DHM_ERROR("Only a single instance of DHMProcessor is permitted");

    this->num_slices = num_slices;
    this->delta_z = delta_z;
    this->z_init = z_init;
    this->memory_kind = memory_kind;

    // camera crap would go here...

    // allocate various buffers / handles
    setup_cuda();

    // pack parameters (for kernels)
    params.N = N;
    params.num_slices = this->num_slices;
    params.DX = DX;
    params.DY = DY;
    params.LAMBDA0 = LAMBDA0;
    params.delta_z = this->delta_z;
    params.z_init = this->z_init;

    // construct filter stack once - gets transferred to GPU in here
    build_filter_stack();

    // initially query all slices
    memset(h_mask, 1, num_slices);
    CUDA_CHECK( cudaMemcpy(d_mask, h_mask, num_slices*sizeof(byte), cudaMemcpyHostToDevice) );

    is_initialized = true;
}

DHMProcessor::~DHMProcessor() {
    cleanup_cuda();

    is_initialized = false;
}

void DHMProcessor::setup_cuda() {
    CUDA_CHECK( cudaDeviceReset() );

    // allow unified memory
    if (memory_kind == DHM_UNIFIED_MEM)
        CUDA_CHECK( cudaSetDeviceFlags(cudaDeviceMapHost) );

    // setup CUDA stream to run copying in background
    CUDA_CHECK( cudaStreamCreateWithFlags(&async_stream, cudaStreamNonBlocking) );

    // setup CUFFT
    CUDA_CHECK( cufftCreate(&fft_plan) );
    CUDA_CHECK( cufftXtMakePlanMany(fft_plan, 2, fft_dims,
                                    NULL, 1, 0, fft_type,
                                    NULL, 1, 0, fft_type,
                                    1, &fft_work_size, fft_type) );

    // only one quadrant stored on host -- point is to minimize transfer time
    CUDA_CHECK( cudaMallocHost(&h_filter, num_slices*(N/2+1)*(N/2+1)*sizeof(complex)) );
    // double buffering on device, allows simultaneous copy and processing
    CUDA_CHECK( cudaMalloc(&d_filter[0], num_slices*N*N*sizeof(complex)) );
    CUDA_CHECK( cudaMalloc(&d_filter[1], num_slices*N*N*sizeof(complex)) );
    buffer_pos = 0;
    // space for frame
    CUDA_CHECK( cudaMallocHost(&h_frame, N*N*sizeof(byte)) );
    if (memory_kind == DHM_STANDARD_MEM)
        CUDA_CHECK( cudaMalloc(&d_frame, N*N*sizeof(byte)) );
    else
        CUDA_CHECK( cudaHostGetDevicePointer(&d_frame, h_frame, 0) );
    // work space for FFT'ing image
    CUDA_CHECK( cudaMalloc(&d_image, N*N*sizeof(complex)) );
    // end result in here
    CUDA_CHECK( cudaMalloc(&d_volume, num_slices*N*N*sizeof(float)) );
    // masks to toggle processing of specific slices
    // these need to be kept separate, even with unified memory
    CUDA_CHECK( cudaMallocHost(&h_mask, num_slices*sizeof(byte)) );
    CUDA_CHECK( cudaMalloc(&d_mask, num_slices*sizeof(byte)) );

    // setup sparse save - just proof of concept, this is really slow
    CUDA_CHECK( cusparseCreate(&handle) );
    CUDA_CHECK( cusparseCreateMatDescr(&descr) );
    CUDA_CHECK( cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL) );
    CUDA_CHECK( cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO) );

    // generate parameters for filter quadrant copy
    memcpy3d_params = { 0 };
    memcpy3d_params.srcPtr.ptr = h_filter;
    memcpy3d_params.srcPtr.pitch = (N/2+1) * sizeof(complex);
    memcpy3d_params.srcPtr.xsize = (N/2+1);
    memcpy3d_params.srcPtr.ysize = (N/2+1);
    memcpy3d_params.dstPtr.ptr = nullptr; //d_filter;
    memcpy3d_params.dstPtr.pitch = N * sizeof(complex);
    memcpy3d_params.dstPtr.xsize = N;
    memcpy3d_params.dstPtr.ysize = N;
    memcpy3d_params.extent.width = (N/2+1) * sizeof(complex);
    memcpy3d_params.extent.height = (N/2+1);
    memcpy3d_params.extent.depth = num_slices;
    memcpy3d_params.kind = cudaMemcpyHostToDevice;
}

void DHMProcessor::cleanup_cuda()
{
    CUDA_CHECK( cusparseDestroyMatDescr(descr) );
    CUDA_CHECK( cusparseDestroy(handle) );

    CUDA_CHECK( cufftDestroy(fft_plan) );

    CUDA_CHECK( cudaStreamDestroy(async_stream) );

    CUDA_CHECK( cudaFreeHost(h_frame) );
    CUDA_CHECK( cudaFreeHost(h_filter) );
    CUDA_CHECK( cudaFreeHost(h_mask) );

    CUDA_CHECK( cudaFree(d_filter[0]) );
    CUDA_CHECK( cudaFree(d_filter[1]) );
    if (memory_kind == DHM_STANDARD_MEM)
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

void DHMProcessor::process_folder(std::string input_dir, std::string output_dir)
{
    input_dir = check_dir(input_dir);
    output_dir = check_dir(output_dir);

    for (std::string &f_in : iter_folder(input_dir, "bmp"))
    {
        load_image(f_in);

        // save_frame(); // only for process_camera

        CUDA_TIMER( process_frame() ); // *this* triggers the volume processing callback
        // TODO: ... and shouldn't hand over control until it's done!

        // TODO: I *think* it's safe to move the callback outside...?
        // (sync up masks at the start, not end)
        // ... don't do that if you will have multiple callbacks at diff stages

        std::string f_out = output_dir + "/" + f_in.substr(f_in.find_last_of("/") + 1) + ".bin";
        CUDA_TIMER( save_volume(f_out) );

        // write volume to disk... what format? HDF5?
    }
}

// will I expose the callback object or no?
void DHMProcessor::set_callback(DHMCallback cb)
{
    callback = cb;
}

// this would be a CALLBACK (no - but part of one, load/store ops...)
void DHMProcessor::process_frame()
{
    // start transferring filter quadrants to alternating buffer, for *next* frame
    // ... waiting for previous ops to finish first
    CUDA_CHECK( cudaDeviceSynchronize() );
    cudaMemcpy3DParms p = memcpy3d_params;
    p.dstPtr.ptr = d_filter[!buffer_pos];
    CUDA_CHECK( cudaMemcpy3DAsync(&p, async_stream) );

    // convert 8-bit image to real channel of complex float
    _b2c<<<N, N>>>(d_frame, d_image);
    KERNEL_CHECK();

    // FFT image in-place
    CUDA_CHECK( cufftXtExec(fft_plan, d_image, d_image, CUFFT_FORWARD) );

    // multiply image with stored quadrant of filter stack
    _quad_mul<<<N/2+1, N/2+1>>>(d_filter[buffer_pos], d_image, d_mask, params);
    KERNEL_CHECK();

    // inverse FFT the product, and take complex magnitude
    for (int i = 0; i < num_slices; i++)
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
    callback(d_volume, d_mask, params);
    KERNEL_CHECK();

    // sync up the host-side and device-side masks, TODO: ensure ONCE IT'S DONE!!!
    CUDA_CHECK( cudaMemcpy(h_mask, d_mask, num_slices*sizeof(byte), cudaMemcpyDeviceToHost) );

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
    float *h_volume = new float[N*N*num_slices];
    load_volume(f_in, h_volume);
    display_volume(h_volume);
    delete[] h_volume;
}

}

