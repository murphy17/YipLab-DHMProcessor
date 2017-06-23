/*
 * DHMProcessor.cu
 *
 *  Created on: Jun 14, 2017
 *      Author: michaelmurphy
 */

#include "DHMCommon.cuh"
#include "DHMProcessor.cuh"

namespace YipLab {

///////////////////////////////////////////////////////////////////////////////
// Complex arithmetic
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ complex conj(const complex a)
{
    complex c;
    c.x = a.x;
    c.y = -a.y;
    return c;
}

__device__ __forceinline__ complex cmul(const complex a, const complex b)
{
    complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

template <typename T>
__device__ __forceinline__ T sym_get(const T *x, const int i, const int j, const int N)
{
    const int offset = i <= j ? i*N+j : j*N+i;
    return x[offset];
}

///////////////////////////////////////////////////////////////////////////////
// Element-wise operations
///////////////////////////////////////////////////////////////////////////////

__global__
void _b2c(const __restrict__ byte *b, complex *z)
{
    const int i = blockIdx.x;
    const int j = threadIdx.x;
    const int N = blockDim.x; // blockDim shall equal N

    z[i*N+j].x = ((float)(b[i*N+j])) / 255.f;
    z[i*N+j].y = 0.f;
}

__global__
void _freq_shift(complex *data)
{
    const int i = blockIdx.x;
    const int j = threadIdx.x;
    const int N = blockDim.x;

    const float a = (float)(1 - 2 * ((i+j) & 1));

    data[i*N+j].x *= a;
    data[i*N+j].y *= a;
}

__global__
void _modulus(const __restrict__ complex *z, float *r)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    r[offset] = hypotf(z[offset].x, z[offset].y);
}

template <typename T>
__global__ void _cudaFill(T *devPtr, T value, size_t count)
{
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < count)
    {
        devPtr[offset] = value;
    }
}
template <typename T>
inline cudaError_t cudaFill(T *devPtr, T value, size_t count)
{
    int num_threads = count < 1024 ? count : 1024;
    _cudaFill<<<(count + 1023) / 1024, num_threads>>>(devPtr, value, count);
    return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////
// Construct filter stack
///////////////////////////////////////////////////////////////////////////////

__global__ void _gen_filter_slice(complex *g, const float z, const DHMParameters p)
{
    const int i = blockIdx.x;
    const int j = threadIdx.x;

    // this is correct, 'FFT-even symmetry' - periodic extension must be symmetric about (0,0)
    float x = (i - p.N/2) * p.DX;
    float y = (j - p.N/2) * p.DY;

    float r = (-2.f / p.LAMBDA0) * norm3df(x, y, z);

    // exp(ix) = cos(x) + isin(x)
    float re, im;
    sincospif(r, &im, &re);

    // also corrects the sign flip above
    r /= -2.f * z / p.LAMBDA0;

    // re(iz) = -im(z), im(iz) = re(z)
    g[i*p.N+j].x = -im / r;
    g[i*p.N+j].y = re / r;
}

void DHMProcessor::build_filter_stack()
{
    complex *slice;
    CUDA_CHECK( cudaMalloc(&slice, N*N*sizeof(complex)) );

    for (int i = 0; i < num_slices; i++)
    {
        _gen_filter_slice<<<N, N>>>(slice, z_init + i * delta_z, params);
        KERNEL_CHECK();

        // FFT in-place
        CUDA_CHECK( cufftXtExec(fft_plan, slice, slice, CUFFT_INVERSE) );

        // frequency shift -- eliminates need to copy later
        _freq_shift<<<N, N>>>(slice);
        KERNEL_CHECK();

        // copy single quadrant to host
        CUDA_CHECK( cudaMemcpy2D(
            h_filter + (N/2+1)*(N/2+1)*i,
            (N/2+1)*sizeof(complex),
            slice,
            N*sizeof(complex),
            (N/2+1)*sizeof(complex),
            N/2+1,
            cudaMemcpyDeviceToHost
        ) );
    }

    // do a 3D copy to buffer for first N_BUF-1 frames
    // 3D allows transferring only a single quadrant
    for (int i = 0; i < N_BUF-1; i++)
    {
        cudaMemcpy3DParms p = memcpy3d_params;
        p.dstPtr.ptr = d_filter[i];
        CUDA_CHECK( cudaMemcpy3D(&p) );
    }

    CUDA_CHECK( cudaFree(slice) );
}

///////////////////////////////////////////////////////////////////////////////
// Quadrant multiply kernel
///////////////////////////////////////////////////////////////////////////////

// using fourfold symmetry of z
__global__
void _quad_mul(
//    complex *f,
    complex *z,
    const __restrict__ complex *w,
    const __restrict__ byte *mask,
    const DHMParameters p
) {
    const int i = blockIdx.x;
    const int j = threadIdx.x;
    const int ii = p.N-i;
    const int jj = p.N-j;

    if ((i>0 && i<p.N/2) && (j>0 && j<p.N/2))
    {
        complex w1 = w[i*p.N+j];
        complex w2 = w[ii*p.N+j];
        complex w3 = w[i*p.N+jj];
        complex w4 = w[ii*p.N+jj];

        for (int k = 0; k < p.num_slices; k++)
        {
            if (mask[k])
            {
                complex z_ij = z[i*p.N+j];
//                complex z_ij = sym_get(z, i, j, p.N);
                z[i*p.N+j] = cmul(w1, z_ij);
                z[ii*p.N+jj] = cmul(w4, z_ij);
                z[ii*p.N+j] = cmul(w2, z_ij);
                z[i*p.N+jj] = cmul(w3, z_ij);
            }
            z += p.N*p.N;
//            f += p.N*(p.N+1)/2;
        }
    }
    else if (i>0 && i<p.N/2)
    {
        complex w1 = w[i*p.N+j];
        complex w2 = w[ii*p.N+j];

        for (int k = 0; k < p.num_slices; k++)
        {
            if (mask[k])
            {
                complex z_ij = z[i*p.N+j];
//                complex z_ij = sym_get(z, i, j, p.N);
                z[i*p.N+j] = cmul(w1, z_ij);
                z[ii*p.N+j] = cmul(w2, z_ij);
            }
            z += p.N*p.N;
//            f += p.N*(p.N+1)/2;
        }
    }
    else if (j>0 && j<p.N/2)
    {
        complex w1 = w[i*p.N+j];
        complex w2 = w[i*p.N+jj];

        for (int k = 0; k < p.num_slices; k++)
        {
            if (mask[k])
            {
                complex z_ij = z[i*p.N+j];
//                complex z_ij = sym_get(z, i, j, p.N);
                z[i*p.N+j] = cmul(w1, z_ij);
                z[i*p.N+jj] = cmul(w2, z_ij);
            }
            z += p.N*p.N;
//            f += p.N*(p.N+1)/2;
        }
    }
    else
    {
        complex w1 = w[i*p.N+j];

        for (int k = 0; k < p.num_slices; k++)
        {
            if (mask[k])
            {
                complex z_ij = z[i*p.N+j];
//                complex z_ij = sym_get(z, i, j, p.N);
                z[i*p.N+j] = cmul(w1, z_ij);
            }
            z += p.N*p.N;
//            f += p.N*(p.N+1)/2;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// I/O ops
///////////////////////////////////////////////////////////////////////////////

// should these happen in separate threads?

// load image and push to GPU
void DHMProcessor::load_image(fs::path path)
{
    cv::Mat mat = cv::imread(path.string(), CV_LOAD_IMAGE_GRAYSCALE);
    if ( mat.cols != N || mat.rows != N )
        DHM_ERROR("Image " + path.string() + " is of size " +
                  std::to_string(mat.rows) + "x" + std::to_string(mat.cols) +
                  ", must be of size 1024x1024");

    memcpy(h_frame, mat.data, N*N*sizeof(byte));

    if (memory_kind == DHM_STANDARD_MEM)
    {
        CUDA_CHECK( cudaMemcpy(d_frame, h_frame, N*N*sizeof(byte), cudaMemcpyHostToDevice) );
    }
    // d_frame already mapped to h_frame in constructor if unified
}

void DHMProcessor::save_depth(fs::path path)
{
    TinyTIFFFile* tif = TinyTIFFWriter_open(path.string().c_str(), 8, N, N);

    if (tif)
    {
        // allocation in here because idk if 8-bit depth suffices
        CUDA_CHECK( cudaMemcpy(h_depth, d_depth, N*N*sizeof(float), cudaMemcpyDeviceToHost) );

        cv::Mat mat_32f(N, N, CV_32F, h_depth);
        cv::Mat mat_8u(N, N, CV_8U);

        // depth is normalized
        cv::normalize(mat_32f, mat_32f, 1.0, 0.0, cv::NORM_MINMAX, -1);
        mat_32f.convertTo(mat_8u, CV_8U, 255.f);

        TinyTIFFWriter_writeImage(tif, mat_8u.data);
    }
    else
    {
        DHM_ERROR("Could not write " + path.string());
    }

    TinyTIFFWriter_close(tif);
}

void DHMProcessor::save_volume(fs::path path)
{
    cv::Mat mat(N, N, CV_8U);

    cv::cuda::GpuMat d_mat_norm(N, N, CV_32F);
    cv::cuda::GpuMat d_mat_byte(N, N, CV_8U);

    TinyTIFFFile* tif = TinyTIFFWriter_open(path.string().c_str(), 8, N, N);

    if (tif)
    {
        for (int i = 0; i < num_slices; i++)
        {
            cv::cuda::GpuMat d_mat(N, N, CV_32F, d_volume + i*N*N);
            cv::cuda::normalize(d_mat, d_mat_norm, 1.0, 0.0, cv::NORM_MINMAX, -1);
            d_mat_norm.convertTo(d_mat_byte, CV_8U, 255.f);
            d_mat_byte.download(mat);

            TinyTIFFWriter_writeImage(tif, mat.data);
        }
    }
    else
    {
        DHM_ERROR("Could not write " + path.string());
    }

    TinyTIFFWriter_close(tif);
}

void DHMProcessor::display_image(byte *h_image)
{
    cv::Mat mat(N, N, CV_8U, h_image);
    cv::namedWindow("Display window", CV_WINDOW_NORMAL);
    cv::imshow("Display window", mat);
    cv::waitKey(0);
}

void DHMProcessor::display_volume(float *h_volume, bool inv)
{
    for (int i = 0; i < num_slices; i++)
    {
        cv::Mat mat(N, N, CV_32F, h_volume + i*N*N);
        cv::normalize(mat, mat, 1.0, 0.0, cv::NORM_MINMAX, -1);
        if (inv)
            cv::subtract(1.0, mat, mat);
        cv::namedWindow("Display window", CV_WINDOW_NORMAL);
        cv::imshow("Display window", mat);
        cv::waitKey(0);
    }
}

//void DHMProcessor::acquire_frame(std::string output_dir)
//{
//
//}

////////////////////////////////////////////////////////////////////////////////
// Filesystem helper functions
////////////////////////////////////////////////////////////////////////////////

// make sure the input directory exists, and resolve any symlinks
fs::path check_dir(fs::path path)
{
    using namespace boost::filesystem;

    if (path.string()[0] == '~')
    {
        path = fs::path(std::string(std::getenv("HOME")) + path.string().substr(1, path.string().size()-1));
    }

    path = canonical(path).string(); // resolve symlinks

    if ( !exists(path) || !is_directory(path) ) DHM_ERROR("Input directory not found");

    return path;
}

// returns an iterator through a folder in alphabetical order, optionally filtering by extension
std::vector<fs::path> iter_folder(fs::path path, std::string ext)
{
    path = check_dir(path);

    // loop thru bitmap files in input folder
    std::vector<fs::path> dir;
    for (auto &i : boost::make_iterator_range(fs::directory_iterator(path), {})) {
        fs::path f = i.path();
        if ( f.stem().string().size() > 0 && f.stem().string()[0] != '.' &&
             (ext.length() == 0 || ext.length() > 0 && f.extension().string() == ext) )
            dir.push_back(f);
    }
    std::sort(dir.begin(), dir.end(), [](const fs::path &a, const fs::path &b) -> bool
                                      {
                                          return strnatcmp(a.stem().c_str(), b.stem().c_str()) < 0;
                                      });

    if ( dir.size() == 0 ) DHM_ERROR("No matching files found");

    return dir;
}

////////////////////////////////////////////////////////////////////////////////
// Callback class
////////////////////////////////////////////////////////////////////////////////

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
    _func(img, mask, params);
}

////////////////////////////////////////////////////////////////////////////////
// Main DHMProcessor class
////////////////////////////////////////////////////////////////////////////////

bool DHMProcessor::is_initialized = false;

DHMProcessor::DHMProcessor(const int num_slices, const float delta_z, const float z_init,
                           const float delta_x, const float delta_y, const float lambda0)
{
    if (is_initialized) DHM_ERROR("Only a single instance of DHMProcessor is permitted");

    this->num_slices = num_slices;
    this->delta_z = delta_z;
    this->z_init = z_init;
    this->DX = delta_x;
    this->DY = delta_y;
    this->LAMBDA0 = lambda0;

    this->memory_kind = DHM_STANDARD_MEM; // unified mem doesn't give worthwhile speedup

    // allocate various buffers / handles
    setup_cuda();

    // pack parameters (for kernels)
    params.N = N;
    params.num_slices = this->num_slices;
    params.DX = this->DX;
    params.DY = this->DY;
    params.LAMBDA0 = this->LAMBDA0;
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

////////////////////////////////////////////////////////////////////////////////
// Memory setup + cleanup
////////////////////////////////////////////////////////////////////////////////

void DHMProcessor::setup_cuda() {
    CUDA_CHECK( cudaDeviceReset() );

    // allow unified memory
    if (memory_kind == DHM_UNIFIED_MEM)
        CUDA_CHECK( cudaSetDeviceFlags(cudaDeviceMapHost) );

    // setup CUDA stream to run copying in background
    for (int i = 0; i < N_BUF; i++)
        CUDA_CHECK( cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking) );

    // setup CUFFT
    CUDA_CHECK( cufftCreate(&fft_plan) );
    CUDA_CHECK( cufftXtMakePlanMany(fft_plan, 2, fft_dims,
                                    NULL, 1, 0, fft_type,
                                    NULL, 1, 0, fft_type,
                                    1, &fft_work_size, fft_type) );

    // only one quadrant stored on host -- point is to minimize transfer time
    CUDA_CHECK( cudaMallocHost(&h_filter, num_slices*(N/2+1)*(N/2+1)*sizeof(complex)) );
    // multiple buffering on device, allows simultaneous copy and processing
    for (int i = 0; i < N_BUF; i++)
        CUDA_CHECK( cudaMalloc(&d_filter[i], num_slices*N*N*sizeof(complex)) );
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
    CUDA_CHECK( cudaMalloc(&d_depth, N*N*sizeof(float)) );
    CUDA_CHECK( cudaMallocHost(&h_depth, N*N*sizeof(float)) );
    // masks to toggle processing of specific slices
    // these need to be kept separate, even with unified memory
    CUDA_CHECK( cudaMallocHost(&h_mask, num_slices*sizeof(byte)) );
    CUDA_CHECK( cudaMalloc(&d_mask, num_slices*sizeof(byte)) );

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
//    CUDA_CHECK( cusparseDestroyMatDescr(descr) );
//    CUDA_CHECK( cusparseDestroy(handle) );

    CUDA_CHECK( cufftDestroy(fft_plan) );

    for (int i = 0; i < N_BUF; i++)
        CUDA_CHECK( cudaStreamDestroy(stream[i]) );

    CUDA_CHECK( cudaFreeHost(h_frame) );
    CUDA_CHECK( cudaFreeHost(h_filter) );
    CUDA_CHECK( cudaFreeHost(h_mask) );
    CUDA_CHECK( cudaFreeHost(h_depth) );

    for (int i = 0; i < N_BUF; i++)
        CUDA_CHECK( cudaFree(d_filter[i]) );
    if (memory_kind == DHM_STANDARD_MEM)
        CUDA_CHECK( cudaFree(d_frame) );
    CUDA_CHECK( cudaFree(d_volume) );
    CUDA_CHECK( cudaFree(d_mask) );
    CUDA_CHECK( cudaFree(d_image) );
    CUDA_CHECK( cudaFree(d_depth) );
}

// wrapper for parts of workflow common to folder and camera
void DHMProcessor::process(fs::path input_path)
{
    std::string f_str = output_dir.string() + "/" + input_path.filename().stem().string();

    generate_volume();

    ////////////////////////////////////////////////////////////////////////
    // VOLUME REDUCTION / DEPTH MAP GENERATION OPS GO HERE
    ////////////////////////////////////////////////////////////////////////

    // i.e. move the callback out of generate_volume

    // save depth map

    // disabled for now, depth map not implemented
//    fs::path depth_path = fs::path(f_str + "_depth.tiff");
//    save_depth(depth_path);

    if (do_save_volume)
    {
        fs::path volume_path = fs::path(f_str + "_volume.tiff");
        save_volume(volume_path);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Public methods
////////////////////////////////////////////////////////////////////////////////

void DHMProcessor::process_folder(fs::path input_dir, fs::path output_dir, bool do_save_volume, int max_frames)
{
    if (is_running) DHM_ERROR("Can only run one operation at a time");
    is_running = true;

    this->do_save_volume = do_save_volume;

    input_dir = check_dir(input_dir);
    this->output_dir = check_dir(output_dir);

    if (max_frames == 0)
    {
        for (auto &it : boost::make_iterator_range(fs::directory_iterator(input_dir), {}))
            max_frames++;
    }

    ImageReader queue(input_dir, 16);
    queue.run();

    frame_num = 0;

    while (max_frames == -1 || frame_num < max_frames)
    {
        // TODO: eliminate some of these copies, there's a lot

        Image img;
        queue.get(&img);

        std::cout << img.str << std::endl;

        memcpy(h_frame, img.mat.data, N*N*sizeof(byte));

        if (memory_kind == DHM_STANDARD_MEM)
        {
            CUDA_CHECK( cudaMemcpy(d_frame, h_frame, N*N*sizeof(byte), cudaMemcpyHostToDevice) );
        }

        process(img.str);
        frame_num++;
    }

    queue.stop();

    is_running = false;
}

// will I expose the callback object or no?
void DHMProcessor::set_callback(DHMCallback cb)
{
    callback = cb;
}

////////////////////////////////////////////////////////////////////////////////
// The actual deconvolution
////////////////////////////////////////////////////////////////////////////////

// this would be a CALLBACK (no - but part of one, load/store ops...)
void DHMProcessor::generate_volume()
{
    // start transferring filter quadrants to alternating buffer, for *next* frame
    // ... waiting for previous ops to finish first
//    CUDA_CHECK( cudaDeviceSynchronize() );
    cudaMemcpy3DParms p = memcpy3d_params;
    p.dstPtr.ptr = d_filter[(buffer_pos + N_BUF - 1) % N_BUF];
    CUDA_CHECK( cudaMemcpy3DAsync(&p, stream[(buffer_pos + N_BUF - 1) % N_BUF]) );
    // ^^^ this transfer is the largest bottleneck on the Titan
    // TODO: only transfer upper triangle of quadrant... I couldn't get it to work

    // convert 8-bit image to real channel of complex float
    _b2c<<<N, N>>>(d_frame, d_image);
    KERNEL_CHECK();

    // FFT image in-place
    CUDA_CHECK( cufftXtExec(fft_plan, d_image, d_image, CUFFT_FORWARD) );

    // multiply image with stored quadrant of filter stack
    CUDA_CHECK( cudaStreamSynchronize(stream[buffer_pos]) ); // wait for queued copy to finish
    _quad_mul<<<N/2+1, N/2+1>>>(d_filter[buffer_pos], d_image, d_mask, params);
    KERNEL_CHECK();


    ////////////////////////////////////////////////////////////////////////////
    // CUSTOM FREQUENCY DOMAIN OPS GO HERE
    ////////////////////////////////////////////////////////////////////////////


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


    ////////////////////////////////////////////////////////////////////////////
    // CUSTOM SPATIAL DOMAIN OPS GO HERE
    ////////////////////////////////////////////////////////////////////////////


    // run the callback ...
    callback(d_volume, d_mask, params);
    KERNEL_CHECK();

    // sync up the host-side and device-side masks once callback returns
    CUDA_CHECK( cudaMemcpy(h_mask, d_mask, num_slices*sizeof(byte), cudaMemcpyDeviceToHost) );

    // advance the buffer
    buffer_pos = (buffer_pos + 1) % N_BUF;
}

}

