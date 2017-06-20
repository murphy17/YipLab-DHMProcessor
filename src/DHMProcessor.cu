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

    // do a 3D copy to buffer for first frame
    // 3D allows transferring only a single quadrant
    cudaMemcpy3DParms p = memcpy3d_params;
    p.dstPtr.ptr = d_filter[0];
    CUDA_CHECK( cudaMemcpy3D(&p) );

    CUDA_CHECK( cudaFree(slice) );
}

///////////////////////////////////////////////////////////////////////////////
// Quadrant multiply kernel
///////////////////////////////////////////////////////////////////////////////

// using fourfold symmetry of z
__global__
void _quad_mul(
    complex *z,
    const __restrict__ complex *w,
    const byte *mask,
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
                z[i*p.N+j] = cmul(w1, z_ij);
                z[ii*p.N+jj] = cmul(w4, z_ij);
                z[ii*p.N+j] = cmul(w2, z_ij);
                z[i*p.N+jj] = cmul(w3, z_ij);
            }
            z += p.N*p.N;
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
                z[i*p.N+j] = cmul(w1, z_ij);
                z[ii*p.N+j] = cmul(w2, z_ij);
            }
            z += p.N*p.N;
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
                z[i*p.N+j] = cmul(w1, z_ij);
                z[i*p.N+jj] = cmul(w2, z_ij);
            }
            z += p.N*p.N;
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
                z[i*p.N+j] = cmul(w1, z_ij);
            }
            z += p.N*p.N;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Convert 3D volume to sparse
///////////////////////////////////////////////////////////////////////////////

// this interface is awful, dumb hack
// and this method is WAY too slow
// some notes: dX, dY fit into chars, can cast V to char; use special packet for new slice
void volume2sparse(const cusparseHandle_t handle, const cusparseMatDescr_t descr,
                   const float *vol, int **x, int m, int **y, int n, int **z, int p, float **v, int *totalNnz,
                   const bool unified_mem = false, int **h_x = nullptr, int **h_y = nullptr, int **h_z = nullptr, float **h_v = nullptr)
{
    int nnz;
    int *dNnzPerRow, *dCsrRowPtrA;
    CUDA_CHECK( cudaMalloc(&dNnzPerRow, m*sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&dCsrRowPtrA, (m+1)*sizeof(int)) );

    CUDA_CHECK( cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, m, n*p, descr, vol, m, dNnzPerRow, totalNnz) );

    if (unified_mem)
    {
        CUDA_CHECK( cudaMallocHost(h_x, (*totalNnz)*sizeof(int)) );
        CUDA_CHECK( cudaMallocHost(h_y, (*totalNnz)*sizeof(int)) );
        CUDA_CHECK( cudaMallocHost(h_z, (*totalNnz)*sizeof(int)) );
        CUDA_CHECK( cudaMallocHost(h_v, (*totalNnz)*sizeof(float)) );
        CUDA_CHECK( cudaHostGetDevicePointer(x, *h_x, 0) );
        CUDA_CHECK( cudaHostGetDevicePointer(y, *h_y, 0) );
        CUDA_CHECK( cudaHostGetDevicePointer(z, *h_z, 0) );
        CUDA_CHECK( cudaHostGetDevicePointer(v, *h_v, 0) );
    }
    else
    {
        CUDA_CHECK( cudaMalloc(x, (*totalNnz)*sizeof(int)) );
        CUDA_CHECK( cudaMalloc(y, (*totalNnz)*sizeof(int)) );
        CUDA_CHECK( cudaMalloc(z, (*totalNnz)*sizeof(int)) );
        CUDA_CHECK( cudaMalloc(v, (*totalNnz)*sizeof(float)) );
    }

    int s = 0;

    for (int k = 0; k < p; k++)
    {
        const float *slice = vol + m * n * k;

        int *dCooRowIndA = *x + s;
        int *dCsrColIndA = *y + s;
        float *dCsrValA = *v + s;

        // count nonzeros in slice
        CUDA_CHECK( cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, m, n, descr, slice, m, dNnzPerRow, &nnz) );

        // first get CSR matrix
        CUDA_CHECK( cusparseSdense2csr(handle, m, n, descr, slice, m, dNnzPerRow,
                                       dCsrValA, dCsrRowPtrA, dCsrColIndA) );

        // then generate COO indices
        CUDA_CHECK( cusparseXcsr2coo(handle, dCsrRowPtrA, nnz, m, dCooRowIndA, CUSPARSE_INDEX_BASE_ZERO) );

        cudaDeviceSynchronize();

        // generate Z indices
        CUDA_CHECK( cudaFill(*z + s, k, nnz) );

        s += nnz;
    }

    CUDA_CHECK( cudaFree(dNnzPerRow) );
    CUDA_CHECK( cudaFree(dCsrRowPtrA) );

    cudaDeviceSynchronize();
}

///////////////////////////////////////////////////////////////////////////////
// I/O ops
///////////////////////////////////////////////////////////////////////////////

// should these happen in separate threads?

// load image and push to GPU
void DHMProcessor::load_image(std::string path)
{
    cv::Mat mat = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    if ( mat.cols != N || mat.rows != N ) DHM_ERROR("Images must be of size NxN");

    memcpy(h_frame, mat.data, N*N*sizeof(byte));

    if (memory_kind == DHM_STANDARD_MEM)
    {
        CUDA_CHECK( cudaMemcpy(d_frame, h_frame, N*N*sizeof(byte), cudaMemcpyHostToDevice) );
    }
    // d_frame already mapped to h_frame in constructor if unified
}

void DHMProcessor::save_image(std::string path)
{
    cv::Mat mat(N, N, CV_8U, h_frame);
    cv::imwrite(path, mat);
}

// compress 3D volume to COO and save
// this method is bad and slow, for a few reasons... consider as placeholder
// separate CPU write thread would be nice
void DHMProcessor::save_volume(std::string path)
{
    // wait to finish
//    if (save_thread.joinable())
//        TIMER( save_thread.join() );

//    save_thread = std::thread([=](){
    float *d_v;
    int *d_x, *d_y, *d_z;
    int nnz;

    std::ofstream f(path, std::ios::out | std::ios::binary);

    if (memory_kind == DHM_STANDARD_MEM)
    {
        volume2sparse(handle, descr, d_volume, &d_x, N, &d_y, N, &d_z, num_slices, &d_v, &nnz);

        char *buffer = new char[nnz*(3*sizeof(int)+sizeof(float))];

        CUDA_CHECK( cudaMemcpy(buffer, d_x, nnz*sizeof(int), cudaMemcpyDeviceToHost) );
        CUDA_CHECK( cudaMemcpy(buffer + nnz*sizeof(int), d_y, nnz*sizeof(int), cudaMemcpyDeviceToHost) );
        CUDA_CHECK( cudaMemcpy(buffer + 2*nnz*sizeof(int), d_z, nnz*sizeof(int), cudaMemcpyDeviceToHost) );
        CUDA_CHECK( cudaMemcpy(buffer + 3*nnz*sizeof(int), d_v, nnz*sizeof(float), cudaMemcpyDeviceToHost) );

        f.write(buffer, 4*nnz*sizeof(int));

        delete[] buffer;
    }
    else
    {
        int *h_x, *h_y, *h_z;
        float *h_v;

        volume2sparse(handle, descr,
                      d_volume, &d_x, N, &d_y, N, &d_z, num_slices, &d_v, &nnz,
                      true, &h_x, &h_y, &h_z, &h_v);

        f.write((char *)h_x, nnz*sizeof(int));
        f.write((char *)h_y, nnz*sizeof(int));
        f.write((char *)h_z, nnz*sizeof(int));
        f.write((char *)h_v, nnz*sizeof(float));
    }

    f.close();
//    });
}

void DHMProcessor::load_volume(std::string path, float *h_volume)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);

    f.seekg(0, f.end);
    int nnz = f.tellg() / (4 * sizeof(int));
    f.seekg(0, f.beg);

    if (nnz == 0) DHM_ERROR("Empty file");

    float *v = new float[nnz];
    int *x = new int[nnz];
    int *y = new int[nnz];
    int *z = new int[nnz];

    f.read((char *)x, nnz*sizeof(int));
    f.read((char *)y, nnz*sizeof(int));
    f.read((char *)z, nnz*sizeof(int));
    f.read((char *)v, nnz*sizeof(float));

    for (int i = 0; i < nnz; i++)
    {
        h_volume[z[i]*N*N+y[i]*N+x[i]] = v[i];
    }

    delete[] x;
    delete[] y;
    delete[] z;
    delete[] v;
}

void DHMProcessor::display_image(byte *h_image)
{
    cv::Mat mat(N, N, CV_8U, h_image);
    cv::namedWindow("Display window", CV_WINDOW_NORMAL);
    cv::imshow("Display window", mat);
    cv::waitKey(0);
}

void DHMProcessor::display_volume(float *h_volume)
{
    for (int i = 0; i < num_slices; i++)
    {
        cv::Mat mat(N, N, CV_32F, h_volume + i*N*N);
        cv::normalize(mat, mat, 1.0, 0.0, cv::NORM_MINMAX, -1);
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
std::string check_dir(std::string path)
{
    using namespace boost::filesystem;

    if (path[0] == '~')
    {
        path = std::string(std::getenv("HOME")) + path.substr(1, path.size()-1);
    }

    path = canonical(path).string(); // resolve symlinks

    if ( !exists(path) || !is_directory(path) ) DHM_ERROR("Input directory not found");

    return path;
}

// returns an iterator through a folder in alphabetical order, optionally filtering by extension
std::vector<std::string> iter_folder(std::string path, std::string ext = "")
{
    using namespace boost::filesystem;

    path = check_dir(path);

    // loop thru bitmap files in input folder
    // ... these would probably be a video ...
    std::vector<std::string> dir;
    for (auto &i : boost::make_iterator_range(directory_iterator(path), {})) {
        std::string f = i.path().string();
        if (ext.length() > 0 && f.substr(f.find_last_of(".") + 1) == ext)
            dir.push_back(f);
    }
    std::sort(dir.begin(), dir.end());

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
    try {
        _func(img, mask, params);
    } catch (...) {
        DHM_ERROR("Callback failure");
    }
}

////////////////////////////////////////////////////////////////////////////////
// Main DHMProcessor class
////////////////////////////////////////////////////////////////////////////////

bool DHMProcessor::is_initialized = false;

DHMProcessor::DHMProcessor(const int num_slices, const float delta_z, const float z_init,
                           const DHMMemoryKind memory_kind = DHM_STANDARD_MEM)
{
    if (is_initialized) DHM_ERROR("Only a single instance of DHMProcessor is permitted");

    this->num_slices = num_slices;
    this->delta_z = delta_z;
    this->z_init = z_init;
    this->memory_kind = memory_kind;

    // allocate various buffers / handles
    setup_cuda();

    // save_thread = std::thread([](){});

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

////////////////////////////////////////////////////////////////////////////////
// Memory setup + cleanup
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
// Public methods
////////////////////////////////////////////////////////////////////////////////

void DHMProcessor::process_folder(std::string input_dir, std::string output_dir)
{
    input_dir = check_dir(input_dir);
    output_dir = check_dir(output_dir);

    for (std::string &f_in : iter_folder(input_dir, "bmp"))
    {
        load_image(f_in);

        // save_frame(); // only for process_camera

        CUDA_TIMER( process_frame() ); // *this* triggers the volume processing callback

        ////////////////////////////////////////////////////////////////////////
        // VOLUME REDUCTION OPS GO HERE
        ////////////////////////////////////////////////////////////////////////

        // semi-placeholder save operation
        std::string f_out = output_dir + "/" + f_in.substr(f_in.find_last_of("/") + 1) + ".bin";
        CUDA_TIMER( save_volume(f_out) );
    }
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

// will I expose the callback object or no?
void DHMProcessor::set_callback(DHMCallback cb)
{
    callback = cb;
}

void DHMProcessor::view_volume(std::string f_in)
{
    float *h_volume = new float[N*N*num_slices];
    load_volume(f_in, h_volume);
    display_volume(h_volume);
    delete[] h_volume;
}

////////////////////////////////////////////////////////////////////////////////
// The actual deconvolution
////////////////////////////////////////////////////////////////////////////////

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

    // flip the buffer
    buffer_pos = !buffer_pos;
}

}

