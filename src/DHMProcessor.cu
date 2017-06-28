/*
 * DHMProcessor.cu
 *
 *  Created on: Jun 14, 2017
 *      Author: michaelmurphy
 */

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

// Generate the wavefront (in spatial domain) at a given distance.
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

// Generate each slice of the filter stack, FFT them, and push them to a host-side buffer.
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

        // trick to perform FFT shift without copying, uses a Fourier transform identity
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

// Pointwise multiply input matrix w by the upper-left quadrant of each slice of
// stack z, writing the result to z, and skipping slices indicated by mask
__global__
void _quad_mul(
    complex *z,
    const __restrict__ complex *w,
    const __restrict__ byte *mask,
    const DHMParameters p
) {
    const int i = blockIdx.x;
    const int j = threadIdx.x;
    const int ii = p.N-i;
    const int jj = p.N-j;

    // note: ordinarily you'd have the if statement inside the for loop,
    // that tends to hurt performance -- hence the repeated code.
    // each case handles either points in the interior of the quadrant,
    // or a certain (horizontal/vertical) boundary

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


////////////////////////////////////////////////////////////////////////////////
// Filesystem helper functions
////////////////////////////////////////////////////////////////////////////////

// make sure the input directory exists, and resolve any symlinks
fs::path check_dir(fs::path path)
{
    if (path.string()[0] == '~')
    {
        path = fs::path(std::string(std::getenv("HOME")) + path.string().substr(1, path.string().size()-1));
    }

    path = fs::canonical(path).string(); // resolve symlinks

    if ( !fs::exists(path) || !fs::is_directory(path) ) DHM_ERROR("Input directory not found");

    return path;
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
    //CUDA_CHECK( cudaMallocHost(&h_frame, N*N*sizeof(byte)) );
    //if (memory_kind == DHM_STANDARD_MEM)
    CUDA_CHECK( cudaMalloc(&d_frame, N*N*sizeof(byte)) );
    //else
    //    CUDA_CHECK( cudaHostGetDevicePointer(&d_frame, h_frame, 0) );
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

    //CUDA_CHECK( cudaFreeHost(h_frame) );
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


////////////////////////////////////////////////////////////////////////////////
// Public methods
////////////////////////////////////////////////////////////////////////////////

void DHMProcessor::process_folder(fs::path input_dir, fs::path output_dir, bool do_save_volume, int max_frames)
{
    if (is_running) DHM_ERROR("Can only run one operation at a time");
    is_running = true;

    input_dir = check_dir(input_dir);
    output_dir = check_dir(output_dir);
    this->output_dir = output_dir;

    // Count valid TIFF files in folder.
    if (max_frames == 0)
    {
        for (auto &i : boost::make_iterator_range(fs::directory_iterator(input_dir), {}))
        {
            fs::path this_path = i.path();
            bool is_valid = this_path.stem().string()[0] != '.';
            bool is_tiff = this_path.extension() == ".tif" || this_path.extension() == ".tiff";
            if (is_valid && is_tiff)
            {
                max_frames++;
            }
        }
    }

    // Start the thread monitoring the input folder, populating a queue of images in memory.
    ImageReader in(input_dir, QUEUE_SIZE);
    in.start();

    // Also start the thread responsible for writing in background.
    ImageWriter out(output_dir, QUEUE_SIZE);
    out.start();

    frame_num = 0;

    while (max_frames == -1 || frame_num < max_frames)
    {
        // Grab an image from the preloaded queue. Pause this thread and wait for a new image if queue empty.
        Image img_in;
        in.get(&img_in);

        std::cout << img_in.str;

        // Transfer the image from our wrapper type to the GPU.
        // ... and transfer it to the GPU.
        CUDA_CHECK( cudaMemcpy(d_frame, img_in.mat.data, N*N*sizeof(byte), cudaMemcpyHostToDevice) );

        // Perform GPU-side processing.
        process();

        std::string out_name = output_dir.string() + "/" + fs::path(img_in.str).filename().stem().string();

        // Transfer the depth map from GPU.
        CUDA_CHECK( cudaMemcpy(h_depth, d_depth, N*N*sizeof(float), cudaMemcpyDeviceToHost) );
        // ... and to our wrapper type.
        Image img_out;
        img_out.mat = cv::Mat(N, N, CV_32F, h_depth);
        img_out.str = out_name + ".tiff";

        // Write depth map to disk.
        out.write(img_out);

        std::cout << " -> " << img_out.str;

        // Save volume to disk if requested -- doesn't run in a separate thread since it takes so long.
        if (do_save_volume)
        {
            fs::path volume_path = fs::path(out_name + "_.tiff");

            cv::Mat mat(N, N, CV_8U);
            cv::cuda::GpuMat d_mat_norm(N, N, CV_32F);
            cv::cuda::GpuMat d_mat_byte(N, N, CV_8U);

            TinyTIFFFile* tif = TinyTIFFWriter_open(volume_path.string().c_str(), 8, N, N);

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
                DHM_ERROR("Could not write " + volume_path.string());
            }

            TinyTIFFWriter_close(tif);

            std::cout << ", " << volume_path.string();
        }

        std::cout << std::endl;

        frame_num++;
    }

    in.finish();
    out.finish();

    is_running = false;
}


////////////////////////////////////////////////////////////////////////////////
// The actual algorithm
////////////////////////////////////////////////////////////////////////////////

void DHMProcessor::process()
{
    // start transferring filter quadrants to alternating buffer, for *next* frame
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
    // d_filter -- NUM_SLICES*N*N complex volume
    ////////////////////////////////////////////////////////////////////////////


    // inverse FFT the product, and take complex magnitude
    for (int i = 0; i < num_slices; i++)
    {
        if (h_mask[i])
        {
            CUDA_CHECK( cufftXtExec(fft_plan, d_filter[buffer_pos] + i*N*N, d_filter[buffer_pos] + i*N*N, CUFFT_INVERSE) );

            _modulus<<<N, N>>>(d_filter[buffer_pos] + i*N*N, d_volume + i*N*N);
            KERNEL_CHECK();
        }
    }


    ////////////////////////////////////////////////////////////////////////////
    // CUSTOM SPATIAL DOMAIN OPS GO HERE
    // d_volume -- NUM_SLICES*N*N float volume
    // d_mask -- NUM_SLICES char vector, 1 to query that slice on next frame
    // d_depth -- N*N float depth map
    ////////////////////////////////////////////////////////////////////////////



    // placeholder depth-map operation (just first slice)
    CUDA_CHECK( cudaMemcpy(d_depth, &d_volume[0], N*N*sizeof(float), cudaMemcpyDeviceToDevice) );



    // placeholder mask generation
    CUDA_CHECK( cudaMemset(d_mask, 1, num_slices*sizeof(byte)) );



    // sync up the host-side and device-side masks
    // note: I first tried to used to use "managed memory" to do this,
    // but CPU and GPU crash when they read the same managed memory simultaneously
    CUDA_CHECK( cudaMemcpy(h_mask, d_mask, num_slices*sizeof(byte), cudaMemcpyDeviceToHost) );

    // advance the buffer
    buffer_pos = (buffer_pos + 1) % N_BUF;
}

}

