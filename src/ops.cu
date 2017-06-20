/*
 * ops.cu
 *
 * Tried to keep all the "business logic" in here
 * ... i realize the organisation is a bit schizophrenic
 *
 *  Created on: Jun 14, 2017
 *      Author: michaelmurphy
 */

#include "ops.cuh"
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
    g[i*p.N+j] = {-im / r, re / r};
}

void DHMProcessor::transfer_filter_async(complex *h_filter, complex *d_filter)
{
    // generate parameters for 3D copy
    cudaMemcpy3DParms q = { 0 };
    q.srcPtr.ptr = h_filter;
    q.srcPtr.pitch = (N/2+1) * sizeof(complex);
    q.srcPtr.xsize = (N/2+1);
    q.srcPtr.ysize = (N/2+1);
    q.dstPtr.ptr = d_filter;
    q.dstPtr.pitch = N * sizeof(complex);
    q.dstPtr.xsize = N;
    q.dstPtr.ysize = N;
    q.extent.width = (N/2+1) * sizeof(complex);
    q.extent.height = (N/2+1);
    q.extent.depth = num_slices;
    q.kind = cudaMemcpyHostToDevice;

    CUDA_CHECK( cudaMemcpy3DAsync(&q, async_stream) );
}

void DHMProcessor::gen_filter_quadrant(complex *h_filter) {
    complex *slice;
    CUDA_CHECK( cudaMalloc(&slice, N*N*sizeof(complex)) );

    for (int i = 0; i < num_slices; i++)
    {
        _gen_filter_slice<<<N, N>>>(slice, z_init + i * delta_z, p);
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

void volume2sparse(const cusparseHandle_t handle, const cusparseMatDescr_t descr,
                   const float *vol, int **x, int m, int **y, int n, int **z, int p, float **v, int *totalNnz)
{
    int nnz;
    int *dNnzPerRow, *dCsrRowPtrA;
    CUDA_CHECK( cudaMalloc(&dNnzPerRow, m*sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&dCsrRowPtrA, (m+1)*sizeof(int)) );

    CUDA_CHECK( cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, m, n*p, descr, vol, m, dNnzPerRow, totalNnz) );

    CUDA_CHECK( cudaMalloc(x, (*totalNnz)*sizeof(int)) );
    CUDA_CHECK( cudaMalloc(y, (*totalNnz)*sizeof(int)) );
    CUDA_CHECK( cudaMalloc(z, (*totalNnz)*sizeof(int)) );
    CUDA_CHECK( cudaMalloc(v, (*totalNnz)*sizeof(float)) );

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
    cv::Mat frame_mat = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    if ( frame_mat.cols != N || frame_mat.rows != N ) DHM_ERROR("Images must be of size NxN");

    memcpy(h_frame, frame_mat.data, N*N*sizeof(byte));

    if (memory_kind == DHM_STANDARD_MEM)
    {
        CUDA_CHECK( cudaMemcpy(d_frame, h_frame, N*N*sizeof(byte), cudaMemcpyHostToDevice) );
    }
    else
    {
        CUDA_CHECK( cudaHostGetDevicePointer(&d_frame, h_frame, 0) );
    }
}

// compress 3D volume to COO and save, using separate thread to do disk write
// this method is bad and slow, for a few reasons... consider as placeholder
void DHMProcessor::save_volume(std::string path)
{
    float *d_v;
    int *d_x, *d_y, *d_z;
    int nnz;

//    static std::shared_future<void> write_task;
//    static bool write_task_ready;

//    if (write_task_ready)
//        write_task.wait();

    volume2sparse(handle, descr, d_volume, &d_x, N, &d_y, N, &d_z, num_slices, &d_v, &nnz);

//    if (!write_task_ready)
//    {
//        write_task = std::shared_future<void>(std::async(std::launch::async, [&]() {
    char *buffer = new char[nnz*(3*sizeof(int)+sizeof(float))];

    CUDA_CHECK( cudaMemcpy(buffer, d_x, nnz*sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(buffer + nnz*sizeof(int), d_y, nnz*sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(buffer + 2*nnz*sizeof(int), d_z, nnz*sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(buffer + 3*nnz*sizeof(int), d_v, nnz*sizeof(float), cudaMemcpyDeviceToHost) );

    std::ofstream f(path, std::ios::out | std::ios::binary);
    f.write(buffer, 4*nnz*sizeof(int));
    f.close();

    delete[] buffer;
//        }));
//        write_task_ready = true;
//    }
//
//    write_task.get();
}

void DHMProcessor::load_volume(std::string path, float *h_volume)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);

    f.seekg(0, f.end);
    int nnz = f.tellg() / (4 * sizeof(int));
    f.seekg(0, f.beg);

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
    for (int i = 0; i< num_slices; i++)
    {
        cv::Mat mat(N, N, CV_32F, h_volume + i*N*N);
        cv::normalize(mat, mat, 1.0, 0.0, cv::NORM_MINMAX, -1);
        cv::namedWindow("Display window", CV_WINDOW_NORMAL);
        cv::imshow("Display window", mat);
        cv::waitKey(0);
    }
}

// returns an iterator through a folder in alphabetical order, optionally filtering by extension
std::vector<std::string> iter_folder(std::string path, std::string ext = "")
{
    using namespace boost::filesystem;

    if ( !exists(path) || !is_directory(path) ) DHM_ERROR("Input directory not found");

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

}
