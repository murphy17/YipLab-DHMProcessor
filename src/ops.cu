/*
 * ops.cu
 *
 *  Created on: Jun 14, 2017
 *      Author: michaelmurphy
 */

#include "ops.cuh"
#include "DHMProcessor.cuh"

///////////////////////////////////////////////////////////////////////////////
// Complex arithmetic
///////////////////////////////////////////////////////////////////////////////

namespace ops {

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

}


///////////////////////////////////////////////////////////////////////////////
// Element-wise operations
///////////////////////////////////////////////////////////////////////////////

namespace ops {

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

}

///////////////////////////////////////////////////////////////////////////////
// Construct "PSF"
///////////////////////////////////////////////////////////////////////////////

namespace ops {

__global__ void _gen_filter_slice(
    complex *g,
    const float z,
    const DHMParameters p
) {
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
    q.extent.depth = NUM_SLICES;
    q.kind = cudaMemcpyHostToDevice;

    CUDA_CHECK( cudaMemcpy3DAsync(&q, async_stream) );
}

void DHMProcessor::gen_filter_quadrant(complex *h_filter) {
    complex *slice;
    CUDA_CHECK( cudaMalloc(&slice, N*N*sizeof(complex)) );

    for (int i = 0; i < NUM_SLICES; i++)
    {
        ops::_gen_filter_slice<<<N, N>>>(slice, Z0 + i * DZ, p);
        KERNEL_CHECK();

        // FFT in-place
        CUDA_CHECK( cufftXtExec(fft_plan, slice, slice, CUFFT_INVERSE) );

        // frequency shift -- eliminates need to copy later
        ops::_freq_shift<<<N, N>>>(slice);
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

namespace ops {

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

        for (int k = 0; k < p.NUM_SLICES; k++)
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

        for (int k = 0; k < p.NUM_SLICES; k++)
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

        for (int k = 0; k < p.NUM_SLICES; k++)
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

        for (int k = 0; k < p.NUM_SLICES; k++)
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

}


///////////////////////////////////////////////////////////////////////////////
// Convert 3D volume to sparse (COO) format
///////////////////////////////////////////////////////////////////////////////

//typedef thrust::counting_iterator<int> IndexIterator;
//typedef thrust::device_vector<float>::iterator FloatIterator;
//typedef thrust::tuple<IndexIterator, FloatIterator> IteratorTuple;
//typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

//namespace ops {
//
//struct _gen_coo_tuple : public thrust::unary_function<thrust::tuple<int, float>,float> {
//    int N;
//    __host__ __device__
//    _gen_coo_tuple(int n)
//    {
//        N = n;
//    }
//    __host__ __device__
//    COOTuple operator()(const thrust::tuple<int, float> &t)
//    {
//        int ii = thrust::get<0>(t);
//        int iz = ii / (N*N);
//        int iy = (ii - iz*N*N) / N;
//        int ix = ii - iz*N*N - iy*N;
//        float v = thrust::get<1>(t);
//        COOTuple c = {ix, iy, iz, v};
//        return c;
//    }
//};
//
//struct _filter_zeros {
//    float ZERO_THR;
//    __host__ __device__
//    _filter_zeros(float thr)
//    {
//        ZERO_THR = thr;
//    }
//    bool operator()(const thrust::tuple<int, float> t)
//    {
//        float v = thrust::get<1>(t);
//        return (v > ZERO_THR);
//    }
//};

namespace ops {

// compact rows
__global__
void _compact_rows(const float *slice, int *x, float *v, int *s, const DHMParameters p)
{
    extern __shared__ int shared_mem[];
    int *xs = shared_mem;
    float *vs = ((float *)shared_mem) + p.N;
    unsigned int *ss = ((unsigned int *)shared_mem) + 2*p.N;

    const float *row = slice + blockIdx.x * blockDim.x;

    float val = row[threadIdx.x];

    if (val > 0)
    {
        int idx = atomicInc(ss, 1);
        xs[idx] = threadIdx.x;
        vs[idx] = val;
    }

    __syncthreads();

    if (threadIdx.x < *ss)
    {
        int offset = blockIdx.x * blockDim.x + threadIdx.x;
        x[offset] = xs[threadIdx.x];
        v[offset] = vs[threadIdx.x];
    }

    if (threadIdx.x == 0)
    {
        s[blockIdx.x] = *ss;
    }
}

}

COOList DHMProcessor::rows_to_list(int *x, int *y, int z, float *v, int *s)
{
    int offset = 0;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < s[i]; j++)
        {
            x[offset] = x[i*N+j];
            y[offset] = i;
            v[offset] = v[i*N+j];

            offset++;
        }
    }

    COOList list;

    for (int i = 0; i < offset; i++)
    {
        COOTuple t = {x[i], y[i], z, v[i]};
        list.push_back(t);
    }

    return list;
}

// not working
COOList DHMProcessor::volume_to_list(float *volume)
{
    int *d_x, *d_y, *d_s;
    float *d_v;

    int *h_x, *h_y, *h_s;
    float *h_v;

    // unified...
    CUDA_CHECK( cudaMalloc(&d_x, N*N*sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&d_y, N*N*sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&d_v, N*N*sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&d_s, N*sizeof(int)) );

    CUDA_CHECK( cudaMallocHost(&h_x, N*N*sizeof(int)) );
    CUDA_CHECK( cudaMallocHost(&h_y, N*N*sizeof(int)) );
    CUDA_CHECK( cudaMallocHost(&h_v, N*N*sizeof(int)) );
    CUDA_CHECK( cudaMallocHost(&h_s, N*sizeof(int)) );

    std::vector<COOTuple> list;

    for (int i = 0; i < NUM_SLICES; i++)
    {
        float *slice = volume + i*N*N;
        ops::_compact_rows<<<N, N, (2*N+1)*sizeof(int)>>>(slice, d_x, d_v, d_s, p);
        KERNEL_CHECK();

        // not needed w/ unified!
        CUDA_CHECK( cudaMemcpy(h_x, d_x, N*N*sizeof(int), cudaMemcpyDeviceToHost) );
        CUDA_CHECK( cudaMemcpy(h_y, d_y, N*N*sizeof(int), cudaMemcpyDeviceToHost) );
        CUDA_CHECK( cudaMemcpy(h_v, d_v, N*N*sizeof(float), cudaMemcpyDeviceToHost) );
        CUDA_CHECK( cudaMemcpy(h_s, d_s, N*sizeof(int), cudaMemcpyDeviceToHost) );

        COOList row_list = rows_to_list(h_x, h_y, i, h_v, h_s);

        list.insert(list.end(), row_list.begin(), row_list.end());
    }

    // unified...
    CUDA_CHECK( cudaFree(d_x) );
    CUDA_CHECK( cudaFree(d_y) );
    CUDA_CHECK( cudaFree(d_v) );
    CUDA_CHECK( cudaFree(d_s) );
    CUDA_CHECK( cudaFree(h_x) );
    CUDA_CHECK( cudaFree(h_y) );
    CUDA_CHECK( cudaFree(h_v) );
    CUDA_CHECK( cudaFree(h_s) );

    return list;
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

    if (!UNIFIED_MEM)
    {
        CUDA_CHECK( cudaMemcpy(d_frame, h_frame, N*N*sizeof(byte), cudaMemcpyHostToDevice) );
    }
    else
    {
        CUDA_CHECK( cudaHostGetDevicePointer(&d_frame, h_frame, 0) );
    }
}

// compress 3D volume to COO and save
// not obvious how to use unified mem here - Thrust allocation rules it out?
void DHMProcessor::save_volume(std::string path)
{
    COOList list = volume_to_list(d_volume);

    std::ofstream f(path, std::ios::out | std::ios::binary);
    f.write((char *)(list.data()), list.size()*sizeof(COOTuple));
    f.close();
}

void DHMProcessor::load_volume(std::string path, float *h_volume)
{
    std::ifstream f(path, std::ios::in | std::ios::binary | std::ios::ate);

    int len = f.tellg();

    COOTuple *list = new COOTuple[len];

    f.read((char *)list, len*sizeof(COOTuple));
    f.close();

    for (int i = 0; i < len; i++)
    {
        COOTuple t = list[i];
        h_volume[t.z*N*N+t.y*N+t.x] = t.v;
    }

    delete[] list;
}







