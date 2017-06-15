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

    CUDA_CHECK( cudaMemcpy3DAsync(&q, copy_stream) );
}

void DHMProcessor::gen_filter_quadrant(complex *h_filter) {
    complex *slice;
    CUDA_CHECK( cudaMalloc(&slice, N*N*sizeof(complex)) );

    for (int i = 0; i < NUM_SLICES; i++)
    {
        ops::_gen_filter_slice<<<N, N, 0, math_stream>>>(slice, Z0 + i * DZ, p);
        KERNEL_CHECK();

        // FFT in-place
        CUDA_CHECK( cufftXtExec(fft_plan, slice, slice, CUFFT_INVERSE) );
        CUDA_CHECK( cudaStreamSynchronize(math_stream) );

        // frequency shift -- eliminates need to copy later
        ops::_freq_shift<<<N, N, 0, math_stream>>>(slice);
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

