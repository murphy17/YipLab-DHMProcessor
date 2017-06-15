/*
 * ops.cu
 *
 *  Created on: Jun 14, 2017
 *      Author: michaelmurphy
 */

#include "DHMCommon.cuh"
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

    z[i*N+j].x = (real)(((float)(b[i*N+j])) / 255.f);
    z[i*N+j].y = (real)0.f;
}

__global__
void _freq_shift(complex *data)
{
    const int i = blockIdx.x;
    const int j = threadIdx.x;
    const int N = blockDim.x;

    const real a = (real)(1 - 2 * ((i+j) & 1));

    data[i*N+j].x *= a;
    data[i*N+j].y *= a;
}

}

///////////////////////////////////////////////////////////////////////////////
// Construct "PSF"
///////////////////////////////////////////////////////////////////////////////

namespace ops {

__global__ void _gen_psf_slice(
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

void DHMProcessor::gen_psf_quadrant(complex *psf) {
    complex *slice;
    CUDA_CHECK( cudaMalloc(&slice, N*N*sizeof(complex)) );

    for (int i = 0; i < NUM_SLICES; i++)
    {
        ops::_gen_psf_slice<<<N, N, 0, math_stream>>>(slice, Z0 + i * DZ, p);
        KERNEL_CHECK();

        // FFT in-place
        CUDA_CHECK( cufftXtExec(fft_plan, slice, slice, CUFFT_INVERSE) );
        CUDA_CHECK( cudaStreamSynchronize(math_stream) );

        // frequency shift -- eliminates need to copy later
        ops::_freq_shift<<<N, N, 0>>>(slice);
        KERNEL_CHECK();

        // copy single quadrant
        CUDA_CHECK( cudaMemcpy2D(
            psf + (N/2+1)*(N/2+1)*i,
            (N/2+1)*sizeof(complex),
            psf,
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
// Frame processing pipeline stages
///////////////////////////////////////////////////////////////////////////////

void DHMProcessor::fft_image(const byte *image, complex *image_f)
{
    ops::_b2c<<<N, N>>>(image, image_f);
    KERNEL_CHECK();

    CUDA_CHECK( cufftXtExec(fft_plan, image_f, image_f, CUFFT_FORWARD) );
}

void DHMProcessor::apply_filter(complex *stack, const complex *image, const byte *mask)
{
    ops::_quad_mul<<<N/2+1, N/2+1>>>(stack, image, mask, p);
    KERNEL_CHECK();
}

//void deconvolve_filter(
//    complex *stack,
//    real *output,
//    byte *mask,
//    DHMParameters params
//) {
//    // inverse FFT the product - batch FFT gave no speedup
//    for (int i = 0; i < NUM_SLICES; i++)
//    {
//        complex *slice = stack + N*N*sizeof(complex);
//
//        if (mask[slice])
//        {
//            checkCudaErrors( cufftXtExec(params.fft_plan,
//                                         slice + N*N*slice,
//                                         slice + N*N*slice,
//                                         CUFFT_INVERSE) );
//
//            modulus<<<N, N, 0, math_stream>>>(in_buffer + N*N*slice, out_buffer + N*N*slice);
//    }
//}
