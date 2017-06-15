/*
 * ops_extra.cu
 *
 *  Created on: Jun 14, 2017
 *      Author: michaelmurphy
 */

//#include "DHMCommon.cuh"
//
//namespace ops {
//
//__global__
//void mirror_quadrants(complex *z)
//{
//    const int i = blockIdx.x;
//    const int j = threadIdx.x;
//    const int ii = N-i;
//    const int jj = N-j;
//
//    if (j>0&&j<N/2) z[i*N+jj] = z[i*N+j];
//    if (i>0&&i<N/2) z[ii*N+j] = z[i*N+j];
//    if (i>0&&i<N/2&&j>0&&j<N/2) z[ii*N+jj] = z[i*N+j];
//}
//
//}

////////////////////////////////////////////////////////////////////////////////
// Prevent accidentally using host pointer on device, and vice versa
////////////////////////////////////////////////////////////////////////////////

//template <class T> class device_T;
//template <class T> class host_T;
//
//template <class T>
//class device_T : T
//{
//private:
//    operator host_T<T>();
//    device_T(host_T<T>);
//};
//
//template <class T>
//class host_T : T
//{
//private:
//    operator device_T<T>();
//    host_T(device_T<T>);
//};
//
//typedef device_T<complex> d_complex;
//typedef host_T<complex> h_complex;
//typedef device_T<real> d_real;
//typedef host_T<real> h_real;
//typedef device_T<byte> d_byte;
//typedef host_T<byte> h_byte;
//
