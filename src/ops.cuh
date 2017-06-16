/*
 * ops.cuh
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#pragma once

#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "DHMCommon.cuh"
#include "DHMProcessor.cuh"

namespace ops {

__global__ void _b2c(const byte*, complex*);

__global__ void _freq_shift(complex*);

__global__ void _modulus(const complex*, float*);

__global__ void _gen_filter_slice(complex*, const float, const DHMParameters);

__global__ void _quad_mul(complex*, const complex*, const byte*, const DHMParameters);

}
