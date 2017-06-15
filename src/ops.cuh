/*
 * ops.cuh
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#pragma once

#include "DHMCommon.cuh"

namespace ops {

__global__ void _b2c(const byte*, complex*);

__global__ void _freq_shift(complex*);

__global__ void _modulus(const complex*, float*);

__global__ void _gen_filter_slice(complex*, const float, const DHMParameters);

__global__ void _quad_mul(complex*, const complex*, const byte*, const DHMParameters);

}
