/*
 * Copyright (C) 2024, Inria, University of Liege, KAUST and University of Oxford
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * TELIM research group, http://www.telecom.ulg.ac.be/
 * IVUL research group, https://ivul.kaust.edu.sa/
 * VGG research group, https://www.robots.ox.ac.uk/~vgg/
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact jan.held@uliege.be
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float* delta,
		const float* sigma,
		const int* num_points_per_convex,
		const int* cumsum_of_points_per_convex,
		const float2* normals,
		const float* offsets,
		const int* num_points_per_convex_view,
		const float4* conic_opacity,
		const float* depths,
		const float2* means2D,
		const float* colors,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		float2* dL_dnormals,
		float* dL_doffsets,
		float* dL_ddelta,
		float* dL_dsigma,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors);

	void preprocess(
		int P, int D, int M,
		const float* convex_points,
		int W, int H,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const float* view,
		const float* proj,
		const int* num_points_per_convex,
		const int* cumsum_of_points_per_convex,
		float4* p_hom,
		float* p_w,
		float3* p_proj,
		float2* p_image,
		int* hull,
		int* indices,
		int* num_points_per_convex_view,
		const float* cov3Ds,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		glm::vec3* dL_dconvex,
		const float2* dL_dnormals,
		const float* dL_doffsets,
		glm::vec3* dL_dmeans,
		float3* dL_dmean2D,
		const float* dL_dconics,
		float* dL_dcov3D,
		float* dL_dcolor,
		float* dL_dsh);
}

#endif