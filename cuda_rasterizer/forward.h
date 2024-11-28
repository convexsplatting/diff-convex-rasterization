/*
 * The original code is under the following copyright:
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE_GS.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * The modifications of the code are under the following copyright:
 * Copyright (C) 2024, University of Liege, KAUST and University of Oxford
 * TELIM research group, http://www.telecom.ulg.ac.be/
 * IVUL research group, https://ivul.kaust.edu.sa/
 * VGG research group, https://www.robots.ox.ac.uk/~vgg/
 * All rights reserved.
 * The modifications are under the LICENSE.md file.
 *
 * For inquiries contact jan.held@uliege.be
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Convex prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* convex_points,
		const float* delta,
		const float* sigma,
		const int* num_points_per_convex,
		const int* cumsum_of_points_per_convex,
		const float* opacities,
		float* scaling,
		float* density_factor,
		const float* shs,
		bool* clamped,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* normals,
		float* offsets,
		int* num_points_per_convex_view,
		float4* p_hom,
		float* p_w,
		float3* p_proj,
		float2* p_image,
		int* hull,
		int* indices,
		float2* points_xy_image,
		float* depths,
		float* colors,
		float4* conic_opacity,
		float* cov3Ds,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* normals,
		const float* offsets,
		const int* num_points_per_convex_view,
		const float2* points_xy_image,
		const float* delta,
		const float* sigma,
		const int* num_points_per_convex,
		const int* cumsum_of_points_per_convex,
		const float* features,
		const float4* conic_opacity,
		const float* depths,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_others);
}


#endif