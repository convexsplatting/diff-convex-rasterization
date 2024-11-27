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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* convex_points,
			const float* delta,
			const float* sigma,
			const int* num_points_per_convex,
			const int* cumsum_of_points_per_convex,
			const int total_nb_points,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			float* scaling,
			float* density_factor,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_others,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* convex_points,
			const float* delta,
			const float* sigma,
			const int* num_points_per_convex,
			const int* cumsum_of_points_per_convex,
			const int total_nb_points,
			const float* shs,
			const float* colors_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmeans3D,
			float* dL_dmeans2D,
			float* dL_dcov3D,
			float* dL_dconvex,
			float* dL_ddelta,
			float* dL_dsigma,
			float* dL_dnormals,
			float* dL_doffsets,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dsh,
			bool debug);
	};
};

#endif