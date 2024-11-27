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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeConvexesCUDA(
	const torch::Tensor& background,
	const torch::Tensor& convex_points,
	const torch::Tensor& delta,
	const torch::Tensor& sigma,
	const torch::Tensor& num_points_per_convex,
	const torch::Tensor& cumsum_of_points_per_convex,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	torch::Tensor& scaling,
	torch::Tensor& density_factor,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const int number_of_points,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeConvexesBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& convex_points,
	const torch::Tensor& delta,
	const torch::Tensor& sigma,
	const torch::Tensor& num_points_per_convex,
	const torch::Tensor& cumsum_of_points_per_convex,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const int number_of_points,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);