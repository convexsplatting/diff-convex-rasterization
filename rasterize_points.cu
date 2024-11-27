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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
	const bool debug)
{
    
  const int P = number_of_points;
  const int H = image_height;
  const int W = image_width;

  auto int_opts = convex_points.options().dtype(torch::kInt32);
  auto float_opts = convex_points.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, convex_points.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  torch::Tensor out_others = torch::full({3+3+1, H, W}, 0.0, float_opts);


  const int total_nb_points = num_points_per_convex[P-1].item<int>() + cumsum_of_points_per_convex[P-1].item<int>();
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data_ptr<float>(),
		W, H,
		convex_points.contiguous().data_ptr<float>(),
		delta.contiguous().data_ptr<float>(),
		sigma.contiguous().data_ptr<float>(),
		num_points_per_convex.contiguous().data_ptr<int>(),
		cumsum_of_points_per_convex.contiguous().data_ptr<int>(),
		total_nb_points,
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data_ptr<float>(), 
		opacity.contiguous().data_ptr<float>(), 
		scaling.contiguous().data_ptr<float>(), 
		density_factor.contiguous().data_ptr<float>(), 
		viewmatrix.contiguous().data_ptr<float>(), 
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data_ptr<float>(),
		out_others.contiguous().data_ptr<float>(),
		radii.contiguous().data_ptr<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_others, radii, geomBuffer, binningBuffer, imgBuffer, scaling, density_factor);
}

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
	const bool debug) 
{
  const int P = number_of_points;
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  const int total_nb_points = num_points_per_convex[P-1].item<int>() + cumsum_of_points_per_convex[P-1].item<int>();

  torch::Tensor dL_dconvex = torch::zeros({total_nb_points, 3}, convex_points.options());
  torch::Tensor dL_ddelta = torch::zeros({P}, convex_points.options());
  torch::Tensor dL_dsigma = torch::zeros({P}, convex_points.options());
  torch::Tensor dL_dnormals = torch::zeros({total_nb_points, 3}, convex_points.options());
  torch::Tensor dL_doffsets = torch::zeros({total_nb_points, 3}, convex_points.options());

  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, convex_points.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, convex_points.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, convex_points.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, convex_points.options());

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, convex_points.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, convex_points.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, convex_points.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data_ptr<float>(),
	  W, H, 
	  convex_points.contiguous().data_ptr<float>(),
	  delta.contiguous().data_ptr<float>(),
	  sigma.contiguous().data_ptr<float>(),
	  num_points_per_convex.contiguous().data_ptr<int>(),
	  cumsum_of_points_per_convex.contiguous().data_ptr<int>(),
	  total_nb_points,
	  sh.contiguous().data_ptr<float>(),
	  colors.contiguous().data_ptr<float>(),
	  viewmatrix.contiguous().data_ptr<float>(),
	  projmatrix.contiguous().data_ptr<float>(),
	  campos.contiguous().data_ptr<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data_ptr<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data_ptr<float>(),
	  dL_dmeans3D.contiguous().data_ptr<float>(),
	  dL_dmeans2D.contiguous().data_ptr<float>(),
	  dL_dcov3D.contiguous().data_ptr<float>(),
	  dL_dconvex.contiguous().data_ptr<float>(),
	  dL_ddelta.contiguous().data_ptr<float>(),
	  dL_dsigma.contiguous().data_ptr<float>(),
	  dL_dnormals.contiguous().data_ptr<float>(),
	  dL_doffsets.contiguous().data_ptr<float>(),
	  dL_dconic.contiguous().data_ptr<float>(),  
	  dL_dopacity.contiguous().data_ptr<float>(),
	  dL_dcolors.contiguous().data_ptr<float>(),
	  dL_dsh.contiguous().data_ptr<float>(),
	  debug);
  }

  return std::make_tuple(dL_dconvex, dL_ddelta, dL_dsigma, dL_dcolors, dL_dopacity, dL_dsh, dL_dmeans2D);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		present.contiguous().data_ptr<bool>());
  }
  
  return present;
}