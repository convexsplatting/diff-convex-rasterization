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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;



// Backward pass for conversion of spherical harmonics to RGB for
// each Convex.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dconvex, glm::vec3* dL_dshs, const int cumsum_for_convex, const int num_points_per_convex)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means;
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Convex to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Convex's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	glm::vec3 scaled_dL_dmean = glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z) / static_cast<float>(num_points_per_convex);

	// Gradients of loss w.r.t. Convex means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.

	for (int i = 0; i < num_points_per_convex; i++)
	{
		dL_dconvex[cumsum_for_convex + i] += scaled_dL_dmean;
	}

}




// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float* convex_points,
	int W, int H,
	const int* radii,
	const float* shs,
	const bool* clamped,
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
	const glm::vec3* campos,
	glm::vec3* dL_dconvex,
	const float2* dL_dnormals,
	const float* dL_doffsets,
	glm::vec3* dL_dmeans,
	float3* dL_dmean2D,
	float* dL_dcov3D,
	float* dL_dcolor,
	float* dL_dsh)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int cumsum_for_convex = cumsum_of_points_per_convex[idx];
	const int offset = 3 * cumsum_for_convex;
	float3 center_convex = {0.0f, 0.0f, 0.0f};
	float sum_x[MAX_NB_POINTS] = {0.0f};
	float sum_y[MAX_NB_POINTS] = {0.0f};
	float sum_z[MAX_NB_POINTS] = {0.0f};
	for (int i = 0; i < num_points_per_convex[idx]; i++) {
		center_convex.x += convex_points[offset + 3 * i];
		center_convex.y += convex_points[offset + 3 * i + 1];
		center_convex.z += convex_points[offset + 3 * i + 2];
	}

	float3 total_sum = {center_convex.x, center_convex.y, center_convex.z};

	center_convex.x /= num_points_per_convex[idx];
	center_convex.y /= num_points_per_convex[idx];
	center_convex.z /= num_points_per_convex[idx];
	
	// Initialize loss accumulators for normals and offsets
	float loss_points_x[MAX_NB_POINTS] = {0.0f};
	float loss_points_y[MAX_NB_POINTS] = {0.0f};

	
	for (int i = 0; i < num_points_per_convex_view[idx]; i++) {
		float dL_dnormal_x = dL_dnormals[cumsum_for_convex + i].x;
		float dL_dnormal_y = dL_dnormals[cumsum_for_convex + i].y;
		float dL_doffset = dL_doffsets[cumsum_for_convex + i];

		float2 p1_conv = p_image[cumsum_for_convex + hull[2 * cumsum_for_convex + i]];
		float2 p2_conv = p_image[cumsum_for_convex + hull[2 * cumsum_for_convex + (i + 1) % (num_points_per_convex_view[idx]+1)]];

		// Calculate the normal vector (90-degree counterclockwise rotation)
		float2 normal = { p2_conv.y - p1_conv.y, -(p2_conv.x - p1_conv.x)};

		// Calculate the gradient of the loss with respect to the points p1 and p2
		// Gradient with respect to p1 (due to normal and offset)
		float2 dL_dp1_conv = {
			(dL_dnormal_y - dL_doffset * p2_conv.y),
			(-dL_dnormal_x + dL_doffset * p2_conv.x)
		};

		float2 dL_dp2_conv = {
			(-dL_dnormal_y + dL_doffset * p1_conv.y),
			(dL_dnormal_x - dL_doffset * p1_conv.x )
		};

		loss_points_x[indices[cumsum_for_convex + hull[2 * cumsum_for_convex + i]]] += dL_dp1_conv.x;
    	loss_points_y[indices[cumsum_for_convex + hull[2 * cumsum_for_convex + i]]] += dL_dp1_conv.y;

		loss_points_x[indices[cumsum_for_convex + hull[2 * cumsum_for_convex + (i + 1) % (num_points_per_convex_view[idx]+1)]]] += dL_dp2_conv.x;
    	loss_points_y[indices[cumsum_for_convex + hull[2 * cumsum_for_convex + (i + 1) % (num_points_per_convex_view[idx]+1)]]] += dL_dp2_conv.y;
	}


	dL_dmean2D[idx].x = 0;
	dL_dmean2D[idx].y = 0;
	dL_dmean2D[idx].z = 0;

	for (int i = 0; i < num_points_per_convex[idx]; i++) {

		dL_dmean2D[idx].x += loss_points_x[i];
		dL_dmean2D[idx].y += loss_points_y[i];
		dL_dmean2D[idx].z += abs(loss_points_x[i]) + abs(loss_points_y[i]);

		float mul1 = (proj[0] * convex_points[offset + 3 * i] + proj[4] * convex_points[offset + 3 * i + 1] + proj[8] * convex_points[offset + 3 * i + 2] + proj[12]) * p_w[cumsum_for_convex + i] * p_w[cumsum_for_convex + i];
		float mul2 = (proj[1] * convex_points[offset + 3 * i] + proj[5] * convex_points[offset + 3 * i + 1] + proj[9] * convex_points[offset + 3 * i + 2] + proj[13]) * p_w[cumsum_for_convex + i] * p_w[cumsum_for_convex + i];
		dL_dconvex[cumsum_for_convex + i].x = (proj[0] * p_w[cumsum_for_convex + i] - proj[3] * mul1) * loss_points_x[i]  + (proj[1] * p_w[cumsum_for_convex + i] - proj[3] * mul2) * loss_points_y[i];
		dL_dconvex[cumsum_for_convex + i].y = (proj[4] * p_w[cumsum_for_convex + i] - proj[7] * mul1) * loss_points_x[i] + (proj[5] * p_w[cumsum_for_convex + i] - proj[7] * mul2) * loss_points_y[i];
		dL_dconvex[cumsum_for_convex + i].z = (proj[8] * p_w[cumsum_for_convex + i] - proj[11] * mul1) * loss_points_x[i] + (proj[9] * p_w[cumsum_for_convex + i] - proj[11] * mul2) * loss_points_y[i];
	
		sum_x[i] = total_sum.x - convex_points[offset + 3 * i];
		sum_y[i] = total_sum.y - convex_points[offset + 3 * i + 1];
		sum_z[i] = total_sum.z - convex_points[offset + 3 * i + 2];

	}

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3)(center_convex.x, center_convex.y, center_convex.z), *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dconvex, (glm::vec3*)dL_dsh, cumsum_for_convex, num_points_per_convex[idx]);

}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float* __restrict__ delta,
	const float* __restrict__ sigma,
	const int* __restrict__ num_points_per_convex,
	const int* __restrict__ cumsum_of_points_per_convex,
	const float2* __restrict__ normals,
	const float* __restrict__ offsets,
	const int* __restrict__ num_points_per_convex_view,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ depths,
	const float2* __restrict__ means2D,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float2* __restrict__ dL_dnormals,
	float* __restrict__ dL_doffsets,
	float* __restrict__ dL_ddelta,
	float* __restrict__ dL_dsigma,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	/*
	ADDED FOR CONVEX PURPOSES ==========================================================================
	*/
	__shared__ float2 collected_normals[BLOCK_SIZE * MAX_NB_POINTS];
	__shared__ float collected_offsets[BLOCK_SIZE * MAX_NB_POINTS];
	__shared__ int collected_num_points_per_convex_view[BLOCK_SIZE];
	__shared__ int collected_cumsum_of_points_per_convex[BLOCK_SIZE];
	__shared__ float collected_delta[BLOCK_SIZE];
	__shared__ float collected_sigma[BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	/*
	===================================================================================================
	*/

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Convex is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	// const float ddelx_dx = 0.5 * W;
	// const float ddely_dy = 0.5 * H;

	// Traverse all Convexes
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			collected_num_points_per_convex_view[block.thread_rank()] = num_points_per_convex_view[coll_id];
			collected_cumsum_of_points_per_convex[block.thread_rank()] = cumsum_of_points_per_convex[coll_id];
			collected_delta[block.thread_rank()] = delta[coll_id];
			collected_sigma[block.thread_rank()] = sigma[coll_id];
			collected_depths[block.thread_rank()] = depths[coll_id];
			collected_xy[block.thread_rank()] = means2D[coll_id];
			for (int k = 0; k < num_points_per_convex_view[coll_id]; k++) {
    			collected_normals[MAX_NB_POINTS * block.thread_rank() + k] = normals[cumsum_of_points_per_convex[coll_id] + k];
			}
			for (int k = 0; k < num_points_per_convex_view[coll_id]; k++) {
    			collected_offsets[MAX_NB_POINTS * block.thread_rank() + k] = offsets[cumsum_of_points_per_convex[coll_id] + k];
			}
		}
		block.sync();

		// Iterate over Convexes
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{

			contributor--;
			if (contributor >= last_contributor)
				continue;

			float4 con_o = collected_conic_opacity[j];
			float distances[MAX_NB_POINTS];
			float max_val = -INFINITY;
			for (int k = 0; k < collected_num_points_per_convex_view[j]; k++) {
				distances[k] = collected_normals[j * MAX_NB_POINTS + k].x * pixf.x + collected_normals[j * MAX_NB_POINTS + k].y * pixf.y + collected_offsets[j * MAX_NB_POINTS + k];
				if (distances[k] > max_val) {
					max_val = distances[k];
				}
			}

			float sum_exp = 0.0f;
			for (int k = 0; k < collected_num_points_per_convex_view[j]; k++) {
				sum_exp += expf(collected_depths[j] * collected_delta[j] * (distances[k]-max_val));
			}

			float phi_x = collected_depths[j] * collected_delta[j]*max_val + logf(sum_exp);

			float Cx = 1.0f / (1.0f + expf(collected_depths[j] * collected_sigma[j] * phi_x));

			const float alpha = min(0.99f, con_o.w * Cx);

			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Convex colors and keep
			// gradients w.r.t. alpha (blending factor for a Convex/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Convex. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Convex.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Helpful reusable temporary variables
			const float dL_dC = con_o.w * dL_dalpha;

			// Calculate gradients w.r.t sigma 
            float dL_dsigma_value = -collected_depths[j] * phi_x * Cx * (1.0f - Cx) * dL_dC;   // remove depth here
            atomicAdd(&(dL_dsigma[global_id]), dL_dsigma_value);

			// Calculate gradient w.r.t phi_x 
            float dL_dphi_x = -collected_sigma[j]  * collected_depths[j] * Cx * (1.0f - Cx) * dL_dC;  // remove depth here

            // Calculate gradients with respect to distances
            float dL_ddistances[MAX_NB_POINTS];
            for (int k = 0; k < collected_num_points_per_convex_view[j]; k++) {
                float exp_val = expf(collected_depths[j] * collected_delta[j] * (distances[k]-max_val));
                dL_ddistances[k] = (exp_val / sum_exp) * dL_dphi_x * collected_delta[j] * collected_depths[j];
            }

			// Gradient with respect to delta
			float dL_ddelta_value = 0.0f;
			for (int k = 0; k < collected_num_points_per_convex_view[j]; k++) {
				float exp_val = expf(collected_delta[j] * collected_depths[j] * (distances[k]-max_val));	
				dL_ddelta_value += collected_depths[j] * (distances[k]-max_val) * exp_val / sum_exp;
			}
			// Multiply by the chain rule term dL_dphi_x
			float dL_ddelta_value_aux = (collected_depths[j] * max_val + dL_ddelta_value) * dL_dphi_x;

			// Apply the gradient update to delta
			atomicAdd(&(dL_ddelta[global_id]), dL_ddelta_value_aux);


            // Calculate gradients w.r.t normals and offsets
            for (int k = 0; k < collected_num_points_per_convex_view[j]; k++) {
				// Gradient w.r.t. nx and ny
				atomicAdd(&(dL_dnormals[collected_cumsum_of_points_per_convex[j] + k].x), dL_ddistances[k] * pixf.x);
				atomicAdd(&(dL_dnormals[collected_cumsum_of_points_per_convex[j] + k].y), dL_ddistances[k] * pixf.y);
				// Gradient w.r.t. d
				atomicAdd(&(dL_doffsets[collected_cumsum_of_points_per_convex[j] + k]), dL_ddistances[k]);
            }

			// Update gradients w.r.t. opacity of the Convex
			atomicAdd(&(dL_dopacity[global_id]), dL_dalpha * Cx);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float* convex_points,
	int W, int H,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const float* viewmatrix,
	const float* projmatrix,
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
	glm::vec3* dL_dmean3D,
	float3* dL_dmean2D,
	const float* dL_dconic,
	float* dL_dcov3D,
	float* dL_dcolor,
	float* dL_dsh
	)
{
	
	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		convex_points,
		W, H,
		radii,
		shs,
		clamped,
		projmatrix,
		num_points_per_convex,
		cumsum_of_points_per_convex,
		p_hom,
		p_w,
		p_proj,
		p_image,
		hull,
		indices,
		num_points_per_convex_view,
		campos,
		(glm::vec3*)dL_dconvex,
		(float2*) dL_dnormals,
		dL_doffsets,
		(glm::vec3*)dL_dmean3D,
		(float3*)dL_dmean2D,
		dL_dcov3D,
		dL_dcolor,
		dL_dsh);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
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
	float* dL_dcolors)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		delta,
		sigma,
		num_points_per_convex,
		cumsum_of_points_per_convex,
		normals,
		offsets,
		num_points_per_convex_view,
		conic_opacity,
		depths,
		means2D,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dnormals,
		dL_doffsets,
		dL_ddelta,
		dL_dsigma,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors
		);
}