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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;



// Forward method for converting the input spherical harmonics
// coefficients of each Convex to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means;
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}



// Perform initial steps for each Convex prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
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
	float* rgb,
	float4* conic_opacity,
	float* cov3Ds,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{

	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Convex will not be processed further.
	
	const int cumsum_for_convex = cumsum_of_points_per_convex[idx];
	const int offset = 3 * cumsum_for_convex;

	radii[idx] = 0;
	tiles_touched[idx] = 0;
	num_points_per_convex_view[idx] = 0;
	scaling[idx] = 0.0f;
	density_factor[idx] = 0.0f;

	
	float3 center_convex = {0.0f, 0.0f, 0.0f};

	for (int i = 0; i < num_points_per_convex[idx]; i++) {
		indices[cumsum_for_convex + i] = i;
		center_convex.x += convex_points[offset + 3 * i];
		center_convex.y += convex_points[offset + 3 * i + 1];
		center_convex.z += convex_points[offset + 3 * i + 2];
	}

	center_convex.x /= num_points_per_convex[idx];
	center_convex.y /= num_points_per_convex[idx];
	center_convex.z /= num_points_per_convex[idx];

	// Perform near culling, quit if outside.
	float3 p_view_convex;
	if (!in_frustum_convex(idx, center_convex, viewmatrix, projmatrix, prefiltered, p_view_convex)){
		return;
	}

	float4 p_hom_center = transformPoint4x4(center_convex, projmatrix);
	float p_w_center = 1.0f / (p_hom_center.w + 0.0000001f);
	float3 center_convex_camera_view = { p_hom_center.x * p_w_center, p_hom_center.y * p_w_center, p_hom_center.z * p_w_center };
	float2 center_convex_2D = { ndc2Pix(center_convex_camera_view.x, W), ndc2Pix(center_convex_camera_view.y, H) };

	// Calculation of points in 2D image space 
	float cov3D[6] = {0.0f};

	for (int i = 0; i < num_points_per_convex[idx]; i++) {
		float3 convex_point = {convex_points[offset + 3 * i], convex_points[offset + 3 * i + 1], convex_points[offset + 3 * i + 2]};
		p_hom[cumsum_for_convex + i] = transformPoint4x4(convex_point, projmatrix);
		p_w[cumsum_for_convex + i] = 1.0f / (p_hom[cumsum_for_convex + i].w + 0.0000001f);
		p_proj[cumsum_for_convex + i] = { p_hom[cumsum_for_convex + i].x * p_w[cumsum_for_convex + i], p_hom[cumsum_for_convex + i].y * p_w[cumsum_for_convex + i], p_hom[cumsum_for_convex + i].z * p_w[cumsum_for_convex + i] };
		p_image[cumsum_for_convex + i] = { ndc2Pix(p_proj[cumsum_for_convex + i].x, W), ndc2Pix(p_proj[cumsum_for_convex + i].y, H) };

		float3 diff = {
			convex_point.x - center_convex.x,
			convex_point.y - center_convex.y,
			convex_point.z - center_convex.z
		};

		cov3D[0] += diff.x * diff.x;
		cov3D[1] += diff.x * diff.y;
		cov3D[2] += diff.x * diff.z;
		cov3D[3] += diff.y * diff.y;
		cov3D[4] += diff.y * diff.z;
		cov3D[5] += diff.z * diff.z;
	}

	// normalize the covariance matrix
	cov3D[0] /= num_points_per_convex[idx];
	cov3D[1] /= num_points_per_convex[idx];
	cov3D[2] /= num_points_per_convex[idx];
	cov3D[3] /= num_points_per_convex[idx];
	cov3D[4] /= num_points_per_convex[idx];
	cov3D[5] /= num_points_per_convex[idx];

	cov3Ds[6 * idx] = cov3D[0];
	cov3Ds[6 * idx + 1] = cov3D[1];
	cov3Ds[6 * idx + 2] = cov3D[2];
	cov3Ds[6 * idx + 3] = cov3D[3];
	cov3Ds[6 * idx + 4] = cov3D[4];
	cov3Ds[6 * idx + 5] = cov3D[5];

	scaling[idx] = max(sqrtf(cov3D[0]), max(sqrtf(cov3D[3]), sqrtf(cov3D[5])));


	float max_distance = sqrtf(
            (p_image[cumsum_for_convex].x - center_convex_2D.x) * (p_image[cumsum_for_convex].x - center_convex_2D.x) +
            (p_image[cumsum_for_convex].y - center_convex_2D.y) * (p_image[cumsum_for_convex].y - center_convex_2D.y)
        );
	float2 ref_point = p_image[cumsum_for_convex];
	
    for (int i = 1; i < num_points_per_convex[idx]; i++) {
		
		float distance = sqrtf(
            (p_image[cumsum_for_convex + i].x - center_convex_2D.x) * (p_image[cumsum_for_convex + i].x - center_convex_2D.x) +
            (p_image[cumsum_for_convex + i].y - center_convex_2D.y) * (p_image[cumsum_for_convex + i].y - center_convex_2D.y)
        );

        // Update max_distance if the current distance is greater
        if (distance > max_distance) {
            max_distance = distance;
        }

        if (p_image[cumsum_for_convex + i].y < ref_point.y || (p_image[cumsum_for_convex + i].y == ref_point.y && p_image[cumsum_for_convex + i].x < ref_point.x)) {
            ref_point = p_image[cumsum_for_convex + i];
        }
    }

	if(max_distance > 10000.0f){
		return;
	}

	// Sort the points based on their polar angle with respect to ref_point.
	// There exist definitely better sorting algos
	for (int i = 0; i < num_points_per_convex[idx] - 1; i++) {
		for (int j = i + 1; j < num_points_per_convex[idx]; j++) {
			float angle1 = atan2f(p_image[cumsum_for_convex + i].y - ref_point.y, p_image[cumsum_for_convex + i].x - ref_point.x);
			float angle2 = atan2f(p_image[cumsum_for_convex + j].y - ref_point.y, p_image[cumsum_for_convex + j].x - ref_point.x);
			if (angle1 > angle2) {
				float2 temp = p_image[cumsum_for_convex + i];
				p_image[cumsum_for_convex + i] = p_image[cumsum_for_convex + j];
				p_image[cumsum_for_convex + j] = temp;

				// Swap their corresponding indices
				int temp_idx = indices[cumsum_for_convex + i];
				indices[cumsum_for_convex + i] = indices[cumsum_for_convex + j];
				indices[cumsum_for_convex + j] = temp_idx;
			}
		}
	}

	// Now we apply the Graham scan algorithm to find the convex hull.
	int hull_size = 0;

	// Lower hull
	for (int i = 0; i < num_points_per_convex[idx]; i++) {
		while (hull_size >= 2 && crossProduct(p_image[cumsum_for_convex + hull[2 * cumsum_for_convex + hull_size - 2]], p_image[cumsum_for_convex + hull[2 * cumsum_for_convex + hull_size - 1]], p_image[cumsum_for_convex + i]) <= 0)
			hull_size--;
		hull[2 * cumsum_for_convex + hull_size++] = i;
	}

	//Upper hull
	int t = hull_size + 1;
	for (int i = num_points_per_convex[idx] - 2; i >= 0; i--) {
		while (hull_size >= t && crossProduct(p_image[cumsum_for_convex + hull[2 * cumsum_for_convex + hull_size - 2]], p_image[cumsum_for_convex + hull[2 * cumsum_for_convex + hull_size - 1]], p_image[cumsum_for_convex + i]) <= 0)
			hull_size--;
		hull[2 * cumsum_for_convex + hull_size++] = i;
	} 

	float max_distance_off = 0.0f;
	float previous_offset = 0.0f;
	int counter = 0;
	float max_distance_x = (2.1f / (p_view_convex.z * p_view_convex.z * delta[idx] * sigma[idx]));

	for (int i = 0; i < hull_size - 1; i++) {
		// Points forming the segment
		float2 p1_conv = p_image[cumsum_for_convex + hull[2 * cumsum_for_convex + i]];
		float2 p2_conv = p_image[cumsum_for_convex + hull[2 * cumsum_for_convex + (i + 1) % hull_size]];

		// Calculate the normal vector (90-degree counterclockwise rotation)
		float2 normal = { p2_conv.y - p1_conv.y, -(p2_conv.x - p1_conv.x)};

		// Calculate the offset (dot product of normal vector and point p1)
		float offset = - (normal.x * p1_conv.x + normal.y * p1_conv.y);

		normals[cumsum_for_convex + i] = normal;
    	offsets[cumsum_for_convex + i] = offset; 

		if (normal.x * center_convex_2D.x + normal.y * center_convex_2D.y + offset < 0) {
			offset -= max_distance_x;
		}else{
			offset += max_distance_x;
		}

		if (i != 0){
			float denominator = normal.x * normals[cumsum_for_convex + (i-1)].y - normal.y * normals[cumsum_for_convex + (i-1)].x; // to avoid division by small numbers

			//  calculate the point of intersection between normals[i] and normals[i-1]
			float2 intersection_point = { (-offset * normals[cumsum_for_convex + (i-1)].y + previous_offset * normal.y) / denominator, (-previous_offset * normal.x + offset * normals[cumsum_for_convex + (i-1)].x) / denominator};

			float angle = acosf( (normal.x * normals[cumsum_for_convex + (i-1)].x + normal.y * normals[cumsum_for_convex + (i-1)].y) / (sqrtf(normal.x * normal.x + normal.y * normal.y) * sqrtf(normals[cumsum_for_convex + (i-1)].x * normals[cumsum_for_convex + (i-1)].x + normals[cumsum_for_convex + (i-1)].y * normals[cumsum_for_convex + (i-1)].y)));
			
			float distance = sqrtf((intersection_point.x - center_convex_2D.x) * (intersection_point.x - center_convex_2D.x) + (intersection_point.y - center_convex_2D.y) * (intersection_point.y - center_convex_2D.y));

			if (angle > 0.1f && angle < 3.0f){
				max_distance_off += distance;
				counter++;
			}	
		}

		previous_offset = offset;
		num_points_per_convex_view[idx] = i + 1;
	}

	if (num_points_per_convex_view[idx] < 3 || counter == 0){
		radii[idx] = 0;
		tiles_touched[idx] = 0;
		num_points_per_convex_view[idx] = 0;
		scaling[idx] = 0.0f;
		density_factor[idx] = 0.0f;
		return;
	}

	max_distance_off = max_distance_off / counter;
	max_distance = ceil(max(max_distance * 1.1f, max_distance_off));
	
	// The max distance should be changed in the future as it also depends on delta. If the transition from 0 to 1 is smoother then the distance of influence of the convex is also larger
	uint2 rect_min_convex, rect_max_convex;
	getRect(center_convex_2D, max_distance, rect_min_convex, rect_max_convex, grid);
	if ((rect_max_convex.x - rect_min_convex.x) * (rect_max_convex.y - rect_min_convex.y) == 0)
		return;


	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3)(center_convex.x, center_convex.y, center_convex.z), *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view_convex.z; 
	radii[idx] = max_distance;
	points_xy_image[idx] = center_convex_2D;
	conic_opacity[idx] = { 0, 0, 0, opacities[idx]};
	tiles_touched[idx] = (rect_max_convex.y - rect_min_convex.y) * (rect_max_convex.x - rect_min_convex.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ normals,
	const float* __restrict__ offsets,
	const int* __restrict__ num_points_per_convex_view,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ delta,
	const float* __restrict__ sigma,
	const int* __restrict__ num_points_per_convex,
	const int* __restrict__ cumsum_of_points_per_convex,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ depths,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	/*
	ADDED FOR CONVEX PURPOSES ==========================================================================
	*/
	__shared__ float2 collected_normals[BLOCK_SIZE * MAX_NB_POINTS];
	__shared__ float collected_offsets[BLOCK_SIZE * MAX_NB_POINTS];
	__shared__ int collected_num_points_per_convex_view[BLOCK_SIZE];
	__shared__ float collected_delta[BLOCK_SIZE];
	__shared__ float collected_sigma[BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	/*
	===================================================================================================
	*/

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Convex data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_num_points_per_convex_view[block.thread_rank()] = num_points_per_convex_view[coll_id];
			collected_delta[block.thread_rank()] = delta[coll_id];
			collected_sigma[block.thread_rank()] = sigma[coll_id];
			collected_depths[block.thread_rank()] = depths[coll_id];
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			for (int k = 0; k < num_points_per_convex_view[coll_id]; k++) {
    			collected_normals[MAX_NB_POINTS * block.thread_rank() + k] = normals[cumsum_of_points_per_convex[coll_id] + k];
			}
			for (int k = 0; k < num_points_per_convex_view[coll_id]; k++) {
    			collected_offsets[MAX_NB_POINTS * block.thread_rank() + k] = offsets[cumsum_of_points_per_convex[coll_id] + k];
			}
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;


			float4 con_o = collected_conic_opacity[j];
			float distances[MAX_NB_POINTS]; // Max 4 distances as per collected_offsets Later, if we have more points per convex, this needs to be updated
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

			float alpha = min(0.99f, con_o.w * Cx); 
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Convex splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
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
	const float* colors,
	const float4* conic_opacity,
	const float* depths,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		normals,
		offsets,
		num_points_per_convex_view,
		points_xy_image,
		delta,
		sigma,
		num_points_per_convex,
		cumsum_of_points_per_convex,
		colors,
		conic_opacity,
		depths,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others);
}

void FORWARD::preprocess(int P, int D, int M,
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
	float2* means2D,
	float* depths,
	float* rgb,
	float4* conic_opacity,
	float* cov3Ds,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		convex_points,
		delta,
		sigma,
		num_points_per_convex,
		cumsum_of_points_per_convex,
		opacities,
		scaling,
		density_factor,
		shs,
		clamped,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		normals,
		offsets,
		num_points_per_convex_view,
		p_hom,
		p_w,
		p_proj,
		p_image,
		hull,
		indices,
		means2D,
		depths,
		rgb,
		conic_opacity,
		cov3Ds,
		grid,
		tiles_touched,
		prefiltered
		);
}