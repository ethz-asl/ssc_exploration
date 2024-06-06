#include <stdio.h>
#include <cuda_runtime.h>


__global__ void depth2Grid(double *  cam_info, double *  vox_info,  double * depth_data, double * vox_binary_GPU, double * depth2voxel_idx ){
  // Get camera information
  int frame_width = cam_info[0];
  //int frame_height = cam_info[1];
  double cam_K[9];
  for (int i = 0; i < 9; ++i)
    cam_K[i] = cam_info[i + 2];
  double cam_pose[16];
  for (int i = 0; i < 16; ++i)
    cam_pose[i] = cam_info[i + 11];

  // Get voxel volume parameters
  double vox_unit = vox_info[0];
  //double vox_margin = vox_info[1];
  int vox_size[3];
  for (int i = 0; i < 3; ++i)
    vox_size[i] = vox_info[i + 2];
  double vox_origin[3];
  for (int i = 0; i < 3; ++i)
    vox_origin[i] = vox_info[i + 5];


  // Get point in world coordinate
  int pixel_x = blockIdx.x;
  int pixel_y = threadIdx.x;

  double point_depth = depth_data[pixel_y * frame_width + pixel_x];

  double point_cam[3] = {0};
  point_cam[0] =  (pixel_x - cam_K[2])*point_depth/cam_K[0];
  point_cam[1] =  (pixel_y - cam_K[5])*point_depth/cam_K[4];
  point_cam[2] =  point_depth;

  double point_base[3] = {0};

  point_base[0] = cam_pose[0 * 4 + 0]* point_cam[0] + cam_pose[0 * 4 + 1]*  point_cam[1] + cam_pose[0 * 4 + 2]* point_cam[2];
  point_base[1] = cam_pose[1 * 4 + 0]* point_cam[0] + cam_pose[1 * 4 + 1]*  point_cam[1] + cam_pose[1 * 4 + 2]* point_cam[2];
  point_base[2] = cam_pose[2 * 4 + 0]* point_cam[0] + cam_pose[2 * 4 + 1]*  point_cam[1] + cam_pose[2 * 4 + 2]* point_cam[2];

  point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
  point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
  point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];


  //printf("vox_origin: %f,%f,%f\n",vox_origin[0],vox_origin[1],vox_origin[2]);
  // World coordinate to grid coordinate
  int z = (int)floor((point_base[0] - vox_origin[0])/vox_unit);
  int x = (int)floor((point_base[1] - vox_origin[1])/vox_unit);
  int y = (int)floor((point_base[2] - vox_origin[2])/vox_unit);
	
	//printf("calculating depth mappings");
  // mark vox_binary_GPU
  if( x >= 0 && x < vox_size[0] && y >= 0 && y < vox_size[1] && z >= 0 && z < vox_size[2]){
    int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
    vox_binary_GPU[vox_idx] = double(1.0);
    //printf("depth mapping at %d,%d,%d is %d\n", x,y,z,vox_idx);
    depth2voxel_idx[pixel_y * frame_width + pixel_x] = vox_idx;
  }
}

__global__ void calculate_occupancy_prob(double *  cam_info, double *  vox_info,  double * depth_data, double * vox_log_GPU ){
  // Get camera information
  int frame_width = cam_info[0];
  //int frame_height = cam_info[1];
  double cam_K[9];
  for (int i = 0; i < 9; ++i)
    cam_K[i] = cam_info[i + 2];
  double cam_pose[16];
  for (int i = 0; i < 16; ++i)
    cam_pose[i] = cam_info[i + 11];

  // Get voxel volume parameters
  double vox_unit = vox_info[0];
  //double vox_margin = vox_info[1];
  int vox_size[3];
  for (int i = 0; i < 3; ++i)
    vox_size[i] = vox_info[i + 2];
  double vox_origin[3];
  for (int i = 0; i < 3; ++i)
    vox_origin[i] = vox_info[i + 5];


  // Get point in world coordinate
  int pixel_x = blockIdx.x;
  int pixel_y = threadIdx.x;

  double point_depth = depth_data[pixel_y * frame_width + pixel_x];

  double point_cam[3] = {0};
  point_cam[0] =  (pixel_x - cam_K[2])*point_depth/cam_K[0];
  point_cam[1] =  (pixel_y - cam_K[5])*point_depth/cam_K[4];
  point_cam[2] =  point_depth;

  double point_base[3] = {0};

  point_base[0] = cam_pose[0 * 4 + 0]* point_cam[0] + cam_pose[0 * 4 + 1]*  point_cam[1] + cam_pose[0 * 4 + 2]* point_cam[2];
  point_base[1] = cam_pose[1 * 4 + 0]* point_cam[0] + cam_pose[1 * 4 + 1]*  point_cam[1] + cam_pose[1 * 4 + 2]* point_cam[2];
  point_base[2] = cam_pose[2 * 4 + 0]* point_cam[0] + cam_pose[2 * 4 + 1]*  point_cam[1] + cam_pose[2 * 4 + 2]* point_cam[2];

  point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
  point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
  point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];


  //printf("vox_origin: %f,%f,%f\n",vox_origin[0],vox_origin[1],vox_origin[2]);
  // World coordinate to grid coordinate
  int z = (int)floor((point_base[0] - vox_origin[0])/vox_unit);
  int x = (int)floor((point_base[1] - vox_origin[1])/vox_unit);
  int y = (int)floor((point_base[2] - vox_origin[2])/vox_unit);
	
	//printf("calculating depth mappings");
  // mark vox_binary_GPU
  float prob_occ = 0.7f;
  float max_prob = 0.97f;
  
  if( x >= 0 && x < vox_size[0] && y >= 0 && y < vox_size[1] && z >= 0 && z < vox_size[2]){
    int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
    vox_log_GPU[vox_idx] = min(vox_log_GPU[vox_idx] + log(prob_occ/(1-prob_occ)), log(max_prob/(1-max_prob)));
  }
}

__global__
void tsdfTransform( double * vox_info, double * vox_tsdf){

  int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (vox_idx >= vox_info[0+2] * vox_info[1+2] * vox_info[2+2]){
    return;
  }
  double value = double(vox_tsdf[vox_idx]);


  double sign;
  if (abs(value) < 0.001)
    sign = 1;
  else
    sign = value/abs(value);

  vox_tsdf[vox_idx] = sign*(max(0.001,(1.0-abs(value))));
}


__global__ void SquaredDistanceTransform(double * cam_info, double * vox_info, double * depth_data, double * vox_binary_GPU , double * vox_tsdf) {
  // Get voxel volume parameters
  double vox_unit = vox_info[0];
  double vox_margin = vox_info[1];
  int vox_size[3];
  for (int i = 0; i < 3; ++i)
    vox_size[i] = vox_info[i + 2];
  double vox_origin[3];
  for (int i = 0; i < 3; ++i)
    vox_origin[i] = vox_info[i + 5];

  int frame_width = cam_info[0];
  int frame_height = cam_info[1];
  double cam_K[9];
  for (int i = 0; i < 9; ++i)
    cam_K[i] = cam_info[i + 2];
  double cam_pose[16];
  for (int i = 0; i < 16; ++i)
    cam_pose[i] = cam_info[i + 11];


  int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (vox_idx >= vox_size[0] * vox_size[1] * vox_size[2]){
    return;
  }

  int z = double((vox_idx / ( vox_size[0] * vox_size[1]))%vox_size[2]) ;
  int y = double((vox_idx / vox_size[0]) % vox_size[1]);
  int x = double(vox_idx % vox_size[0]);
  int search_region = (int)round(vox_margin/vox_unit);

  if (vox_binary_GPU[vox_idx] >0 ){
    vox_tsdf[vox_idx] = 0;
    return;
  }

  // Get point in world coordinates (XYZ) from grid coordinates (YZX)  
  double point_base[3] = {0};
  point_base[0] = double(z) * vox_unit + vox_origin[0];
  point_base[1] = double(x) * vox_unit + vox_origin[1];
  point_base[2] = double(y) * vox_unit + vox_origin[2];

  // Get point in current camera coordinates
  double point_cam[3] = {0};
  point_base[0] = point_base[0] - cam_pose[0 * 4 + 3];
  point_base[1] = point_base[1] - cam_pose[1 * 4 + 3];
  point_base[2] = point_base[2] - cam_pose[2 * 4 + 3];
  point_cam[0] = cam_pose[0 * 4 + 0] * point_base[0] + cam_pose[1 * 4 + 0] * point_base[1] + cam_pose[2 * 4 + 0] * point_base[2];
  point_cam[1] = cam_pose[0 * 4 + 1] * point_base[0] + cam_pose[1 * 4 + 1] * point_base[1] + cam_pose[2 * 4 + 1] * point_base[2];
  point_cam[2] = cam_pose[0 * 4 + 2] * point_base[0] + cam_pose[1 * 4 + 2] * point_base[1] + cam_pose[2 * 4 + 2] * point_base[2];
  if (point_cam[2] <= 0){
    return;
  }

  // Project point to 2D
  int pixel_x = roundf(cam_K[0] * (point_cam[0] / point_cam[2]) + cam_K[2]);
  int pixel_y = roundf(cam_K[4] * (point_cam[1] / point_cam[2]) + cam_K[5]);
  if (pixel_x < 0 || pixel_x >= frame_width || pixel_y < 0 || pixel_y >= frame_height){ // outside FOV
    return;
  }


  // Get depth
  double point_depth = depth_data[pixel_y * frame_width + pixel_x];
  if (point_depth < double(0.5f) || point_depth > double(8.0f)){
    return;
  }
  if (roundf(point_depth) == 0){ // mising depth
    vox_tsdf[vox_idx] = double(-1.0);
    return;
  }


  // Get depth difference
  double sign;
  if (abs(point_depth - point_cam[2]) < 0.0001){
    sign = 1; // avoid NaN
  }else{
    sign = (point_depth - point_cam[2])/abs(point_depth - point_cam[2]);
  }
  vox_tsdf[vox_idx] = double(sign);
  for (int iix = max(0,x-search_region); iix < min((int)vox_size[0],x+search_region+1); iix++){
    for (int iiy = max(0,y-search_region); iiy < min((int)vox_size[1],y+search_region+1); iiy++){
      for (int iiz = max(0,z-search_region); iiz < min((int)vox_size[2],z+search_region+1); iiz++){
        int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
        if (vox_binary_GPU[iidx] > 0){
          double xd = abs(x - iix);
          double yd = abs(y - iiy);
          double zd = abs(z - iiz);
          double tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/(double)search_region;
          if (tsdf_value < abs(vox_tsdf[vox_idx])){
            vox_tsdf[vox_idx] = double(tsdf_value*sign);
          }
        }
      }
    }
  }

}

void ComputeTSDF(double * cam_info_CPU, double * vox_info_CPU,
                 double * depth_data_CPU,  double * vox_tsdf_CPU, double * depth_mapping_idxs_CPU, double * occupancy) {

  int frame_width  = cam_info_CPU[0];
  int frame_height = cam_info_CPU[1];
  int vox_size[3];
  for (int i = 0; i < 3; ++i)
    vox_size[i] = vox_info_CPU[i + 2];
  int num_crop_voxels = vox_size[0] * vox_size[1] * vox_size[2];


  // allocate voxel occupancy
  double * vox_binary_CPU = (double*)malloc((int)(num_crop_voxels * sizeof(double)));
	memset(vox_binary_CPU, 0, num_crop_voxels * sizeof(double));

  //  Copy from host to device
  double *  vox_binary_GPU;
  cudaMalloc(&vox_binary_GPU, num_crop_voxels * sizeof(double));
  cudaMemcpy(vox_binary_GPU, vox_binary_CPU, num_crop_voxels * sizeof(double), cudaMemcpyHostToDevice);
  //GPU_set_zeros(num_crop_voxels, vox_binary_GPU);

  // copy cam info to gpu
  double * cam_info_GPU;
  cudaMalloc(&cam_info_GPU, 27 * sizeof(double));
  cudaMemcpy(cam_info_GPU, cam_info_CPU, 27 * sizeof(double), cudaMemcpyHostToDevice);

  // copy vox info to gpu
  double * vox_info_GPU;
  cudaMalloc(&vox_info_GPU, 8 * sizeof(double));
  cudaMemcpy(vox_info_GPU, vox_info_CPU, 8 * sizeof(double), cudaMemcpyHostToDevice);

  //copy depth data to gpu
  double * depth_data_GPU;
  cudaMalloc(&depth_data_GPU, frame_height * frame_width * sizeof(double));
  cudaMemcpy(depth_data_GPU, depth_data_CPU, frame_height * frame_width * sizeof(double), cudaMemcpyHostToDevice);

  // copy depth mapping to gpu
  double * depth_mapping_idxs_GPU;
  cudaMalloc(&depth_mapping_idxs_GPU, frame_height * frame_width * sizeof(double));
  cudaMemcpy(depth_mapping_idxs_GPU, depth_mapping_idxs_CPU, frame_height * frame_width * sizeof(double), cudaMemcpyHostToDevice);

  // copy voxel tsd to gpu
  double * vox_tsdf_GPU;
  cudaMalloc(&vox_tsdf_GPU, num_crop_voxels * sizeof(double));
  cudaMemcpy(vox_tsdf_GPU, vox_tsdf_CPU, num_crop_voxels * sizeof(double), cudaMemcpyHostToDevice);

  // from depth map to binaray voxel representation 
  depth2Grid<<<frame_width,frame_height>>>(cam_info_GPU, vox_info_GPU, depth_data_GPU, vox_binary_GPU, depth_mapping_idxs_GPU);

  //cudaGetLastError();

  // distance transform 
  int THREADS_NUM = 512;  // 1024
  int BLOCK_NUM = int((num_crop_voxels + size_t(THREADS_NUM) - 1) / THREADS_NUM);

  SquaredDistanceTransform <<< BLOCK_NUM, THREADS_NUM >>> (cam_info_GPU, vox_info_GPU, depth_data_GPU, vox_binary_GPU, vox_tsdf_GPU);
  
  // invert TSDF
  tsdfTransform <<< BLOCK_NUM, THREADS_NUM >>> (vox_info_GPU, vox_tsdf_GPU);
  
  // copy computed TSDF back to CPU
  cudaMemcpy(vox_tsdf_CPU, vox_tsdf_GPU, num_crop_voxels * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(depth_mapping_idxs_CPU, depth_mapping_idxs_GPU, frame_height * frame_width * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(occupancy, vox_binary_GPU,  num_crop_voxels * sizeof(double), cudaMemcpyDeviceToHost);
  
  // deallocation
  cudaFree(vox_info_GPU);
  cudaFree(cam_info_GPU);
  cudaFree(depth_data_GPU);
  cudaFree(depth_mapping_idxs_GPU);
  cudaFree(vox_tsdf_GPU);
  cudaFree(vox_binary_GPU);
  free(vox_binary_CPU);
}

void calculateOccupancyProb(double * cam_info_CPU, double * vox_info_CPU,
                 double * depth_data_CPU, double * log_odds_occupancy) {

  int frame_width  = cam_info_CPU[0];
  int frame_height = cam_info_CPU[1];
  int vox_size[3];
  for (int i = 0; i < 3; ++i)
    vox_size[i] = vox_info_CPU[i + 2];
  int num_crop_voxels = vox_size[0] * vox_size[1] * vox_size[2];


  // allocate voxel occupancy
  double * vox_prob_CPU = (double*)malloc((int)(num_crop_voxels * sizeof(double)));
	memset(vox_prob_CPU, 0, num_crop_voxels * sizeof(double));

  //  Copy from host to device
  double *  vox_prob_GPU;
  cudaMalloc(&vox_prob_GPU, num_crop_voxels * sizeof(double));
  cudaMemcpy(vox_prob_GPU, vox_prob_CPU, num_crop_voxels * sizeof(double), cudaMemcpyHostToDevice);
  //GPU_set_zeros(num_crop_voxels, vox_binary_GPU);

  // copy cam info to gpu
  double * cam_info_GPU;
  cudaMalloc(&cam_info_GPU, 27 * sizeof(double));
  cudaMemcpy(cam_info_GPU, cam_info_CPU, 27 * sizeof(double), cudaMemcpyHostToDevice);

  // copy vox info to gpu
  double * vox_info_GPU;
  cudaMalloc(&vox_info_GPU, 8 * sizeof(double));
  cudaMemcpy(vox_info_GPU, vox_info_CPU, 8 * sizeof(double), cudaMemcpyHostToDevice);

  //copy depth data to gpu
  double * depth_data_GPU;
  cudaMalloc(&depth_data_GPU, frame_height * frame_width * sizeof(double));
  cudaMemcpy(depth_data_GPU, depth_data_CPU, frame_height * frame_width * sizeof(double), cudaMemcpyHostToDevice);


  // from depth map to binary voxel representation 
  calculate_occupancy_prob<<<frame_width,frame_height>>>(cam_info_GPU, vox_info_GPU, depth_data_GPU, vox_prob_GPU);

  // copy computed log odds back to CPU
  cudaMemcpy(log_odds_occupancy, vox_prob_GPU,  num_crop_voxels * sizeof(double), cudaMemcpyDeviceToHost);
  
  // deallocation
  cudaFree(vox_info_GPU);
  cudaFree(cam_info_GPU);
  cudaFree(depth_data_GPU);
  cudaFree(vox_prob_GPU);
  free(vox_prob_CPU);
}
//int main() {return 0;}
