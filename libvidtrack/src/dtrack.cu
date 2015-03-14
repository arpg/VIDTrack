#include <stdio.h>
//#include <limits>
//#include <float.h>
//#include <string>

//#include <thrust/count.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <vidtrack/dtrack.cuh>


///////////////////////////////////////////////////////////////////////////
///  Device Functions
///////////////////////////////////////////////////////////////////////////
__global__ void dEstimate()
{
//  int px = blockDim.x*blockIdx.x + threadIdx.x;
//  int py = blockDim.y*blockIdx.y + threadIdx.y;

}



///////////////////////////////////////////////////////////////////////////
///  Host Functions
///////////////////////////////////////////////////////////////////////////
cuDTrack::cuDTrack(unsigned int max_height, unsigned int max_width)
{
  cudaMalloc((void**)&ref_image_, max_width*max_height*sizeof(unsigned char));
  cudaMalloc((void**)&ref_depth_, max_width*max_height*sizeof(float));
  cudaMalloc((void**)&live_image_, max_width*max_height*sizeof(unsigned char));

  // Storage for Least Squares System
  // 21 for upper diagonal of LHS, 6 for RHS, 1 for squared_error, 1 for num_obs.
  cudaMalloc((void**)&lss_, max_width*max_height*sizeof(float)*29);

#if 0
  cudaMalloc((void**)&lss_.jacobian, max_width*max_height*sizeof(float)*6);
  cudaMalloc((void**)&lss_.error, max_width*max_height*sizeof(float));
  cudaMalloc((void**)&lss_.weight, max_width*max_height*sizeof(float));
  cudaMalloc((void**)&lss_.obs, max_width*max_height*sizeof(bool));
#endif
}

///////////////////////////////////////////////////////////////////////////
cuDTrack::~cuDTrack()
{
  if (ref_image_ != NULL) {
    cudaFree(ref_image_);
  }
  if (ref_depth_ != NULL) {
    cudaFree(ref_depth_);
  }
  if (live_image_ != NULL) {
    cudaFree(live_image_);
  }
  if (lss_ != NULL) {
    cudaFree(lss_);
  }
}

///////////////////////////////////////////////////////////////////////////
void cuDTrack::_LaunchEstimate(unsigned int image_height, unsigned int image_width)
{
  dim3 gridSize, blockSize;
  gridSize.x = _GCD(image_width, 32);
  gridSize.y = _GCD(image_height, 32);
  blockSize.x = image_width / gridSize.x;
  blockSize.y = image_height / gridSize.y;

  dEstimate<<<gridSize, blockSize>>>();

#if 0
  thrust::device_ptr<float> ptr = thrust::device_pointer_cast(lss_.error);
  float max = thrust::reduce(ptr, &ptr[image_height*image_width], -1.0f,
      thrust::maximum<float>());
#endif
  _CheckErrors("Estimate");
}

///////////////////////////////////////////////////////////////////////////
int cuDTrack::_GCD(int a, int b)
{
  if (a % b == 0) {
    return b; //(a >= b) ? b : a;
  }
  return _GCD(a, a % b);
}

///////////////////////////////////////////////////////////////////////////
void cuDTrack::_CheckErrors(const char* label)
{
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error [%s]: %s\n", label, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

///////////////////////////////////////////////////////////////////////////
unsigned int cuDTrack::_CheckMemory()
{
  cudaError_t err;

  size_t avail;
  size_t total;

  err = cudaMemGetInfo(&avail, &total);
  if (err != cudaSuccess) {
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
      fprintf(stderr, "CheckMemoryCUDA: Unable to reset device: %s \n", cudaGetErrorString(err));
    } else {
      err = cudaMemGetInfo(&avail, &total);
    }
  }

  if (err == cudaSuccess) {
    size_t used = total - avail;
    const unsigned bytes_per_mb = 1024*1000;
    fprintf(stdout, "- Checking CUDA Memory: Total = %lu,  Available = %lu, Used = %lu\n",
            total/bytes_per_mb, avail/bytes_per_mb, used/bytes_per_mb);
    return avail/bytes_per_mb;
  } else {
    fprintf(stderr, "CheckMemoryCUDA: There is an irrecoverable error: %s\n", cudaGetErrorString(err));
    fprintf(stderr, "CheckMemoryCUDA: There is an irrecoverable error: %s\n", cudaGetErrorString(err));
  }
  return 0;
}

