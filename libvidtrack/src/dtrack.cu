#include <limits>
#include <float.h>
#include <stdio.h>
#include <string>

#ifndef HAVE_CLANG
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#endif

#include "idtam.cuh"
#include "math.cuh"


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///  Device Functions


///////////////////////////////////////////////////////////////////////////
__global__ void dTransformVolume3D(float* vol, float* temp_vol, float* frames,
                                   float* temp_frames, float* T, float* K,
                                   float* Kinv,int width, int height, int levels,
                                   float ksi_min, float ksi_max)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;
  int pz = blockDim.z*blockIdx.z + threadIdx.z;

  const float delta = (ksi_max-ksi_min)/levels;
  const float inv_depth = pz*delta + ksi_min;

  float ray[4];
  float ray_t[4];

  ray[0] = px;
  ray[1] = py;
  ray[2] = 1;
  ray[3] = 1;

  ray_t[0] = Kinv[0]*ray[0] + Kinv[3]*ray[1] + Kinv[6]*ray[2];
  ray_t[1] = Kinv[1]*ray[0] + Kinv[4]*ray[1] + Kinv[7]*ray[2];
  ray_t[2] = Kinv[2]*ray[0] + Kinv[5]*ray[1] + Kinv[8]*ray[2];

  ray_t[0] /= inv_depth;
  ray_t[1] /= inv_depth;
  ray_t[2] /= inv_depth;
  ray_t[3] = 1;

  ray[0] = T[0]*ray_t[0] + T[4]*ray_t[1] + T[8]*ray_t[2] + T[12]*ray_t[3];
  ray[1] = T[1]*ray_t[0] + T[5]*ray_t[1] + T[9]*ray_t[2] + T[13]*ray_t[3];
  ray[2] = T[2]*ray_t[0] + T[6]*ray_t[1] + T[10]*ray_t[2] + T[14]*ray_t[3];
  ray[3] = T[3]*ray_t[0] + T[7]*ray_t[1] + T[11]*ray_t[2] + T[15]*ray_t[3];

  ray[0] /= ray[3];
  ray[1] /= ray[3];
  ray[2] /= ray[3];

  ray_t[0] = K[0]*ray[0] + K[3]*ray[1] + K[6]*ray[2];
  ray_t[1] = K[1]*ray[0] + K[4]*ray[1] + K[7]*ray[2];
  ray_t[2] = K[2]*ray[0] + K[5]*ray[1] + K[8]*ray[2];

  ray[0] = ray_t[0] / ray_t[2];
  ray[1] = ray_t[1] / ray_t[2];

  ray[2] = 1.0f / ray[2];
  ray[2] -= ksi_min;
  ray[2] /= delta;

  if ((ray[0] > 0) && (ray[0] < width-1) && (ray[1] > 0) && (ray[1] < height-1)
      && (ray[2] > 0) && (ray[2] < levels-1)) {
    const unsigned int index = px + py*width + pz*width*height;
    temp_vol[index] = Trilinear(vol, ray[0], ray[1], ray[2], width,
        height, levels, ksi_min, ksi_max);
    temp_frames[index] = Trilinear(frames, ray[0], ray[1], ray[2], width,
        height, levels, ksi_min, ksi_max);
  }
  //    else ( (ray[0] >= 0) && (ray[0] < width) && (ray[1] >= 0) && (ray[1] < height) && (ray[2] >= 0) && (ray[2] < levels)) {  // Ensure that the borders are caught
  //        int x, y, z;
  //        TempVol[px + py*width + pz*width*height] = Trilinear<float>( CostVol, ray[0], ray[1], ray[2], width, height, levels, ksi_min, ksi_max);
  //        TempFrames[px + py*width + pz*width*height] = Trilinear<float> ( Frames, ray[0], ray[1], ray[2], width, height, levels, ksi_min, ksi_max);
  //    }
}

///////////////////////////////////////////////////////////////////////////
__global__ void dTransformVolume2D(float* vol, float* temp_vol,
                                   float* frames, float* temp_frames,
                                   float* T, float* K, float* Kinv,
                                   int width, int height, int levels,
                                   float ksi_min, float ksi_max)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;

  for (int pz = 0; pz < levels; ++pz) {
    const float delta = (ksi_max-ksi_min)/levels;
    const float inv_depth = pz*delta + ksi_min;

    float ray[4];
    float ray_t[4];

    ray[0] = px;
    ray[1] = py;
    ray[2] = 1;
    ray[3] = 1;

    ray_t[0] = Kinv[0]*ray[0] + Kinv[3]*ray[1] + Kinv[6]*ray[2];
    ray_t[1] = Kinv[1]*ray[0] + Kinv[4]*ray[1] + Kinv[7]*ray[2];
    ray_t[2] = Kinv[2]*ray[0] + Kinv[5]*ray[1] + Kinv[8]*ray[2];

    ray_t[0] /= inv_depth;
    ray_t[1] /= inv_depth;
    ray_t[2] /= inv_depth;
    ray_t[3] = 1;

    ray[0] = T[0]*ray_t[0] + T[4]*ray_t[1] + T[8]*ray_t[2] + T[12]*ray_t[3];
    ray[1] = T[1]*ray_t[0] + T[5]*ray_t[1] + T[9]*ray_t[2] + T[13]*ray_t[3];
    ray[2] = T[2]*ray_t[0] + T[6]*ray_t[1] + T[10]*ray_t[2] + T[14]*ray_t[3];
    ray[3] = T[3]*ray_t[0] + T[7]*ray_t[1] + T[11]*ray_t[2] + T[15]*ray_t[3];

    ray[0] /= ray[3];
    ray[1] /= ray[3];
    ray[2] /= ray[3];

    ray_t[0] = K[0]*ray[0] + K[3]*ray[1] + K[6]*ray[2];
    ray_t[1] = K[1]*ray[0] + K[4]*ray[1] + K[7]*ray[2];
    ray_t[2] = K[2]*ray[0] + K[5]*ray[1] + K[8]*ray[2];

    ray[0] = ray_t[0] / ray_t[2];
    ray[1] = ray_t[1] / ray_t[2];

    ray[2] = 1.0f / ray[2];
    ray[2] -= ksi_min;
    ray[2] /= delta;

    if ((ray[0] > 0) && (ray[0] < width-1) && (ray[1] > 0) && (ray[1] < height-1)
        && (ray[2] > 0) && (ray[2] < levels-1)) {
      const unsigned int index = px + py*width + pz*width*height;
      temp_vol[index] = Trilinear(vol, ray[0], ray[1], ray[2], width,
          height, levels, ksi_min, ksi_max);
      temp_frames[index] = Trilinear(frames, ray[0], ray[1], ray[2], width,
          height, levels, ksi_min, ksi_max);
    }
  }
  //    else ( (ray[0] >= 0) && (ray[0] < width) && (ray[1] >= 0) && (ray[1] < height) && (ray[2] >= 0) && (ray[2] < levels)) {  // Ensure that the borders are caught
  //        int x, y, z;
  //        TempVol[px + py*width + pz*width*height] = Trilinear<float>( CostVol, ray[0], ray[1], ray[2], width, height, levels, ksi_min, ksi_max);
  //        TempFrames[px + py*width + pz*width*height] = Trilinear<float> ( Frames, ray[0], ray[1], ray[2], width, height, levels, ksi_min, ksi_max);
  //    }
}

///////////////////////////////////////////////////////////////////////////
// TODO(jmf) Try a 2D launch.
__global__ void dScaleVolume(const float* cost, const float* frames,
                             float* scaled_cost, int width, int height)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;
  int pz = blockDim.z*blockIdx.z + threadIdx.z;

  const int index = px + py*width + pz*width*height;
  if (frames[index] == 0) {
    scaled_cost[index] = FLT_MAX;
  } else {
    scaled_cost[index] = cost[index] / frames[index];
  }
}

///////////////////////////////////////////////////////////////////////////
__global__ void dSimpleMinimum(float* scaled_cost, float* d, int width,
                               int height, int levels, float kmin, float kmax)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;
  const unsigned int index = px + py*width;

  int min_level = 0;
  float min_cost = scaled_cost[index];
  for (int ii = 1; ii < levels; ++ii) {
    float cost = scaled_cost[index + ii*width*height];

    if (cost < min_cost) {
      min_cost = cost;
      min_level = ii;
    }
  }

  if (min_cost != FLT_MAX) {
    // Subvoxel refinement.
    float step = 0.0;
    const float delta = (kmax - kmin) / levels;
    // TODO(jmf) Do central differences??
    if (min_level < levels - 2) {
      float Ep = scaled_cost[index + (min_level+1)*width*height];
      float Ep2 = scaled_cost[index + (min_level+2)*width*height];
      float E0 = min_cost;
      float d1 = (Ep - E0);
      float d2 = 2*(Ep2 + E0 - 2*Ep);
      step = (d1 / d2)*delta;

      if (isnan(step) || (step*step > delta*delta)) {
        step = 0.0;
      }
    }

    d[index] = (min_level*delta + kmin) - step;
  } else {
    // TODO(jmf) Set to NAN?
//    d[index] = 0.0f / 0.0f;
    d[index] = kmin + (kmax - kmin) / 2.0f;
  }
}

/////////////////////////////////////////////////////////////////////////////
__global__ void dUpdateQ(float* Q, float* D, float* weight, float sq,
                         float epsilon, int width, int height)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;

  const float g = weight[px + width*py];

  float2 q;
  q.x = Q[px + py*width];
  q.y = Q[px + py*width + width*height];
  float2 depthGrad = grad(D, px, py, width, height);
  q.x = (q.x + sq * g * depthGrad.x) / (1.0f + sq*epsilon);
  q.y = (q.y + sq * g * depthGrad.y) / (1.0f + sq*epsilon);
  q = Pi(q);
  Q[px + py*width] = q.x;
  Q[px + py*width + width*height] = q.y;
}

///////////////////////////////////////////////////////////////////////////
__global__ void dUpdateD(float* D, float* A, float* Q, float* weight,
                         float inv_theta, float sd, int width, int height)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;

  float g = weight[px + py*width];
  float d = D[px + py*width];
  float a = A[px + py*width];

  float div = Div(Q, px, py, width, height);
  D[px + py*width] = (d + sd*(g*div + a*inv_theta)) / (1.0f + sd*inv_theta);
}

///////////////////////////////////////////////////////////////////////////
__global__ void dUpdateA(float* scaled_vol, float* diff, float* A, float* D,
                         float theta, float lambda, int width, int height,
                         int levels, float kmin, float kmax)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;

  const int index = px + py*width;
  const int imageSize = height*width;
  const float delta = (kmax - kmin) / levels;

  const float d = D[index];
  const float inv2theta = 1.0f / (2.0f*theta);

  int imin, imax;

  // Cuadratic speed-up.
  float r = sqrtf(2*theta*lambda*diff[px + py*width]);
  if ((r == 0) || (isnan(r))) {
    r = kmax;
  }

  imin = floor(((d - r) - kmin) / delta);
  if (imin < 0) {
    imin = 0;
  }

  imax = ceil(((d + r) - kmin) / delta);
  if (imax > levels) {
    imax = levels;
  }

  // Look for minimum cost.
  int minId = imin;
  float min = FLT_MAX;
  for (int ii = imin; ii < imax; ++ii ) {
    float a = ii*delta + kmin;
    float ref = (((d - a)*(d - a))*inv2theta) +
        lambda*(scaled_vol[index + ii*imageSize]);

    if ((ref < min) && (!isnan(ref))) {
      min = ref;
      minId = ii;
    }
  }

  if (min != FLT_MAX) {
    float a = delta*minId + kmin;
    // Subvoxel refinement.
    if (0 < minId && minId < levels - 1) {
      // Newton Step
      const float dl = (minId-1)*delta + kmin;
      const float dr = (minId+1)*delta + kmin;
      const float sl = inv2theta*(d-dl)*(d-dl) + lambda*scaled_vol[index + (minId-1)*imageSize];
      const float sr = inv2theta*(d-dr)*(d-dr) + lambda*scaled_vol[index + (minId+1)*imageSize];

      const float subpixdisp = a - delta*(sr-sl) / (2.0f*(sr-2.0f*min+sl));

      // Check that minima is sensible. Otherwise assume bad data.
      if (dl < subpixdisp && subpixdisp < dr) {
        a = subpixdisp;
      }
    }
    A[px + py*width] = a;
  } else {
    // TODO(jmf) Set this to NAN?
    A[px + py*width] = kmin + (kmax - kmin) / 2.0f;
  }
}

///////////////////////////////////////////////////////////////////////////
__global__ void dAddToCostVolume(float* cost, float* frames, float* Ir,
                                 float* Im, int width, int height,
                                 int levels, float kmin, float kmax, float* K,
                                 float* Kinv, float* T, float omega)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;
  int pz = blockDim.z*blockIdx.z + threadIdx.z;

  const int image_size = width*height;

  const float ref = Ir[px + py*width];

  float ray[4];
  float ray_t[4];

  const float inv_depth = pz*(kmax - kmin)/levels + kmin;

  ray[0] = px;
  ray[1] = py;
  ray[2] = 1;
  ray[3] = 1;

  ray_t[0] = Kinv[0]*ray[0] + Kinv[3]*ray[1] + Kinv[6]*ray[2];
  ray_t[1] = Kinv[1]*ray[0] + Kinv[4]*ray[1] + Kinv[7]*ray[2];
  ray_t[2] = Kinv[2]*ray[0] + Kinv[5]*ray[1] + Kinv[8]*ray[2];

  ray_t[0] /= inv_depth;
  ray_t[1] /= inv_depth;
  ray_t[2] /= inv_depth;
  ray_t[3] = 1;

  ray[0] = T[0]*ray_t[0] + T[4]*ray_t[1] + T[8]*ray_t[2] + T[12]*ray_t[3];
  ray[1] = T[1]*ray_t[0] + T[5]*ray_t[1] + T[9]*ray_t[2] + T[13]*ray_t[3];
  ray[2] = T[2]*ray_t[0] + T[6]*ray_t[1] + T[10]*ray_t[2] + T[14]*ray_t[3];
  ray[3] = T[3]*ray_t[0] + T[7]*ray_t[1] + T[11]*ray_t[2] + T[15]*ray_t[3];

  ray[0] /= ray[3];
  ray[1] /= ray[3];
  ray[2] /= ray[3];

  ray_t[0] = K[0]*ray[0] + K[3]*ray[1] + K[6]*ray[2];
  ray_t[1] = K[1]*ray[0] + K[4]*ray[1] + K[7]*ray[2];
  ray_t[2] = K[2]*ray[0] + K[5]*ray[1] + K[8]*ray[2];

  ray[0] = ray_t[0]/ray_t[2];
  ray[1] = ray_t[1]/ray_t[2];

  if (InBounds(ray[0], ray[1], width-1, height-1)) {
    float comp = bilinear(Im, ray[0], ray[1], width);
    if (isnan(comp)) {
      return;
    }
    comp = ref - comp;
    if (comp < 0) {
      comp = 0 - comp;
    }
    cost[px + width*py + image_size*pz] = omega*cost[px + width*py + image_size*pz] + comp;
    frames[px + width*py + image_size*pz] = omega*frames[px + width*py + image_size*pz] + 1.0f;
  }
}

///////////////////////////////////////////////////////////////////////////
__global__ void dAddLayerToCostVolume(float* cost, float* frames, float* Ir,
                                      unsigned char* Ir_mask, float* Im,  int id,
                                      int width, int height, int levels,
                                      float kmin, float kmax, float* K,
                                      float* Kinv, float* T, float omega)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;
  int pz = blockDim.z*blockIdx.z + threadIdx.z;

  const int index = px + py*width;

  if (Ir_mask[index] != id) {
    return;
  }

  const float ref = Ir[index];
  const int image_size = width*height;

  float ray[4];
  float ray_t[4];

  const float inv_depth = pz*(kmax - kmin)/levels + kmin;

  ray[0] = px;
  ray[1] = py;
  ray[2] = 1;
  ray[3] = 1;

  ray_t[0] = Kinv[0]*ray[0] + Kinv[3]*ray[1] + Kinv[6]*ray[2];
  ray_t[1] = Kinv[1]*ray[0] + Kinv[4]*ray[1] + Kinv[7]*ray[2];
  ray_t[2] = Kinv[2]*ray[0] + Kinv[5]*ray[1] + Kinv[8]*ray[2];

  ray_t[0] /= inv_depth;
  ray_t[1] /= inv_depth;
  ray_t[2] /= inv_depth;
  ray_t[3] = 1;

  ray[0] = T[0]*ray_t[0] + T[4]*ray_t[1] + T[8]*ray_t[2] + T[12]*ray_t[3];
  ray[1] = T[1]*ray_t[0] + T[5]*ray_t[1] + T[9]*ray_t[2] + T[13]*ray_t[3];
  ray[2] = T[2]*ray_t[0] + T[6]*ray_t[1] + T[10]*ray_t[2] + T[14]*ray_t[3];
  ray[3] = T[3]*ray_t[0] + T[7]*ray_t[1] + T[11]*ray_t[2] + T[15]*ray_t[3];

  ray[0] /= ray[3];
  ray[1] /= ray[3];
  ray[2] /= ray[3];

  ray_t[0] = K[0]*ray[0] + K[3]*ray[1] + K[6]*ray[2];
  ray_t[1] = K[1]*ray[0] + K[4]*ray[1] + K[7]*ray[2];
  ray_t[2] = K[2]*ray[0] + K[5]*ray[1] + K[8]*ray[2];

  ray[0] = ray_t[0]/ray_t[2];
  ray[1] = ray_t[1]/ray_t[2];

  if (InBounds(ray[0], ray[1], width-1, height-1)) {
    float comp = bilinear(Im, ray[0], ray[1], width);
    if (isnan(comp)) {
      return;
    }
    comp = ref - comp;
    if (comp < 0) {
      comp = 0 - comp;
    }
    cost[px + width*py + image_size*pz] = omega*cost[px + width*py + image_size*pz] + comp;
    frames[px + width*py + image_size*pz] = omega*frames[px + width*py + image_size*pz] + 1.0f;
  }
}

///////////////////////////////////////////////////////////////////////////
__global__ void dInitWeights(float* weight, float* image, float alpha,
                             float beta, int width, int height)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;

  float2 g;
  g.x = g.y = 0.0f;
  if (px > 0 && px < width-1) {
    g.x = (image[px+1 + py*width] - image[px-1 + py*width]) / 2.0f;
  }
  if (py > 0 && py < height-1) {
    g.y = (image[px + (py+1)*width] - image[px + (py-1)*width]) / 2.0f;
  }

  const float mag = sqrtf(g.x*g.x + g.y*g.y);
  weight[px + py*width] = expf(-alpha*powf(mag, beta));
}

///////////////////////////////////////////////////////////////////////////
__global__ void dRescaleVolume(float* new_vol, float* vol, int height,
                               int width, int levels, float kmin_new,
                               float kmax_new, float kmin_old, float kmax_old)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;
  int pz = blockDim.z*blockIdx.z + threadIdx.z;

  const float delta_new = (kmax_new - kmin_new) / levels;
  const float inv_depth = kmin_new + pz*delta_new; // new inverse depth at level pz

  float toRet = 0;

  // Check if new inv_depth falls in old volume ...
  if ((inv_depth <= kmax_old) && (inv_depth >= kmin_old)) {
    //    // Check if shrinking or expanding ...
    //    if (delta_new <= delta_old) {
    //      // Shrinking...
    //      toRet = dInterpolate(Vol, inv_depth, kmin_old, kmax_old, levels);
    //    } else {
    //      // Expanding...
    //      float delta = (delta_new / delta_old);
    //      float sum = 0;
    //      int n = 0;
    //      for (float l = -delta/2.0f; l <= delta/2.0f; l += delta_old) {
    //        float local_idx = l + inv_depth;
    //        if ((local_idx <= kmax_old) && (local_idx >= kmin_old)) {
    //          sum += dInterpolate(Vol, local_idx, kmin_old, kmax_old, levels);
    //          n++;
    //        }
    //      }

    //      if (n > 0) {
    //        toRet = sum / n;
    //      }
    //    }
    //    if (delta_new <= delta_old) {
    toRet = dInterpolate(vol, inv_depth, kmin_old, kmax_old, levels);
    //    } else if (delta_new > delta_old) {
    //      float delta = (delta_new / delta_old);
    //      float sum = 0;
    //      int n = 0;
    //      for (float l = -delta /2; l <= delta / 2; l += delta_old) {
    //        float local_idx = l + inv_depth;
    //        if ((local_idx <= kmax_old) && (local_idx >= kmin_old)) {
    //          sum += dInterpolate(Vol, local_idx, kmin_old, kmax_old, levels);
    //          n++;
    //        }
    //      }

    //      if (n > 0) {
    //        toRet = sum / n;
    //      }
    //    }
  }

  new_vol[px + py*width + pz*width*height] = toRet;
}

///////////////////////////////////////////////////////////////////////////
__global__ void dInvert(float* out, float* in, int width)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;

  if (!isnan(in[px + py*width]))
    out[px + py*width] = 1.0f / in[px + py*width];
  else
    out[px + py*width] = 0;
}

///////////////////////////////////////////////////////////////////////////
__global__ void dCopyDualDepth(float* Q, float* A, int width, int height)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;

  float2 g = grad(A, px, py, width, height);
  g = Pi(g);
  Q[px + py*width] = g.x;
  Q[px + py*width + width*height] = g.y;
}

///////////////////////////////////////////////////////////////////////////
__global__ void dInitDiff(float* scaled_vol, float* diff, int width,
                          int height, int levels)
{
  int px = blockDim.x*blockIdx.x + threadIdx.x;
  int py = blockDim.y*blockIdx.y + threadIdx.y;

  const int index = px + py*width;
  const int imageSize = width*height;

  float max = 0;
  float min = FLT_MAX;

  for (int ii = 0; ii < levels; ++ii) {
    float val = scaled_vol[index + ii*imageSize];
    if ((val > max) && (val != FLT_MAX)){
      max = val;
    }
    if (val < min) {
      min = val;
    }
  }

  diff[index] = max - min > 0 ? max - min : 0;
}

///////////////////////////////////////////////////////////////////////////
__global__ void dCalculateRelativeTransform(float* Twa, float* Twb, float* Tab)
{

  // Find Taw = inv(Twa)
  float Taw[16];

  Taw[0] = Twa[0];
  Taw[1] = Twa[4];
  Taw[2] = Twa[8];
  Taw[3] = 0.0f;

  Taw[4] = Twa[1];
  Taw[5] = Twa[5];
  Taw[6] = Twa[9];
  Taw[7] = 0.0f;

  Taw[8] = Twa[2];
  Taw[9] = Twa[6];
  Taw[10] = Twa[10];
  Taw[11] = 0.0f;

  Taw[12] = -Twa[0]*Twa[12] - Twa[1]*Twa[13] - Twa[2]*Twa[14];
  Taw[13] = -Twa[4]*Twa[12] - Twa[5]*Twa[13] - Twa[6]*Twa[14];
  Taw[14] = -Twa[8]*Twa[12] - Twa[9]*Twa[13] - Twa[10]*Twa[14];
  Taw[15] = 1.0f;


  // Trw * Twm
  Tab[0] = Taw[0]*Twb[0] + Taw[4]*Twb[1] + Taw[8]*Twb[2];
  Tab[1] = Taw[1]*Twb[0] + Taw[5]*Twb[1] + Taw[9]*Twb[2];
  Tab[2] = Taw[2]*Twb[0] + Taw[6]*Twb[1] + Taw[10]*Twb[2];
  Tab[3] = 0.0f;

  Tab[4] = Taw[0]*Twb[4] + Taw[4]*Twb[5] + Taw[8]*Twb[6];
  Tab[5] = Taw[1]*Twb[4] + Taw[5]*Twb[5] + Taw[9]*Twb[6];
  Tab[6] = Taw[2]*Twb[4] + Taw[6]*Twb[5] + Taw[10]*Twb[6];
  Tab[7] = 0.0f;

  Tab[8] = Taw[0]*Twb[8] + Taw[4]*Twb[9] + Taw[8]*Twb[10];
  Tab[9] = Taw[1]*Twb[8] + Taw[5]*Twb[9] + Taw[9]*Twb[10];
  Tab[10] = Taw[2]*Twb[8] + Taw[6]*Twb[9] + Taw[10]*Twb[10];
  Tab[11] = 0.0f;

  Tab[12] = Taw[0]*Twb[12] + Taw[4]*Twb[13] + Taw[8]*Twb[14] + Taw[12]*Twb[15];
  Tab[13] = Taw[1]*Twb[12] + Taw[5]*Twb[13] + Taw[9]*Twb[14] + Taw[13]*Twb[15];
  Tab[14] = Taw[2]*Twb[12] + Taw[6]*Twb[13] + Taw[10]*Twb[14] + Taw[14]*Twb[15];
  Tab[15] = 1.0f;
}



///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///  Host Functions


///////////////////////////////////////////////////////////////////////////
inline void CheckErrors(const char* label)
{
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error [%s]: %s\n", label, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

///////////////////////////////////////////////////////////////////////////
iDTAM::iDTAM() : width_(0), height_(0), kmin_(0), kmax_(0), levels_(0),
  incremental_vol_(false)
{
  K_            = NULL;
  Kinv_         = NULL;
  Iv_           = NULL;
  Twv_          = NULL;
  Tvm_          = NULL;
  Ir_           = NULL;
  Ir_mask_      = NULL;
  Twr_          = NULL;
  window_full_  = false;
  window_idx_   = -1;
  diff_         = NULL;
  weight_       = NULL;
  Q_            = NULL;
  D_            = NULL;
  A_            = NULL;
  cost_         = NULL;
  icost_        = NULL;
  frames_       = NULL;
  iframes_      = NULL;
  scaled_cost_  = NULL;
  depth_        = NULL;
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::Init(unsigned int width, unsigned int height,
                 float kmin, float kmax, unsigned int levels,
                 bool incremental_volume, float* K,
                 float* Kinv, Parameters params)
{
  width_            = width;
  height_           = height;
  kmin_             = kmin;
  kmax_             = kmax;
  levels_           = levels;
  incremental_vol_  = incremental_volume;
  params_           = params;

  dim3 gridSize;
  gridSize.x = Gcd(width_, kBlockSizeCUDA);
  gridSize.y = Gcd(height_, kBlockSizeCUDA);
  gridSize.z = Gcd(levels_, kBlockSizeCUDA);
  printf("iDTAM: Grid Size: %dx%dx%d\n", gridSize.x, gridSize.y, gridSize.z);

  // Free previous memory (if used).
  _FreeAll();
  CheckErrors("Init FreeAll");

  // Check memory requirements.
  const size_t bytes_per_mb = 1024*1000;
  const size_t bytes_per_image = width_*height_*sizeof(float);
  const size_t bytes_per_volume = levels_*bytes_per_image;
  const int num_images = kWindowSize+8;
  const int num_volumes = 5;
  const unsigned int memory_required = ((num_images*bytes_per_image)
      + (num_volumes*bytes_per_volume))/bytes_per_mb;
  const size_t cuda_memory = _CheckMemoryCUDA();
  if(cuda_memory < memory_required) {
    fprintf(stderr, "- Low memory! Required memory is approximately %d MB. Aborting...\n", memory_required);
    exit(EXIT_FAILURE);
  }

  // Allocate memory.
  cudaMalloc((void**) &K_, 9*sizeof(float));
  cudaMalloc((void**) &Kinv_, 9*sizeof(float));
  cudaMalloc((void**) &Iv_, height_*width_*sizeof(float));
  cudaMalloc((void**) &Twv_, 16*sizeof(float));
  cudaMalloc((void**) &Tvm_, 16*sizeof(float));
  cudaMalloc((void**) &Ir_, height_*width_*sizeof(float));
  cudaMalloc((void**) &Ir_mask_, height_*width_);
  cudaMalloc((void**) &Twr_, 16*sizeof(float));
  for (size_t ii = 0; ii < kWindowSize; ++ii) {
    cudaMalloc((void**) &Im_[ii], height_*width_*sizeof(float));
    cudaMalloc((void**) &Twm_[ii], 16*sizeof(float));
    cudaMalloc((void**) &Tmr_[ii], 16*sizeof(float));
  }
  cudaMalloc((void**) &diff_, width_*height_*sizeof(float));
  cudaMalloc((void**) &weight_, width_*height_*sizeof(float));
  cudaMalloc((void**) &Q_, 2*width_*height_*sizeof(float));
  cudaMalloc((void**) &D_, width_*height_*sizeof(float));
  cudaMalloc((void**) &A_, width_*height_*sizeof(float));
  cudaMalloc((void**) &icost_, width_*height_*levels_*sizeof(float));
  cudaMalloc((void**) &cost_, width_*height_*levels_*sizeof(float));
  cudaMalloc((void**) &iframes_, width_*height_*levels_*sizeof(float));
  cudaMalloc((void**) &frames_, width_*height_*levels_*sizeof(float));
  cudaMalloc((void**) &scaled_cost_, width_*height_*levels_*sizeof(float));
  cudaMalloc((void**) &depth_, width_*height_*sizeof(float));
  cudaThreadSynchronize();
  CheckErrors("Init Malloc");

  // Copy K matrices.
  cudaMemcpy(K_, K, 9*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Kinv_, Kinv, 9*sizeof(float), cudaMemcpyHostToDevice);

  // Reset window index.
  window_full_ = false;
  window_idx_  = -1;

  // Reset memory.
  cudaMemset(Iv_, 0, width_*height_*sizeof(float));
  cudaMemset(Twv_, 0, 16*sizeof(float));
  cudaMemset(Tvm_, 0, 16*sizeof(float));
  cudaMemset(Ir_, 0, width_*height_*sizeof(float));
  cudaMemset(Twr_, 0, 16*sizeof(float));
  for (size_t ii = 0; ii < kWindowSize; ++ii) {
    cudaMemset(Im_[ii], 0, width_*height_*sizeof(float));
    cudaMemset(Twm_[ii], 0, 16*sizeof(float));
    cudaMemset(Tmr_[ii], 0, 16*sizeof(float));
  }
  cudaMemset(diff_, 0, width_*height_*sizeof(float));
  cudaMemset(weight_, 0, width_*height_*sizeof(float));
  cudaMemset(Q_, 0, 2*width_*height_*sizeof(float));
  cudaMemset(D_, 0, width_*height_*sizeof(float));
  cudaMemset(A_, 0, width_*height_*sizeof(float));
  cudaMemset(icost_, 0, width_*height_*levels_*sizeof(float));
  cudaMemset(cost_, 0, width_*height_*levels_*sizeof(float));
  cudaMemset(iframes_, 0, width_*height_*levels_*sizeof(float));
  cudaMemset(frames_, 0, width_*height_*levels_*sizeof(float));
  cudaMemset(scaled_cost_, 0, width_*height_*levels_*sizeof(float));
  cudaMemset(depth_, 0, width_*height_*sizeof(float));

  cudaThreadSynchronize();
  CheckErrors("Init");
}

///////////////////////////////////////////////////////////////////////////
iDTAM::~iDTAM()
{
  _FreeAll();
  cudaThreadSynchronize();
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::_FreeAll()
{
  if (K_ != NULL) {
    cudaFree(K_);
  }
  if (Kinv_ != NULL) {
    cudaFree(Kinv_);
  }
  if (Iv_ != NULL) {
    cudaFree(Iv_);
  }
  if (Twv_ != NULL) {
    cudaFree(Twv_);
  }
  if (Tvm_ != NULL) {
    cudaFree(Tvm_);
  }
  if (Ir_ != NULL) {
    cudaFree(Ir_);
  }
  if (Ir_mask_ != NULL) {
    cudaFree(Ir_mask_);
  }
  if (Twr_ != NULL) {
    cudaFree(Twr_);
  }
  if (window_idx_ != -1) {
    for (size_t ii = 0; ii < kWindowSize; ++ii) {
      cudaFree(Im_[ii]);
      cudaFree(Twm_[ii]);
      cudaFree(Tmr_[ii]);
    }
  }
  if (diff_ != NULL) {
    cudaFree(diff_);
  }
  if (weight_ != NULL) {
    cudaFree(weight_);
  }
  if (Q_ != NULL) {
    cudaFree(Q_);
  }
  if (D_ != NULL) {
    cudaFree(D_);
  }
  if (A_ != NULL) {
    cudaFree(A_);
  }
  if (icost_ != NULL) {
    cudaFree(icost_);
  }
  if (cost_ != NULL) {
    cudaFree(cost_);
  }
  if (iframes_ != NULL) {
    cudaFree(iframes_);
  }
  if (frames_ != NULL) {
    cudaFree(frames_);
  }
  if (scaled_cost_ != NULL) {
    cudaFree(scaled_cost_);
  }
  if (depth_ != NULL) {
    cudaFree(depth_);
  }
  cudaThreadSynchronize();
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::_Iterate(float theta)
{
  _UpdateQ();
  _UpdateD(theta);
  _UpdateA(theta);
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::_UpdateQ()
{
  dim3 gridSize, blockSize;
  gridSize.x = Gcd(width_, kBlockSizeCUDA);
  gridSize.y = Gcd(height_, kBlockSizeCUDA);
  blockSize.x = width_ / gridSize.x;
  blockSize.y = height_ / gridSize.y;

  dUpdateQ<<<gridSize, blockSize>>>(Q_, D_, weight_, params_.sig_q,
                                    params_.epsilon, width_, height_);
  CheckErrors("UpdateQ");
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::_UpdateD(float theta)
{
  dim3 gridSize, blockSize;
  gridSize.x = Gcd(width_, kBlockSizeCUDA);
  gridSize.y = Gcd(height_, kBlockSizeCUDA);
  blockSize.x = width_ / gridSize.x;
  blockSize.y = height_ / gridSize.y;

  dUpdateD<<<gridSize, blockSize>>>(D_, A_, Q_, weight_, 1.0f/theta,
                                    params_.sig_d, width_, height_);
  CheckErrors("UpdateD");
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::_UpdateA(float theta)
{
  dim3 gridSize, blockSize;
  gridSize.x = Gcd(width_, kBlockSizeCUDA);
  gridSize.y = Gcd(height_, kBlockSizeCUDA);
  blockSize.x = width_ / gridSize.x;
  blockSize.y = height_ / gridSize.y;

  dUpdateA<<<gridSize, blockSize>>>(scaled_cost_, diff_, A_, D_, theta,
                                    params_.lambda, width_, height_, levels_,
                                    kmin_, kmax_);

#ifdef HAVE_CLANG
  const float mid_range = ((kmax_-kmin_)/2.0)+kmin_;
  params_.lambda = 1.0 / (1.0 + 0.5/mid_range);
#else
  // Adjust Lambda dynamically.
  thrust::device_ptr<float> ptr = thrust::device_pointer_cast(D_);
  float max = thrust::reduce(ptr, &ptr[height_*width_], -1.0f,
      thrust::maximum<float>());
  params_.lambda = 1.0 / (1.0 + 0.5/max);
#endif

  CheckErrors("UpdateA");
}

///////////////////////////////////////////////////////////////////////////
bool iDTAM::_CheckNaN(char var)
{
  float* temp = (float*)malloc(sizeof(float)*height_*width_);
  switch(var) {
  case 'D' : cudaMemcpy(temp, D_, sizeof(float)*height_*width_, cudaMemcpyDeviceToHost); break;
  case 'A' : cudaMemcpy(temp, A_, sizeof(float)*height_*width_, cudaMemcpyDeviceToHost); break;
  case 'X' : cudaMemcpy(temp, Q_, sizeof(float)*height_*width_, cudaMemcpyDeviceToHost); break;
  case 'Y' : cudaMemcpy(temp, &Q_[height_*width_], sizeof(float)*height_*width_, cudaMemcpyDeviceToHost); break;
  }

  bool return_val = false;
  for (int jj = 0; jj < height_; jj++) {
    for (int ii = 0; ii < width_; ii++) {
      if (isnan(temp[ii + jj*width_])) {
        fprintf(stderr, "%c has a NaN at (%d, %d)\n", var, jj, ii);
        return_val = true;
      }
    }
  }
  fflush(stdout);
  free(temp);
  return return_val;
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::_InitializeDAQ()
{
  // Scale cost volume.
  dim3 gridSize3, blockSize3;
  gridSize3.x = Gcd(width_, kBlockSizeCUDA);
  gridSize3.y = Gcd(height_, kBlockSizeCUDA);
  gridSize3.z = Gcd(levels_, kBlockSizeCUDA);
  blockSize3.x = width_ / gridSize3.x;
  blockSize3.y = height_ / gridSize3.y;
  blockSize3.z = levels_ / gridSize3.z;
  dScaleVolume<<<gridSize3, blockSize3>>>(cost_, frames_, scaled_cost_, width_,
                                          height_);

  // Find minimum through cost volume.
  dim3 gridSize2, blockSize2;
  gridSize2.x = gridSize3.x;
  gridSize2.y = gridSize3.y;
  blockSize2.x = blockSize3.x;
  blockSize2.y = blockSize3.y;
  dSimpleMinimum<<<gridSize2, blockSize2>>>(scaled_cost_, A_, width_, height_,
                                            levels_, kmin_, kmax_);

  _CheckNaN('A');

  // Copy A to D.
  cudaMemcpy(D_, A_, height_*width_*sizeof(float), cudaMemcpyDeviceToDevice);

  // Initialize Q.
  // TODO(jmf) Check if this works or start at 0.
  // NOTE(jmf) Seems to do the same either way.
  cudaMemset(Q_, 0.0f, 2*height_*width_*sizeof(float));
//  dCopyDualDepth<<<gridSize2, blockSize2>>>(Q_, A_, width_, height_);

  // Find max-min difference for cuadratic speed-up.
  dInitDiff<<<gridSize2, blockSize2>>>(scaled_cost_, diff_, width_, height_,
                                       levels_);
  CheckErrors("InitializeDepth");
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::GetCostVolumeSlice(float* host_buffer, int n)
{
  if (host_buffer != NULL && n >= 0 && n < levels_) {
    cudaMemcpy(host_buffer, &scaled_cost_[n*height_*width_],
        width_*height_*sizeof(float), cudaMemcpyDeviceToHost);
    CheckErrors("GetSlice");
  }
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::GetInverseDepth(float* host_buffer)
{
  if (host_buffer != NULL) {
    cudaMemcpy(host_buffer, D_, height_*width_*sizeof(float), cudaMemcpyDeviceToHost);
    CheckErrors("GetInverseDepth");
  }
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::PushImage(float* host_image, float* host_pose)
{
  if (host_image == NULL || host_pose == NULL) {
    return;
  }

  // Push old reference image into support image queue.
  if (window_idx_ != -1) {
    // Check if we are using incremental volume or not...
    if (incremental_vol_ && window_full_) {
      // Using incremental volume.
      // Since frame is to be dropped, save pose and fuse into incremental
      // volume before dropping.

      // Find relative T between Twv_ and Twm_.
      dim3 gridSize1, blockSize1;  // Defaults to 1.
      dCalculateRelativeTransform<<<gridSize1, blockSize1>>>(Twv_,
                                                             Twm_[window_idx_],
                                                             Tvm_);

      // Reset old volume.
      cudaMemset(cost_, 0, width_*height_*levels_*sizeof(float));
      cudaMemset(cost_, 0, width_*height_*levels_*sizeof(float));

      // Transform ivolumes into new volume.
      dim3 gridSize2, blockSize2;
      gridSize2.x = Gcd(width_, kBlockSizeCUDA);
      gridSize2.y = Gcd(height_, kBlockSizeCUDA);
      blockSize2.x = width_ / gridSize2.x;
      blockSize2.y = height_ / gridSize2.y;

      dTransformVolume2D<<<gridSize2, blockSize2>>>(icost_, cost_, iframes_,
                                                    frames_, Tvm_, K_, Kinv_,
                                                    width_, height_, levels_,
                                                    kmin_, kmax_);

      // Add popped image into cost volume.
      _AddToCostVolume(Im_[window_idx_], Iv_, Tvm_, params_.omega);

      // Swap volume pointers.
      float* tmp_ptr = icost_;
      icost_ = cost_;
      cost_ = tmp_ptr;
      tmp_ptr = iframes_;
      iframes_ = frames_;
      frames_ = tmp_ptr;

      // Swap image/pose pointers.
      tmp_ptr = Im_[window_idx_];
      Im_[window_idx_] = Ir_;
      Ir_ = Iv_;
      Iv_ = tmp_ptr;

      tmp_ptr = Twm_[window_idx_];
      Twm_[window_idx_] = Twr_;
      Twr_ = Twv_;
      Twv_ = tmp_ptr;
    } else {
      // Not using incremental volume. Simply swap pointers.
      float* tmp_ptr = Im_[window_idx_];
      Im_[window_idx_] = Ir_;
      Ir_ = tmp_ptr;

      tmp_ptr = Twm_[window_idx_];
      Twm_[window_idx_] = Twr_;
      Twr_ = tmp_ptr;
    }
  }

  // Increment index.
  window_idx_++;

  // Wrap index if outside window.
  if (window_idx_ == kWindowSize) {
    window_idx_ = 0;
    window_full_ = true;
  }

  // New frame becomes reference frame.
  cudaMemcpy(Ir_, host_image, height_*width_*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Twr_, host_pose, 16*sizeof(float), cudaMemcpyHostToDevice);

  _Optimize();
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::_InitWeights()
{
  dim3 gridSize, blockSize;
  gridSize.x = Gcd(width_, kBlockSizeCUDA);
  gridSize.y = Gcd(height_, kBlockSizeCUDA);
  gridSize.z = Gcd(levels_, kBlockSizeCUDA);
  blockSize.x = width_ / gridSize.x;
  blockSize.y = height_ / gridSize.y;
  blockSize.z = levels_ / gridSize.z;

  dInitWeights<<<gridSize, blockSize>>>(weight_, Ir_, params_.g_alpha,
                                        params_.g_beta, width_, height_);

  CheckErrors("InitGradImage");
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::_CalculateRelativePoses()
{
  dim3 gridSize, blockSize;  // Defaults to 1.

  // This has to be called this way, since Twm_ is a HOST pointer of DEVICE
  // pointers rather than a DEVICE pointer itself.
  for (size_t ii = 0; ii < kWindowSize; ++ii) {
    dCalculateRelativeTransform<<<gridSize, blockSize>>>(Twm_[ii], Twr_,
                                                         Tmr_[ii]);
  }
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::_Optimize()
{
  // If no support images, do nothing.
  if ((window_full_ == false) && (window_idx_ == 0)) {
    return;
  }

  // Find relative transforms between reference image and support images.
  _CalculateRelativePoses();

  // Reset volumes.
  cudaMemset(cost_, 0, width_*height_*levels_*sizeof(float));
  cudaMemset(frames_, 0, width_*height_*levels_*sizeof(float));

  // If incremental volume used, initialize volume with transformed volume.
  if (incremental_vol_ && window_full_) {
    dim3 gridSize1, blockSize1;  // Defaults to 1.
    dCalculateRelativeTransform<<<gridSize1, blockSize1>>>(Twv_, Twr_, Tvm_);

    dim3 gridSize2, blockSize2;
    gridSize2.x = Gcd(width_, kBlockSizeCUDA);
    gridSize2.y = Gcd(height_, kBlockSizeCUDA);
    blockSize2.x = width_ / gridSize2.x;
    blockSize2.y = height_ / gridSize2.y;

    dTransformVolume2D<<<gridSize2, blockSize2>>>(icost_, cost_, iframes_,
                                                  frames_, Tvm_, K_, Kinv_,
                                                  width_, height_, levels_,
                                                  kmin_, kmax_);
  }

  // Add images to cost volume.
  const size_t num_frames = window_full_ ? kWindowSize : window_idx_;
  for (size_t ii = 0; ii < num_frames; ++ii) {
    _AddToCostVolume(Ir_, Im_[ii], Tmr_[ii], 1.0);
  }

  // Initialize weight for huber norm based on reference image.
  _InitWeights();

  // Initialize DA from SimpleMinimum and Q.
  _InitializeDAQ();

  // Set theta start and iterate...
  float theta = params_.theta_start;
  int step = 0;
  while ((theta > params_.theta_end)) {
    _Iterate(theta);

    if (theta > 0.001) {
      theta = theta*(1 - step*0.001);
    } else {
      theta = theta*(1 - step*params_.beta);
    }
    step++;
  }

  const int bilateral_patch_size = 0;
  if (bilateral_patch_size != 0) {
    dim3 gridSize, blockSize;
    gridSize.x = Gcd(width_, kBlockSizeCUDA);
    gridSize.y = Gcd(height_, kBlockSizeCUDA);
    blockSize.x = width_ / gridSize.x;
    blockSize.y = height_ / gridSize.y;

    dBilateral<<<gridSize, blockSize>>>(depth_, D_, bilateral_patch_size, 2, 5,
                                        width_, height_);
    dInvert<<<gridSize, blockSize>>>(D_, depth_, width_);
  }
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::OptimizeLayers()
{
  // Initialize weight for huber norm based on reference image.
  _InitWeights();

  // Initialize DA from SimpleMinimum and Q.
  _InitializeDAQ();

  // Set theta start and iterate...
  float theta = params_.theta_start;
  int step = 0;
  while ((theta > params_.theta_end)) {
    _Iterate(theta);

    if (theta > 0.001) {
      theta = theta*(1 - step*0.001);
    } else {
      theta = theta*(1 - step*params_.beta);
    }
    step++;
  }
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::GetDepth(float* host_buffer)
{
  // Convert inverse depth to depth.
  dim3 gridSize, blockSize;
  gridSize.x = Gcd(width_, kBlockSizeCUDA);
  gridSize.y = Gcd(height_, kBlockSizeCUDA);
  blockSize.x = width_ / gridSize.x;
  blockSize.y = height_ / gridSize.y;
  dInvert<<<gridSize, blockSize>>>(depth_, D_, width_);

  if (host_buffer != NULL) {
    cudaMemcpy(host_buffer, depth_, height_*width_*sizeof(float), cudaMemcpyDeviceToHost);
    CheckErrors("GetDepth");
  }
}

///////////////////////////////////////////////////////////////////////////
/// Adds Im into Cost Volume. Cost Volume is in Ir's frame, T is Tmr.
void iDTAM::_AddToCostVolume(float* Ir, float* Im, float* Tmr, float omega)
{
  dim3 gridSize, blockSize;
  gridSize.x = Gcd(width_, kBlockSizeCUDA);
  gridSize.y = Gcd(height_, kBlockSizeCUDA);
  gridSize.z = Gcd(levels_, kBlockSizeCUDA);
  blockSize.x = width_ / gridSize.x;
  blockSize.y = height_ / gridSize.y;
  blockSize.z = levels_ / gridSize.z;

  dAddToCostVolume<<<gridSize, blockSize>>>(cost_, frames_, Ir, Im, width_,
                                            height_, levels_, kmin_,
                                            kmax_, K_, Kinv_, Tmr, omega);
  CheckErrors("AddToCostVolume");
}

///////////////////////////////////////////////////////////////////////////
/// TODO: This is temporal test for moving objects depth.
/// Adds Im into Cost Volume. Cost Volume is in Ir's frame, T is Tmr.
void iDTAM::AddLayerToCostVolume(float* Ir, unsigned char* Ir_mask, float* Im,
                                 int id, float* Tmr, float omega)
{

  cudaMemcpy(Ir_, Ir, height_*width_*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Im_[0], Im, height_*width_*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Ir_mask_, Ir_mask, height_*width_, cudaMemcpyHostToDevice);
  cudaMemcpy(Tvm_, Tmr, 16*sizeof(float), cudaMemcpyHostToDevice);

  dim3 gridSize, blockSize;
  gridSize.x = Gcd(width_, kBlockSizeCUDA);
  gridSize.y = Gcd(height_, kBlockSizeCUDA);
  gridSize.z = Gcd(levels_, kBlockSizeCUDA);
  blockSize.x = width_ / gridSize.x;
  blockSize.y = height_ / gridSize.y;
  blockSize.z = levels_ / gridSize.z;

  dAddLayerToCostVolume<<<gridSize, blockSize>>>(cost_, frames_, Ir_, Ir_mask_,
                                                 Im_[0], id, width_,
                                                 height_, levels_, kmin_,
                                                 kmax_, K_, Kinv_, Tvm_, omega);
  CheckErrors("AddToCostVolume");
}

///////////////////////////////////////////////////////////////////////////
void iDTAM::RescaleVolume(float threshold, float dz = 0, float NKMin = 0, float NKMax = 0)
{
#if 0
  // Select free volume to use as a temp variable.
  float* tempVolume;
  if(switch_ == false) {
    tempVolume = cost2_;
  } else {
    tempVolume = cost1_;
  }

  const float delta = (kmax_ - kmin_) / levels_;

  float kmin, kmax;

  int minId = 0;
  int maxId = levels_ - 1;

  // Inc/dec limits based on threshold.
  while(m_fHist[minId] < threshold) {
    minId++;
  }
  while(m_fHist[maxId] < threshold) {
    maxId--;
  }

  // Expand limits +1.
  if(minId > 0) {
    minId--;
  }
  if(maxId < levels_-1) {
    maxId++;
  }

  // If zeros, automatically resize.
  if((NKMin == 0) && (NKMax == 0)) {

    float maxSlope, slope;

    ///-------------------- Max check.
    // If max is at limit, and greater than threshold ...
    if ((maxId ==  levels_-1) && (m_fHist[maxId] > threshold)) {
      // ... EXPAND
      /*
      maxSlope = -m_fHist[maxId] / (3*delta);
      slope = (m_fHist[maxId-1] - m_fHist[maxId]) / delta;
      slope = (slope < maxSlope) ? slope : maxSlope;
      kmax = ((maxId+1)*delta + m_fKmin) - (m_fHist[maxId]) / slope;
      */
      kmax = kmax_ + delta*3;
      printf("-- Expanding kmax.\n");
    } else {
      // ... SHRINK
      kmax = delta*maxId + kmin_;
      printf("-- Shrinking kmax.\n");
    }


    ///-------------------- Min check.
    // If min is at limit, and greater than threshold ...
    if((minId == 0) && (m_fHist[minId] > threshold)) {
      // ... EXPAND
      kmin = kmin_ - delta*3;
      kmin = std::max(kmin, 0.01f);
      printf("-- Expand kmin.\n");
    } else {
      // ... SHRINK////
      kmin = delta*minId + kmin_;
      printf("-- Shrinking kmin.\n");
    }
    fprintf(stdout, "slope max: %f --- slope: %f\n", maxSlope, slope);
    fprintf(stdout, "OLD: kmin - %f kmax - %f\n", kmin_, kmax_);
    fprintf(stdout, "NEW: kmin - %f kmax - %f\n", kmin, kmax);
  } else {
    float tkmin, tkmax;

    kmax = NKMax;
    kmin = NKMin;
    if (minId > 0) minId--;
    if (maxId < levels_ - 1) maxId++;
    tkmin = minId*delta + kmin_;
    tkmax = maxId*delta + kmin_;
    fprintf(stdout, "// hist[%d] = %f - hist[%d] = %f //  %f - %f ----- %f - %f\n", minId, m_fHist[minId], maxId, m_fHist[maxId], kmin, kmax, tkmin, tkmax);
    if (minId == 0) {
      kmin = std::min(tkmin, NKMin);
      kmin = std::max(kmin, 0.01f);
    }
    else {
      kmin = tkmin;
    }
    if (maxId == levels_ - 1) {
      kmax = std::max(tkmax, NKMax);
      kmax = std::min(kmax, 10.0f);
    }
    else {
      kmax = tkmax;
    }

    if (kmin >= kmax) { // this should never happen!  something went wrong
      fprintf(stdout, "Something crapped out in the kmin/kmax estimation.\n");
      return;
    }

    //  Don't forget to take the simple motion model into account!
    //    fprintf(stdout, "before motion model: (%f, %f)\n", kmin, kmax);
    if (dz != 0) {
      if ((1 / kmax) > dz)
        kmax = 1 / ((1 / kmax) - dz);
      if ((1 / kmin) > dz)
        kmin = 1 / ((1 / kmin) - dz);
    }
    //    kmax = std::min(kmax, 1.2f*m_fKmax);
    //    fprintf(stdout, "dz = %f : (%f, %f) vs (%f, %f) = (%f, %f)\n", dz, tkmin, tkmax, NKMin, NKMax, kmin, kmax);
  }

  float diff_min;
  float diff_max;
  diff_max = (kmax > kmax_) ? (kmax - kmax_) : (kmax_ - kmax);
  diff_max /= kmax_ ;

  diff_min = (kmin > kmin_) ? (kmin - kmin_) : (kmin_ - kmin);
  diff_min /= kmin_;
  //  fprintf(stdout, "diff_min = %f diff_max = %f\n", fabs(diff_min), fabs(diff_max));

  //  We don't want to change every iteration
  //  if((diff_min < 0.1) && (diff_max < 0.1))
  //    return;

  dim3 gridSize3, blockSize3;
  gridSize3.x = Gcd(width_, CUDA_BLOCK_SIZE);
  gridSize3.y = Gcd(height_, CUDA_BLOCK_SIZE);
  gridSize3.z = Gcd(levels_, CUDA_BLOCK_SIZE);
  blockSize3.x = width_ / gridSize3.x;
  blockSize3.y = height_ / gridSize3.y;
  blockSize3.z = levels_ / gridSize3.z;

  //  kmin = m_fKmin; /// TODO(jmf) take me out!

  //  cudaMemset(tempVolume, 0.0f, sizeof(float)*m_nHeight*m_nWidth*m_nLevels);
  //  dRescaleVolume<<<gridSize3, blockSize3>>>(tempVolume, dVol, m_nHeight, m_nWidth, m_nLevels, kmin, kmax, m_fKmin, m_fKmax);
  //  cudaMemcpy(dVol, tempVolume, sizeof(float)*m_nHeight*m_nWidth*m_nLevels, cudaMemcpyDeviceToDevice);

  //  cudaMemset(tempVolume, 0.0f, sizeof(float)*m_nHeight*m_nWidth*m_nLevels);
  //  dRescaleVolume<<<gridSize3, blockSize3>>>(tempVolume, dFrames, m_nHeight, m_nWidth, m_nLevels, kmin, kmax, m_fKmin, m_fKmax);
  //  cudaMemcpy(dFrames, tempVolume, sizeof(float)*m_nHeight*m_nWidth*m_nLevels, cudaMemcpyDeviceToDevice);

  cudaMemset(tempVolume, 0.0f, sizeof(float)*height_*width_*levels_);
  dRescaleVolume<<<gridSize3, blockSize3>>>(tempVolume, cost_, height_, width_, levels_, NKMin, NKMax, kmin_, kmax_);
  cudaMemcpy(cost_, tempVolume, sizeof(float)*height_*width_*levels_, cudaMemcpyDeviceToDevice);

  cudaMemset(tempVolume, 0.0f, sizeof(float)*height_*width_*levels_);
  dRescaleVolume<<<gridSize3, blockSize3>>>(tempVolume, frames_, height_, width_, levels_, NKMin, NKMax, kmin_, kmax_);
  cudaMemcpy(frames_, tempVolume, sizeof(float)*height_*width_*levels_, cudaMemcpyDeviceToDevice);

  kmin_ = kmin;
  kmax_ = kmax;

  CheckErrors("RescaleVolume");
#endif
}

///////////////////////////////////////////////////////////////////////////
size_t iDTAM::_CheckMemoryCUDA()
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
