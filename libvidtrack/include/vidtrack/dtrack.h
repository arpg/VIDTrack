/*
 * Copyright (c) 2015  Juan M. Falquez,
 *                     University of Colorado - Boulder
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <vector>

#include <vidtrack/config.h>

#ifdef VIDTRACK_USE_TBB
#include <tbb/tbb.h>
#endif


#ifdef VIDTRACK_USE_CUDA
#include <vidtrack/dtrack.cuh>
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif
#include <opencv2/opencv.hpp>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <sophus/se3.hpp>
#include <calibu/cam/CameraRig.h>


/////////////////////////////////////////////////////////////////////////////
namespace Eigen {
typedef Matrix<double, 2, 3> Matrix2x3d;
typedef Matrix<double, 3, 4> Matrix3x4d;
typedef Matrix<double, 6, 6> Matrix6d;
typedef Matrix<double, 6, 1> Vector6d;
}  // namespace Eigen


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
class DTrack
{
public:
  ///////////////////////////////////////////////////////////////////////////
  DTrack(unsigned int pyramid_levels);

  ///////////////////////////////////////////////////////////////////////////
  ~DTrack();

  ///////////////////////////////////////////////////////////////////////////
  void SetParams(
      const calibu::CameraModelGeneric<double>&   live_grey_cmod,
      const calibu::CameraModelGeneric<double>&   ref_grey_cmod,
      const calibu::CameraModelGeneric<double>&   ref_depth_cmod,
      const Sophus::SE3d&                         Tgd
      );

  ///////////////////////////////////////////////////////////////////////////
  void SetKeyframe(
      const cv::Mat&    ref_grey,  // Input: Reference image (float format, normalized).
      const cv::Mat&    ref_depth  // Input: Reference depth (float format, meters).
      );

  ///////////////////////////////////////////////////////////////////////////
  double Estimate(
      bool                      use_pyramid,  // Input: Flag to enable full pyramid.
      const cv::Mat&            live_grey,    // Input: Live image (float format, normalized).
      Sophus::SE3Group<double>& Trl,          // Input/Output: Transform between grey cameras (vision frame/input is hint).
      Eigen::Matrix6d&          covariance,   // Output: Covariance.
      unsigned int&             num_obs,      // Output: Number of observations.
      const cv::Mat&            live_depth
    );

private:
  ///////////////////////////////////////////////////////////////////////////
  calibu::CameraModelGeneric<double> _ScaleCM(
      calibu::CameraModelGeneric<double>  cam_model,  // Input: Camera Model.
      unsigned int                        level       // Input: Number of pyramid levels.
    );

  ///////////////////////////////////////////////////////////////////////////
  /// Calculates image gradients.
  void _CalculateGradients(
      const unsigned char*      image_ptr,    //< Input: Image pointer.
      int                       image_width,  //< Input: Image width.
      int                       image_height, //< Input: Image height.
      float*                    gradX_ptr,    //< Output: Gradient in X.
      float*                    gradY_ptr     //< Output: Gradient in Y.
    );

  ///////////////////////////////////////////////////////////////////////////
  /// Tukey robust norm.
  double _NormTukey(
      double      r,    //< Input: Error.
      double      c     //< Input: Norm parameter.
    );

  ///////////////////////////////////////////////////////////////////////////
  /// Adjust mean and variance of Image1 brightness to be closer to Image2.
  void _BrightnessCorrectionImagePair(
      unsigned char*  img1_ptr,     //< Input: Pointer 1
      unsigned char*  img2_ptr,     //< Input: Pointer 2
      size_t          image_size    //< Input: Number of pixels in image
    );


  ///////////////////////////////////////////////////////////////////////////
public:
  const unsigned int kPyramidLevels;

private:
#ifdef VIDTRACK_USE_CUDA
  cuDTrack*                                        cu_dtrack_;
#endif
#ifdef VIDTRACK_USE_TBB
  tbb::task_scheduler_init                         tbb_scheduler_;
#endif
  std::vector<cv::Mat>                             live_grey_pyramid_;
  std::vector<cv::Mat>                             live_depth_pyramid_;
  std::vector<cv::Mat>                             ref_grey_pyramid_;
  std::vector<cv::Mat>                             ref_depth_pyramid_;
  std::vector<calibu::CameraModelGeneric<double> > live_grey_cam_model_;
  std::vector<calibu::CameraModelGeneric<double> > ref_grey_cam_model_;
  std::vector<calibu::CameraModelGeneric<double> > ref_depth_cam_model_;
  Sophus::SE3d                                     Tgd_;
};
