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

#undef VIDTRACK_USE_TBB

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

#include <calibu/Calibu.h>
#include <sophus/se3.hpp>


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
      const Eigen::Matrix3d&    live_grey_K,  // Input: K matrix for live grey camera.
      const Eigen::Matrix3d&    ref_grey_K,   // Input: K matrix for reference grey camera.
      const Eigen::Matrix3d&    ref_depth_K,  // Input: K matrix for reference depth camera.
      const Sophus::SE3d&       Tgd           // Input: Grey-Depth camera transform.
    );

  ///////////////////////////////////////////////////////////////////////////
  void SetKeyframe(
      const cv::Mat&    ref_grey,  // Input: Reference image (unsigned char format).
      const cv::Mat&    ref_depth  // Input: Reference depth (float format, meters).
      );

  ///////////////////////////////////////////////////////////////////////////
  double Estimate(
      bool                      use_pyramid,  // Input: Flag to enable full pyramid.
      const cv::Mat&            live_grey,    // Input: Live image (unsigned char format).
      Sophus::SE3d&             Trl,          // Input/Output: Transform between grey cameras (vision frame/input is hint).
      Eigen::Matrix6d&          covariance,   // Output: Covariance.
      unsigned int&             num_obs       // Output: Number of observations.
    );

  ///////////////////////////////////////////////////////////////////////////
  void BuildProblem(const Sophus::SE3d& Trl,
                    Eigen::Matrix6d&    LHS,
                    Eigen::Vector6d&    RHS,
                    double&             squared_error,
                    double&             number_observations,
                    uint                pyramid_lvl);

  ///////////////////////////////////////////////////////////////////////////
  void ComputeGradient(uint pyramid_lvl);

private:
  ///////////////////////////////////////////////////////////////////////////
  Eigen::Matrix3d  _ScaleCM(
      const Eigen::Matrix3d&    K,      // Input: Camera model matrix K.
      unsigned int              level   // Input: Pyramid level to scale K by.
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
  const double       kGreySigma = 1.0;
  const double       kDepthSigma = 0.01;
  const unsigned int kPyramidLevels;
private:
#ifdef VIDTRACK_USE_CUDA
  cuDTrack*                       cu_dtrack_;
#endif
#ifdef VIDTRACK_USE_TBB
  tbb::task_scheduler_init        tbb_scheduler_;
#endif
  cv::Mat                         gradient_x_live_;
  cv::Mat                         gradient_y_live_;
  cv::Mat                         gradient_x_ref_;
  cv::Mat                         gradient_y_ref_;

  std::vector<cv::Mat>            live_grey_pyramid_;
  std::vector<cv::Mat>            live_depth_pyramid_;
  std::vector<cv::Mat>            ref_grey_edges_;
  std::vector<cv::Mat>            ref_grey_pyramid_;
  std::vector<cv::Mat>            ref_depth_pyramid_;
  std::vector<Eigen::Matrix3d>    live_grey_cam_model_;
  std::vector<Eigen::Matrix3d>    ref_grey_cam_model_;
  std::vector<Eigen::Matrix3d>    ref_depth_cam_model_;
  Sophus::SE3d                    Tgd_;
};
