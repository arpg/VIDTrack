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

#include <tbb/tbb.h>

#include <opencv2/opencv.hpp>

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
        const calibu::CameraModelGeneric<double>& live_grey_cmod,
        const calibu::CameraModelGeneric<double>& ref_grey_cmod,
        const calibu::CameraModelGeneric<double>& ref_depth_cmod,
        const Sophus::SE3d&                       Tgd
      );

    ///////////////////////////////////////////////////////////////////////////
    void SetKeyframe(
        const cv::Mat& ref_grey,  // Input: Reference image (float format, normalized).
        const cv::Mat& ref_depth  // Input: Reference depth (float format, meters).
      );

    ///////////////////////////////////////////////////////////////////////////
    double Estimate(
        const cv::Mat&            live_grey,          // Input: Live image (float format, normalized).
        Sophus::SE3Group<double>& Trl,                // Input/Output: Transform between grey cameras (input is hint).
        Eigen::Matrix6d&          covariance,         // Output: Covariance
        bool                      use_pyramid = true  // Input: Options.
      );

  private:
    ///////////////////////////////////////////////////////////////////////////
    calibu::CameraModelGeneric<double> _ScaleCM(
        calibu::CameraModelGeneric<double> cam_model,
        unsigned int                       level
      );


    ///////////////////////////////////////////////////////////////////////////
  public:
    const unsigned int kPyramidLevels;

  private:
    tbb::task_scheduler_init                         tbb_scheduler_;
    std::vector<cv::Mat>                             live_grey_pyramid_;
    std::vector<cv::Mat>                             ref_grey_pyramid_;
    std::vector<cv::Mat>                             ref_depth_pyramid_;
    std::vector<calibu::CameraModelGeneric<double> > live_grey_cam_model_;
    std::vector<calibu::CameraModelGeneric<double> > ref_grey_cam_model_;
    std::vector<calibu::CameraModelGeneric<double> > ref_depth_cam_model_;
    Sophus::SE3d                                     Tgd_;
};
