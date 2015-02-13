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

#include <deque>

#include <Eigen/Eigen>

#include <ba/BundleAdjuster.h>
#include <ba/InterpolationBuffer.h>
#include <ba/Types.h>

#include <devil/dtrack.h>


namespace devil {

class Tracker {

public:

  ///////////////////////////////////////////////////////////////////////////
  Tracker(unsigned int window_size = 5, unsigned int pyramid_levels = 5);


  ///////////////////////////////////////////////////////////////////////////
  ~Tracker();


  ///////////////////////////////////////////////////////////////////////////
  /// Simplified configuration. Assumes live, reference and depth have same
  /// camera models and depth is aligned to greyscale image.
  void ConfigureDTrack(
      const cv::Mat&                            keyframe_grey,
      const cv::Mat&                            keyframe_depth,
      double                                    time,
      const calibu::CameraModelGeneric<double>& cmod
    );


  ///////////////////////////////////////////////////////////////////////////
  void ConfigureDTrack(
      const cv::Mat&                            keyframe_grey,
      const cv::Mat&                            keyframe_depth,
      double                                    time,
      const calibu::CameraModelGeneric<double>& live_grey_cmod,
      const calibu::CameraModelGeneric<double>& ref_grey_cmod,
      const calibu::CameraModelGeneric<double>& ref_depth_cmod,
      const Sophus::SE3d&                       Tgd
    );


  ///////////////////////////////////////////////////////////////////////////
  /// Simplified configuration. Default BA options.
  void ConfigureBA(const calibu::CameraRig& rig);


  ///////////////////////////////////////////////////////////////////////////
  void ConfigureBA(
      const calibu::CameraRig&    rig,
      const ba::Options<double>&  options
    );


  ///////////////////////////////////////////////////////////////////////////
  void Estimate(
      const cv::Mat&  grey_image,
      const cv::Mat&  depth_image,
      double          time,
      Sophus::SE3d&   pose
    );


  ///////////////////////////////////////////////////////////////////////////
  void AddInertialMeasurement(
      const Eigen::Vector3d&  accel,
      const Eigen::Vector3d&  gyro,
      double                  time
    );




  // For debugging. Remove later.
  const ba::ImuResidualT<double,9,9>& GetImuResidual(const uint32_t id)
  {
    return bundle_adjuster_.GetImuResidual(id);
  }

  // For debugging. Remove later.
  const std::vector<uint32_t>& GetImuResidualIds()
  {
    return imu_residual_ids_;
  }


  ///
  ///////////////////////////////////////////////////////////////////////////

public:
  const unsigned int kWindowSize;
  const unsigned int kPyramidLevels;


private:
  struct DTrackPose {
    Sophus::SE3d      T_ab;
    double            time_a;
    double            time_b;
    Eigen::Matrix6d   covariance;
  };

  typedef ba::ImuMeasurementT<double>   ImuMeasurement;


private:
  bool                                              config_ba_;
  bool                                              config_dtrack_;
  bool                                              ba_has_converged_;

  calibu::CameraRig                                 rig_;
  Sophus::SE3d                                      current_pose_;
  double                                            current_time_;

  /// DTrack variables.
  Sophus::SE3d                                      Trv_;
  DTrack                                            dtrack_;
  Eigen::Matrix6d                                   dtrack_covariance_;
  std::deque<DTrackPose>                            dtrack_window_;

  /// BA variables.
  ba::BundleAdjuster<double, 0, 9, 0>               bundle_adjuster_;
  ba::Options<double>                               options_;
  std::deque<ba::PoseT<double> >                    ba_window_;
  ba::InterpolationBufferT<ImuMeasurement, double>  imu_buffer_;

  // TODO(jfalquez) Remove later. Only for debugging.
  std::vector<uint32_t>                             imu_residual_ids_;

};


} /* devil namespace */