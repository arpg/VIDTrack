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

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <ba/BundleAdjuster.h>
#include <ba/InterpolationBuffer.h>
#include <ba/Types.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <calibu/cam/CameraRig.h>

#include <devil/dtrack.h>


namespace devil {

/////////////////////////////////////////////////////////////////////////////
/// Convert greyscale image to float and normalizes.
inline cv::Mat ConvertAndNormalize(const cv::Mat& in)
{
  cv::Mat out;
  in.convertTo(out, CV_32FC1);
  out /= 255.0;
  return out;
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
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
      Sophus::SE3d&   global_pose,
      Sophus::SE3d&   rel_pose,
      Sophus::SE3d&   vo_pose
    );


  ///////////////////////////////////////////////////////////////////////////
  void AddInertialMeasurement(
      const Eigen::Vector3d&  accel,
      const Eigen::Vector3d&  gyro,
      double                  time
    );




  // For debugging. Remove later.
  const ba::ImuResidualT<double,15,15>& GetImuResidual(const uint32_t id)
  {
    return bundle_adjuster_.GetImuResidual(id);
  }

  // For debugging. Remove later.
  const std::vector<uint32_t>& GetImuResidualIds()
  {
    return imu_residual_ids_;
  }

  // For debugging. Remove later.
  const ba::ImuCalibrationT<double>& GetImuCalibration()
  {
    return bundle_adjuster_.GetImuCalibration();
  }

  // For debugging. Remove later.
  const ba::PoseT<double>& GetPose(const uint32_t id)
  {
    return bundle_adjuster_.GetPose(id);
  }

  // For debugging. Remove later.
  const ba::InterpolationBufferT<ba::ImuMeasurementT<double>, double>& GetImuBuffer()
  {
    return imu_buffer_;
  }

  // For debugging. Remove later.
  const std::deque<ba::PoseT<double> >& GetAdjustedPoses()
  {
    return ba_window_;
  }


  ///
  ///////////////////////////////////////////////////////////////////////////

public:
  const unsigned int kWindowSize;
  const unsigned int kMinWindowSize;
  const unsigned int kPyramidLevels;

  const double       kTimeOffset = 0.0;
//  const double       kTimeOffset = -0.00195049; // Old
//  const double       kTimeOffset = -0.00490676; // New

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
  Sophus::SE3d                                      Tic_;
  Sophus::SE3d                                      current_pose_;
  double                                            current_time_;

  /// DTrack variables.
  DTrack                                            dtrack_;
  Sophus::SE3d                                      last_estimated_pose_;
  std::deque<DTrackPose>                            dtrack_window_;

  /// BA variables.
  ba::BundleAdjuster<double, 0, 15, 0>              bundle_adjuster_;
  ba::Options<double>                               options_;
  std::deque<ba::PoseT<double> >                    ba_window_;
  ba::InterpolationBufferT<ImuMeasurement, double>  imu_buffer_;

  // TODO(jfalquez) Remove later. Only for debugging.
  std::vector<uint32_t>                             imu_residual_ids_;

};


} /* devil namespace */
