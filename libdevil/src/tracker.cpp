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

#include <devil/tracker.h>

#include <miniglog/logging.h>

using namespace devil;

inline Eigen::Vector3d R2Cart(const Eigen::Matrix3d& R) {
  Eigen::Vector3d rpq;
  // roll
  rpq[0] = atan2(R(2, 1), R(2, 2));

  // pitch
  double det = -R(2, 0) * R(2, 0) + 1.0;
  if (det <= 0) {
    if (R(2, 0) > 0) {
      rpq[1] = -M_PI / 2.0;
    } else {
      rpq[1] = M_PI / 2.0;
    }
  } else {
    rpq[1] = -asin(R(2, 0));
  }
  // yaw
  rpq[2] = atan2(R(1, 0), R(0, 0));
  return rpq;
}


inline Eigen::Matrix<double, 6, 1> T2Cart(const Eigen::Matrix4d& T) {
  Eigen::Matrix<double, 6, 1> Cart;
  Eigen::Vector3d rpq = R2Cart(T.block<3, 3>(0, 0));
  Cart[0] = T(0, 3);
  Cart[1] = T(1, 3);
  Cart[2] = T(2, 3);
  Cart[3] = rpq[0];
  Cart[4] = rpq[1];
  Cart[5] = rpq[2];
  return Cart;
}





///////////////////////////////////////////////////////////////////////////
Tracker::Tracker(unsigned int window_size, unsigned int pyramid_levels)
  : kWindowSize(window_size), kMinWindowSize(5), kPyramidLevels(pyramid_levels),
    config_ba_(false), config_dtrack_(false), ba_has_converged_(false),
    dtrack_(pyramid_levels)
{
}


///////////////////////////////////////////////////////////////////////////
Tracker::~Tracker()
{
}


///////////////////////////////////////////////////////////////////////////
void Tracker::ConfigureDTrack(
    const cv::Mat&                            keyframe_grey,
    const cv::Mat&                            keyframe_depth,
    double                                    time,
    const calibu::CameraModelGeneric<double>& cmod
    )
{
  ConfigureDTrack(keyframe_grey, keyframe_depth, time, cmod, cmod, cmod,
                  Sophus::SE3d());
}


///////////////////////////////////////////////////////////////////////////
void Tracker::ConfigureDTrack(
    const cv::Mat&                            keyframe_grey,
    const cv::Mat&                            keyframe_depth,
    double                                    time,
    const calibu::CameraModelGeneric<double>& live_grey_cmod,
    const calibu::CameraModelGeneric<double>& ref_grey_cmod,
    const calibu::CameraModelGeneric<double>& ref_depth_cmod,
    const Sophus::SE3d&                       Tgd
    )
{
  if (config_dtrack_) {
    LOG(WARNING) << "DTrack is already configured. Ignoring new configuration.";
  } else {
    dtrack_.SetParams(live_grey_cmod, ref_grey_cmod, ref_depth_cmod, Tgd);
    dtrack_.SetKeyframe(keyframe_grey, keyframe_depth);
    current_time_ = time;

    // Add initial pose to BA.
    ba::PoseT<double> initial_pose;
    initial_pose.t_wp = Sophus::SE3d();
    initial_pose.time = time + kTimeOffset;
    ba_window_.push_back(initial_pose);

    config_dtrack_ = true;
  }
}


///////////////////////////////////////////////////////////////////////////
void Tracker::ConfigureBA(const calibu::CameraRig& rig)
{
  // Set up default BA options.
  ba::Options<double> options;
  // The window is always considered batch since no pose is inactive,
  // so if this is set to true it will NOT change biases much.
  // If using datasets with no bias, set to true.
  options.regularize_biases_in_batch  = true;
  options.use_triangular_matrices     = false;
  options.use_sparse_solver           = false;
  options.use_dogleg                  = true;

  ConfigureBA(rig, options);
}


///////////////////////////////////////////////////////////////////////////
void Tracker::ConfigureBA(const calibu::CameraRig& rig,
                          const ba::Options<double>& options)
{
  if (config_ba_) {
    LOG(WARNING) << "BA is already configured. Ignoring new configuration.";
  } else {
    // Standardize camera rig and IMU-Camera transform.
    rig_ = calibu::ToCoordinateConvention(rig, calibu::RdfRobotics);

    // Set up robotic to vision permuation matrix.
    Sophus::SE3d Trv;
    Trv.so3() = calibu::RdfRobotics;

    // Tic holds IMU-Camera transform. It is used to convert camera poses from
    // vision to robotics, and then applies the transform of camera with
    // respect to IMU -- thus bringing poses to IMU reference frame, required
    // by BA. NOTE: When using Vicalib, IMU is "world" origin.
    Tic_ = rig_.cameras[0].T_wc * Trv;
    LOG(INFO) << "Twc:" << std::endl << rig_.cameras[0].T_wc.matrix();
    LOG(INFO) << "Tic:" << std::endl << Tic_.matrix();

    // Transform officially all cameras in rig. This is done for compatibility,
    // in case actual reprojection residuals are used.
    for (calibu::CameraModelAndTransform& model : rig_.cameras) {
      model.T_wc = model.T_wc * Trv;
    }

    // Set gravity.
    Eigen::Matrix<double, 3, 1> gravity;
//    gravity << 0, -9.806, 0; // CityBlock
//    gravity << 0, 0, 9.806; // PathGen
    gravity << 0, 0, -9.806; // Rig
    bundle_adjuster_.SetGravity(gravity);

    // Set up BA options.
    options_ = options;

    // Set up debug options for BA.
    bundle_adjuster_.debug_level_threshold = -1;

    config_ba_ = true;
  }
}

#define USE_IMU 1

///////////////////////////////////////////////////////////////////////////
void Tracker::Estimate(
    const cv::Mat&  grey_image,
    const cv::Mat&  depth_image,
    double          time,
    Sophus::SE3d&   global_pose,
    Sophus::SE3d&   rel_pose,
    Sophus::SE3d&   vo_pose
  )
{
  CHECK(config_ba_ && config_dtrack_)
      << "DTrack and BA must be configured first before calling this method!";

  // Adjust time offset.
  time = time + kTimeOffset;

  Sophus::SE3d        rel_pose_estimate;
  Eigen::Matrix6d     dtrack_covariance;

  ///--------------------
  /// If BA has converged, integrate IMU measurements (if available) instead
  /// of doing full pyramid.
  bool use_pyramid = true;
  if (ba_has_converged_) {
    // Get IMU measurements between keyframe and current frame.
    CHECK_LT(current_time_, time);
    std::vector<ImuMeasurement> imu_measurements =
        imu_buffer_.GetRange(current_time_, time);

    if (imu_measurements.size() < 3) {
      LOG(WARNING) << "Not integrating IMU since few measurements were found between: " <<
                      current_time_ << " and " << time;
      LOG(WARNING) << "Doing full pyramid visual only estimation instead.";
    } else {
      std::vector<ba::ImuPoseT<double> > imu_poses;

      ba::PoseT<double>& last_adjusted_pose = ba_window_.back();
      CHECK_EQ(current_time_, last_adjusted_pose.time);

      ba::ImuPoseT<double> new_pose =
          decltype(bundle_adjuster_)::ImuResidual::IntegrateResidual(
            last_adjusted_pose, imu_measurements,
            last_adjusted_pose.b.head<3>(), last_adjusted_pose.b.tail<3>(),
            bundle_adjuster_.GetImuCalibration().g_vec, imu_poses);

      // Get new relative transform to seed ESM.
      rel_pose_estimate = last_adjusted_pose.t_wp.inverse() * new_pose.t_wp;

      // Transform rel pose from IMU to camera frame.
      rel_pose_estimate = Tic_.inverse() * rel_pose_estimate * Tic_;

      // Do not use pyramid since the IMU should already have us at basin.
      use_pyramid = false;
    }
  }


  ///--------------------
  /// RGBD pose estimation.
  double dtrack_error;
  if (use_pyramid) {
    // TODO(jfalquez) If constant velocity model is to be used, this is the
    // place to add it before calling DTrack's estimate. Do not use it on the
    // else statement, since rel_pose_estimate should have IMU integration.
    dtrack_error = dtrack_.Estimate(grey_image, rel_pose_estimate,
                                    dtrack_covariance, true);
  } else {
    dtrack_error = dtrack_.Estimate(grey_image, rel_pose_estimate,
                                    dtrack_covariance, false);
  }

  // Transfer covariance to IMU frame.
//  dtrack_covariance = rel_pose_estimate.Adj() * dtrack_covariance
//      * rel_pose_estimate.Adj().inverse();

  // Transfer relative pose to IMU frame.
  rel_pose_estimate = Tic_ * rel_pose_estimate * Tic_.inverse();

  vo_pose = rel_pose_estimate;

  // Push pose estimate into DTrack window.
  DTrackPose dtrack_rel_pose;
  dtrack_rel_pose.T_ab        = rel_pose_estimate;
  dtrack_rel_pose.covariance  = dtrack_covariance; // 10000;
  dtrack_rel_pose.time_a      = current_time_;
  dtrack_rel_pose.time_b      = time;
  dtrack_window_.push_back(dtrack_rel_pose);

  // Set current frame as new keyframe.
  dtrack_.SetKeyframe(grey_image, depth_image);

  // Get latest adjusted pose.
  ba::PoseT<double>& latest_adjusted_pose = ba_window_.back();

  // Add most recent pose to BA.
  ba::PoseT<double> latest_pose;
  latest_pose.t_wp = latest_adjusted_pose.t_wp * dtrack_rel_pose.T_ab;
  latest_pose.time = time;
  ba_window_.push_back(latest_pose);


  ///--------------------
  /// Windowed BA.
  if (dtrack_window_.size() >= kMinWindowSize) {
    // Sanity check.
    CHECK_EQ(ba_window_.size(), dtrack_window_.size()+1)
        << "BA: " << ba_window_.size() << " DTrack: " << dtrack_window_.size();

    bundle_adjuster_.Init(options_, kWindowSize, kWindowSize*10);

    // Reset IMU residuals IDs.
    imu_residual_ids_.clear();

    // Push first pose and keep track of ID.
    int cur_id, prev_id;
    ba::PoseT<double>& front_adjusted_pose = ba_window_.front();
    prev_id = bundle_adjuster_.AddPose(front_adjusted_pose.t_wp,
                                       front_adjusted_pose.cam_params,
                                       front_adjusted_pose.v_w,
                                       front_adjusted_pose.b, true,
                                       front_adjusted_pose.time);

    // Set this pose as root ID.
    bundle_adjuster_.SetRootPoseId(prev_id);

    // Push rest of BA poses.
    for (size_t ii = 1; ii < ba_window_.size(); ++ii) {
      ba::PoseT<double>& adjusted_pose = ba_window_[ii];
      cur_id = bundle_adjuster_.AddPose(adjusted_pose.t_wp,
                                        adjusted_pose.cam_params,
                                        adjusted_pose.v_w,
                                        adjusted_pose.b, true,
                                        adjusted_pose.time);

      DTrackPose& dtrack_rel_pose = dtrack_window_[ii-1];

      CHECK_EQ(adjusted_pose.time, dtrack_rel_pose.time_b);

      // Add binary constraints.
      CHECK_EQ(cur_id-1, prev_id);
      bundle_adjuster_.AddBinaryConstraint(prev_id, cur_id,
                                           dtrack_rel_pose.T_ab,
                                           dtrack_rel_pose.covariance);

      // Get IMU measurements between frames.
      std::vector<ImuMeasurement> imu_measurements =
          imu_buffer_.GetRange(dtrack_rel_pose.time_a, dtrack_rel_pose.time_b);

#if USE_IMU
      // Add IMU constraints.
      imu_residual_ids_.push_back(
            bundle_adjuster_.AddImuResidual(prev_id, cur_id, imu_measurements));
#endif

      // Update pose IDs.
      prev_id = cur_id;
    }

    // Solve.
    bundle_adjuster_.Solve(1000, 1.0);
    ba_has_converged_ = true;

    // Get adjusted poses.
    ba_window_.clear();
    for (size_t ii = 0; ii < bundle_adjuster_.GetNumPoses(); ++ii) {
      ba_window_.push_back(bundle_adjuster_.GetPose(ii));
    }

    // Pop front element of DTrack estimates.
    if (dtrack_window_.size() == kWindowSize) {
      dtrack_window_.pop_front();
      ba_window_.pop_front();
    }
  }

  // If BA has not converged yet, return visual only global pose.
  // Otherwise, return BA's adjusted pose.
  if (ba_has_converged_ == false) {
    current_pose_ *= rel_pose_estimate;
    rel_pose = rel_pose_estimate;
  } else {
    ba::PoseT<double>& last_adjusted_pose = ba_window_.back();
    current_pose_ = last_adjusted_pose.t_wp;
    ba::PoseT<double>& last_last_adjusted_pose = ba_window_[ba_window_.size()-2];
    rel_pose = last_last_adjusted_pose.t_wp.inverse() * last_adjusted_pose.t_wp;
  }

  // Update last estimated pose.
  last_estimated_pose_ = global_pose;

  // Update time.
  current_time_ = time;

  // Update return pose.
  global_pose = current_pose_;
}


///////////////////////////////////////////////////////////////////////////
void Tracker::AddInertialMeasurement(
    const Eigen::Vector3d&  accel,
    const Eigen::Vector3d&  gyro,
    double                  time
    )
{
//  std::cout << "Accel: " << accel.transpose() << std::endl;
  ImuMeasurement imu(gyro, accel, time);
  imu_buffer_.AddElement(imu);
}


