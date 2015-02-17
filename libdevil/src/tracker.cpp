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
  : kWindowSize(window_size), kPyramidLevels(pyramid_levels), config_ba_(false),
    config_dtrack_(false), ba_has_converged_(false), dtrack_(pyramid_levels)
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
    config_dtrack_ = true;
  }
}


///////////////////////////////////////////////////////////////////////////
void Tracker::ConfigureBA(const calibu::CameraRig& rig)
{
  // Set up default BA options.
  ba::Options<double> options;
  options.regularize_biases_in_batch  = true;
  options.use_triangular_matrices = false;
  options.use_sparse_solver = false;
  options.use_dogleg = true;

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

    // Transform offiially all cameras in rig. This is done for compatibility,
    // in case actual reprojection residuals are used.
    for (calibu::CameraModelAndTransform& model : rig_.cameras) {
      model.T_wc = model.T_wc * Trv;
    }

    // Set gravity.
    Eigen::Matrix<double, 3, 1> gravity;
//    gravity << 0, -9.806, 0;
    gravity << 0, 0, -9.806;
//    bundle_adjuster_.SetGravity(gravity);

    // Set up BA options.
    options_ = options;
    ba::debug_level_threshold = -1;

    current_pose_ = Tic_;

    config_ba_ = true;
  }
}


///////////////////////////////////////////////////////////////////////////
void Tracker::Estimate(
    const cv::Mat&  grey_image,
    const cv::Mat&  depth_image,
    double          time,
    Sophus::SE3d&   pose
    )
{
  CHECK(config_ba_ && config_dtrack_)
      << "DTrack and BA must be configured first before calling this method!";

  Sophus::SE3d rel_pose_estimate;

  Eigen::Matrix6d cov;
  cov.setIdentity();
  cov *= 1e16;

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

      ba::PoseT<double> last_adjusted_pose =
          bundle_adjuster_.GetPose(bundle_adjuster_.GetNumPoses()-1);

      ba::ImuPoseT<double> new_pose =
          decltype(bundle_adjuster_)::ImuResidual::IntegrateResidual(
            last_adjusted_pose, imu_measurements,
            last_adjusted_pose.b.head<3>(), last_adjusted_pose.b.tail<3>(),
            bundle_adjuster_.GetImuCalibration().g_vec, imu_poses);

      // Get new relative transform to seed ESM.
      rel_pose_estimate = last_adjusted_pose.t_wp.inverse() * new_pose.t_wp;

      std::cout << "Integrated Pose: " << rel_pose_estimate.log().transpose() << std::endl;

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
                                    dtrack_covariance_, true);
  } else {
    dtrack_error = dtrack_.Estimate(grey_image, rel_pose_estimate,
                                    dtrack_covariance_, false);
  }

  // Push pose estimate into DTrack window.
  DTrackPose dtrack_rel_pose;
  dtrack_rel_pose.T_ab        = rel_pose_estimate;
  dtrack_rel_pose.covariance  = dtrack_covariance_;
  dtrack_rel_pose.time_a      = current_time_;
  dtrack_rel_pose.time_b      = time;
  dtrack_window_.push_back(dtrack_rel_pose);

  std::cout << "VO Pose Estimate: " << Sophus::SE3::log(rel_pose_estimate).transpose() << std::endl;

  // Set current frame as new keyframe.
  dtrack_.SetKeyframe(grey_image, depth_image);


  ///--------------------
  /// Windowed BA.
  if (dtrack_window_.size() == kWindowSize) {
    bundle_adjuster_.Init(options_, 10, 1000);
    bundle_adjuster_.AddCamera(rig_.cameras[0].camera, rig_.cameras[0].T_wc);

    // Reset IMU residuals IDs.
    imu_residual_ids_.clear();

    // The first time BA runs is special, since no BA poses have yet been created.
    if (ba_has_converged_ == false) {
      // kWindowSize holds size of deque of relative DTrack transforms.
      // BA window holds global poses, so we need one more pose to hold
      // all poses in window.
      // First global pose is world origin.
      Sophus::SE3d global_pose = Tic_;

      // Push first pose and keep track of ID.
      int cur_id, prev_id;
      prev_id = bundle_adjuster_.AddPose(global_pose, true,
                                         dtrack_window_[0].time_a);
      std::cout << "Pose Added: " << T2Cart((global_pose).matrix()).transpose() << std::endl;

      // Push rest of DTrack binary constraints.
      for (size_t ii = 0; ii < dtrack_window_.size(); ++ii) {
        DTrackPose& dtrack_rel_pose = dtrack_window_[ii];
        global_pose *= dtrack_rel_pose.T_ab;

        std::cout << "Pose Added: " << T2Cart((global_pose).matrix()).transpose() << std::endl;

        cur_id = bundle_adjuster_.AddPose(global_pose, true,
                                          dtrack_rel_pose.time_b);

        bundle_adjuster_.AddBinaryConstraint(prev_id, cur_id,
                                             dtrack_rel_pose.T_ab,
                                             dtrack_rel_pose.covariance);
//                                             cov);

        // Get IMU measurements between frames.
        std::vector<ImuMeasurement> imu_measurements =
            imu_buffer_.GetRange(dtrack_rel_pose.time_a, dtrack_rel_pose.time_b);

        // Add IMU constraints.
        imu_residual_ids_.push_back(
              bundle_adjuster_.AddImuResidual(prev_id, cur_id, imu_measurements));

        // Update pose IDs.
        prev_id = cur_id;
      }
    } else {
      // Push first pose and keep track of ID.
      int cur_id, prev_id;
      ba::PoseT<double>& front_adjusted_pose = ba_window_.front();
      prev_id = bundle_adjuster_.AddPose(front_adjusted_pose.t_wp,
                                         front_adjusted_pose.t_vs,
                                         front_adjusted_pose.cam_params,
                                         front_adjusted_pose.v_w,
                                         front_adjusted_pose.b, true,
                                         front_adjusted_pose.time);
      std::cout << "Pose Added: " << T2Cart(front_adjusted_pose.t_wp.matrix()).transpose() << std::endl;

      CHECK_EQ(ba_window_.size(), dtrack_window_.size())
          << "BA: " << ba_window_.size() << " DTrack: " << dtrack_window_.size();

      // Push rest of BA poses.
      for (size_t ii = 1; ii < ba_window_.size(); ++ii) {
        ba::PoseT<double>& adjusted_pose = ba_window_[ii];
        cur_id = bundle_adjuster_.AddPose(adjusted_pose.t_wp,
                                          adjusted_pose.t_vs,
                                          adjusted_pose.cam_params,
                                          adjusted_pose.v_w,
                                          adjusted_pose.b, true,
                                          adjusted_pose.time);
        std::cout << "Pose Added: " << T2Cart(adjusted_pose.t_wp.matrix()).transpose() << std::endl;

        DTrackPose& dtrack_rel_pose = dtrack_window_[ii-1];

        CHECK_EQ(adjusted_pose.time, dtrack_rel_pose.time_b);

        // Add binary constraints.
        bundle_adjuster_.AddBinaryConstraint(prev_id, cur_id,
                                             dtrack_rel_pose.T_ab,
                                             dtrack_rel_pose.covariance);
//                                             cov);

        // Get IMU measurements between frames.
        std::vector<ImuMeasurement> imu_measurements =
            imu_buffer_.GetRange(dtrack_rel_pose.time_a, dtrack_rel_pose.time_b);

        // Add IMU constraints.
        imu_residual_ids_.push_back(
              bundle_adjuster_.AddImuResidual(prev_id, cur_id, imu_measurements));

        // Update pose IDs.
        prev_id = cur_id;
      }

      // Create new pose and add to BA.
      ba::PoseT<double>& last_adjusted_pose = ba_window_.back();
      CHECK_EQ(kWindowSize, dtrack_window_.size());
      DTrackPose& dtrack_rel_pose = dtrack_window_[kWindowSize-1];
      std::cout << "Last Adjusted Pose: " << T2Cart(last_adjusted_pose.t_wp.matrix()).transpose() << std::endl;
      std::cout << "Estimated Rel T: " << Sophus::SE3::log(dtrack_rel_pose.T_ab).transpose() << std::endl;
      Sophus::SE3d global_pose = last_adjusted_pose.t_wp * dtrack_rel_pose.T_ab;

      cur_id = bundle_adjuster_.AddPose(global_pose, true, dtrack_rel_pose.time_b);

      std::cout << "Pose Added: " << T2Cart(global_pose.matrix()).transpose() << std::endl;

      bundle_adjuster_.AddBinaryConstraint(prev_id, cur_id,
                                           dtrack_rel_pose.T_ab,
                                           dtrack_rel_pose.covariance);
//                                           cov);

      // Get IMU measurements between frames.
      std::vector<ImuMeasurement> imu_measurements =
          imu_buffer_.GetRange(dtrack_rel_pose.time_a, dtrack_rel_pose.time_b);

      // Add IMU constraints.
      imu_residual_ids_.push_back(
            bundle_adjuster_.AddImuResidual(prev_id, cur_id, imu_measurements));
    }

    // Solve.
    bundle_adjuster_.Solve(1000, 1.0);

    // Get adjusted poses.
    if (ba_has_converged_ == false) {
      // Since this is the first time it runs, push ALL poses.
      CHECK_EQ(bundle_adjuster_.GetNumPoses(), kWindowSize+1);
      for (size_t ii = 0; ii < bundle_adjuster_.GetNumPoses(); ++ii) {
        ba_window_.push_back(bundle_adjuster_.GetPose(ii));
      }

      std::cout << "First pose should not have moved: " <<
                   ba_window_[0].t_wp.log().transpose() << std::endl;

      ba_has_converged_ = true;
    } else {
      // Push in last adjusted pose to BA deque.
      ba_window_.push_back(
            bundle_adjuster_.GetPose(bundle_adjuster_.GetNumPoses() - 1));
    }

    ba::PoseT<double>& last_adjusted_pose = ba_window_.back();

    std::cout << "Last Adjusted Pose: " <<
                 T2Cart(last_adjusted_pose.t_wp.matrix()).transpose() << std::endl;


    // Pop first elements of each queue
    dtrack_window_.pop_front();
    ba_window_.pop_front();
  }

  // If BA has not converged yet, return visual only global pose.
  // Otherwise, return BA's adjusted pose.
  if (ba_has_converged_ == false) {
    current_pose_ *= rel_pose_estimate;
  } else {
    CHECK_EQ(bundle_adjuster_.GetNumPoses(), kWindowSize+1);
    ba::PoseT<double> last_adjusted_pose =
        bundle_adjuster_.GetPose(bundle_adjuster_.GetNumPoses() - 1);
    current_pose_ = last_adjusted_pose.t_wp;
  }

  // Update time.
  current_time_ = time;

  // Update return pose.
  pose = current_pose_;
}


///////////////////////////////////////////////////////////////////////////
void Tracker::AddInertialMeasurement(
    const Eigen::Vector3d&  accel,
    const Eigen::Vector3d&  gyro,
    double                  time
    )
{
  ImuMeasurement imu(gyro, accel, time);
  imu_buffer_.AddElement(imu);
}
