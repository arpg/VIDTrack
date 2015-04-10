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

#include <vidtrack/tracker.h>

#include <miniglog/logging.h>

using namespace vid;

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

inline Eigen::Matrix4d Cart2T(
    double x,
    double y,
    double z,
    double r,
    double p,
    double q) {
  Eigen::Matrix4d T;
  // psi = roll, th = pitch, phi = yaw
  double cq, cp, cr, sq, sp, sr;
  cr = cos(r);
  cp = cos(p);
  cq = cos(q);

  sr = sin(r);
  sp = sin(p);
  sq = sin(q);

  T(0, 0) = cp * cq;
  T(0, 1) = -cr * sq + sr * sp * cq;
  T(0, 2) = sr * sq + cr * sp * cq;

  T(1, 0) = cp * sq;
  T(1, 1) = cr * cq + sr * sp * sq;
  T(1, 2) = -sr * cq + cr * sp * sq;

  T(2, 0) = -sp;
  T(2, 1) = sr * cp;
  T(2, 2) = cr * cp;

  T(0, 3) = x;
  T(1, 3) = y;
  T(2, 3) = z;
  T.row(3) = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  return T;
}


///////////////////////////////////////////////////////////////////////////
/// SAD score image1 with image2 -- images are assumed to be same type
/// and dimensions.
/// returns: SAD score
template<typename T>
inline float ScoreImages(
    const cv::Mat&              image1,
    const cv::Mat&              image2
  )
{
  float score = 0;
  for (int ii = 0; ii < image1.rows; ii++) {
    for (int jj = 0; jj < image1.cols; jj++) {
      score += fabs(image1.at<T>(ii, jj) - image2.at<T>(ii, jj));
    }
  }
  return score;
}

///////////////////////////////////////////////////////////////////////////
bool _CompareNorm(
    std::pair<unsigned int, float> lhs,
    std::pair<unsigned int, float> rhs
  )
{
  return std::get<1>(lhs) < std::get<1>(rhs);
}


///////////////////////////////////////////////////////////////////////////
void Tracker::FindLoopClosureCandidates(
    int                                             margin,
    int                                             id,
    const cv::Mat&                                  thumbnail,
    float                                           max_intensity_change,
    std::vector<std::pair<unsigned int, float> >&   candidates
  )
{
  CHECK_GE(margin, 0);
  CHECK_GE(id, 0);
  CHECK_LE(static_cast<size_t>(id), dtrack_vector_.size());

  const float max_score = max_intensity_change
                          * (thumbnail.rows * thumbnail.cols);
  for (unsigned int ii = 0; ii < dtrack_vector_.size(); ++ii) {
    if (abs(id - ii) >= margin) {
      DTrackPoseOut& dtrack_estimate = dtrack_vector_[ii];

      float score = ScoreImages<unsigned char>(thumbnail,
                                               dtrack_estimate.thumbnail);

      if (score < max_score) {
        candidates.push_back(std::pair<unsigned int, float>(ii, score));
      }
    }
  }

  // Sort vector by score.
  std::sort(candidates.begin(), candidates.end(), _CompareNorm);
}



///////////////////////////////////////////////////////////////////////////
Tracker::Tracker(unsigned int window_size, unsigned int pyramid_levels)
  : kWindowSize(window_size), kMinWindowSize(2), kPyramidLevels(pyramid_levels),
    config_ba_(false), config_dtrack_(false), ba_has_converged_(false),
    dtrack_(pyramid_levels), dtrack_refine_(pyramid_levels)
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
    dtrack_refine_.SetParams(live_grey_cmod, ref_grey_cmod, ref_depth_cmod, Tgd);
    dtrack_.SetParams(live_grey_cmod, ref_grey_cmod, ref_depth_cmod, Tgd);
    dtrack_.SetKeyframe(keyframe_grey, keyframe_depth);
    current_time_ = time;

    // Add initial pose to BA.
    ba::PoseT<double> initial_pose;
    initial_pose.SetZero();
    initial_pose.t_wp = Sophus::SE3d();
    initial_pose.time = time + kTimeOffset;
    ba_window_.push_back(initial_pose);

    // NOTE(jfalquez) This first one is not used during optimization.
    // It is only required to store the images of the first pose.
    DTrackPoseOut dtrack_rel_pose_out;
    dtrack_rel_pose_out.time_a      = 0;
    dtrack_rel_pose_out.time_b      = 0;
    dtrack_rel_pose_out.grey_img    = keyframe_grey.clone();
    dtrack_rel_pose_out.depth_img   = keyframe_depth.clone();
    dtrack_rel_pose_out.thumbnail   = GenerateThumbnail(keyframe_grey).clone();
    dtrack_vector_.push_back(dtrack_rel_pose_out);

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
  options.use_triangular_matrices     = true;
  // NOTE(jfalquez) When this is set to true, and no IMU residuals are added
  // an error/warning shows up.
  options.use_sparse_solver           = true;
  options.use_dogleg                  = true;

  // IMU Sigmas.
  options.accel_sigma       = 0.001883649; //0.0392266
  options.accel_bias_sigma  = 1.2589254e-2;
  options.gyro_sigma        = 5.3088444e-5; //0.00104719755
  options.gyro_bias_sigma   = 1.4125375e-4;

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
    gravity << 0, 0, 9.806; // PathGen
//    gravity << 0, 0, -9.806; // Rig
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

  ///--------------------
  /// If BA has converged, integrate IMU measurements (if available) instead
  /// of doing full pyramid.
  bool use_pyramid = true;
  if (ba_has_converged_ && false) {
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
  unsigned int        dtrack_num_obs;
  double              dtrack_error;
  Eigen::Matrix6d     dtrack_covariance;

  if (use_pyramid) {
    // TODO(jfalquez) If constant velocity model is to be used, this is the
    // place to add it before calling DTrack's estimate. Do not use it on the
    // else statement, since rel_pose_estimate should have IMU integration.
    dtrack_error = dtrack_.Estimate(true, grey_image, rel_pose_estimate,
                                    dtrack_covariance, dtrack_num_obs,
                                    depth_image);
  } else {
    dtrack_error = dtrack_.Estimate(false, grey_image, rel_pose_estimate,
                                    dtrack_covariance, dtrack_num_obs,
                                    depth_image);
  }

  LOG_IF(WARNING, dtrack_num_obs < (grey_image.cols*grey_image.rows*0.3))
      << "Number of observations for DTrack is less than 30%!";

  // Transfer covariance to IMU frame.
//  dtrack_covariance = rel_pose_estimate.Adj().inverse() * dtrack_covariance
//      * rel_pose_estimate.Adj();

  // Transfer relative pose to IMU frame.
  rel_pose_estimate = Tic_ * rel_pose_estimate * Tic_.inverse();

  vo_pose = rel_pose_estimate;

  // Push pose estimate into DTrack window.
  DTrackPose dtrack_rel_pose;
  dtrack_rel_pose.T_ab        = rel_pose_estimate;
  dtrack_rel_pose.covariance  = dtrack_covariance;
  dtrack_rel_pose.time_a      = current_time_;
  dtrack_rel_pose.time_b      = time;
  dtrack_window_.push_back(dtrack_rel_pose);

  // Set current frame as new keyframe.
  dtrack_.SetKeyframe(grey_image, depth_image);

  // Get latest adjusted pose.
  ba::PoseT<double>& latest_adjusted_pose = ba_window_.back();

  // Add most recent pose to BA.
  // TODO(jfalquez) Find out the behavior of this pose... does setting
  // velocities or anything affect BA?
  ba::PoseT<double> latest_pose;
  latest_pose = latest_adjusted_pose;
  latest_pose.t_wp = latest_adjusted_pose.t_wp * dtrack_rel_pose.T_ab;
  latest_pose.time = time;
  ba_window_.push_back(latest_pose);


  ///--------------------
  /// Windowed BA.
  if (dtrack_window_.size() >= kMinWindowSize && false) {
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
    bundle_adjuster_.Solve(1000, 1.0, false);

    // NOTE(jfalquez) This is a hack since BA has that weird memory problem
    // and the minimum window has to be set to 2. However, the real minimum
    // window is controlled here.
    if (ba_window_.size() == 10) {
      ba_has_converged_ = true;
    }

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

  // "Map".
  DTrackPoseOut dtrack_rel_pose_out;
  dtrack_rel_pose_out.T_ab        = rel_pose;
  dtrack_rel_pose_out.covariance  = dtrack_covariance;
  dtrack_rel_pose_out.time_a      = current_time_;
  dtrack_rel_pose_out.time_b      = time;
  dtrack_rel_pose_out.grey_img    = grey_image.clone();
  dtrack_rel_pose_out.depth_img   = depth_image.clone();
  dtrack_rel_pose_out.thumbnail   = GenerateThumbnail(grey_image).clone();
  dtrack_vector_.push_back(dtrack_rel_pose_out);
}


///////////////////////////////////////////////////////////////////////////
void Tracker::RefinePose(
    const cv::Mat&    grey_image,
    int               keyframe_id,
    Sophus::SE3d&     Twp
  )
{
  // Localize against that keyframe. Potentially localize against previous
  // keyframe too, for robustness. Refine with keyframes?
  DTrackMap& map_frame = dtrack_map_[keyframe_id];

  // Set keyframe.
  dtrack_refine_.SetKeyframe(map_frame.grey_img, map_frame.depth_img);

  // Find relative transform between current pose and keyframe.
  Sophus::SE3d Tkc = map_frame.T_wp.inverse() * Twp;

  // Set up robotic to vision permuation matrix.
  Sophus::SE3d Trv;
  Trv.so3() = calibu::RdfRobotics;

  Tkc = Trv.inverse() * Tkc * Trv;

  /// RGBD pose estimation.
  unsigned int        dtrack_num_obs;
  double              dtrack_error;
  Eigen::Matrix6d     dtrack_covariance;

  dtrack_error = dtrack_refine_.Estimate(true, grey_image, Tkc,
                                  dtrack_covariance, dtrack_num_obs,
                                  grey_image);

  LOG_IF(WARNING, dtrack_num_obs < (grey_image.cols*grey_image.rows*0.3))
      << "Number of observations for DTrack is less than 30%!";

  Tkc = Trv * Tkc * Trv.inverse();

  Twp = map_frame.T_wp * Tkc;
}


///////////////////////////////////////////////////////////////////////////
int Tracker::FindClosestKeyframe(
    int               last_frame_id,
    Sophus::SE3d      Twp,
    int               range
  )
{
  DTrackMap& map_frame = dtrack_map_[last_frame_id];

  int     closest_id = last_frame_id;
  double  closest_distance = (Twp.inverse() * map_frame.T_wp).translation().norm();

  int ii = last_frame_id;
  int jj = last_frame_id;
  // This doesn't take in consideration rotation.
  for (int xx = 0; xx < range; ++xx) {
    ii--;
    jj++;

    if (ii < 0) {
      ii = dtrack_map_.size() - 1;
    }
    if (static_cast<size_t>(jj) >= dtrack_map_.size()) {
      jj = 0;
    }

    if (ii == jj) {
      break;
    }

    DTrackMap& map_frame_ii = dtrack_map_[ii];
    double ii_distance = (Twp.inverse() * map_frame_ii.T_wp).translation().norm();
    if (ii_distance < closest_distance) {
      closest_distance = ii_distance;
      closest_id = ii;
    }

    DTrackMap& map_frame_jj = dtrack_map_[jj];
    double jj_distance = (Twp.inverse() * map_frame_jj.T_wp).translation().norm();
    if (jj_distance < closest_distance) {
      closest_distance = jj_distance;
      closest_id = jj;
    }
  }
  return closest_id;
}

///////////////////////////////////////////////////////////////////////////
void Tracker::ExportMap()
{
  std::ofstream fw;
  fw.open("map/poses.txt");

  for (size_t ii = 0; ii < dtrack_vector_.size(); ++ii) {
    DTrackPoseOut& dtrack_pose = dtrack_vector_[ii];

    // Export file of poses.
    fw << T2Cart(dtrack_pose.T_wp.matrix()).transpose() << std::endl;

    // Export grey and depth images.
    const std::string depth_file_prefix = "map/depth_";
    char index[10];
    int  i_idx = static_cast<int>(ii);
    sprintf(index, "%05d", i_idx);
    std::string depth_filename;
    depth_filename = depth_file_prefix + index + ".pdm";
    std::ofstream file(depth_filename.c_str(), std::ios::out | std::ios::binary);
    file << "P7" << std::endl;
    file << dtrack_pose.depth_img.cols << " " << dtrack_pose.depth_img.rows << std::endl;
    unsigned int size = dtrack_pose.depth_img.elemSize1()
        * dtrack_pose.depth_img.rows * dtrack_pose.depth_img.cols;
    file << 4294967295 << std::endl;
    file.write((const char*)dtrack_pose.depth_img.data, size);
    file.close();

    // Save grey image.
    std::string grey_prefix = "map/grey_";
    std::string grey_filename;
    grey_filename = grey_prefix + index + ".pgm";
    cv::imwrite(grey_filename, dtrack_pose.grey_img);

    std::cout << "-- Saving: " << depth_filename << " " <<
              grey_filename << std::endl;
  }
}


///////////////////////////////////////////////////////////////////////////
void Tracker::ImportMap(const std::string& map_path)
{
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "Importing map..." << std::endl;
  std::cout << "Path: " << map_path << std::endl;

  std::string poses_file = map_path + "/poses.txt";

  FILE* fd = fopen(poses_file.c_str(), "r");
  float x, y, z, p, q, r;

  int index = 0;
  while (fscanf(fd, "%f\t%f\t%f\t%f\t%f\t%f", &x, &y, &z, &p, &q, &r) != EOF) {
    Sophus::SE3d frame_pose(Cart2T(x, y, z, p, q,r));

    DTrackMap map_frame;
    map_frame.T_wp = frame_pose;

    char index_string[10];
    int  i_idx = static_cast<int>(index);
    sprintf(index_string, "%05d", i_idx);

    const std::string depth_file_prefix = "/depth_";
    std::string depth_filename;
    depth_filename = map_path + depth_file_prefix + index_string + ".pdm";

    std::ifstream depth_file(depth_filename.c_str());

    unsigned int        depth_width;
    unsigned int        depth_height;
    long unsigned int   image_size;

    if (depth_file.is_open()) {
      std::string file_type;
      depth_file >> file_type;
      depth_file >> depth_width;
      depth_file >> depth_height;
      depth_file >> image_size;

      image_size = 4 * depth_width * depth_height;

      map_frame.depth_img = cv::Mat(depth_height, depth_width, CV_32FC1);

      depth_file.seekg(depth_file.tellg() + (std::ifstream::pos_type)1, std::ios::beg);
      depth_file.read((char*)map_frame.depth_img.data, image_size);
      depth_file.close();
    }

    std::string grey_prefix = "/grey_";
    std::string grey_filename;
    grey_filename = map_path + grey_prefix + index_string + ".pgm";
    map_frame.grey_img = cv::imread(grey_filename, -1);
    // Build pyramids.
    std::vector<cv::Mat> pyramid;
    const int thumb_level = 4;
    cv::buildPyramid(map_frame.grey_img, pyramid, thumb_level);
    map_frame.thumbnail = pyramid[thumb_level-1].clone();


    dtrack_map_.push_back(map_frame);

    index++;
  }
  fclose(fd);
  std::cout << "Size of map loaded: " << dtrack_map_.size() << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
}


///////////////////////////////////////////////////////////////////////////
void Tracker::RunBatchBAwithLC()
{
  // Init pose only BA.
  pose_relaxer_.Init(options_, dtrack_vector_.size(), dtrack_vector_.size()*5);

  // Reset IMU residuals IDs.
  imu_residual_ids_.clear();

  ///-------------------- PUSH VO AND IMU CONSTRAINTS
  // Push first pose and keep track of ID.
  int cur_id, prev_id;
  Sophus::SE3d global_pose;
  DTrackPoseOut& dtrack_estimate = dtrack_vector_[1];
  prev_id = pose_relaxer_.AddPose(global_pose, true, dtrack_estimate.time_a);

  // Push rest of BA poses.
  for (size_t ii = 2; ii < dtrack_vector_.size(); ++ii) {
    DTrackPoseOut& dtrack_estimate = dtrack_vector_[ii-1];
    global_pose = global_pose * dtrack_estimate.T_ab;

    cur_id = pose_relaxer_.AddPose(global_pose, true, dtrack_estimate.time_b);

    // Add VO constraint.
    pose_relaxer_.AddBinaryConstraint(prev_id, cur_id,
                                         dtrack_estimate.T_ab,
                                         dtrack_estimate.covariance);

#if 0
    // Get IMU measurements between frames.
    std::vector<ImuMeasurement> imu_measurements =
        imu_buffer_.GetRange(dtrack_estimate.time_a, dtrack_estimate.time_b);

    // Add IMU constraint.
    imu_residual_ids_.push_back(
          pose_relaxer_.AddImuResidual(prev_id, cur_id, imu_measurements));
#endif

    // Update pose IDs.
    prev_id = cur_id;
  }


  ///-------------------- CHECK LOOP CLOSURES AND ADD LC CONSTRAINTS

  for (size_t ii = 0; ii < dtrack_vector_.size(); ++ii) {
    DTrackPoseOut& dtrack_estimate = dtrack_vector_[ii];

    std::vector<std::pair<unsigned int, float> > candidates;
    FindLoopClosureCandidates(30, ii, dtrack_estimate.thumbnail,
                              10.0, candidates);

#if 0
    if (!candidates.empty()) {
      cv::imshow("Keyframe", dtrack_estimate.grey_img);
      for (size_t ii = 0; ii < candidates.size(); ++ii) {
        int index = std::get<0>(candidates[ii]);
        DTrackPoseOut& dtrack_match = dtrack_vector_[index];
        cv::imshow("Match", dtrack_match.grey_img);
        cv::waitKey(5000);
      }
    }
#endif

    // If loop closure candidates found, chose the "best" one and track against it.
    if (!candidates.empty()) {
      dtrack_.SetKeyframe(dtrack_estimate.grey_img, dtrack_estimate.depth_img);

      int index = std::get<0>(candidates[0]);
      DTrackPoseOut& dtrack_match = dtrack_vector_[index];

      double              dtrack_error;
      Sophus::SE3d        Trl;
      unsigned int        dtrack_num_obs;
      Eigen::Matrix6d     dtrack_covariance;
      dtrack_error = dtrack_.Estimate(true, dtrack_match.grey_img, Trl,
                                      dtrack_covariance, dtrack_num_obs,
                                      dtrack_match.depth_img);

      // Transfer relative pose to IMU frame.
      Trl = Tic_ * Trl * Tic_.inverse();

      // If tracking error is less than threshold, accept as loop closure.
      if (dtrack_error < 15.0) {
        LOG(INFO) << "Loop closure found!";
        pose_relaxer_.AddBinaryConstraint(ii, index, Trl, dtrack_covariance);
#if 0
        cv::imshow("Keyframe", dtrack_estimate.grey_img);
        cv::imshow("Match", dtrack_match.grey_img);
        cv::waitKey(8000);
#endif
      }
    }
  }

  // Solve.
  pose_relaxer_.Solve(1000, 1.0, false);

  // Update adjusted poses.
  for (size_t ii = 0; ii < pose_relaxer_.GetNumPoses(); ++ii) {
    const ba::PoseT<double>& pose = pose_relaxer_.GetPose(ii);
    DTrackPoseOut& dtrack_pose = dtrack_vector_[ii];
    dtrack_pose.T_wp = pose.t_wp;
  }
}

///////////////////////////////////////////////////////////////////////////
bool Tracker::WhereAmI(
    const cv::Mat&  image,
    int&            frame_id,
    Sophus::SE3d&   Twp
  )
{
  // Reset output.
  frame_id = -1;
  Twp = Sophus::SE3d();

  cv::Mat thumbnail = GenerateThumbnail(image);

  std::vector<std::pair<unsigned int, float> > candidates;
  const float max_score = 5.0 * (thumbnail.rows * thumbnail.cols);
  for (unsigned int ii = 0; ii < dtrack_map_.size(); ++ii) {
    DTrackMap& map_frame = dtrack_map_[ii];

    float score = ScoreImages<unsigned char>(thumbnail,
                                             map_frame.thumbnail);

    if (score < max_score) {
      candidates.push_back(std::pair<unsigned int, float>(ii, score));
    }
  }

  // Sort vector by score.
  std::sort(candidates.begin(), candidates.end(), _CompareNorm);

  // If loop closure candidates found, chose the "best" one and track against it.
  if (candidates.empty()) {
    return false;
  } else {
    int index = std::get<0>(candidates[0]);
    DTrackMap& map_frame = dtrack_map_[index];

    dtrack_refine_.SetKeyframe(map_frame.grey_img, map_frame.depth_img);

    double              dtrack_error;
    Sophus::SE3d        Tkc;
    unsigned int        dtrack_num_obs;
    Eigen::Matrix6d     dtrack_covariance;
    dtrack_error = dtrack_refine_.Estimate(true, image, Tkc,
                                    dtrack_covariance, dtrack_num_obs,
                                    image);

    // If tracking error is less than threshold, accept as loop closure.
    if (dtrack_error > 15.0) {
      return false;
    } else {
      frame_id = index;
      Twp = map_frame.T_wp * Tkc;
#if 0
      std::cout << "-- ID: " << frame_id << std::endl;
      std::cout << "-- Pose: " << T2Cart(Twp.matrix()).transpose() << std::endl;

      cv::imshow("Image", image);
      cv::imshow("Match", map_frame.grey_img);
      cv::waitKey(15000);
#endif
      return true;
    }
  }
}

///////////////////////////////////////////////////////////////////////////
cv::Mat Tracker::GenerateThumbnail(const cv::Mat& image)
{
  cv::Mat thumbnail;
  const int thumb_level = 4;
  std::vector<cv::Mat> pyramid;
  cv::buildPyramid(image, pyramid, thumb_level);
  thumbnail = pyramid[thumb_level-1].clone();
  return thumbnail;
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
