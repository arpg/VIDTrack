#include <unistd.h>
#include <deque>

#include <Eigen/Eigen>
#include <sophus/sophus.hpp>
#include <opencv2/opencv.hpp>

#include <ba/BundleAdjuster.h>
#include <ba/Types.h>
#include <ba/InterpolationBuffer.h>
#include <calibu/Calibu.h>
#include <calibu/calib/LocalParamSe3.h>
#include <calibu/calib/CostFunctionAndParams.h>
#include <HAL/Utils/GetPot>
#include <HAL/Utils/TicToc.h>
#include <HAL/Camera/CameraDevice.h>
#include <HAL/IMU/IMUDevice.h>
#include <pangolin/pangolin.h>
#include <SceneGraph/SceneGraph.h>

#include "AuxGUI/AnalyticsView.h"
#include "AuxGUI/Timer.h"
#include "AuxGUI/TimerView.h"
#include "AuxGUI/GLPathRel.h"
#include "AuxGUI/GLPathAbs.h"

#include "dtrack.h"
#include "muse.h"
#include "ceres_dense_ba.h"

#define DEVIL_DEBUG 0

///////////////////////////////////////////////////////////////////////////
/// Generates a "heat map" based on an error image provided.
cv::Mat GenerateHeatMap(const cv::Mat& input)
{
  cv::Mat output(input.rows, input.cols, CV_8UC3);

  // Get min/max to normalize.
  double min, max;
  cv::minMaxIdx(input, &min, &max);
  const double mean = cv::mean(input).val[0];
  max = 3*mean;
  for (int vv = 0; vv < input.rows; ++vv) {
    for (int uu = 0; uu < input.cols; ++uu) {
      float n_val = (input.at<float>(vv, uu) - min) / (max - min);
      if (n_val < 0.5) {
        output.at<cv::Vec3b>(vv, uu) = cv::Vec3b(255*n_val, 0, 128);
      } else {
        output.at<cv::Vec3b>(vv, uu) = cv::Vec3b(255, 0, 128*n_val);
      }
    }
  }
  return output;
}

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
struct Pose {
  Sophus::SE3d      Tab;
  Eigen::Matrix6d   covariance;
  double            timeA;
  double            timeB;
};


/////////////////////////////////////////////////////////////////////////////
typedef ba::ImuMeasurementT<double>               ImuMeasurement;
ba::InterpolationBufferT<ImuMeasurement, double>  imu_buffer;
std::mutex                                        imu_mutex;

void IMU_Handler(pb::ImuMsg& IMUdata) {
  Eigen::Vector3d a(IMUdata.accel().data(0),
                    IMUdata.accel().data(1),
                    IMUdata.accel().data(2));
  Eigen::Vector3d w(IMUdata.gyro().data(0),
                    IMUdata.gyro().data(1),
                    IMUdata.gyro().data(2));

#if DEVIL_DEBUG
  std::cout << "=== Adding ->> T: " << IMUdata.system_time()
            << " A: " << a.transpose() << "  G:"
            << w.transpose() << std::endl;
#endif

  ImuMeasurement imu(w, a, IMUdata.system_time());
  imu_mutex.lock();
  imu_buffer.AddElement(imu);
  imu_mutex.unlock();
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  std::cout << "Starting DEVIL ..." << std::endl;

  ///----- Initialize Camera.
  GetPot cl_args(argc, argv);
  if (!cl_args.search("-cam")) {
    std::cerr << "Camera arguments missing!" << std::endl;
    exit(EXIT_FAILURE);
  }
  hal::Camera camera(cl_args.follow("", "-cam"));

  const int image_width = camera.Width();
  const int image_height = camera.Height();
  std::cout << "- Image Dimensions: " << image_width <<
               "x" << image_height << std::endl;


  ///----- Initialize IMU.
  if (!cl_args.search("-imu")) {
    std::cerr << "IMU arguments missing!" << std::endl;
    exit(EXIT_FAILURE);
  }
  hal::IMU imu(cl_args.follow("", "-imu"));
  imu.RegisterIMUDataCallback(&IMU_Handler);
  std::cout << "- Registering IMU device." << std::endl;


  ///----- Set up GUI.
  pangolin::CreateGlutWindowAndBind("DEVIL", 1600, 800);

  // Set up panel.
  const unsigned int panel_size = 180;
  pangolin::CreatePanel("ui").SetBounds(0, 1, 0, pangolin::Attach::Pix(panel_size));
  pangolin::Var<bool>           ui_camera_follow("ui.Camera Follow", false, true);
  pangolin::Var<bool>           ui_reset("ui.Reset", true, false);
  pangolin::Var<bool>           ui_use_gt_depth("ui.Use GT Depth", true, true);
  pangolin::Var<bool>           ui_use_gt_poses("ui.Use GT Poses", false, true);
  pangolin::Var<bool>           ui_use_constant_velocity("ui.Use Const Vel Model", false, true);
  pangolin::Var<bool>           ui_use_imu_estimates("ui.Use IMU Estimates", false, true);
  pangolin::Var<bool>           ui_use_pyramid("ui.Use Pyramid", true, true);
  pangolin::Var<bool>           ui_windowed_ba("ui.Windowed BA", true, true);
  pangolin::Var<unsigned int>   ui_ba_window_size("ui.BA Window Size", 5, 0, 20);
  pangolin::Var<bool>           ui_show_vo_path("ui.Show VO Path", true, true);
  pangolin::Var<bool>           ui_show_ba_path("ui.Show BA Path", true, true);
  pangolin::Var<bool>           ui_show_gt_path("ui.Show GT Path", true, true);

  // Set up container.
  pangolin::View& container = pangolin::CreateDisplay();
  container.SetBounds(0, 1, pangolin::Attach::Pix(panel_size), 0.65);
  container.SetLayout(pangolin::LayoutEqual);
  pangolin::DisplayBase().AddDisplay(container);

  // Set up timer.
  Timer     timer;
  TimerView timer_view;
  timer_view.SetBounds(0.5, 1, 0.65, 1.0);
  pangolin::DisplayBase().AddDisplay(timer_view);
  timer_view.InitReset();

  // Set up analytics.
  std::map<std::string, float>  analytics;
  AnalyticsView                 analytics_view;
  analytics_view.SetBounds(0, 0.5, 0.65, 1.0);
  pangolin::DisplayBase().AddDisplay(analytics_view);
  analytics_view.InitReset();

  // Set up 3D view for container.
  SceneGraph::GLSceneGraph gl_graph;
  SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

  // Reset background color to black.
  glClearColor(0, 0, 0, 1);

  // Add path.
  GLPathRel gl_path_vo;
  GLPathAbs gl_path_ba;
  GLPathAbs gl_path_gt;
  gl_path_vo.SetPoseDisplay(5);
  gl_path_ba.SetPoseDisplay(5);
  gl_path_gt.SetPoseDisplay(5);
  gl_path_ba.SetLineColor(0, 1.0, 0);
  gl_path_gt.SetLineColor(0, 0, 1.0);
  gl_graph.AddChild(&gl_path_vo);
  gl_graph.AddChild(&gl_path_ba);
  gl_graph.AddChild(&gl_path_gt);
  std::vector<Sophus::SE3d>& path_vo_vec = gl_path_vo.GetPathRef();
  std::vector<Sophus::SE3d>& path_ba_vec = gl_path_ba.GetPathRef();
  std::vector<Sophus::SE3d>& path_gt_vec = gl_path_gt.GetPathRef();

  // Add grid.
  SceneGraph::GLGrid gl_grid(50, 1);
#if 0
  {
    Sophus::SE3d vision_RDF;
    Sophus::SO3d& rotation = vision_RDF.so3();
    rotation = calibu::RdfRobotics.inverse();
    gl_grid.SetPose(vision_RDF.matrix());
  }
#endif
  gl_graph.AddChild(&gl_grid);

  pangolin::View view_3d;
  const double far = 10*1000;
  const double near = 1E-3;

  pangolin::OpenGlRenderState stacks3d(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, near, far),
        pangolin::ModelViewLookAt(-5, 0, -8, 0, 0, 0, pangolin::AxisNegZ)
        );

  view_3d.SetHandler(new SceneGraph::HandlerSceneGraph(gl_graph, stacks3d))
      .SetDrawFunction(SceneGraph::ActivateDrawFunctor(gl_graph, stacks3d));

  // Add all subviews to container.
  SceneGraph::ImageView image_view;
  image_view.SetAspect(640.0 / 480.0);
  container.AddDisplay(image_view);

  SceneGraph::ImageView depth_view;
  container.AddDisplay(depth_view);

  container.AddDisplay(view_3d);

  // GUI aux variables.
  bool capture_flag = false;
  bool paused       = true;
  bool step_once    = false;
  bool run_batch_ba = false;


  ///----- Load camera model.
  calibu::CameraRig old_rig;
  if (camera.GetDeviceProperty(hal::DeviceDirectory).empty() == false) {
    std::cout<<"- Loaded camera: " <<
               camera.GetDeviceProperty(hal::DeviceDirectory) + '/'
               + cl_args.follow("cameras.xml", "-cmod") << std::endl;
    old_rig = calibu::ReadXmlRig(camera.GetDeviceProperty(hal::DeviceDirectory)
                             + '/' + cl_args.follow("cameras.xml", "-cmod"));
  } else {
    old_rig = calibu::ReadXmlRig(cl_args.follow("cameras.xml", "-cmod"));
  }
  Eigen::Matrix3f K = old_rig.cameras[0].camera.K().cast<float>();
  Eigen::Matrix3f Kinv = K.inverse();
  std::cout << "-- K is: " << std::endl << K << std::endl;

  // Convert old rig to new rig.
  calibu::Rig<double> rig;
  calibu::CreateFromOldRig(&old_rig, &rig);

  ///----- Init DTrack stuff.
  cv::Mat keyframe_image, keyframe_depth;
  DTrack dtrack;
  dtrack.Init();
  dtrack.SetParams(old_rig.cameras[0].camera, old_rig.cameras[0].camera,
      old_rig.cameras[0].camera, Sophus::SE3d());

  ///----- Init BA stuff.
  std::vector<uint32_t>                             imu_residual_ids;
  ba::BundleAdjuster<double, 0, 9, 0>               bundle_adjuster;
  ba::Options<double>                               options;

  ///----- Load file of ground truth poses (required).
  std::vector<Sophus::SE3d> poses;
  {
    std::string pose_file = cl_args.follow("", "-poses");
    if (pose_file.empty()) {
      std::cerr << "- NOTE: No poses file given. It is required!" << std::endl;
      exit(EXIT_FAILURE);
    }
    pose_file = camera.GetDeviceProperty(hal::DeviceDirectory) + "/" + pose_file;
    FILE* fd = fopen(pose_file.c_str(), "r");
    Eigen::Matrix<double, 6, 1> pose;
    float x, y, z, p, q, r;

    std::cout << "- Loading pose file: '" << pose_file << "'" << std::endl;
    if (cl_args.search("-V")) {
      // Vision convention.
      std::cout << "- NOTE: File is being read in VISION frame." << std::endl;
    } else if (cl_args.search("-C")) {
      // Custom convention.
      std::cout << "- NOTE: File is being read in *****CUSTOM***** frame." << std::endl;
    } else if (cl_args.search("-T")) {
      // Tsukuba convention.
      std::cout << "- NOTE: File is being read in TSUKUBA frame." << std::endl;
    } else {
      // Robotics convention (default).
      std::cout << "- NOTE: File is being read in ROBOTICS frame." << std::endl;
    }

    while (fscanf(fd, "%f\t%f\t%f\t%f\t%f\t%f", &x, &y, &z, &p, &q, &r) != EOF) {
      pose(0) = x;
      pose(1) = y;
      pose(2) = z;
      pose(3) = p;
      pose(4) = q;
      pose(5) = r;

      Sophus::SE3d T(SceneGraph::GLCart2T(pose));

      // Flag to load poses as a particular convention.
      if (cl_args.search("-V")) {
        // Vision convention.
        poses.push_back(T);
      } else if (cl_args.search("-C")) {
        // Custom setting.
        pose(0) *= -1;
        pose(2) *= -1;
        Sophus::SE3d Tt(SceneGraph::GLCart2T(pose));
        poses.push_back(Tt);
      } else if (cl_args.search("-T")) {
        // Tsukuba convention.
        Eigen::Matrix3d tsukuba_convention;
        tsukuba_convention << -1,  0,  0,
                               0, -1,  0,
                               0,  0, -1;
        Sophus::SO3d tsukuba_convention_sophus(tsukuba_convention);
        poses.push_back(calibu::ToCoordinateConvention(T,
                                        tsukuba_convention_sophus.inverse()));
      } else {
        // Robotics convention (default).
        poses.push_back(calibu::ToCoordinateConvention(T,
                                        calibu::RdfRobotics.inverse()));
      }
    }
    std::cout << "- NOTE: " << poses.size() << " poses loaded." << std::endl;
    fclose(fd);
  }

  ///----- Register callbacks.
  // Hide/Show panel.
  pangolin::RegisterKeyPressCallback('~', [&](){
    static bool fullscreen = true;
    fullscreen = !fullscreen;
    if (fullscreen) {
      container.SetBounds(0, 1, pangolin::Attach::Pix(panel_size), 0.65);
    } else {
      container.SetBounds(0, 1, 0, 1);
    }
    analytics_view.Show(fullscreen);
    timer_view.Show(fullscreen);
    pangolin::Display("ui").Show(fullscreen);
  });

  // Container view handler.
  const char keyShowHide[] = {'1','2','3','4','5','6','7','8','9','0'};
  const char keySave[]     = {'!','@','#','$','%','^','&','*','(',')'};
  for (int ii = 0; ii < container.NumChildren(); ii++) {
    pangolin::RegisterKeyPressCallback(keyShowHide[ii], [&container,ii]() {
      container[ii].ToggleShow(); });
    pangolin::RegisterKeyPressCallback(keySave[ii], [&container,ii]() {
      container[ii].SaveRenderNow("screenshot", 4); });
  }

  pangolin::RegisterKeyPressCallback(' ', [&paused] { paused = !paused; });
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_RIGHT,
                                     [&step_once] {
                                        step_once = !step_once; });
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'r',
                                     [&ui_reset] {
                                        ui_reset = true; });
  pangolin::RegisterKeyPressCallback('o',
                                     [&run_batch_ba] {
                                        run_batch_ba = !run_batch_ba; });

  ///----- Init general variables.
  std::deque<Pose>      dtrack_map;
  unsigned int          frame_index;
  ba::PoseT<double>     last_adjusted_pose;
  Sophus::SE3d          current_pose;
  Sophus::SE3d          pose_estimate;
  Eigen::Matrix6d       pose_covariance;
  double                current_timestamp;
  double                keyframe_timestamp;
  bool                  ba_has_run;

  // Image holder.
  std::shared_ptr<pb::ImageArray> images = pb::ImageArray::Create();

  // Permutation matrix to bring things into robotic reference frame.
//  Sophus::SE3d permutation = rig.t_wc_[0].inverse();
  Eigen::Matrix4d tmp;
//  tmp = SceneGraph::GLCart2T(0, 0, 0, M_PI/2.0, 0, M_PI/2.0);
//  Sophus::SE3d permutation(tmp);
  Sophus::SE3d permutation;
  permutation.so3() = calibu::RdfRobotics;


  /////////////////////////////////////////////////////////////////////////////
  ///---- MAIN LOOP
  ///
  while (!pangolin::ShouldQuit()) {

    // Start timer.
    timer.Tic();

    ///----- Init reset ...
    if (pangolin::Pushed(ui_reset)) {
      // Reset timer and analytics.
      timer_view.InitReset();
      analytics_view.InitReset();

      // Reset GUI path.
      path_vo_vec.clear();
      path_ba_vec.clear();
      path_gt_vec.clear();

      // Re-initialize camera.
      if (!camera.GetDeviceProperty(hal::DeviceDirectory).empty()) {
//       camera = hal::Camera(cl_args.follow("", "-cam"));
      }

      // Reset frame counter.
      frame_index = 0;

      // Reset BA flag.
      ba_has_run = false;

      // Reset map and current pose.
      dtrack_map.clear();
      current_pose = permutation;
      last_adjusted_pose.t_wp = permutation;
      path_vo_vec.push_back(current_pose);
      path_ba_vec.push_back(last_adjusted_pose.t_wp);
      path_gt_vec.push_back(permutation * poses[0].inverse() * poses[frame_index]);

      // Capture first image.
      capture_flag = camera.Capture(*images);
      cv::Mat current_image = ConvertAndNormalize(images->at(0)->Mat());

      // Reset reference image for DTrack.
      keyframe_image = current_image;
      keyframe_depth = images->at(1)->Mat();
      keyframe_timestamp = images->at(0)->Timestamp();

      // Increment frame counter.
      frame_index++;
    }


    ///----- Step forward ...
    if (!paused || pangolin::Pushed(step_once)) {
      //  Capture the new image.
      capture_flag = camera.Capture(*images);
      std::cout << "==================================================" << std::endl;
      std::cout << "Frame#: " << frame_index << std::endl;

      if (capture_flag == false) {
        paused = true;
      } else {
        // Convert to float and normalize.
        cv::Mat current_image = ConvertAndNormalize(images->at(0)->Mat());
        current_timestamp = images->at(0)->Timestamp();

#if DEVIL_DEBUG
        std::cout << "=== Image timestamps is: " << current_timestamp << std::endl;
#endif

        // Get pose for this image.
        timer.Tic("DTrack");

        // Reset pose estimate to identity if no constant velocity model is used.
        if (!ui_use_constant_velocity) {
          pose_estimate = Sophus::SE3d();
        }


        ///--------------------
        /// Integrate IMU to get an initial pose estimate to seed VO.
        if (ui_use_imu_estimates) {
          if (ba_has_run == false) {
            std::cerr << "- BA has to be run at least once before seeding with "
                      << "IMU estimates!" << std::endl;
          } else {

            // Get IMU measurements between previous frame and current frame.
            std::vector<ImuMeasurement> imu_measurements =
                imu_buffer.GetRange(keyframe_timestamp, current_timestamp);

            if (imu_measurements.size() == 0) {
              std::cerr << "Could not find imu measurements between : " <<
                           keyframe_timestamp << " and " <<
                           current_timestamp << std::endl;
              exit(EXIT_FAILURE);
            }

            std::vector<ba::ImuPoseT<double>> imu_poses;

            ba::PoseT<double> last_pose =
                bundle_adjuster.GetPose(bundle_adjuster.GetNumPoses() - 1);

            ba::ImuPoseT<double> new_pose =
                decltype(bundle_adjuster)::ImuResidual::IntegrateResidual(last_pose,
                imu_measurements, last_pose.b.head<3>(), last_pose.b.tail<3>(),
                bundle_adjuster.GetImuCalibration().g_vec, imu_poses);

            // Get new relative transform to seed ESM.
            pose_estimate = last_pose.t_wp.inverse() * new_pose.t_wp;

#if DTRACK_DEBUG
            Sophus::SE3d real_pose_transform;
            real_pose_transform = last_pose.t_wp.inverse() * new_pose.t_wp;
            std::cout << "Real Pose Transform: " <<
                  Sophus::SE3::log(poses[frame_index-1].inverse() * poses[frame_index]).transpose()
                          << std::endl;

            std::cout << "Integrated Pose: " <<
                         Sophus::SE3::log(real_pose_transform).transpose() << std::endl;
#endif
          }
        }


        ///--------------------
        /// RGBD pose estimation.
        dtrack.SetKeyframe(keyframe_image, keyframe_depth);
        double dtrack_error = dtrack.Estimate(current_image, pose_estimate,
                                              pose_covariance, ui_use_pyramid);
        std::cout << "VO Pose Estimate[" << frame_index << "]: " << Sophus::SE3::log(pose_estimate).transpose() << std::endl;
        Pose pose;
        pose.Tab        = pose_estimate;
        pose.covariance = pose_covariance;
        pose.timeA      = keyframe_timestamp;
        pose.timeB      = current_timestamp;
        dtrack_map.push_back(pose);
        analytics["DTrack RMS"] = dtrack_error;
        timer.Toc("DTrack");

        // If using ground-truth poses, override pose estimate with GT pose.
        if (ui_use_gt_poses) {
          Sophus::SE3d gt_relative_pose = poses[frame_index-1].inverse()
              * poses[frame_index];
          pose_estimate = gt_relative_pose;
        }


        ///--------------------
        /// Windowed BA.
        if (ui_windowed_ba == true && frame_index > ui_ba_window_size) {
          // Init BA.
          options.regularize_biases_in_batch  = true;
          options.use_triangular_matrices = false;
          options.use_sparse_solver = false;
          options.use_dogleg = true;
          ba::debug_level_threshold = -1;
          bundle_adjuster.Init(options, 10, 1000);
          bundle_adjuster.AddCamera(rig.cameras_[0], rig.t_wc_[0]);

#if 0
          Eigen::Matrix<double, 3, 1> gravity;
          gravity << 0, 0, -9.806;
          bundle_adjuster.SetGravity(gravity);
#endif

          // Reset IMU residuals IDs.
          imu_residual_ids.clear();

          // Push last adjusted pose.
          int cur_id, prev_id;
          Sophus::SE3d global_pose = last_adjusted_pose.t_wp;
          prev_id = bundle_adjuster.AddPose(last_adjusted_pose.t_wp,
                                            last_adjusted_pose.t_vs,
                                            last_adjusted_pose.cam_params,
                                            last_adjusted_pose.v_w,
                                            last_adjusted_pose.b, true,
                                            last_adjusted_pose.time);

          Eigen::Matrix6d cov;
          cov.setIdentity();
          cov *= 1e-6;

          // Push rest of poses.
          for (size_t ii = 0; ii < dtrack_map.size(); ++ii) {
            Pose& pose = dtrack_map[ii];
            global_pose *= pose.Tab;
            cur_id = bundle_adjuster.AddPose(global_pose, true, pose.timeB);

//            bundle_adjuster.AddBinaryConstraint(prev_id, cur_id, pose.Tab, cov);
            bundle_adjuster.AddBinaryConstraint(prev_id, cur_id, pose.Tab, pose.covariance);
//            bundle_adjuster.AddBinaryConstraint(prev_id, cur_id, poses[ii].inverse() * poses[ii+1], cov);

            // Get IMU measurements between frames.
            std::vector<ImuMeasurement> imu_measurements =
                imu_buffer.GetRange(pose.timeA, pose.timeB);

            // Add IMU constraints.
            imu_residual_ids.push_back(
                  bundle_adjuster.AddImuResidual(prev_id, cur_id, imu_measurements));

            // Update pose IDs.
            prev_id = cur_id;
          }

          // Run solver.
          bundle_adjuster.Solve(1000, 1.0);

          // Reset flag.
          ba_has_run = true;

          // Pop front and update last adjusted pose.
          ba::PoseT<double> tmp = bundle_adjuster.GetPose(0);
          last_adjusted_pose = bundle_adjuster.GetPose(1);
          dtrack_map.pop_front();
          std::cout << "BA Pose Popped[" << path_ba_vec.size() << "]: " << (tmp.t_wp.inverse()*last_adjusted_pose.t_wp).log().transpose() << std::endl;
          path_ba_vec.push_back(last_adjusted_pose.t_wp);

#if 0
          ///----- Update GUI objects.
          const size_t dtrack_map_size = dtrack_map.size();
          for (size_t ii = 0; ii < ui_ba_window_size; ++ii) {
            ba::PoseT<double> pose = bundle_adjuster.GetPose(ii);
            path_ba_vec[dtrack_map_size-ui_ba_window_size+ii] = pose.t_wp;
          }

          // Update error.
          ba::PoseT<double> pose = bundle_adjuster.GetPose(frame_index-1);
          Sophus::SE3d gt_pose = permutation * (poses[0].inverse() * poses[frame_index-1]);
          analytics["BA Path Error"] =
              Sophus::SE3::log(pose.t_wp.inverse() * gt_pose).head(3).norm();

          // Update analytics.
          analytics_view.Update(analytics);
#endif
        }


        ///----- Update GUI objects.
        // Update poses.
        Sophus::SE3d gt_pose = permutation *
                          (poses[0].inverse() * poses[frame_index]);
        current_pose = current_pose * pose_estimate;

        // Update error.
        analytics["Path Error"] =
            Sophus::SE3::log(current_pose.inverse() * gt_pose).head(3).norm();

        // Reset reference image for DTrack.
        keyframe_image = current_image;
        keyframe_depth = images->at(1)->Mat();
        keyframe_timestamp = current_timestamp;

        // Update path.
        path_vo_vec.push_back(pose_estimate);
        path_gt_vec.push_back(permutation*poses[0].inverse()*poses[frame_index]);

        // Update analytics.
        analytics_view.Update(analytics);

        // Increment frame counter.
        frame_index++;
      }
    }

    ///----- Run Batch BA ...
    if (pangolin::Pushed(run_batch_ba)) {

        // Init BA.
        options.regularize_biases_in_batch  = true;
        options.use_triangular_matrices = false;
        options.use_sparse_solver = false;
        options.use_dogleg = true;
        //ba::debug_level_threshold = 1;
        bundle_adjuster.Init(options, 200, 2000);
        bundle_adjuster.AddCamera(rig.cameras_[0], rig.t_wc_[0]);

        Eigen::Matrix<double, 3, 1> gravity;
        gravity << 0, 0, -9.806;
        bundle_adjuster.SetGravity(gravity);

        // Reset IMU residuals IDs.
        imu_residual_ids.clear();

        // Push first pose.
        Sophus::SE3d global_pose;
        {
          Sophus::SO3d& rot = global_pose.so3();
          rot = calibu::RdfRobotics;
        }

        int cur_id, prev_id;
        prev_id = bundle_adjuster.AddPose(global_pose, true, 0);

        Eigen::Matrix6d cov;
        cov.setIdentity();
        cov *= 1e-6;

        // Push rest of poses.
        std::cout << "DTrack Map Size: " << dtrack_map.size() << std::endl;
        for (size_t ii = 0; ii < dtrack_map.size(); ++ii) {
          Pose& pose = dtrack_map[ii];
          global_pose *= pose.Tab;
          cur_id = bundle_adjuster.AddPose(global_pose, true, pose.timeB);

#if DEVIL_DEBUG
          std::cout << "GT Rel Pose: " <<
                Sophus::SE3::log(poses[ii].inverse() * poses[ii+1]).transpose()
                        << std::endl;

          std::cout << "VO Rel Pose: " <<
                Sophus::SE3::log(pose.Tab).transpose() << std::endl;

          std::cout << "ERROR Rel Pose: " <<
                Sophus::SE3::log(pose.Tab.inverse()*poses[ii].inverse()*poses[ii+1]).transpose() << std::endl;
#endif

//          bundle_adjuster.AddBinaryConstraint(prev_id, cur_id, pose.Tab, cov);
          bundle_adjuster.AddBinaryConstraint(prev_id, cur_id, pose.Tab, pose.covariance);
//          bundle_adjuster.AddBinaryConstraint(prev_id, cur_id, poses[ii].inverse() * poses[ii+1], cov);

          // Get IMU measurements between frames.
          std::vector<ImuMeasurement> imu_measurements =
              imu_buffer.GetRange(pose.timeA, pose.timeB);

#if DEVIL_DEBUG
          std::cout << "Number of IMU measurements between frames: " <<
                       imu_measurements.size() << std::endl;
#endif

          // Add IMU constraints.
          imu_residual_ids.push_back(
                bundle_adjuster.AddImuResidual(prev_id, cur_id, imu_measurements));

          // Update pose IDs.
          prev_id = cur_id;
      }

      // Run solver.
      bundle_adjuster.Solve(1000, 1.0);

      // Reset flag.
      ba_has_run = true;

      ///----- Update GUI objects.
      path_ba_vec.clear();
      for (size_t ii = 0; ii < frame_index; ++ii) {
        ba::PoseT<double> pose = bundle_adjuster.GetPose(ii);
        path_ba_vec.push_back(pose.t_wp);
      }


      // Update error.
      ba::PoseT<double> pose = bundle_adjuster.GetPose(frame_index-1);
      Sophus::SE3d gt_pose = permutation * (poses[0].inverse() * poses[frame_index-1]);
      analytics["BA Path Error"] =
          Sophus::SE3::log(pose.t_wp.inverse() * gt_pose).head(3).norm();

      // Update analytics.
      analytics_view.Update(analytics);
    }



    /////////////////////////////////////////////////////////////////////////////
    ///---- Render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (capture_flag) {
      image_view.SetImage(images->at(0)->data(), image_width, image_height,
                          GL_RGB8, GL_LUMINANCE, GL_UNSIGNED_BYTE);

      depth_view.SetImage(images->at(1)->data(), image_width, image_height,
                          GL_RGB8, GL_LUMINANCE, GL_FLOAT, true);
    }

    if (ui_camera_follow) {
      stacks3d.Follow(current_pose.matrix());
    }

    gl_path_vo.SetVisible(ui_show_vo_path);
    gl_path_ba.SetVisible(ui_show_ba_path);
    gl_path_gt.SetVisible(ui_show_gt_path);


    // Update path using NIMA's code.
    {
      view_3d.ActivateAndScissor(stacks3d);
      const ba::ImuCalibrationT<double>& imu = bundle_adjuster.GetImuCalibration();
      std::vector<ba::ImuPoseT<double>> imu_poses;

      for (uint32_t id : imu_residual_ids) {
        const auto& res = bundle_adjuster.GetImuResidual(id);
        const ba::PoseT<double>& pose = bundle_adjuster.GetPose(res.pose1_id);
        std::vector<ba::ImuMeasurementT<double> > meas =
            imu_buffer.GetRange(res.measurements.front().time,
                                res.measurements.back().time);
        res.IntegrateResidual(pose, meas, pose.b.head<3>(), pose.b.tail<3>(),
                              imu.g_vec, imu_poses);
        if (pose.is_active) {
          glColor3f(1.0, 0.0, 1.0);
        } else {
          glColor3f(1.0, 0.2, 0.5);
        }

        for (size_t ii = 1 ; ii < imu_poses.size() ; ++ii) {
          ba::ImuPoseT<double>& prev_imu_pose = imu_poses[ii - 1];
          ba::ImuPoseT<double>& imu_pose = imu_poses[ii];
          pangolin::glDrawLine(prev_imu_pose.t_wp.translation()[0],
              prev_imu_pose.t_wp.translation()[1],
              prev_imu_pose.t_wp.translation()[2],
              imu_pose.t_wp.translation()[0],
              imu_pose.t_wp.translation()[1],
              imu_pose.t_wp.translation()[2]);
        }
      }
    }

    // Sleep a bit.
    usleep(1e6/60.0);

    // Stop timer and update.
    timer.Toc();
    timer_view.Update(10, timer.GetNames(3), timer.GetTimes(3));

    pangolin::FinishFrame();
  }

  return 0;
}
