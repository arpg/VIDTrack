#include <unistd.h>
#include <deque>

#include <Eigen/Eigen>
#include <sophus/sophus.hpp>
#include <opencv2/opencv.hpp>

#include <HAL/Utils/GetPot>
#include <HAL/Utils/TicToc.h>
#include <HAL/Camera/CameraDevice.h>
#include <pangolin/pangolin.h>
#include <SceneGraph/SceneGraph.h>
#include <calibu/Calibu.h>
#include <calibu/calib/LocalParamSe3.h>
#include <calibu/calib/CostFunctionAndParams.h>
#include <ba/BundleAdjuster.h>
#include <ba/Types.h>
#include <ba/InterpolationBuffer.h>

#include "AuxGUI/AnalyticsView.h"
#include "AuxGUI/Timer.h"
#include "AuxGUI/TimerView.h"
#include "AuxGUI/GLPathRel.h"
#include "AuxGUI/GLPathAbs.h"

#include "dtrack.h"
#include "ceres_dense_ba.h"


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
  Sophus::SE3d      Trl;
  Eigen::Matrix6d   covariance;
  double            time;
};



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


  ///----- Set up GUI.
  pangolin::CreateGlutWindowAndBind("DEVIL", 1600, 800);

  // Set up panel.
  const unsigned int panel_size = 180;
  pangolin::CreatePanel("ui").SetBounds(0, 1, 0, pangolin::Attach::Pix(panel_size));
  pangolin::Var<bool>  ui_camera_follow("ui.Camera Follow", false, true);
  pangolin::Var<bool>  ui_reset("ui.Reset", true, false);
  pangolin::Var<bool>  ui_use_gt_poses("ui.Use GT Poses", false, true);
  pangolin::Var<bool>  ui_use_constant_velocity("ui.Use Const Vel Model", false, true);
  pangolin::Var<bool>  ui_use_imu_estimates("ui.Use IMU Estimates", false, true);
  pangolin::Var<bool>  ui_show_vo_path("ui.Show VO Path", true, true);
  pangolin::Var<bool>  ui_show_ba_path("ui.Show BA Path", true, true);
  pangolin::Var<bool>  ui_show_gt_path("ui.Show GT Path", true, true);

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
  bool run_ba       = false;


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
  typedef ba::ImuMeasurementT<double>               ImuMeasurement;
  std::vector<uint32_t>                             imu_residual_ids;
  ba::InterpolationBufferT<ImuMeasurement, double>  imu_buffer;
  ba::BundleAdjuster<double, 1, 15, 0>              bundle_adjuster;
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

      path_gt_vec.push_back(T);

      // Flag to load poses as a particular convention.
      if (cl_args.search("-V")) {
        // Vision convention.
        poses.push_back(T);
      } else if (cl_args.search("-C")) {
        // Custom setting.
        Sophus::SE3d Tt(SceneGraph::GLCart2T(0, 0, 0, 0, -M_PI/2.0, -M_PI/2.0));
        poses.push_back(calibu::ToCoordinateConvention(T,
                                                       calibu::RdfRobotics.inverse())*Tt);
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

  ///----- Load file of IMU measurements (required).
  {
    std::string imu_file = cl_args.follow("", "-imu");
    if (imu_file.empty()) {
      std::cerr << "- NOTE: No IMU file given. It is required!" << std::endl;
      exit(EXIT_FAILURE);
    }
    imu_file = camera.GetDeviceProperty(hal::DeviceDirectory) + "/" + imu_file;
    FILE* fd = fopen(imu_file.c_str(), "r");
    float timestamp, accelX, accelY, accelZ, gyroX, gyroY, gyroZ;

    std::cout << "- Loading IMU measurements file: '" << imu_file << "'" << std::endl;

    int imu_count = 0;
    while (fscanf(fd, "%f\t%f\t%f\t%f\t%f\t%f\t%f", &timestamp, &accelX, &accelY,
                  &accelZ, &gyroX, &gyroY, &gyroZ) != EOF) {

      Eigen::Vector3d a(accelX, accelY, accelZ);
      Eigen::Vector3d w(gyroX, gyroY, gyroZ);

      ImuMeasurement imu(w, a, timestamp);
      imu_buffer.AddElement(imu);
      imu_count++;
    }
    std::cout << "- NOTE: " << imu_count << " IMU measurements loaded." << std::endl;
    fclose(fd);
  }

  ///----- Load image timestamps.
  std::vector<double> image_timestamps;
  {
    std::string timestamps_file = cl_args.follow("", "-timestamps");
    if (timestamps_file.empty()) {
      std::cerr << "- NOTE: No timestamps file given. It is required!" << std::endl;
      exit(EXIT_FAILURE);
    }
    timestamps_file = camera.GetDeviceProperty(hal::DeviceDirectory) + "/" + timestamps_file;
    FILE* fd = fopen(timestamps_file.c_str(), "r");
    double timestamp;

    std::cout << "- Loading timestamps file: '" << timestamps_file << "'" << std::endl;

    while (fscanf(fd, "%lf", &timestamp) != EOF) {
      image_timestamps.push_back(timestamp);
    }
    fclose(fd);
    std::cout << "- NOTE: " << image_timestamps.size() << " timestamps loaded." << std::endl;
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
                                     [&run_ba] {
                                        run_ba = !run_ba; });

  ///----- Init general variables.
  std::vector<Pose> dtrack_map;
  unsigned int frame_index = 0;
  Sophus::SE3d gt_pose;
  Sophus::SE3d current_pose;
  Sophus::SE3d pose_estimate;
  Eigen::Matrix6d pose_covariance;
  std::shared_ptr<pb::ImageArray> images = pb::ImageArray::Create();


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

      // Re-initialize camera.
      if (!camera.GetDeviceProperty(hal::DeviceDirectory).empty()) {
       camera = hal::Camera(cl_args.follow("", "-cam"));
      }

      // Reset frame counter.
      frame_index = 0;

      // Reset map and current pose.
      dtrack_map.clear();
      current_pose = Sophus::SE3d();
      path_vo_vec.push_back(current_pose);

      // Capture first image.
      capture_flag = camera.Capture(*images);
      cv::Mat current_image = ConvertAndNormalize(images->at(0)->Mat());

      // Reset reference image for DTrack.
      keyframe_image = current_image;
      keyframe_depth = images->at(1)->Mat();

      // Increment frame counter.
      frame_index++;
    }


    ///----- Step forward ...
    if (!paused || pangolin::Pushed(step_once)) {
      //  Capture the new image.
      capture_flag = camera.Capture(*images);

      if (capture_flag == false) {
        paused = true;
      } else {
        // Convert to float and normalize.
        cv::Mat current_image = ConvertAndNormalize(images->at(0)->Mat());

        // Get pose for this image.
        timer.Tic("DTrack");

        // Reset pose estimate to identity if no constant velocity model is used.
        if (!ui_use_constant_velocity) {
          pose_estimate = Sophus::SE3d();
        }

        // Integrate IMU to get an initial pose estimate to seed VO.
        if (ui_use_imu_estimates) {

          // Get IMU measurements between previous frame and current frame.
          std::vector<ImuMeasurement> imu_measurements =
              imu_buffer.GetRange(image_timestamps[frame_index-1],
              image_timestamps[frame_index]);

          if (imu_measurements.size() == 0) {
            std::cerr << "Could not find imu measurements between : " <<
                         image_timestamps[frame_index-1] << " and " <<
                         image_timestamps[frame_index] << std::endl;
            exit(EXIT_FAILURE);
          }

          ba::PoseT<double> last_pose = bundle_adjuster.GetPose(frame_index-1);
          std::vector<ba::ImuPoseT<double>> imu_poses;

          ba::ImuPoseT<double> new_pose =
              decltype(bundle_adjuster)::ImuResidual::IntegrateResidual(last_pose,
              imu_measurements, last_pose.b.head<3>(), last_pose.b.tail<3>(),
              bundle_adjuster.GetImuCalibration().g_vec, imu_poses);

          // Get new relative transform to seed ESM.
          pose_estimate = last_pose.t_wp.inverse() * new_pose.t_wp;

          // Make sure BA is ran at the end.
          run_ba = true;
        }

        // RGBD pose estimation.
        dtrack.SetKeyframe(keyframe_image, keyframe_depth);
        double dtrack_error = dtrack.Estimate(current_image, pose_estimate,
                                              pose_covariance);
        std::cout << "Pose Estimate: " << SceneGraph::GLT2Cart(pose_estimate.matrix()).transpose() << std::endl;
        Pose pose;
        pose.Trl        = pose_estimate;
        pose.covariance = pose_covariance;
        pose.time       = image_timestamps[frame_index];
        dtrack_map.push_back(pose);
        analytics["DTrack RMS"] = dtrack_error;

        // Calculate pose error.
        Sophus::SE3d gt_relative_pose = poses[frame_index-1].inverse()
            * poses[frame_index];
        timer.Toc("DTrack");

        // If using ground-truth poses, override pose estimate with GT pose.
        if (ui_use_gt_poses) {
          pose_estimate = gt_relative_pose;
        }

        // Update pose.
        gt_pose = gt_pose * gt_relative_pose;
        current_pose = current_pose * pose_estimate;
        path_vo_vec.push_back(pose_estimate);


        ///----- Update GUI objects.
        // Update error.
        analytics["Path Error"] =
            Sophus::SE3::log(current_pose.inverse() * gt_pose).head(3).norm();

        // Reset reference image for DTrack.
        keyframe_image = current_image;
        keyframe_depth = images->at(1)->Mat();

        // Increment frame counter.
        frame_index++;

        // Update analytics.
        analytics_view.Update(analytics);
      }
    }


    ///----- Run BA ...
    if (pangolin::Pushed(run_ba)) {

#if 0
      // Init BA.
      options.regularize_biases_in_batch  = false;
      bundle_adjuster.Init(options, 200, 2000);
      bundle_adjuster.AddCamera(rig.cameras_[0], rig.t_wc_[0]);
      Eigen::Matrix<double, 3, 1> gravity;
      gravity << 0, 9.8, 0;
      bundle_adjuster.SetGravity(gravity);

      // Reset IMU residuals IDs.
      imu_residual_ids.clear();

      // Push initial pose.
      Sophus::SE3d global_pose; // Identity.
      bundle_adjuster.AddPose(global_pose, true, 0);
      {
        Eigen::Matrix6d cov;
        cov.setIdentity();
        cov *= 1e-6;
        bundle_adjuster.AddUnaryConstraint(0, global_pose, cov);
      }

      // Push DTrack estimates.
      double previous_time = 0;
      for (size_t ii = 0; ii < dtrack_map.size(); ++ii) {
        Pose& pose = dtrack_map[ii];
        global_pose *= pose.Trl;
        bundle_adjuster.AddPose(global_pose, true, pose.time);

        // Add DTrack constraints.
//        bundle_adjuster.AddBinaryConstraint(ii, ii+1, pose.Twp, pose.covariance);

        Eigen::Matrix6d cov;
        cov.setIdentity();
        cov *= 1e-6;
        gt_pose = poses[0].inverse() * poses[ii+1];
        bundle_adjuster.AddUnaryConstraint(ii+1, gt_pose, cov);

        // Get IMU measurements between frames.
        std::vector<ImuMeasurement> imu_measurements =
            imu_buffer.GetRange(previous_time, pose.time);

        // Add IMU constraints.
        imu_residual_ids.push_back(
              bundle_adjuster.AddImuResidual(ii, ii+1, imu_measurements));

        // Update time.
        previous_time = pose.time;
#endif

#if 1
        // Init BA.
        options.regularize_biases_in_batch  = false;
        bundle_adjuster.Init(options, 200, 2000);
        bundle_adjuster.AddCamera(rig.cameras_[0], rig.t_wc_[0]);
        Eigen::Matrix<double, 3, 1> gravity;
        gravity << 0, 9.8, 0;
        bundle_adjuster.SetGravity(gravity);

        // Reset IMU residuals IDs.
        imu_residual_ids.clear();

        // Push first pose.
        bundle_adjuster.AddPose(poses[0], true, image_timestamps[0]);
        {
          Eigen::Matrix6d cov;
          cov.setIdentity();
          cov *= 1e-6;
          bundle_adjuster.AddUnaryConstraint(0, poses[0], cov);
        }

        // Push rest of poses.
        double previous_time = image_timestamps[0];
        for (size_t ii = 1; ii < dtrack_map.size(); ++ii) {
          bundle_adjuster.AddPose(poses[ii], true, image_timestamps[ii]);

          Eigen::Matrix6d cov;
          cov.setIdentity();
          cov *= 1e-6;
          bundle_adjuster.AddUnaryConstraint(ii, poses[ii], cov);

          // Get IMU measurements between frames.
          std::vector<ImuMeasurement> imu_measurements =
              imu_buffer.GetRange(previous_time, image_timestamps[ii]);

          // Add IMU constraints.
          imu_residual_ids.push_back(
                bundle_adjuster.AddImuResidual(ii-1, ii, imu_measurements));

          // Update time.
          previous_time = image_timestamps[ii];
#endif
      }

      // Run solver.
      bundle_adjuster.Solve(25, 1.0);

      ///----- Update GUI objects.
      path_ba_vec.clear();
      for (size_t ii = 0; ii < frame_index-1; ++ii) {
        ba::PoseT<double> pose = bundle_adjuster.GetPose(ii);
        path_ba_vec.push_back(pose.t_wp);
      }
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
        const ba::ImuResidualT<double>& res = bundle_adjuster.GetImuResidual(id);
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
