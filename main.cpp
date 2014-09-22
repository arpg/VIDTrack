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
  pangolin::Var<bool>  ui_camera_follow("ui.Camera Follow", true, true);
  pangolin::Var<bool>  ui_reset("ui.Reset", true, false);
  pangolin::Var<bool>  ui_use_gt_poses("ui.Use GT Poses", false, true);
  pangolin::Var<bool>  ui_use_constant_velocity("ui.Use Const Vel Model", false, true);
  pangolin::Var<bool>  ui_use_imu_estimates("ui.Use IMU Estimates", false, true);
  pangolin::Var<bool>  ui_update_ba_poses("ui.Update BA Poses", true, false);
  pangolin::Var<bool>  ui_show_orig_path("ui.Show Original Path", true, true);
  pangolin::Var<bool>  ui_show_ba_path("ui.Show BA Path", true, true);

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
  GLPathRel gl_path_orig;
  GLPathAbs gl_path_ba;
  gl_path_orig.SetPoseDisplay(5);
  gl_path_ba.SetPoseDisplay(5);
  gl_path_ba.SetLineColor(0, 1.0, 0);
  gl_graph.AddChild(&gl_path_orig);
  gl_graph.AddChild(&gl_path_ba);
  std::vector<Sophus::SE3d>& path_orig_vec = gl_path_orig.GetPathRef();
  std::vector<Sophus::SE3d>& path_ba_vec = gl_path_ba.GetPathRef();

  // Add grid.
  SceneGraph::GLGrid gl_grid(50, 1);
  {
    Sophus::SE3d vision_RDF;
    Sophus::SO3d& rotation = vision_RDF.so3();
    rotation = calibu::RdfRobotics.inverse();
    gl_grid.SetPose(vision_RDF.matrix());
  }
  gl_graph.AddChild(&gl_grid);

  pangolin::View view_3d;
  const double far = 10*1000;
  const double near = 1E-3;

  pangolin::OpenGlRenderState stacks3d(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, near, far),
        pangolin::ModelViewLookAt(0, -8, -5, 0, 0, 0, pangolin::AxisNegY)
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
  bool capture_flag;
  bool paused = true;
  bool step_once = false;


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
  options.regularize_biases_in_batch = false;
  options.error_change_threshold = 1e-3;
  bundle_adjuster.Init(options);
  bundle_adjuster.AddCamera(rig.cameras_[0], rig.t_wc_[0]);
  Eigen::Matrix<double, 3, 1> gravity;
  gravity << 0, 9.8, 0;
  bundle_adjuster.SetGravity(gravity);

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
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + GLUT_KEY_RIGHT,
                                     [&step_once] {
                                        step_once = !step_once; });
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'r',
                                     [&ui_reset] {
                                        ui_reset = true; });
  pangolin::RegisterKeyPressCallback('o',
                                     [&bundle_adjuster, &ui_update_ba_poses] {
                                      bundle_adjuster.Solve(25, 0.2);
                                      ui_update_ba_poses = true; });

  ///----- Init general variables.
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

      // Reset path.
      path_orig_vec.clear();
      path_ba_vec.clear();

      // Re-initialize camera.
      if (!camera.GetDeviceProperty(hal::DeviceDirectory).empty()) {
       camera = hal::Camera(cl_args.follow("", "-cam"));
      }

      // Reset frame counter.
      frame_index = 0;

      // Reset pose.
      current_pose = Sophus::SE3d();

      // Capture first image.
      capture_flag = camera.Capture(*images);
      cv::Mat current_image = ConvertAndNormalize(images->at(0)->Mat());

      // Reset reference image for DTrack.
      keyframe_image = current_image;
      keyframe_depth = images->at(1)->Mat();

      // Add initial pose for BA.
      double timestamp = image_timestamps[frame_index];
      bundle_adjuster.AddPose(current_pose, true, timestamp);

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

        // Get IMU measurements between previous frame and current frame.
        std::vector<ImuMeasurement> imu_measurements =
            imu_buffer.GetRange(image_timestamps[frame_index-1],
            image_timestamps[frame_index]);

        // Integrate IMU to get an initial pose estimate to seed VO.
        if (ui_use_imu_estimates) {

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
              gravity, imu_poses);

          pose_estimate = new_pose.t_wp;
        }

        // RGBD pose estimation.
        dtrack.SetKeyframe(keyframe_image, keyframe_depth);
        double dtrack_error = dtrack.Estimate(current_image, pose_estimate,
                                              pose_covariance);
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
        path_orig_vec.push_back(pose_estimate);

        // Push new pose to BA.
        double timestamp = image_timestamps[frame_index];
        bundle_adjuster.AddPose(current_pose, true, timestamp);
        bundle_adjuster.AddBinaryConstraint(frame_index-1, frame_index,
                                            pose_estimate, pose_covariance);
        imu_residual_ids.push_back(
              bundle_adjuster.AddImuResidual(frame_index-1, frame_index,
                                             imu_measurements));


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

    ///----- Draw BA poses ...
    if (pangolin::Pushed(ui_update_ba_poses)) {

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

    gl_path_orig.SetVisible(ui_show_orig_path);
    gl_path_ba.SetVisible(ui_show_ba_path);

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
