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

#include <unistd.h>
#include <deque>
#include <fstream>

#include <Eigen/Eigen>
#include <sophus/sophus.hpp>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif
#include <opencv2/opencv.hpp>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <calibu/Calibu.h>
#include <HAL/Camera/CameraDevice.h>
#include <HAL/IMU/IMUDevice.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <pangolin/pangolin.h>
#include <SceneGraph/SceneGraph.h>

#include <vidtrack/vidtrack.h>

#include <libGUI/AnalyticsView.h>
#include <libGUI/Timer.h>
#include <libGUI/TimerView.h>
#include <libGUI/GLPathRel.h>
#include <libGUI/GLPathAbs.h>



/////////////////////////////////////////////////////////////////////////////
/// IMU auxilary variables.
const size_t    filter_size = 0;
std::deque<std::tuple<Eigen::Vector3d, Eigen::Vector3d, double> >   filter;

/////////////////////////////////////////////////////////////////////////////
/// IMU callback.
void IMU_Handler(hal::ImuMsg& IMUdata, vid::Tracker* vid_tracker) {
  Eigen::Vector3d a(IMUdata.accel().data(0),
                    IMUdata.accel().data(1),
                    IMUdata.accel().data(2));
  Eigen::Vector3d w(IMUdata.gyro().data(0),
                    IMUdata.gyro().data(1),
                    IMUdata.gyro().data(2));


  // If filter is used...
  if (filter_size > 0) {
    filter.push_back(std::make_tuple(a, w, IMUdata.system_time()));

    // If filter is full, start using it.
    if (filter.size() == filter_size) {
      // Average variables.
      double          at = 0;
      Eigen::Vector3d aa, aw;
      aa.setZero(); aw.setZero();

      // Average.
      for (size_t ii = 0; ii < filter_size; ++ii) {
        aa += std::get<0>(filter[ii]);
        aw += std::get<1>(filter[ii]);
        at += std::get<2>(filter[ii]);
      }
      aa /= filter_size;
      aw /= filter_size;
      at /= filter_size;

      // Push filtered IMU data to BA.
      vid_tracker->AddInertialMeasurement(aa, aw, at);

      // Pop oldest measurement.
      filter.pop_front();
    }
  } else {
    // Push IMU data to BA.
    vid_tracker->AddInertialMeasurement(a, w, IMUdata.system_time());
  }
}


#if 0
/////////////////////////////////////////////////////////////////////////////
/// Generate depthmap from stereo.
cv::Mat GenerateDepthmap(
    Elas*             elas,
    const cv::Mat&    left_image,
    const cv::Mat&    right_image,
    double            fu,
    double            baseline
  )
{
  // Store dimensions.
  int32_t dims[3];
  dims[0] = left_image.cols;
  dims[1] = left_image.rows;
  dims[2] = dims[0];

  // Allocate memory for disparity.
  cv::Mat disparity = cv::Mat(left_image.rows, left_image.cols, CV_32FC1);

  // Run ELAS.
  elas->process(left_image.data, right_image.data,
                reinterpret_cast<float*>(disparity.data), nullptr, dims);

  // Convert disparity to depth.
  disparity = 1.0 / disparity;
  disparity = disparity * (fu * baseline);

  return disparity;
}
#endif


/////////////////////////////////////////////////////////////////////////////
/// G-FLAGS
DEFINE_string(cam, "", "Camera arguments for HAL driver.");
DEFINE_string(cmod, "cameras.xml", "Camera mode file to load.");
DEFINE_string(imu, "", "IMU arguments for HAL driver.");
DEFINE_string(map, "", "Path containing pre-saved map.");
DEFINE_string(poses, "", "Text file containing ground truth poses.");
DEFINE_string(poses2, "", "Text file containing ground truth poses.");
DEFINE_string(poses_convention, "robotics", "Convention of poses file being loaded: vision, tsukuba, robotics");
DEFINE_int32(frame_skip, 0, "Number of frames to skip between iterations.");
DEFINE_int32(downsample, 0, "How many times to downsample image.");


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  static Eigen::IOFormat kLongCsvFmt(Eigen::FullPrecision, 0, ", ", "\n", "", "");

  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  std::cout << "Starting VIDTrack ..." << std::endl;
  vid::Tracker vid_tracker(15, 4);

  bool use_map = false;
  if (!FLAGS_map.empty()) {
    // Import map.
    vid_tracker.ImportMap(FLAGS_map);

    // Set flag.
    use_map = true;
  }

  ///----- Initialize Camera.
  if (FLAGS_cam.empty()) {
    std::cerr << "Camera arguments missing!" << std::endl;
    exit(EXIT_FAILURE);
  }
  hal::Camera camera(FLAGS_cam);

  const int image_width = camera.Width() >> FLAGS_downsample;
  const int image_height = camera.Height() >> FLAGS_downsample;
  std::cout << "- Image Dimensions: " << image_width <<
               "x" << image_height << std::endl;

  if (camera.NumChannels() != 2) {
    std::cerr << "Two images (stereo pair) are required in order to" \
                 " use this program!" << std::endl;
    exit(EXIT_FAILURE);
  }


  ///----- Initialize IMU.
  if (FLAGS_imu.empty()) {
    std::cerr << "IMU arguments missing!" << std::endl;
    exit(EXIT_FAILURE);
  }
  hal::IMU imu;
  if (use_map == false) {
    imu = hal::IMU(FLAGS_imu);
    using std::placeholders::_1;
    std::function<void (hal::ImuMsg&)> callback
                      = std::bind(IMU_Handler, _1, &vid_tracker);
    imu.RegisterIMUDataCallback(callback);
    std::cout << "- Registering IMU device." << std::endl;
  }

  ///----- Set up GUI.
  pangolin::CreateGlutWindowAndBind("VIDTrack", 1600, 800);

  // Set up panel.
  const unsigned int panel_size = 180;
  pangolin::CreatePanel("ui").SetBounds(0, 1, 0, pangolin::Attach::Pix(panel_size));
  pangolin::Var<bool>           ui_camera_follow("ui.Camera Follow", false, true);
  pangolin::Var<bool>           ui_reset("ui.Reset", true, false);
  pangolin::Var<bool>           ui_show_vo_path("ui.Show VO Path", false, true);
  pangolin::Var<bool>           ui_show_ba_path("ui.Show BA Path", false, true);
  pangolin::Var<bool>           ui_show_ba_rel_path("ui.Show BA Rel Path", true, true);
  pangolin::Var<bool>           ui_show_ba_win_path("ui.Show BA Win Path", false, true);
  pangolin::Var<bool>           ui_show_gt_path("ui.Show GT Path", true, true);

  // Set up container.
  pangolin::View& container = pangolin::CreateDisplay();
  container.SetBounds(0, 1, pangolin::Attach::Pix(panel_size), 0.75);
  container.SetLayout(pangolin::LayoutEqual);
  pangolin::DisplayBase().AddDisplay(container);

  // Set up timer.
  Timer     timer;
  TimerView timer_view;
  timer_view.SetBounds(0.5, 1, 0.75, 1.0);
  pangolin::DisplayBase().AddDisplay(timer_view);
  timer_view.InitReset();

  // Set up analytics.
  std::map<std::string, float>  analytics;
  AnalyticsView                 analytics_view;
  analytics_view.SetBounds(0, 0.5, 0.75, 1.0);
  pangolin::DisplayBase().AddDisplay(analytics_view);
  analytics_view.InitReset();

  // Set up 3D view for container.
  SceneGraph::GLSceneGraph gl_graph;
  SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

  // Reset background color to black.
  glClearColor(0, 0, 0, 1);

  // Add path.
  GLPathAbs gl_path_vo;
  GLPathAbs gl_path_ba;
  GLPathAbs gl_path_ba_rel;
  GLPathAbs gl_path_ba_win;
  GLPathAbs gl_path_gt;
  gl_path_vo.SetPoseDisplay(5);
  gl_path_ba.SetPoseDisplay(5);
  gl_path_ba_rel.SetPoseDisplay(5);
  gl_path_ba_win.SetPoseDisplay(5);
  gl_path_gt.SetPoseDisplay(5);
  gl_path_vo.SetLineColor(1.0, 0, 1.0);
  gl_path_ba.SetLineColor(0, 1.0, 0);
  gl_path_ba_rel.SetLineColor(0, 1.0, 1.0);
  gl_path_ba_win.SetLineColor(1.0, 0, 0);
  gl_path_gt.SetLineColor(0, 0, 1.0);
  gl_graph.AddChild(&gl_path_vo);
  gl_graph.AddChild(&gl_path_ba);
  gl_graph.AddChild(&gl_path_ba_rel);
  gl_graph.AddChild(&gl_path_ba_win);
  gl_graph.AddChild(&gl_path_gt);
  std::vector<Sophus::SE3d>& path_vo_vec = gl_path_vo.GetPathRef();
  std::vector<Sophus::SE3d>& path_ba_vec = gl_path_ba.GetPathRef();
  std::vector<Sophus::SE3d>& path_ba_rel_vec = gl_path_ba_rel.GetPathRef();
  std::vector<Sophus::SE3d>& path_ba_win_vec = gl_path_ba_win.GetPathRef();
  std::vector<Sophus::SE3d>& path_gt_vec = gl_path_gt.GetPathRef();

  // Add grid.
  SceneGraph::GLGrid gl_grid(150, 1);
  gl_graph.AddChild(&gl_grid);

  pangolin::View view_3d;

  pangolin::OpenGlRenderState stacks3d(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 1E-3, 10*1000),
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


  ///----- Load camera models.
  std::shared_ptr<calibu::Rig<double>> rig;
  if (camera.GetDeviceProperty(hal::DeviceDirectory).empty() == false) {
    std::cout<<"- Loaded camera: " <<
               camera.GetDeviceProperty(hal::DeviceDirectory) + '/'
               + FLAGS_cmod << std::endl;
    rig = calibu::ReadXmlRig(camera.GetDeviceProperty(hal::DeviceDirectory)
                             + '/' + FLAGS_cmod);
  } else {
    rig = calibu::ReadXmlRig(FLAGS_cmod);
  }
  // Standardize camera rig and IMU-Camera transform.
  rig = calibu::ToCoordinateConvention(rig, calibu::RdfRobotics);
  // If downsampling, scale camera model appropriately.
  if (FLAGS_downsample != 0) {
    const float scale = 1.0 / std::powf(2.0, FLAGS_downsample);
    rig->cameras_[0]->Scale(scale);
    rig->cameras_[1]->Scale(scale);
  }
  std::cout << "Twc: " << std::endl << rig->cameras_[0]->Pose().matrix().transpose() << std::endl;
  const Eigen::Matrix3f K = rig->cameras_[0]->K().cast<float>();
  std::cout << "-- K is: " << std::endl << K << std::endl;

  ///----- Aux variables.
  Sophus::SE3d  current_pose;
  int           current_keyframe_id;
  double        current_time;
  cv::Mat       current_grey_image, current_depth_map;

  ///----- Load file of ground truth poses (optional).
  bool have_gt;
  std::vector<Sophus::SE3d> poses;
  {
    if (FLAGS_poses.empty()) {
      std::cerr << "- NOTE: No poses file given. Not comparing against ground truth!" << std::endl;
      have_gt = false;
      ui_show_gt_path = false;
    } else {
      std::string pose_file = camera.GetDeviceProperty(hal::DeviceDirectory)
          + "/" + FLAGS_poses;
      FILE* fd = fopen(pose_file.c_str(), "r");
      Eigen::Matrix<double, 6, 1> pose;
      float x, y, z, p, q, r;

      std::cout << "- Loading pose file: '" << pose_file << "'" << std::endl;
      if (FLAGS_poses_convention == "vision") {
        // Vision convention.
        std::cout << "- NOTE: File is being read in VISION frame." << std::endl;
      } else if (FLAGS_poses_convention == "custom") {
        // Custom convention.
        std::cout << "- NOTE: File is being read in *****CUSTOM***** frame." << std::endl;
      } else if (FLAGS_poses_convention == "tsukuba") {
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
        if (FLAGS_poses_convention == "vision") {
          // Vision convention.
          poses.push_back(T);
        } else if (FLAGS_poses_convention == "custom") {
          // Custom setting.
          pose(0) *= -1;
          pose(2) *= -1;
          Sophus::SE3d Tt(SceneGraph::GLCart2T(pose));
          poses.push_back(calibu::ToCoordinateConvention(Tt,
                                                         calibu::RdfRobotics));
        } else if (FLAGS_poses_convention == "tsukuba") {
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
                                          calibu::RdfRobotics));
        }
      }
      std::cout << "- NOTE: " << poses.size() << " poses loaded." << std::endl;
      fclose(fd);
      have_gt = true;
    }
    // This is the file provided by Mike.
    // It supercedes the original "-poses" argument.
    // Convert camera position, look at, and up into a pose.
    if (FLAGS_poses2.empty() == false) {
      poses.clear();

      std::string pose_file = camera.GetDeviceProperty(hal::DeviceDirectory)
          + "/" + FLAGS_poses2;
      FILE* fd = fopen(pose_file.c_str(), "r");
      float cx, cy, cz, lx, ly, lz, ux, uy, uz;

      while (fscanf(fd, "%f,%f,%f,%f,%f,%f,%f,%f,%f",
                        &cx, &cy, &cz, &lx, &ly, &lz, &ux, &uy, &uz) != EOF) {

        Eigen::Vector3d x;
        x(0) = lx - cx;
        x(1) = ly - cy;
        x(2) = lz - cz;

        x.normalize();

        Eigen::Vector3d u;
        u << ux, uy, uz;

        Eigen::Vector3d y;
        y = x.cross(u);
        y.normalize();

        Eigen::Vector3d z;
        z = x.cross(y);

        Eigen::Matrix3d R;
        R.block<3,1>(0,0) = x;
        R.block<3,1>(0,1) = y;
        R.block<3,1>(0,2) = z;

        Eigen::Vector3d cam_center;
        cam_center << cx, cy, cz;

        Eigen::Matrix4d cam_pose;
        cam_pose.setIdentity();
        cam_pose.block<3,3>(0,0) = R;
        cam_pose.block<3,1>(0,3) = cam_center;

        Sophus::SE3d T(cam_pose);

        poses.push_back(T);
      }

      std::cout << "- NOTE: " << poses.size() << " poses loaded." << std::endl;
      fclose(fd);
      have_gt = true;
      ui_show_gt_path = true;
    }
  }

  ///----- Register callbacks.
  // Hide/Show panel.
  pangolin::RegisterKeyPressCallback('~', [&](){
    static bool fullscreen = true;
    fullscreen = !fullscreen;
    if (fullscreen) {
      container.SetBounds(0, 1, pangolin::Attach::Pix(panel_size), 0.75);
    } else {
      container.SetBounds(0, 1, 0, 1);
    }
    analytics_view.Show(fullscreen);
    timer_view.Show(fullscreen);
    pangolin::Display("ui").Show(fullscreen);
  });

  bool run_batch_ba = false;
  pangolin::RegisterKeyPressCallback('o',
                                       [&run_batch_ba] {
                                          run_batch_ba = !run_batch_ba; });

  pangolin::RegisterKeyPressCallback('e',
                                       [&vid_tracker] {
                                          vid_tracker.ExportMap(); });
  // Container view handler.
  const char keyShowHide[] = {'1','2','3','4','5','6','7','8','9','0'};
  const char keySave[]     = {'!','@','#','$','%','^','&','*','(',')'};
  for (size_t ii = 0; ii < container.NumChildren(); ii++) {
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

  ///----- Init general variables.
  unsigned int                      frame_index;
  Sophus::SE3d                      vo_pose;
  Sophus::SE3d                      ba_accum_rel_pose;
  Sophus::SE3d                      ba_global_pose;

  // Image holder.
  std::shared_ptr<hal::ImageArray> images = hal::ImageArray::Create();

  // IMU-Camera transform through robotic to vision conversion.
  Sophus::SE3d Tic = rig->cameras_[0]->Pose();
  Sophus::SE3d Trv;
  Trv.so3() = calibu::RdfRobotics;
  Sophus::SE3d Ticv = rig->cameras_[0]->Pose() * Trv;

  // Open file for saving poses.
  std::ofstream output_file;
  output_file.open("poses.txt");

  double total_trajectory = 0;
  double trajectory_error = 0;

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
      path_ba_rel_vec.clear();
      path_ba_win_vec.clear();
      path_gt_vec.clear();

      // Reset frame counter.
      frame_index = 0;

      // Reset map and current pose.
      vo_pose = Sophus::SE3d();
      ba_global_pose = Sophus::SE3d();
      ba_accum_rel_pose = Sophus::SE3d();
      path_vo_vec.push_back(vo_pose);
      path_ba_vec.push_back(ba_global_pose);
      path_ba_rel_vec.push_back(ba_accum_rel_pose);
      if (have_gt) {
        path_gt_vec.push_back(poses[0].inverse() * poses[frame_index]);
      }

      // Save first pose.
//      output_file << SceneGraph::GLT2Cart(ba_accum_rel_pose.matrix()).transpose()
//                  << std::endl;
      Eigen::Map<Eigen::Matrix<double,7,1>> mat(ba_accum_rel_pose.data());
      output_file << mat.transpose().format(kLongCsvFmt) << std::endl;


      // Capture first image.
      for (size_t ii = 0; ii < filter_size; ++ii) {
        capture_flag = camera.Capture(*images);
        usleep(100);
      }
      capture_flag = camera.Capture(*images);

      // Set images.
      if (FLAGS_downsample != 0) {
        std::vector<cv::Mat> grey_pyramid;
        std::vector<cv::Mat> depth_pyramid;
        cv::buildPyramid(images->at(0)->Mat(), grey_pyramid, FLAGS_downsample);
        cv::buildPyramid(images->at(1)->Mat(), depth_pyramid, FLAGS_downsample);
        current_grey_image = grey_pyramid[FLAGS_downsample];
        current_depth_map = depth_pyramid[FLAGS_downsample];
      } else {
        current_grey_image = images->at(0)->Mat().clone();
        current_depth_map = images->at(1)->Mat().clone();
      }

#if 0
      double min, max;
      cv::minMaxLoc(current_depth_map, &min, &max, nullptr, nullptr);
      std::cout << "Min depth: " << min << "-- Max depth: " << max << std::endl;
#endif

      // Post-process images.
      cv::Mat maskNAN = cv::Mat(current_depth_map != current_depth_map);
      current_depth_map.setTo(0, maskNAN);
      // Trim left-most margin.
      for (int ii = 0; ii < image_height; ++ii) {
        for (int jj = 0; jj < 20; ++jj) {
//          current_depth_map.at<float>(ii, jj) = 0;
        }
      }

      // Depth map sanity check.
      int non_zero = cv::countNonZero(current_depth_map);
      if (non_zero < image_height*image_width*0.5) {
        std::cerr << "warning: Depth map is less than 50% complete!" << std::endl;
      }

      // Get current time.
      current_time = images->at(0)->Timestamp();

      // Init VIDTrack.
      vid_tracker.ConfigureBA(rig);
      vid_tracker.ConfigureDTrack(current_grey_image, current_depth_map,
                                  current_time, rig->cameras_[0]->K());

      // If map is used, find where we initially are and set current_pose.
      if (use_map) {
        const bool ret = vid_tracker.WhereAmI(current_grey_image, current_keyframe_id,
                                              current_pose);
        if (ret ==  false) {
          std::cerr << "Could not find suitable match in map for initial pose estimate!" << std::endl;
          exit(EXIT_FAILURE);
        }
      }


      // Increment frame counter.
      frame_index++;
    }


    ///----- Step forward ...
    if (!paused || pangolin::Pushed(step_once)) {
      //  Capture the new image.
      for (int ii = 0; ii < FLAGS_frame_skip; ++ii) {
        capture_flag = camera.Capture(*images);
        usleep(100);
      }
      if (frame_index == 1850) {
        capture_flag = false;
      } else {
        capture_flag = camera.Capture(*images);
      }

      if (capture_flag == false) {
        std::cout << "Last Pose: " << SceneGraph::GLT2Cart(ba_accum_rel_pose.matrix()).transpose() << std::endl;
        std::cout << "Final Mean Error: " << trajectory_error/frame_index << std::endl;
        std::cout << "Total Trajectory: " << total_trajectory << std::endl;
        paused = true;
      } else {
        // Set images.
        if (FLAGS_downsample != 0) {
          std::vector<cv::Mat> grey_pyramid;
          std::vector<cv::Mat> depth_pyramid;
          cv::buildPyramid(images->at(0)->Mat(), grey_pyramid, FLAGS_downsample);
          cv::buildPyramid(images->at(1)->Mat(), depth_pyramid, FLAGS_downsample);
          current_grey_image = grey_pyramid[FLAGS_downsample];
          current_depth_map = depth_pyramid[FLAGS_downsample];
        } else {
          current_grey_image = images->at(0)->Mat().clone();
          current_depth_map = images->at(1)->Mat().clone();
        }

        // Post-process images.
        cv::Mat maskNAN = cv::Mat(current_depth_map != current_depth_map);
        current_depth_map.setTo(0, maskNAN);
        // Trim left-most margin.
        for (int ii = 0; ii < image_height; ++ii) {
          for (int jj = 0; jj < 20; ++jj) {
//            current_depth_map.at<float>(ii, jj) = 0;
          }
        }

        // Depth map sanity check.
        int non_zero = cv::countNonZero(current_depth_map);
        if (non_zero < image_height*image_width*0.5) {
          std::cerr << "warning: Depth map is less than 50% complete!" << std::endl;
        }

        // Get current time.
        current_time = images->at(0)->Timestamp();

        // Get pose for this image.
        timer.Tic("Tracker");
        Sophus::SE3d rel_pose, vo;
        if (use_map) {
          int keyframe_id = vid_tracker.FindClosestKeyframe(current_keyframe_id,
                                                            current_pose, 500);
//          std::cout << "Closest keyframe: " << keyframe_id << std::endl;
          current_keyframe_id = keyframe_id;
          vid_tracker.RefinePose(current_grey_image, current_keyframe_id,
                                 current_pose);
          ba_global_pose = current_pose;
          ba_accum_rel_pose = current_pose;
        } else {
          vid_tracker.Estimate(current_grey_image, current_depth_map,
                             current_time, ba_global_pose, rel_pose, vo);

          // Uncomment this if poses are to be seen in camera frame (robotics).
//          ba_accum_rel_pose *= Tic.inverse() * rel_pose * Tic;
          // Uncomment this if poses are to be seen in camera frame (vision).
//          ba_accum_rel_pose *= Ticv.inverse() * rel_pose * Ticv;
          // Uncomment this for regular robotic IMU frame.
          ba_accum_rel_pose *= rel_pose;

          // Uncomment this for regular robotic IMU frame.
          vo_pose *= vo;
          // Uncomment this if poses are to be seen in camera frame (robotics).
//          vo_pose *= Tic.inverse() * vo * Tic;
        }
        timer.Toc("Tracker");



        // Save poses.
//        Eigen::Vector6d tmp = SceneGraph::GLT2Cart(ba_accum_rel_pose.matrix());
//        output_file << tmp.transpose() << std::endl;
        Eigen::Map<Eigen::Matrix<double,7,1>> mat(ba_accum_rel_pose.data());
        output_file << mat.transpose().format(kLongCsvFmt) << std::endl;

        ///----- Update GUI objects.
        // Update poses.
        Sophus::SE3d gt_pose;
        if (have_gt) {
          // Use this to bring poses file from camera frame to IMU frame.
          gt_pose = ((poses[0] * Tic.inverse()).inverse() * poses[frame_index] * Tic.inverse());
          // Use this to use the poses file as is.
//          gt_pose = poses[0].inverse() * poses[frame_index];
          // Update errors.
          analytics["BA Global Path Error"] =
              (ba_global_pose.inverse() * gt_pose).translation().norm();
          analytics["BA Rel Path Error"] =
              (ba_accum_rel_pose.inverse() * gt_pose).translation().norm();
          analytics["VO Path Error"] =
              (vo_pose.inverse() * gt_pose).translation().norm();
          trajectory_error += (ba_accum_rel_pose.inverse() * gt_pose).translation().norm();
          total_trajectory += ((poses[frame_index-1] * Tic.inverse()).inverse() * poses[frame_index] * Tic.inverse()).translation().norm();
        }

        // Update path.
        path_vo_vec.push_back(vo_pose);
        path_ba_vec.push_back(ba_global_pose);
        path_ba_rel_vec.push_back(ba_accum_rel_pose);
        if (have_gt) {
          path_gt_vec.push_back(gt_pose);
        }
        path_ba_win_vec.clear();
        const std::deque<ba::PoseT<double> > ba_poses = vid_tracker.GetAdjustedPoses();
        for (size_t ii = 0; ii < ba_poses.size(); ++ii) {
          path_ba_win_vec.push_back(ba_poses[ii].t_wp);
        }

        // Update analytics.
        analytics_view.Update(analytics);

        // Increment frame counter.
        frame_index++;
      }
    }


    ///----- Run full BA ...
    if (pangolin::Pushed(run_batch_ba)) {
      vid_tracker.RunBatchBAwithLC();
      path_ba_vec.clear();
      for (size_t ii = 0; ii < vid_tracker.GetNumPosesRelaxer(); ++ii) {
        const ba::PoseT<double>& pose = vid_tracker.GetPoseRelaxer(ii);
        path_ba_vec.push_back(pose.t_wp);
      }
    }


    /////////////////////////////////////////////////////////////////////////////
    ///---- Render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (capture_flag) {
      image_view.SetImage(current_grey_image.data, image_width, image_height,
                          GL_RGB8, GL_LUMINANCE, GL_UNSIGNED_BYTE);

      depth_view.SetImage(current_depth_map.data, image_width, image_height,
                          GL_RGB8, GL_LUMINANCE, GL_FLOAT, true);
    }

    if (ui_camera_follow) {
      stacks3d.Follow(ba_accum_rel_pose.matrix());
    }

    gl_path_vo.SetVisible(ui_show_vo_path);
    gl_path_ba.SetVisible(ui_show_ba_path);
    gl_path_ba_rel.SetVisible(ui_show_ba_rel_path);
    gl_path_ba_win.SetVisible(ui_show_ba_win_path);
    gl_path_gt.SetVisible(ui_show_gt_path);


#if 1
    // Update path using NIMA's code.
    {
      const std::vector<uint32_t>& imu_residual_ids = vid_tracker.GetImuResidualIds();

      view_3d.ActivateAndScissor(stacks3d);
      const ba::ImuCalibrationT<double>& imu = vid_tracker.GetImuCalibration();
      std::vector<ba::ImuPoseT<double>> imu_poses;
      const ba::InterpolationBufferT<ba::ImuMeasurementT<double>, double>& imu_buffer
          = vid_tracker.GetImuBuffer();

      for (uint32_t id : imu_residual_ids) {
        const auto& res = vid_tracker.GetImuResidual(id);
        const ba::PoseT<double>& pose = vid_tracker.GetPose(res.pose1_id);
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
#endif

    // Sleep a bit if paused.
    if (paused) {
      usleep(1e6/60.0);
    }

    // Stop timer and update.
    timer.Toc();
    timer_view.Update(10, timer.GetNames(3), timer.GetTimes(3));

    pangolin::FinishFrame();
  }

  return 0;
}
