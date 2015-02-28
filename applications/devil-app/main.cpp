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
#include <opencv.hpp>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <ba/BundleAdjuster.h>
#include <ba/Types.h>
#include <ba/InterpolationBuffer.h>
#include <calibu/Calibu.h>
#include <HAL/Utils/GetPot>
#include <HAL/Utils/TicToc.h>
#include <HAL/Camera/CameraDevice.h>
#include <HAL/IMU/IMUDevice.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <pangolin/pangolin.h>
#include <SceneGraph/SceneGraph.h>

#include <libGUI/AnalyticsView.h>
#include <libGUI/Timer.h>
#include <libGUI/TimerView.h>
#include <libGUI/GLPathRel.h>
#include <libGUI/GLPathAbs.h>

#include <devil/tracker.h>


/////////////////////////////////////////////////////////////////////////////
devil::Tracker                                    dvi_track(10, 4);

/////////////////////////////////////////////////////////////////////////////
/// IMU Variables.
const size_t    filter_size = 0;
std::deque<std::tuple<Eigen::Vector3d, Eigen::Vector3d, double> >   filter;

void IMU_Handler(pb::ImuMsg& IMUdata) {
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
      dvi_track.AddInertialMeasurement(aa, aw, at);

      // Pop oldest measurement.
      filter.pop_front();
    }
  } else {
    // Push IMU data to BA.
    dvi_track.AddInertialMeasurement(a, w, IMUdata.system_time());
  }
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  std::cout << "Starting DEVIL ..." << std::endl;

  GetPot cl_args(argc, argv);
  const int frame_skip = cl_args.follow(0, "-skip");

  ///----- Initialize Camera.
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
  pangolin::Var<bool>           ui_show_vo_path("ui.Show VO Path", true, true);
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
  SceneGraph::GLGrid gl_grid(50, 1);
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
  std::cout << "-- K is: " << std::endl << K << std::endl;

  // Convert old rig to new rig.
  calibu::Rig<double> rig;
  calibu::CreateFromOldRig(&old_rig, &rig);

  ///----- DTrack aux variables.
  cv::Mat keyframe_image, keyframe_depth;

  ///----- Load file of ground truth poses (optional).
  bool have_gt;
  std::vector<Sophus::SE3d> poses;
  {
    std::string pose_file = cl_args.follow("", "-poses");
    if (pose_file.empty()) {
      std::cerr << "- NOTE: No poses file given. Not comparing against ground truth!" << std::endl;
      have_gt = false;
      ui_show_gt_path = false;
    } else {
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
          poses.push_back(calibu::ToCoordinateConvention(Tt,
                                                         calibu::RdfRobotics));
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
                                          calibu::RdfRobotics));
        }
      }
      std::cout << "- NOTE: " << poses.size() << " poses loaded." << std::endl;
      fclose(fd);
      have_gt = true;
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
  double                            keyframe_timestamp;

  // IMU-Camera transform through robotic to vision conversion.
  Sophus::SE3d Trv;
  Trv.so3() = calibu::RdfRobotics;
  Sophus::SE3d Tic = rig.t_wc_[0] * Trv;

  // Image holder.
  std::shared_ptr<pb::ImageArray> images = pb::ImageArray::Create();

  // Open file for saving poses.
  std::ofstream output_file;
  output_file.open("poses.txt");

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
      output_file << SceneGraph::GLT2Cart(ba_accum_rel_pose.matrix()).transpose()
                  << std::endl;

      // Capture first image.
      for (size_t ii = 0; ii < filter_size; ++ii) {
        capture_flag = camera.Capture(*images);
        usleep(100);
      }
      capture_flag = camera.Capture(*images);
      cv::Mat current_image = devil::ConvertAndNormalize(images->at(0)->Mat());

      // Reset reference image for DTrack.
      keyframe_image = current_image;
      keyframe_depth = images->at(1)->Mat();
      cv::Mat maskNAN = cv::Mat(keyframe_depth != keyframe_depth);
      keyframe_depth.setTo(0, maskNAN);
      keyframe_timestamp = images->at(0)->Timestamp();

      // Init DEVIL.
      dvi_track.ConfigureBA(old_rig);
      dvi_track.ConfigureDTrack(keyframe_image, keyframe_depth,
                                keyframe_timestamp, old_rig.cameras[0].camera);

      // Increment frame counter.
      frame_index++;
    }


    ///----- Step forward ...
    if (!paused || pangolin::Pushed(step_once)) {
      //  Capture the new image.
      for (int ii = 0; ii < frame_skip; ++ii) {
        capture_flag = camera.Capture(*images);
        usleep(100);
      }
      capture_flag = camera.Capture(*images);

      if (capture_flag == false) {
        paused = true;
      } else {
        // Convert to float and normalize.
        cv::Mat current_image = devil::ConvertAndNormalize(images->at(0)->Mat());
        const double current_timestamp = images->at(0)->Timestamp();

        // Get pose for this image.
        timer.Tic("DEVIL");
        Sophus::SE3d rel_pose, vo;
        dvi_track.Estimate(current_image, images->at(1)->Mat(),
                           current_timestamp, ba_global_pose, rel_pose, vo);

        // Uncomment this if poses are to be seen in vision and camera frame.
        ba_accum_rel_pose *= Tic.inverse() * rel_pose * Tic;
        // Uncomment this for regular robotic IMU frame.
//        ba_accum_rel_pose *= rel_pose;

        vo_pose *= vo;
        timer.Toc("DEVIL");

        // Save poses.
        Eigen::Vector6d tmp = SceneGraph::GLT2Cart(ba_accum_rel_pose.matrix());
        output_file << tmp.transpose() << std::endl;

        ///----- Update GUI objects.
        // Update poses.
        if (have_gt) {
          Sophus::SE3d gt_pose = (poses[0].inverse() * poses[frame_index]);
          // Update errors.
          analytics["BA Global Path Error"] =
              Sophus::SE3::log(ba_global_pose.inverse() * gt_pose).head(3).norm();
          analytics["BA Rel Path Error"] =
              Sophus::SE3::log(ba_accum_rel_pose.inverse() * gt_pose).head(3).norm();
          analytics["VO Path Error"] =
              Sophus::SE3::log(vo_pose.inverse() * gt_pose).head(3).norm();
        }

        // Reset reference image for DTrack.
        keyframe_image = current_image;
        keyframe_depth = images->at(1)->Mat();
        cv::Mat maskNAN = cv::Mat(keyframe_depth != keyframe_depth);
        keyframe_depth.setTo(0, maskNAN);

        // Update path.
        path_vo_vec.push_back(vo_pose);
        path_ba_vec.push_back(ba_global_pose);
        path_ba_rel_vec.push_back(ba_accum_rel_pose);
        if (have_gt) {
          path_gt_vec.push_back(poses[0].inverse()*poses[frame_index]);
        }
        path_ba_win_vec.clear();
        const std::deque<ba::PoseT<double> > ba_poses = dvi_track.GetAdjustedPoses();
        for (size_t ii = 0; ii < ba_poses.size(); ++ii) {
          path_ba_win_vec.push_back(ba_poses[ii].t_wp);
        }

        // Update analytics.
        analytics_view.Update(analytics);

        // Increment frame counter.
        frame_index++;
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
      const std::vector<uint32_t>& imu_residual_ids = dvi_track.GetImuResidualIds();

      view_3d.ActivateAndScissor(stacks3d);
      const ba::ImuCalibrationT<double>& imu = dvi_track.GetImuCalibration();
      std::vector<ba::ImuPoseT<double>> imu_poses;
      const ba::InterpolationBufferT<ba::ImuMeasurementT<double>, double>& imu_buffer
          = dvi_track.GetImuBuffer();

      for (uint32_t id : imu_residual_ids) {
        const auto& res = dvi_track.GetImuResidual(id);
        const ba::PoseT<double>& pose = dvi_track.GetPose(res.pose1_id);
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
