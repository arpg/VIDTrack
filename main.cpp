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

#include "AuxGUI/AnalyticsView.h"
#include "AuxGUI/Timer.h"
#include "AuxGUI/TimerView.h"
#include "AuxGUI/GLPath.h"

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
  GetPot clArgs(argc, argv);
  if (!clArgs.search("-cam")) {
    std::cerr << "Camera arguments missing!" << std::endl;
    exit(EXIT_FAILURE);
  }

  hal::Camera camera(clArgs.follow("", "-cam"));

  const int image_width = camera.Width();
  const int image_height = camera.Height();
  std::cout << "- Image Dimensions: " << image_width <<
               "x" << image_height << std::endl;


  ///----- Set up GUI.
  pangolin::CreateGlutWindowAndBind("DEVIL", 1600, 800);

  // Set up panel.
  const unsigned int panel_size = 180;
  pangolin::CreatePanel("ui").SetBounds(0, 1, 0, pangolin::Attach::Pix(panel_size));
  pangolin::Var<bool>  ui_reset("ui.Reset", true, false);
  pangolin::Var<bool>  ui_use_gt_poses("ui.Use GT Poses", true, true);

  // Set up container.
  pangolin::View& container = pangolin::CreateDisplay();
  container.SetBounds(0, 1, pangolin::Attach::Pix(panel_size), 0.65);
  container.SetLayout(pangolin::LayoutEqual);
  pangolin::DisplayBase().AddDisplay(container);

  // Set up timer.
  Timer timer;
  TimerView timer_view;
  timer_view.SetBounds(0.5, 1, 0.65, 1.0);
  pangolin::DisplayBase().AddDisplay(timer_view);
  timer_view.InitReset();

  // Set up analytics.
  std::map<std::string, float> analytics;
  AnalyticsView analytics_view;
  analytics_view.SetBounds(0, 0.5, 0.65, 1.0);
  pangolin::DisplayBase().AddDisplay(analytics_view);
  analytics_view.InitReset();

  // Set up 3D view for container.
  SceneGraph::GLSceneGraph gl_graph;
  SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

  // Reset background color to black.
  glClearColor(0, 0, 0, 1);

  // Add path.
  GLPath gl_path;
  gl_graph.AddChild(&gl_path);
  std::vector<Sophus::SE3d>& path_vec = gl_path.GetPathRef();

  // Add axis.
  SceneGraph::GLAxis gl_axis;
  gl_graph.AddChild(&gl_axis);

  // Add grid.
  SceneGraph::GLGrid gl_grid(50, 1);
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
  bool capture_flag;
  bool paused = true;
  bool step_once = false;


  ///----- Load camera model.
  calibu::CameraRig rig;
  if (camera.GetDeviceProperty(hal::DeviceDirectory).empty() == false) {
    std::cout<<"- Loaded camera: " <<
               camera.GetDeviceProperty(hal::DeviceDirectory) + '/'
               + clArgs.follow("cameras.xml", "-cmod") << std::endl;
    rig = calibu::ReadXmlRig(camera.GetDeviceProperty(hal::DeviceDirectory)
                             + '/' + clArgs.follow("cameras.xml", "-cmod"));
  } else {
    rig = calibu::ReadXmlRig(clArgs.follow("cameras.xml", "-cmod"));
  }
  Eigen::Matrix3f K = rig.cameras[0].camera.K().cast<float>();
  Eigen::Matrix3f Kinv = K.inverse();
  std::cout << "-- K is: " << std::endl << K << std::endl;

  ///----- Init DTrack stuff.
  cv::Mat keyframe_image, keyframe_depth;
  DTrack dtrack;
  dtrack.Init();
  dtrack.SetParams(rig.cameras[0].camera, rig.cameras[0].camera,
      rig.cameras[0].camera, Sophus::SE3d());

  ///----- Load file of ground truth poses (required).
  std::vector<Sophus::SE3d> poses;
  {
    std::string pose_file = clArgs.follow("", "-poses");
    if (pose_file.empty()) {
      std::cerr << "- NOTE: No poses file given. It is required!" << std::endl;
      exit(EXIT_FAILURE);
    }
    pose_file = camera.GetDeviceProperty(hal::DeviceDirectory) + "/" + pose_file;
    FILE* fd = fopen(pose_file.c_str(), "r");
    Eigen::Matrix<double, 6, 1> pose;
    float x, y, z, p, q, r;

    std::cout << "- Loading pose file: '" << pose_file << "'" << std::endl;
    if (clArgs.search("-V")) {
      // Vision convention.
      std::cout << "- NOTE: File is being read in VISION frame." << std::endl;
    } else if (clArgs.search("-T")) {
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
      if (clArgs.search("-V")) {
        // Vision convention.
        poses.push_back(T);
      } else if (clArgs.search("-T")) {
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
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + GLUT_KEY_RIGHT,
                                     [&step_once] {
                                        step_once = !step_once; });
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'r',
                                     [&ui_reset] {
                                        ui_reset = true; });


  ///----- Init general variables.
  unsigned int current_frame = 0;
  Sophus::SE3d current_pose;
  Sophus::SE3d pose_estimate;
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
      path_vec.clear();
      // Path expects poses in robotic convetion.
      {
        current_pose = Sophus::SE3d();
        Sophus::SO3d& rotation = current_pose.so3();
        rotation = calibu::RdfRobotics;
        path_vec.push_back(current_pose);
      }

      // Re-initialize camera.
      if (!camera.GetDeviceProperty(hal::DeviceDirectory).empty()) {
       camera = hal::Camera(clArgs.follow("", "-cam"));
      }

      // Reset frame counter.
      current_frame = 0;

      // Capture first image.
      capture_flag = camera.Capture(*images);
      cv::Mat current_image = ConvertAndNormalize(images->at(0)->Mat());

      // Reset reference image for DTrack.
      keyframe_image = current_image;
      keyframe_depth = images->at(1)->Mat();

      // Increment frame counter.
      current_frame++;
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

        // RGBD pose estimation.
        pose_estimate = Sophus::SE3d(); // Reset pose estimate.
        dtrack.SetKeyframe(keyframe_image, keyframe_depth);
        double dtrack_error = dtrack.Estimate(current_image, pose_estimate);
        analytics["DTrack RMS"] = dtrack_error;

        // Calculate pose error.
        Sophus::SE3d gt_pose = poses[current_frame-1].inverse()
            * poses[current_frame];
        analytics["DTrack Error"] =
            (Sophus::SE3::log(pose_estimate.inverse() * gt_pose).head(3).norm()
             / Sophus::SE3::log(gt_pose).head(3).norm()) * 100.0;
        timer.Toc("DTrack");

        // If using ground-truth poses, override pose estimate with GT pose.
        if (ui_use_gt_poses) {
          pose_estimate = gt_pose;
        }

        // Update pose.
        current_pose = current_pose * pose_estimate;
        path_vec.push_back(pose_estimate);

        // Reset reference image for DTrack.
        keyframe_image = current_image;
        keyframe_depth = images->at(1)->Mat();

        // Increment frame counter.
        current_frame++;

        // Update analytics.
        analytics_view.Update(analytics);
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

    gl_axis.SetPose(current_pose.matrix());
    stacks3d.Follow(current_pose.matrix());

    // Sleep a bit.
    usleep(1e6/60.0);

    // Stop timer and update.
    timer.Toc();
    timer_view.Update(10, timer.GetNames(3), timer.GetTimes(3));

    pangolin::FinishFrame();
  }

  return 0;
}
