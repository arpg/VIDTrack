#include <unistd.h>
#include <deque>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

#include <HAL/Utils/GetPot>
#include <HAL/Utils/TicToc.h>
#include <HAL/Camera/CameraDevice.h>
#include <pangolin/pangolin.h>
#include <SceneGraph/SceneGraph.h>
#include <calibu/Calibu.h>
#include <calibu/calib/LocalParamSe3.h>
#include <calibu/calib/CostFunctionAndParams.h>
#include <idtam/idtam.cuh>

#include "dtrack.h"

#include "GUI/AnalyticsView.h"
#include "GUI/Timer.h"
#include "GUI/TimerView.h"


///////////////////////////////////////////////////////////////////////////
/// Converts HSV color to RGB. Used to generate "heat map".
inline void HSV2RGB(const Eigen::Vector3d& hsv, Eigen::Vector3d& rgb) {
  float S, H, V, F, M, N, K;
  int   I;

  S = hsv[1];  /* Saturation */
  H = hsv[0];  /* Hue */
  V = hsv[2];  /* value or brightness */

  if ( S == 0.0 ) {
    /*
       * Achromatic case, set level of grey
       */
    rgb[0] = V;
    rgb[1] = V;
    rgb[2] = V;
  } else {
    /*
       * Determine levels of primary colours.
       */
    if (H >= 1.0) {
      H = 0.0;
    } else {
      H = H * 6;
    } /* end if */
    I = (int) H;   /* should be in the range 0..5 */
    F = H - I;     /* fractional part */

    M = V * (1 - S);
    N = V * (1 - S * F);
    K = V * (1 - S * (1 - F));

    if (I == 0) { rgb[0] = V; rgb[1] = K; rgb[2] = M; }
    if (I == 1) { rgb[0] = N; rgb[1] = V; rgb[2] = M; }
    if (I == 2) { rgb[0] = M; rgb[1] = V; rgb[2] = K; }
    if (I == 3) { rgb[0] = M; rgb[1] = N; rgb[2] = V; }
    if (I == 4) { rgb[0] = K; rgb[1] = M; rgb[2] = V; }
    if (I == 5) { rgb[0] = V; rgb[1] = M; rgb[2] = N; }
  } /* end if */
}


///////////////////////////////////////////////////////////////////////////
/// Generates a "heat map" based on an error image provided.
cv::Mat GenerateHeatMap(const cv::Mat& input)
{
  cv::Mat output(input.rows, input.cols, CV_8UC3);

#if 1
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
#else
  // Get standard deviation.
  cv::Scalar     input_mean;
  cv::Scalar     input_stddev;
  cv::meanStdDev(input, input_mean, input_stddev);
  double mean = input_mean.val[0];
  double stddev = input_stddev.val[0];

  double min, max;
  min = mean - 2.0*stddev;
  max = mean + 2.0*stddev;

  Eigen::Vector3d hsv, rgb;
  hsv[1] = 0.6; // Saturation.
  hsv[2] = 1.0; // Value.

  for (int vv = 0; vv < input.rows; ++vv) {
    for (int uu = 0; uu < input.cols; ++uu) {
      // Normalize.
      float n_val = (input.at<float>(vv, uu) - min) / (max - min);

      // Clamp.
      if (n_val > 1.0)
        n_val = 1.0;

      if (n_val < 0.0)
        n_val = 0.0;

      // Set up HSV color.
      hsv[0] = n_val;

      // Get RGB equiv.
      HSV2RGB(hsv, rgb);

      output.at<cv::Vec3b>(vv, uu) = cv::Vec3b(rgb[0]*255,
                                               rgb[1]*255, rgb[2]*255);
    }
  }
 #endif

  return output;
}

///////////////////////////////////////////////////////////////////////////
/// Normalizes slice of cost volume (for display).
void NormalizeSliceBuffer(float* data, unsigned int imageWidth,
                          unsigned int imageHeight)
{
  float max = 0;
  float min = FLT_MAX;
  for (int jj = 0; jj < imageHeight; jj++){
    for (int ii = 0; ii < imageWidth; ii++) {
      if ((data[ii + jj*imageWidth] > max) && (data[ii + jj*imageWidth] != FLT_MAX))
        max = data[ii + jj*imageWidth];
      if (data[ii + jj*imageWidth] < min)
        min = data[ii + jj*imageWidth];
    }
  }

  for (int jj = 0; jj < imageHeight; jj++){
    for (int ii = 0; ii < imageWidth; ii++) {
      if (data[ii + jj*imageWidth] == FLT_MAX)
        data[ii + jj*imageWidth] = 0;
      else {
        float delta = max - min;
        if (delta == 0) delta++;
        data[ii + jj*imageWidth] = 1 - ((data[ii + jj*imageWidth] - min) / delta);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
/// Saves greyscale and estimated depth map to disk.
inline void Export(unsigned int frame_number, const cv::Mat& depth_map,
                   const cv::Mat& grey_image)
{
  const std::string depth_file_prefix = "depth_dtam_";
  char index[10];
  sprintf(index, "%05d", frame_number);
  std::string depth_filename;
  depth_filename = depth_file_prefix + index + ".pdm";
  std::ofstream file(depth_filename.c_str(), std::ios::out | std::ios::binary);
  file << "P7" << std::endl;
  file << depth_map.cols << " " << depth_map.rows << std::endl;
  unsigned int size = depth_map.elemSize1() * depth_map.rows * depth_map.cols;
  file << 4294967295 << std::endl;
  file.write((const char*)depth_map.data, size);
  file.close();

  // Save grey image.
  std::string grey_prefix = "grey_";
  std::string grey_filename;
  grey_filename = grey_prefix + index + ".pgm";
  cv::imwrite(grey_filename, grey_image);

  std::cout << "-- Saving: " << depth_filename << " " <<
               grey_filename << std::endl;
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
/// Intensity interpolation function.
inline float Interpolate(float          x,            // Input: X coordinate.
                         float          y,            // Input: Y coordinate.
                         const float*   image,        // Input: Pointer to image.
                         const int      image_width   // Input: Image width.
                         )
{
  const int    px  = static_cast<int>(x);  /* top-left corner */
  const int    py  = static_cast<int>(y);
  const float  ax  = x-px;
  const float  ay  = y-py;
  const float  ax1 = 1.0f-ax;
  const float  ay1 = 1.0f-ay;

  const float* p0  = image+(image_width*py)+px;

  float        p1  = p0[0];
  float        p2  = p0[1];
  float        p3  = p0[image_width];
  float        p4  = p0[image_width+1];

  p1 *= ay1;
  p2 *= ay1;
  p3 *= ay;
  p4 *= ay;
  p1 += p3;
  p2 += p4;
  p1 *= ax1;
  p2 *= ax;

  return p1+p2;
}


/////////////////////////////////////////////////////////////////////////////
/// CERES Stuff.

/// Aux functions.
namespace ceres {

/// Function that checks if template type is JET.
template <class T>
struct TypeIsJet
{
  static const bool value = false;
};

/// Function that checks if template type is JET.
template<>
struct TypeIsJet< ceres::Jet<double, 8> >
{
  static const bool value = true;
};

/// Weight of smoothness term for DOUBLE.
inline double SmoothWeight(double u, double v, float* image, int image_width,
                           double g_alpha, double g_beta)
{
  const double du = (Interpolate(u+1, v, image, image_width)
      - Interpolate(u-1, v, image, image_width)) / 2.0;
  const double dv = (Interpolate(u, v+1, image, image_width)
      - Interpolate(u, v-1, image, image_width)) / 2.0;

  const double mag = sqrtf(du*du + dv*dv);
  const double g = expf(-g_alpha*powf(mag, g_beta));

  return g;
}

/// Weight of smoothness term for JET.
template <typename T, int N> inline
Jet<T, N> SmoothWeight(const Jet<T, N>& u, const Jet<T, N>& v, float* image,
                     int image_width, double g_alpha, double g_beta)
{
  const T term = T(SmoothWeight(u.a, v.a, image, image_width, g_alpha, g_beta));

  const T du = T(SmoothWeight(u.a+1, v.a, image, image_width, g_alpha, g_beta)
                 - SmoothWeight(u.a-1, v.a, image, image_width, g_alpha, g_beta))
                / T(2.0);
  const T dv = T(SmoothWeight(u.a, v.a+1, image, image_width, g_alpha, g_beta)
                 - SmoothWeight(u.a, v.a-1, image, image_width, g_alpha, g_beta))
                / T(2.0);
  return Jet<T, N>(term, du*u.v + dv*v.v);
}

/// Image accesor DOUBLE.
inline double ImageSubPix(double u, double v, float* image, int image_width)
{
  return Interpolate(u, v, image, image_width);
}

/// Image accesor JET.
template <typename T, int N> inline
Jet<T, N> ImageSubPix(const Jet<T, N>& u, const Jet<T, N>& v, float* image,
                      int image_width)
{
  const T intensity = Interpolate(u.a, v.a, image, image_width);
  const T du = T(Interpolate(u.a+1, v.a, image, image_width)
                 - Interpolate(u.a-1, v.a, image, image_width)) / T(2.0);
  const T dv = T(Interpolate(u.a, v.a+1, image, image_width)
                 - Interpolate(u.a, v.a-1, image, image_width)) / T(2.0);
  return Jet<T, N>(intensity, du*u.v + dv*v.v);
}

/// InBounds DOUBLE.
inline bool InBounds(double u, double v, int image_width, int image_height)
{
  if ((u >= 2.0) && (u <= image_width-3) && (v >= 2.0) && (v <= image_height-3)) {
    return true;
  } else {
    return false;
  }
}

/// InBounds JET.
template <typename T, int N> inline
bool InBounds(const Jet<T, N>& u, const Jet<T, N>& v, int image_width,
              int image_height)
{
  if ((u.a >= 2.0) && (u.a <= image_width-3)
      && (v.a >= 2.0) && (v.a <= image_height-3)) {
    return true;
  } else {
    return false;
  }
}

} /* namespace */


/// Cost Function: Data Term.
struct DataTerm {
  DataTerm(const Eigen::Matrix3d& K, const cv::Mat& reference_image,
           const cv::Mat& support_image, const Eigen::Vector2d& pr,
           const double lambda)
    : K_(K), reference_image_(reference_image), support_image_(support_image),
      pr_(pr), lambda_(lambda)
  { }

  template<typename T>
  bool operator()(const T* const _Tsr, const T* const _depth, T* residual) const
  {
    const Eigen::Map<const Sophus::SE3Group<T>> Tsr(_Tsr);
    const T depth = *(_depth);

    if (ceres::IsNaN(depth)) {
      return true;
    }

    const Eigen::Matrix<T, 3, 3> K = K_.cast<T>();
    const Eigen::Matrix<T, 2, 1> pr = pr_.cast<T>();

    // Reference image intensity.
    const T Ir = ceres::ImageSubPix(pr(0), pr(1),
                                    reinterpret_cast<float*>(reference_image_.data),
                                    reference_image_.cols);

    // Back-project point.
    Eigen::Matrix<T, 4, 1> hPr;
    hPr(0) = depth*(pr(0)-K(0,2))/K(0,0);
    hPr(1) = depth*(pr(1)-K(1,2))/K(1,1);
    hPr(2) = depth;
    hPr(3) = T(1.0);

    // Transfer 3d point from reference to support image.
    Eigen::Matrix<T, 3, 1> Ps = Tsr.matrix3x4() * hPr;

    // Project.
    Eigen::Matrix<T, 2, 1> ps;
    ps(0) = (Ps(0)*K(0,0)/Ps(2)) + K(0,2);
    ps(1) = (Ps(1)*K(1,1)/Ps(2)) + K(1,2);

    // Check if projected point is within image bounds ...
    if (ceres::InBounds(ps(0), ps(1), support_image_.cols,
                        support_image_.rows) == false) {
      // NOTE(jmf) This is meh. Should find a way to tell Ceres to
      // disregard this point.
      residual[0] = T(0.0);
      return true;
    }

    const T Is = ceres::ImageSubPix(ps(0), ps(1),
                                    reinterpret_cast<float*>(support_image_.data),
                                    support_image_.cols);

    residual[0] = T(lambda_) * (Is - Ir);

    return true;
  }

  const Eigen::Matrix3d&    K_;
  const cv::Mat&            reference_image_;
  const cv::Mat&            support_image_;
  const Eigen::Vector2d     pr_;
  const double              lambda_;
};


/// Cost Function: Smoothness Term.
struct SmoothnessTerm {
  SmoothnessTerm(const cv::Mat& reference_image,
                 const Eigen::Vector2d& pr, const double g_alpha,
                 const double g_beta, const double epsilon)
    : reference_image_(reference_image), pr_(pr),
      g_alpha_(g_alpha), g_beta_(g_beta), epsilon_(epsilon)
  { }

  template<typename T>
  bool operator()(const T* const _depth, const T* const _depthU,
                  const T* const _depthV, T* residual) const
  {
    const Eigen::Matrix<T, 2, 1> pr = pr_.cast<T>();

    if (ceres::InBounds(pr(0), pr(1), reference_image_.cols,
                        reference_image_.rows) == false) {
      residual[0] = T(0.0);
      return true;
    }
    const T inv_depth = T(1.0) / *(_depth);
    const T inv_depthU = T(1.0) / *(_depthU);
    const T inv_depthV = T(1.0) / *(_depthV);

    if (ceres::IsNaN(inv_depth) || ceres::IsNaN(inv_depthU)
        || ceres::IsNaN(inv_depthV)) {
      residual[0] = T(0.0);
      return true;
    }

    const T weight = ceres::SmoothWeight(pr(0), pr(1),
                                         reinterpret_cast<float*>(reference_image_.data),
                                         reference_image_.cols, g_alpha_, g_beta_);

    const T du = inv_depthU - inv_depth;
    const T dv = inv_depthV - inv_depth;
    const T d_norm = sqrt(du*du + dv*dv);

    T term;

    if (d_norm < T(epsilon_)) {
      term =  pow(d_norm, 2) / (2.0*T(epsilon_));
    } else {
      term = d_norm - (T(epsilon_)/T(2.0));
    }

    residual[0] = weight * term;
    return true;
  }

  const cv::Mat&            reference_image_;
  const Eigen::Vector2d     pr_;
  const double              g_alpha_;
  const double              g_beta_;
  const double              epsilon_;
};



/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  std::cout << "Starting iDTAM ..." << std::endl;

  ///----- Initialize Camera.
  GetPot clArgs(argc, argv);
  if (!clArgs.search("-cam")) {
    std::cerr << "Camera arguments missing!" << std::endl;
    exit(EXIT_FAILURE);
  }

  hal::Camera camera(clArgs.follow("", "-cam"));

  const int image_width = camera.Width();
  const int image_height = camera.Height();
  std::cout << "- Image Dimensions: " << image_width << "x" << image_height << std::endl;


  ///----- Set up GUI.
  pangolin::CreateGlutWindowAndBind("iDTAM", 1600, 800);

  // Set up panel.
  const unsigned int panel_size = 180;
  pangolin::CreatePanel("ui").SetBounds(0, 1, 0, pangolin::Attach::Pix(panel_size));
  pangolin::Var<int>   ui_num_levels("ui.Num Levels", 32, 1, 128);
  pangolin::Var<int>   ui_cur_level("ui.CurLevel", 0, 0, ui_num_levels-1);
  pangolin::Var<bool>  ui_reset("ui.Reset", true, false);
  pangolin::Var<bool>  ui_inc_vol("ui.Incremental Volume", false, true);
  pangolin::Var<bool>  ui_use_gt_poses("ui.Use GT Poses", true, true);
  pangolin::Var<float> ui_kmin("ui.Kmin", 1/66.0, 0.01, 1);
  pangolin::Var<float> ui_kmax("ui.KMax", 1/20.0, 0.1, 3);
  pangolin::Var<float> ui_epsilon("ui.Epsilon", 1e-4, 1e-5, 1);
  pangolin::Var<float> ui_theta_start("ui.Theta Start", 1, 0, 1);
  pangolin::Var<float> ui_theta_end("ui.Theta End", 1e-4, 1e-5, 1);
  pangolin::Var<float> ui_lambda("ui.Lambda", 1, 0, 20);
  pangolin::Var<float> ui_g_alpha("ui.gAlpha", 100, 0, 200);
  pangolin::Var<float> ui_g_beta("ui.gBeta", 1.6, 0, 10);
  pangolin::Var<float> ui_beta("ui.Beta", 1e-4, 0, 0.01);
  pangolin::Var<float> ui_sig_d("ui.Sigma D", 0.7, 0, 1);
  pangolin::Var<float> ui_sig_q("ui.Sigma Q", 0.7, 0, 1);
  pangolin::Var<float> ui_omega("ui.Omega", 0.5, 0, 1);
  pangolin::Var<bool>  ui_export_data("ui.Export Data", false, true);
  pangolin::Var<int>   ui_window_size("ui.Window Size", 5, 1, 12);
  pangolin::Var<bool>  ui_show_ba_depth("ui.Show BA Depth", false, true);

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

  // Add axis.
  SceneGraph::GLAxis gl_axis(0.1);
  gl_graph.AddChild(&gl_axis);

  pangolin::View view_3d;
  const double far = 10*1000;
  const double near = 1E-3;

  pangolin::OpenGlRenderState stacks3d(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, near, far),
        pangolin::ModelViewLookAt(0, 0, 0, 0, 0, 1, pangolin::AxisNegY)
        );

  view_3d.SetHandler(new SceneGraph::HandlerSceneGraph(gl_graph, stacks3d))
      .SetDrawFunction(SceneGraph::ActivateDrawFunctor(gl_graph, stacks3d))
      .SetAspect(640.0 / 480.0);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer, image_width*image_height, GL_FLOAT, 3);
  const int vbo_size = image_width*image_height*3*sizeof(float);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer, 3*image_width*image_height, GL_UNSIGNED_BYTE, 3);
  const int cbo_size = image_width*image_height*3;
  pangolin::GlBuffer ibo;
  pangolin::MakeTriangleStripIboForVbo(ibo, image_width, image_height);
  SceneGraph::GLVbo glVbo(&vbo, nullptr, &cbo);
  gl_axis.AddChild(&glVbo);

  // Add all subviews to container.
  container.AddDisplay(view_3d);

  SceneGraph::ImageView image_view;
  container.AddDisplay(image_view);

  SceneGraph::ImageView slice_view;
  container.AddDisplay(slice_view);
  container[2].Show(false);

  SceneGraph::ImageView depth_view;
  container.AddDisplay(depth_view);

  SceneGraph::ImageView depth_error_view;
  container.AddDisplay(depth_error_view);
  if (camera.NumChannels() < 2) {
    container[4].Show(false);
  }

  SceneGraph::ImageView ba_depth_view;
  container.AddDisplay(ba_depth_view);

  SceneGraph::ImageView ba_depth_error_view;
  container.AddDisplay(ba_depth_error_view);
  if (camera.NumChannels() < 2) {
    container[6].Show(false);
  }

  SceneGraph::ImageView ba_covariance_view;
  container.AddDisplay(ba_covariance_view);


  // GUI aux variables.
  bool paused = true;
  bool step_once = false;
  bool run_ba = false;
  float* depth_buffer = (float*)malloc(image_width*image_height*sizeof(float));
  float* slice_buffer = (float*)malloc(image_width*image_height*sizeof(float));
  float* vbo_buffer = (float*)malloc(image_width*image_height*3*sizeof(float));
  unsigned char* cbo_buffer = (unsigned char*)malloc(image_width*image_height*3);


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

  ///----- Init iDTAM stuff.
  iDTAM idtam;
  iDTAM::Parameters params;

  ///----- Init DTrack stuff.
  cv::Mat keyframe_image;
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
        tsukuba_convention << -1, 0, 0,
            0, -1, 0,
            0, 0, -1;
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


  ///----- Init frame variables.
  Sophus::SE3d current_pose;
  unsigned int current_frame = 0;
  std::shared_ptr<pb::ImageArray> image = pb::ImageArray::Create();


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
    pangolin::RegisterKeyPressCallback(keyShowHide[ii], [&container,ii]() { container[ii].ToggleShow(); });
    pangolin::RegisterKeyPressCallback(keySave[ii], [&container,ii]() { container[ii].SaveRenderNow("screenshot", 4); });
  }

  pangolin::RegisterKeyPressCallback('.', [&ui_cur_level, &ui_num_levels] {
    ui_cur_level = ui_cur_level + 1;
    if (ui_cur_level >= ui_num_levels)
      ui_cur_level = ui_num_levels - 1;
  });
  pangolin::RegisterKeyPressCallback(',', [&ui_cur_level] {
    ui_cur_level = ui_cur_level - 1;
    if (ui_cur_level < 0)
      ui_cur_level = 0;
  });
  pangolin::RegisterKeyPressCallback(' ', [&paused] { paused = !paused; });
  pangolin::RegisterKeyPressCallback('/', [&step_once] { step_once = !step_once; });
  pangolin::RegisterKeyPressCallback('o', [&run_ba] { run_ba = !run_ba; });
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'r', [&ui_reset] { ui_reset = true; });


  ///----- CERES variables.
  cv::Mat ba_depth_double;
  cv::Mat ba_depth_float(image_height, image_width, CV_32FC1);
  cv::Mat ba_covariance(image_height, image_width, CV_64FC1);
  std::deque<std::pair<cv::Mat, Sophus::SE3d>> ba_frames;
  ceres::Problem::Options problem_options;
  problem_options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);
  calibu::LocalParameterizationSe3 local_param_se3;


  /////////////////////////////////////////////////////////////////////////////
  ///---- MAIN LOOP
  ///

  while (!pangolin::ShouldQuit()) {

    timer.Tic();

    ///----- Init reset ...
    if (pangolin::Pushed(ui_reset)) {
      // Clear GUI variables.
      memset(depth_buffer, 0, image_width*image_height*sizeof(float));
      depth_error_view.SetImage(depth_buffer, image_width, image_height,
                                GL_RGB8, GL_LUMINANCE, GL_FLOAT, true);

      // Reset timer and analytics.
      timer_view.InitReset();
      analytics_view.InitReset();

      // Re-initialize camera.
      if (!camera.GetDeviceProperty(hal::DeviceDirectory).empty()) {
        camera = hal::Camera(clArgs.follow("", "-cam"));
      }

      // Set parameters from GUI.
      params.beta         = ui_beta;
      params.epsilon      = ui_epsilon;
      params.g_alpha      = ui_g_alpha;
      params.g_beta       = ui_g_beta;
      params.lambda       = ui_lambda;
      params.omega        = ui_omega;
      params.sig_d        = ui_sig_d;
      params.sig_q        = ui_sig_q;
      params.theta_end    = ui_theta_end;
      params.theta_start  = ui_theta_start;

      idtam.Init(image_width, image_height, ui_kmin, ui_kmax, ui_num_levels,
                 ui_inc_vol, K.data(), Kinv.data(), params);

      // Reset frame counter.
      current_frame = 0;

      // Capture first image.
      camera.Capture(*image);

      // Convert to float and normalize.
      cv::Mat current_image = ConvertAndNormalize(image->at(0)->Mat());

      // Get pose for this image.
      current_pose = Sophus::SE3d();
      Eigen::Matrix4f current_pose_eigen = current_pose.matrix().cast<float>();

      // Push image.
      idtam.PushImage(reinterpret_cast<float*>(current_image.data),
                      current_pose_eigen.data());
      idtam.GetDepth(depth_buffer);

      // Set initial reference image for DTrack.
      keyframe_image = current_image;

      // Increment frame counter.
      current_frame++;

      // Reset BA frame window and insert first frame.
      ba_frames.clear();
      ba_frames.push_back(std::make_pair(current_image, current_pose));
      memset(ba_covariance.data, 0, image_width*image_width*sizeof(double));
    }


    ///----- Step forward ...
    if (!paused || pangolin::Pushed(step_once)) {
      //  Capture the new image.
      camera.Capture(*image);

      // Convert to float and normalize.
      cv::Mat current_image = ConvertAndNormalize(image->at(0)->Mat());

      // Get pose for this image.
      timer.Tic("DTrack");

      // RGBD pose estimation.
      Sophus::SE3d pose_delta;
      cv::Mat keyframe_depth(image_height, image_width, CV_32FC1, depth_buffer);
      dtrack.SetKeyframe(keyframe_image, keyframe_depth);
      double dtrack_error = dtrack.Estimate(current_image, pose_delta);
      analytics["DTrack RMS"] = dtrack_error;

      // Reset reference image for DTrack.
      keyframe_image = current_image;

      // Calculate pose error.
      Sophus::SE3d gt_pose = poses[current_frame-1].inverse()
          * poses[current_frame];
      analytics["DTrack Error"] =
          (Sophus::SE3::log(pose_delta.inverse() * gt_pose).head(3).norm()
           / Sophus::SE3::log(gt_pose).head(3).norm()) * 100.0;
      timer.Toc("DTrack");

      // If using ground-truth poses, override pose estimate with GT pose.
      if (ui_use_gt_poses) {
        pose_delta = gt_pose;
      }

      // Update pose.
      current_pose = current_pose * pose_delta;
      Eigen::Matrix4f current_pose_eigen = current_pose.matrix().cast<float>();

      // Push image.
      timer.Tic("iDTAM");
      idtam.PushImage(reinterpret_cast<float*>(current_image.data),
                      current_pose_eigen.data());
      idtam.GetDepth(depth_buffer);
      timer.Toc("iDTAM");

      // Insert frame into BA queue.
      ba_frames.push_back(std::make_pair(current_image, current_pose));

      // Pop frame from BA queue if exceeds window size.
      if (ba_frames.size() > ui_window_size+1) {
        ba_frames.pop_front();
      }

      // Export data if enabled.
      if (ui_export_data) {
        cv::Mat depth_wrapper(image_height, image_width, CV_32FC1, depth_buffer);
        Export(current_frame, depth_wrapper, image->at(0)->Mat());
      }

      // Increment frame counter.
      current_frame++;

      ///-------------------- Update GUI objects.

      // Calculate error if true depth map is provided.
      if (image->Size() > 2) {
        cv::Mat true_depth = image->at(2)->Mat().clone();
        cv::Mat depth_wrapper(image_height, image_width, CV_32FC1, depth_buffer);
        cv::Mat estimated_depth = depth_wrapper.clone();
        // Take out NANs.
        cv::Mat maskNanEst = cv::Mat(estimated_depth == estimated_depth);
        cv::Mat maskNanTrue = cv::Mat(true_depth == true_depth);
        cv::Mat mask = maskNanEst & maskNanTrue;
        cv::Mat depth_error;
        cv::absdiff(true_depth, estimated_depth, depth_error);
        const double mean = cv::mean(depth_error, mask).val[0];
        analytics["Depth Error"] = mean;
        cv::Mat error_heat_map = GenerateHeatMap(depth_error);
        depth_error_view.SetImage(error_heat_map.data, image_width, image_height,
                                  GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);

        std::ofstream file4("depth_error", std::ios::out | std::ios::binary);
        file4.write((const char*)depth_error.data, depth_error.rows*depth_error.cols*sizeof(float));
        file4.close();

#if 0
        double min, max;
        cv::minMaxIdx(true_depth, &min, &max, NULL, NULL, maskNanTrue);
        printf("- True Depth: Max = %f     Min = %f\n", max, min);
#endif
      }

      analytics_view.Update(analytics);
    }

    ///----- Run dense BA ...
    if (pangolin::Pushed(run_ba)) {

      std::cout << "Running dense BA..." << std::endl;

      if (ba_frames.size() < ui_window_size+1) {
        std::cout << "Window is not yet full. Doing nothing..." << std::endl;
      } else {
        // Convert depths to inverse depth, and from floats to doubles.
        cv::Mat depth_image(image_height, image_width, CV_32FC1, depth_buffer);
        depth_image.convertTo(ba_depth_double, CV_64FC1);
        double* ba_depth_ptr = reinterpret_cast<double*>(ba_depth_double.data);

#if 0
        cv::Mat depth_image(image_height, image_width, CV_32FC1,
                            ba_depth_buffer);
        cv::imshow("Ref Img", ba_frames[ui_window_size].first);
        cv::imshow("Support Img", ba_frames[0].first);
        depth_image /= 80.0;
        cv::imshow("Depth Img", depth_image);
        cv::waitKey(10);
#endif

        std::vector<Sophus::SE3d> frame_poses;
        frame_poses.resize(ui_window_size);
        Sophus::SE3d reference_pose = ba_frames[ui_window_size].second;
        Eigen::Vector6d noise;
        noise << 0.5, 0.3, 0.1, 0.01, 0, 0.1;
        Eigen::Matrix4d noiseT = SceneGraph::GLCart2T(noise);
        Sophus::SE3d noisy_pose(noiseT);
        for (unsigned int ii = 0; ii < ui_window_size; ++ii) {
          frame_poses[ii] = ba_frames[ii].second.inverse()*reference_pose;
//          frame_poses[ii] = frame_poses[ii] * noisy_pose;
          problem.AddParameterBlock(frame_poses[ii].data(), 7, &local_param_se3);
        }

        // Find lambda for smoothness term.
        double min, max;
        // NOTE: Max inverse depth is minimum depth.
        cv::minMaxIdx(ba_depth_double, &min, &max);
//        const double lambda = 1.0 / (1.0 + 0.5/max);
        const double lambda = 1.0 / (1.0 + 0.5/(1.0/min));
//        const double lambda = ui_lambda;

//        const int width_margin = 280;
//        const int height_margin = 200;
        const int width_margin = 280;
        const int height_margin = 200;

        cv::Rect ROI(image_width/2-width_margin, image_height/2-height_margin,
                     2*width_margin, 2*height_margin);


        for (unsigned int ii = 0; ii < ui_window_size; ++ii) {
          std::cout << "True Pose: " << std::endl << (ba_frames[ii].second.inverse()*reference_pose).matrix() << std::endl;
          std::cout << "Initial Pose: " << std::endl << frame_poses[ii].matrix() << std::endl;

#if 1
//          for (size_t vv = 0; vv < image_height; ++vv) {
//            for (size_t uu = 0; uu < image_width; ++uu) {
          for (size_t vv = image_height/2-height_margin; vv < image_height/2+height_margin; ++vv) {
            for (size_t uu = image_width/2-width_margin; uu < image_width/2+width_margin; ++uu) {
              Eigen::Vector2d pr;
              pr << uu, vv;

              // Data term.
              calibu::CostFunctionAndParams* cost_func_data =
                  new calibu::CostFunctionAndParams();

              cost_func_data->Cost() =
                  new ceres::AutoDiffCostFunction
                  <DataTerm, 1, Sophus::SE3d::num_parameters, 1>
                  (new DataTerm(rig.cameras[0].camera.K(),
                   keyframe_image, ba_frames[ii].first, pr, lambda));

              cost_func_data->Loss() = NULL;

              cost_func_data->Params() = std::vector<double*> {
                  frame_poses[ii].data(), &ba_depth_ptr[uu + vv*image_width] };

              problem.AddResidualBlock(cost_func_data->Cost(),
                                       cost_func_data->Loss(),
                                       cost_func_data->Params());
            }
          }
#endif

#if 1
          // Smoothness term.
//          for (size_t vv = 0; vv < image_height-1; ++vv) {
//            for (size_t uu = 0; uu < image_width-1; ++uu) {
          for (size_t vv = image_height/2-height_margin; vv < image_height/2+height_margin-1; ++vv) {
            for (size_t uu = image_width/2-width_margin; uu < image_width/2+width_margin-1; ++uu) {
              Eigen::Vector2d pr;
              pr << uu, vv;

              calibu::CostFunctionAndParams* cost_func_smooth =
                  new calibu::CostFunctionAndParams();

              cost_func_smooth->Cost() =
                  new ceres::AutoDiffCostFunction
                  <SmoothnessTerm, 1, 1, 1, 1>
                  (new SmoothnessTerm(keyframe_image, pr,
                                      ui_g_alpha, ui_g_beta, ui_epsilon));

              cost_func_smooth->Loss() = NULL;

              cost_func_smooth->Params() = std::vector<double*> {
                  &ba_depth_ptr[uu + vv*image_width],
                  &ba_depth_ptr[uu+1 + vv*image_width],
                  &ba_depth_ptr[uu + (vv+1)*image_width] };

              problem.AddResidualBlock(cost_func_smooth->Cost(),
                                       cost_func_smooth->Loss(),
                                       cost_func_smooth->Params());
            }
          }
#endif
        }

        // Set center depth as constant so as to fix scale ambiguity.
        // TODO(jmf) Find smallest error depth and use that instead?
//        problem.SetParameterBlockConstant(&ba_depth_ptr[320 + 240*image_width]);
        problem.SetParameterBlockConstant(frame_poses[0].data());

        // Solve.
        ceres::Solver::Options solver_options;
        solver_options.num_threads = 1;
        solver_options.trust_region_strategy_type = ceres::DOGLEG;
        solver_options.max_num_iterations = 50;
        ceres::Solver::Summary solver_summary;
        timer.Tic("Dense BA");
        ceres::Solve(solver_options, &problem, &solver_summary);
        timer.Toc("Dense BA");
        std::cout << solver_summary.FullReport() << std::endl;

        // Get covariance.
        ceres::Covariance::Options covariance_options;
//        covariance_options.algorithm_type = ceres::DENSE_SVD;
        ceres::Covariance covariance(covariance_options);

#if 0
        double* ba_covariance_ptr = reinterpret_cast<double*>(ba_covariance.data);
//        for (size_t vv = 0; vv < image_height; ++vv) {
//          for (size_t uu = 0; uu < image_width; ++uu) {
        for (size_t vv = image_height/2-height_margin; vv < image_height/2+height_margin; ++vv) {
          for (size_t uu = image_width/2-width_margin; uu < image_width/2+width_margin; ++uu) {
            std::vector<std::pair<const double*, const double*> > covariance_blocks;
            covariance_blocks.push_back(
                  std::make_pair(&ba_depth_ptr[uu + vv*image_width],
                                 &ba_depth_ptr[uu + vv*image_width]));
            bool flag = covariance.Compute(covariance_blocks, &problem);
            if (flag == false) {
              std::cerr << "Error for depth at (" << vv << ", " << uu << ")" << std::endl;
              ba_covariance_ptr[uu + vv*image_width] = 0.0;
            } else {
              covariance.GetCovarianceBlock(&ba_depth_ptr[uu + vv*image_width],
                  &ba_depth_ptr[uu + vv*image_width],
                  &ba_covariance_ptr[uu + vv*image_width]);
            }
          }
        }
#endif

#if 0
        std::vector<std::pair<const double*, const double*> > covariance_blocks;
        for (size_t vv = image_height/2-height_margin; vv < image_height/2+height_margin; ++vv) {
          for (size_t uu = image_width/2-width_margin; uu < image_width/2+width_margin; ++uu) {
            covariance_blocks.push_back(
                  std::make_pair(&ba_depth_ptr[uu + vv*image_width],
                                 &ba_depth_ptr[uu + vv*image_width]));
          }
        }

        std::cout << "Computing COVARIANCE ..." << std::endl;
        covariance.Compute(covariance_blocks, &problem);

        std::cout << "Getting COVARIANCE ..." << std::endl;
        double* ba_covariance_ptr = reinterpret_cast<double*>(ba_covariance.data);
        for (size_t vv = image_height/2-height_margin; vv < image_height/2+height_margin; ++vv) {
          for (size_t uu = image_width/2-width_margin; uu < image_width/2+width_margin; ++uu) {
            covariance.GetCovarianceBlock(&ba_depth_ptr[uu + vv*image_width],
                &ba_depth_ptr[uu + vv*image_width],
                &ba_covariance_ptr[uu + vv*image_width]);
          }
        }

#endif

        for (unsigned int ii = 0; ii < ui_window_size; ++ii) {
          std::cout << "Final Pose #" << ii << ": " << std::endl << frame_poses[ii].matrix() << std::endl;
        }

        ///-------------------- Update GUI objects.

        // Calculate error if true depth map is provided.
        if (image->Size() > 2) {
          cv::Mat true_depth = image->at(2)->Mat().clone();
          // Convert inverse depth back to depth and float.
          ba_depth_double.convertTo(ba_depth_float, CV_32FC1);
          // Take out NANs.
          cv::Mat maskNanEst = cv::Mat(ba_depth_float == ba_depth_float);
          cv::Mat maskNanTrue = cv::Mat(true_depth == true_depth);
          cv::Mat mask = maskNanEst & maskNanTrue;
          cv::Mat depth_error;
          cv::absdiff(true_depth, ba_depth_float, depth_error);
          const double mean = cv::mean(depth_error, mask).val[0];
          analytics["BA Depth Error"] = mean;
          cv::Mat error_heat_map = GenerateHeatMap(depth_error);
          ba_depth_error_view.SetImage(error_heat_map.data, image_width, image_height,
                                       GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
          cv::Mat covariance;
          ba_covariance.convertTo(covariance, CV_32FC1);
          cv::Mat mini_cov;
          covariance(ROI).copyTo(mini_cov);
//          cv::Mat cov_heat_map = GenerateHeatMap(covariance);
//          ba_covariance_view.SetImage(cov_heat_map.data, image_width, image_height,
//                                       GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
          cv::Mat cov_heat_map = GenerateHeatMap(mini_cov);
          ba_covariance_view.SetImage(cov_heat_map.data, 2*width_margin,
                                      2*height_margin, GL_RGB, GL_RGB,
                                      GL_UNSIGNED_BYTE);


          std::ofstream file4("ba_depth_error", std::ios::out | std::ios::binary);
          file4.write((const char*)depth_error.data, depth_error.rows*depth_error.cols*sizeof(float));
          file4.close();
          std::ofstream file1("covariance", std::ios::out | std::ios::binary);
          file1.write((const char*)covariance.data, covariance.rows*covariance.cols*sizeof(float));
          file1.close();
          std::ofstream file2("covariance_mini", std::ios::out | std::ios::binary);
          file2.write((const char*)mini_cov.data, mini_cov.rows*mini_cov.cols*sizeof(float));
          file2.close();
        }
        analytics_view.Update(analytics);
      }
    }


    /////////////////////////////////////////////////////////////////////////////
    ///---- Render
    timer.Tic("Update GUI");
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    image_view.SetImage(image->at(0)->data(), image_width, image_height,
                        GL_RGB8, GL_LUMINANCE, GL_UNSIGNED_BYTE);

    idtam.GetCostVolumeSlice(slice_buffer, ui_cur_level);
    NormalizeSliceBuffer(slice_buffer, image_width, image_height);
    slice_view.SetImage(slice_buffer, image_width, image_height, GL_RGB8,
                        GL_LUMINANCE, GL_FLOAT);

    depth_view.SetImage(depth_buffer, image_width, image_height, GL_RGB8,
                        GL_LUMINANCE, GL_FLOAT, true);

    ba_depth_view.SetImage(ba_depth_float.data, image_width, image_height,
                           GL_RGB8, GL_LUMINANCE, GL_FLOAT, true);

    // Update VBO.
    {
      Eigen::Vector3f P, u;
      const unsigned char* pImage = image->at(0)->data();
      for (int jj = 0; jj < image_height; ++jj) {
        for (int ii = 0; ii < image_width; ++ii) {
          const int index = ii + jj*image_width;
          u << ii, jj, 1;
          P = Kinv*u;
          if (ui_show_ba_depth) {
            P *= ba_depth_float.at<float>(jj, ii);
          } else {
            P *= depth_buffer[index];
          }

          vbo_buffer[3*index + 0] = P(0);
          vbo_buffer[3*index + 1] = P(1);
          vbo_buffer[3*index + 2] = P(2);

          cbo_buffer[3*index + 0] = pImage[index];
          cbo_buffer[3*index + 1] = pImage[index];
          cbo_buffer[3*index + 2] = pImage[index];
        }
      }
    }
    cbo.Upload(cbo_buffer, cbo_size);
    vbo.Upload(vbo_buffer, vbo_size);

    timer.Toc("Update GUI");

    // Sleep a bit.
    usleep(1e6/60.0);

    // Update timer last.
    timer.Toc();
    timer_view.Update(10, timer.GetNames(3), timer.GetTimes(3));

    pangolin::FinishFrame();
  }

  // Free GUI buffers.
  free(depth_buffer);
  free(slice_buffer);
  free(vbo_buffer);
  free(cbo_buffer);

  return 0;
}
