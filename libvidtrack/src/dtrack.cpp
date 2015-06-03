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

#include <vidtrack/dtrack.h>

#include <glog/logging.h>


DEFINE_bool(discard_saturated, true,
            "Discard under/over saturated pixels during pose estimation.");
DEFINE_double(min_depth, 0.10,
              "Minimum depth to consider for pose estimation.");
DEFINE_double(max_depth, 20.0,
              "Maxmimum depth to consider for pose estimation.");
DEFINE_double(norm_param, 10.0,
              "Tukey norm parameter for robust norm.");
DEFINE_bool(semi_dense, false,
            "Use semi-dense approach for VO rather than full dense.");


#undef VIDTRACK_USE_TBB


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
template<typename T>
inline double interp(
    double                x,            // Input: X coordinate.
    double                y,            // Input: Y coordinate.
    const T*              image_ptr,    // Input: Pointer to image.
    const unsigned int    image_width,  // Input: Image width.
    const unsigned int    image_height  // Input: Image height.
    )
{
  if (!((x >= 0) && (y >= 0) && (x <= image_width-2)
        && (y <= image_height-2))) {
    LOG(FATAL) << "Bad point: " << x << ", " << y;
  }

  x = std::max(std::min(x, static_cast<double>(image_width)-2.0), 2.0);
  y = std::max(std::min(y, static_cast<double>(image_height)-2.0), 2.0);

  const int    px  = static_cast<int>(x);  /* top-left corner */
  const int    py  = static_cast<int>(y);
  const double  ax  = x-px;
  const double  ay  = y-py;
  const double  ax1 = 1.0-ax;
  const double  ay1 = 1.0-ay;

  const T* p0  = image_ptr+(image_width*py)+px;

  double        p1  = p0[0];
  double        p2  = p0[1];
  double        p3  = p0[image_width];
  double        p4  = p0[image_width+1];

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


#ifdef VIDTRACK_USE_TBB
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
class PoseRefine
{
public:
  ///////////////////////////////////////////////////////////////////////////
  PoseRefine(
      const cv::Mat&           live_grey,
      const cv::Mat&           ref_grey,
      const cv::Mat&           ref_depth,
      const cv::Mat&           grad_x_live,
      const cv::Mat&           grad_y_live,
      const cv::Mat&           grad_x_ref,
      const cv::Mat&           grad_y_ref,
      const Eigen::Matrix3d&   Klg,
      const Eigen::Matrix3d&   Krg,
      const Eigen::Matrix3d&   Krd,
      const Eigen::Matrix4d&   Tgd,
      const Eigen::Matrix4d&   Tlr,
      const Eigen::Matrix3x4d& KlgTlr,
      double                   sigma,
      float                    norm_param,
      bool                     discard_saturated,
      float                    min_depth,
      float                    max_depth
      ):
    error(0),
    num_obs(0),
    live_grey_(live_grey),
    ref_grey_(ref_grey),
    ref_depth_(ref_depth),
    grad_x_live_(grad_x_live),
    grad_y_live_(grad_y_live),
    grad_x_ref_(grad_x_ref),
    grad_y_ref_(grad_y_ref),
    Klg_(Klg),
    Krg_(Krg),
    Krd_(Krd),
    Tgd_(Tgd),
    Tlr_(Tlr),
    KlgTlr_(KlgTlr),
    grey_sigma_(sigma),
    norm_param_(norm_param),
    discard_saturated_(discard_saturated),
    min_depth_(min_depth),
    max_depth_(max_depth)
  {
    hessian.setZero();
    LHS.setZero();
    RHS.setZero();
  }

  ///////////////////////////////////////////////////////////////////////////
  PoseRefine(
      const PoseRefine& x,
      tbb::split
      ):
    error(0),
    num_obs(0),
    live_grey_(x.live_grey_),
    ref_grey_(x.ref_grey_),
    ref_depth_(x.ref_depth_),
    grad_x_live_(x.grad_x_live_),
    grad_y_live_(x.grad_y_live_),
    grad_x_ref_(x.grad_x_ref_),
    grad_y_ref_(x.grad_x_ref_),
    Klg_(x.Klg_),
    Krg_(x.Krg_),
    Krd_(x.Krd_),
    Tgd_(x.Tgd_),
    Tlr_(x.Tlr_),
    KlgTlr_(x.KlgTlr_),
    grey_sigma_(x.grey_sigma_),
    norm_param_(x.norm_param_),
    discard_saturated_(x.discard_saturated_),
    min_depth_(x.min_depth_),
    max_depth_(x.max_depth_)
  {
    hessian.setZero();
    LHS.setZero();
    RHS.setZero();
  }

  ///////////////////////////////////////////////////////////////////////////
  void operator()(const tbb::blocked_range<size_t>& r)
  {
    // Local pointer for optimization apparently.
    for (size_t ii = r.begin(); ii != r.end(); ++ii) {
      const unsigned int u = ii%ref_depth_.cols;
      const unsigned int v = ii/ref_depth_.cols;

      // 2d point in reference depth camera.
      Eigen::Vector2d pr_d;
      pr_d << u, v;

      // Get depth.
      const double depth = ref_depth_.at<float>(v, u);

      // Check if depth is NAN.
      if (depth != depth) {
        continue;
      }

      if (depth < min_depth_ || depth > max_depth_) {
        continue;
      }

      // 3d point in reference depth camera.
      Eigen::Vector4d hPr_d;
      hPr_d(0) = depth * (pr_d(0)-Krd_(0,2))/Krd_(0,0);
      hPr_d(1) = depth * (pr_d(1)-Krd_(1,2))/Krd_(1,1);
      hPr_d(2) = depth;
      hPr_d(3) = 1;

      // 3d point in reference grey camera (homogenized).
      // If depth and grey cameras are aligned, Tgd_ = I4.
      const Eigen::Vector4d hPr_g = Tgd_ * hPr_d;

      // Project to reference grey camera's image coordinate.
      Eigen::Vector2d pr_g;
      pr_g(0) = (hPr_g(0)*Krg_(0,0)/hPr_g(2)) + Krg_(0,2);
      pr_g(1) = (hPr_g(1)*Krg_(1,1)/hPr_g(2)) + Krg_(1,2);

      // Check if point is out of bounds.
      if ((pr_g(0) < 2) || (pr_g(0) >= ref_grey_.cols-3) || (pr_g(1) < 2)
          || (pr_g(1) >= ref_grey_.rows-3)) {
        continue;
      }

      // Homogenized 3d point in live grey camera.
      const Eigen::Vector4d hPl_g = Tlr_ * hPr_g;

      // Project to live grey camera's image coordinate.
      Eigen::Vector2d pl_g;
      pl_g(0) = (hPl_g(0)*Klg_(0,0)/hPl_g(2)) + Klg_(0,2);
      pl_g(1) = (hPl_g(1)*Klg_(1,1)/hPl_g(2)) + Klg_(1,2);

      // Check if point is out of bounds.
      if ((pl_g(0) < 2) || (pl_g(0) >= live_grey_.cols-3) || (pl_g(1) < 2)
          || (pl_g(1) >= live_grey_.rows-3)) {
        continue;
      }

      // Get intensities.
      const double Il =
          interp<unsigned char>(pl_g(0), pl_g(1), live_grey_.data,
                 live_grey_.cols, live_grey_.rows);
      const double Ir =
          interp<unsigned char>(pr_g(0), pr_g(1), ref_grey_.data,
                 ref_grey_.cols, ref_grey_.rows);

      // Discard under/over-saturated pixels.
      if (discard_saturated_) {
        if (Il == 0.0 || Il == 255.0 || Ir == 0.0 || Ir == 255.0) {
          continue;
        }
      }

      // Calculate photometric error.
      const double y = Il-Ir;

      ///-------------------- Forward Compositional
      // Image derivative.
      const double Il_xr =
          interp<unsigned char>(pl_g(0)+0.5, pl_g(1), live_grey_.data,
                 live_grey_.cols, live_grey_.rows);
      const double Il_xl =
          interp<unsigned char>(pl_g(0)-0.5, pl_g(1), live_grey_.data,
                 live_grey_.cols, live_grey_.rows);
      const double Il_yu =
          interp<unsigned char>(pl_g(0), pl_g(1)-0.5, live_grey_.data,
                 live_grey_.cols, live_grey_.rows);
      const double Il_yd =
          interp<unsigned char>(pl_g(0), pl_g(1)+0.5, live_grey_.data,
                 live_grey_.cols, live_grey_.rows);

      Eigen::Matrix<double, 1, 2> dIl;
//      dIl << (Il_xr-Il_xl)/2.0, (Il_yd-Il_yu)/2.0;
//      dIl << ((Il_xr-Il)+(Il-Il_xl))/2.0, ((Il_yd-Il)+(Il-Il_yu))/2.0;
      dIl << ((Il_xr-Il)+(Il-Il_xl))/1.0, ((Il_yd-Il)+(Il-Il_yu))/1.0;
      if (u == 10 && v == 10) {
        std::cout << "hPrg: " << hPr_g.transpose() << std::endl;
        std::cout << "Tlr: " << std::endl << Tlr_.matrix() << std::endl;
        std::cout << "hPlg: " << hPl_g.transpose() << std::endl;
        std::cout << "pl: " << pl_g.transpose() << std::endl;
        std::cout << "dIl: " << dIl << std::endl;
      }
      dIl(0) = interp<float>(pl_g(0), pl_g(1),
                             reinterpret_cast<float*>(grad_x_live_.data),
                             grad_x_live_.cols, grad_x_live_.rows);
      dIl(1) = interp<float>(pl_g(0), pl_g(1),
                             reinterpret_cast<float*>(grad_y_live_.data),
                             grad_y_live_.cols, grad_y_live_.rows);
      if (u == 10 && v == 10) {
        std::cout << "dIl: " << dIl << std::endl;
      }


      ///-------------------- Inverse Compositional
      // Image derivative.
      const double Ir_xr =
          interp<unsigned char>(pr_g(0)+0.5, pr_g(1), ref_grey_.data,
                 ref_grey_.cols, ref_grey_.rows);
      const double Ir_xl =
          interp<unsigned char>(pr_g(0)-0.5, pr_g(1), ref_grey_.data,
                 ref_grey_.cols, ref_grey_.rows);
      const double Ir_yu =
          interp<unsigned char>(pr_g(0), pr_g(1)-0.5, ref_grey_.data,
                 ref_grey_.cols, ref_grey_.rows);
      const double Ir_yd =
          interp<unsigned char>(pr_g(0), pr_g(1)+0.5, ref_grey_.data,
                 ref_grey_.cols, ref_grey_.rows);

      Eigen::Matrix<double, 1, 2> dIr;
//      dIr << (Ir_xr-Ir_xl)/2.0, (Ir_yd-Ir_yu)/2.0;
//      dIr << ((Ir_xr-Ir)+(Ir-Ir_xl))/2.0, ((Ir_yd-Ir)+(Ir-Ir_yu))/2.0;
      dIr << ((Ir_xr-Ir)+(Ir-Ir_xl))/1.0, ((Ir_yd-Ir)+(Ir-Ir_yu))/1.0;
      if (u == 10 && v == 10) {
        std::cout << "pr: " << pr_g.transpose() << std::endl;
        std::cout << "dIr: " << dIr << std::endl;
      }
      dIr(0) = interp<float>(pr_g(0), pr_g(1),
                             reinterpret_cast<float*>(grad_x_ref_.data),
                             grad_x_ref_.cols, grad_x_ref_.rows);
      dIr(1) = interp<float>(pr_g(0), pr_g(1),
                             reinterpret_cast<float*>(grad_y_ref_.data),
                             grad_y_ref_.cols, grad_y_ref_.rows);
      if (u == 10 && v == 10) {
        std::cout << "dIr: " << dIr << std::endl;
      }

      // Projection & dehomogenization derivative.
      Eigen::Vector3d KlPl = Klg_ * hPl_g.head(3);

      Eigen::Matrix<double, 2, 3> dPl;
      dPl << 1.0/KlPl(2), 0, -KlPl(0)/(KlPl(2)*KlPl(2)),
          0, 1.0/KlPl(2), -KlPl(1)/(KlPl(2)*KlPl(2));

      const Eigen::Matrix<double, 1, 4> dIesm_dPl_KlgTlr
                                              = ((dIl+dIr)/2.0)*dPl*KlgTlr_;

      // J = dIesm_dPl_KlgTlr * gen_i * Pr
      Eigen::Matrix<double, 1, 6> J;
      J << dIesm_dPl_KlgTlr(0),
           dIesm_dPl_KlgTlr(1),
           dIesm_dPl_KlgTlr(2),
          -dIesm_dPl_KlgTlr(1)*hPr_g(2) + dIesm_dPl_KlgTlr(2)*hPr_g(1),
          +dIesm_dPl_KlgTlr(0)*hPr_g(2) - dIesm_dPl_KlgTlr(2)*hPr_g(0),
          -dIesm_dPl_KlgTlr(0)*hPr_g(1) + dIesm_dPl_KlgTlr(1)*hPr_g(0);


      ///-------------------- Depth Derivative
      // Homogenization derivative.
      Eigen::Matrix<double, 4, 3> dPinv4;
      dPinv4 << 1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                0, 0, 0;

      // Homogenized depth pixel.
      Eigen::Vector3d hpr_d;
      hpr_d << pr_d(0), pr_d(1), 1;

      // Depth derivative on live image.
      // Jdl = dIl * dPl * Kl * Tlr * Tgd * dPinv * Kdinv * pr_d
      const double Jdl = dIl * dPl * KlgTlr_ * Tgd_ * dPinv4
                             * Krd_.inverse() * hpr_d;

      // Depth derivative on reference image.
      // Projection & dehomogenization derivative.
      Eigen::Vector3d KrPr = Krg_ * hPr_g.head(3);

      Eigen::Matrix<double, 2, 3> dPr;
      dPr << 1.0/KrPr(2), 0, -KrPr(0)/(KrPr(2)*KrPr(2)),
          0, 1.0/KrPr(2), -KrPr(1)/(KrPr(2)*KrPr(2));

      // Jdr = dIr * dPr * Kr * Tgd * dPinv * Kdinv * pr_d
      const double Jdr = dIr * dPr * Krg_ * Tgd_.block<3,4>(0,0) * dPinv4
                                    * Krd_.inverse() * hpr_d;


      if (dIl(0) != 0 && dIl(1) != 0 && u == 10 && v == 10 && false) {
        std::cout << "----------------------------" << std::endl;
        std::cout << "Jd-a: " << Jdl - Jdr << std::endl;
        Eigen::Vector2d tmp = dPl * KlgTlr_ * Tgd_ * dPinv4
                                     * Krd_.inverse() * hpr_d;
//        std::cout << "Jd-a Pix: " << tmp.transpose() << std::endl;

        double epsilon = 1e-6;

        // Get depth.
        const double depthf = depth + epsilon;
        const double depthb = depth - epsilon;

        // 3d point in reference depth camera.
        Eigen::Vector4d hPr_df;
        hPr_df(0) = depthf * (pr_d(0)-Krd_(0,2))/Krd_(0,0);
        hPr_df(1) = depthf * (pr_d(1)-Krd_(1,2))/Krd_(1,1);
        hPr_df(2) = depthf;
        hPr_df(3) = 1;
        Eigen::Vector4d hPr_db;
        hPr_db(0) = depthb * (pr_d(0)-Krd_(0,2))/Krd_(0,0);
        hPr_db(1) = depthb * (pr_d(1)-Krd_(1,2))/Krd_(1,1);
        hPr_db(2) = depthb;
        hPr_db(3) = 1;

        // 3d point in reference grey camera (homogenized).
        // If depth and grey cameras are aligned, Tgd_ = I4.
        const Eigen::Vector4d hPr_gf = Tgd_ * hPr_df;
        const Eigen::Vector4d hPr_gb = Tgd_ * hPr_db;

        // Project to reference grey camera's image coordinate.
        Eigen::Vector2d pr_gf;
        pr_gf(0) = (hPr_gf(0)*Krg_(0,0)/hPr_gf(2)) + Krg_(0,2);
        pr_gf(1) = (hPr_gf(1)*Krg_(1,1)/hPr_gf(2)) + Krg_(1,2);
        Eigen::Vector2d pr_gb;
        pr_gb(0) = (hPr_gb(0)*Krg_(0,0)/hPr_gb(2)) + Krg_(0,2);
        pr_gb(1) = (hPr_gb(1)*Krg_(1,1)/hPr_gb(2)) + Krg_(1,2);

        // Homogenized 3d point in live grey camera.
        const Eigen::Vector4d hPl_gf = Tlr_ * hPr_gf;
        const Eigen::Vector4d hPl_gb = Tlr_ * hPr_gb;

        // Project to live grey camera's image coordinate.
        Eigen::Vector2d pl_gf;
        pl_gf(0) = (hPl_gf(0)*Klg_(0,0)/hPl_gf(2)) + Klg_(0,2);
        pl_gf(1) = (hPl_gf(1)*Klg_(1,1)/hPl_gf(2)) + Klg_(1,2);
        Eigen::Vector2d pl_gb;
        pl_gb(0) = (hPl_gb(0)*Klg_(0,0)/hPl_gb(2)) + Klg_(0,2);
        pl_gb(1) = (hPl_gb(1)*Klg_(1,1)/hPl_gb(2)) + Klg_(1,2);

        // Get intensities.
        const double Ilf =
            interp(pl_gf(0), pl_gf(1), live_grey_.data,
                   live_grey_.cols, live_grey_.rows);
        const double Irf =
            interp(pr_gf(0), pr_gf(1), ref_grey_.data,
                   ref_grey_.cols, ref_grey_.rows);
        const double Ilb =
            interp(pl_gb(0), pl_gb(1), live_grey_.data,
                   live_grey_.cols, live_grey_.rows);
        const double Irb =
            interp(pr_gb(0), pr_gb(1), ref_grey_.data,
                   ref_grey_.cols, ref_grey_.rows);


        std::cout << "Jd-fd: " << ((Ilf-Irf) - (Ilb-Irb))/(depthf-depthb) << std::endl;
//        std::cout << "Jd-fd Pix: " << ((pl_gf-pl_gb)/(depthf-depthb)).transpose() << std::endl;
      }


      // Final depth Jacobian: Jd = Jdl - Jdr
      double Jd = Jdl - Jdr;
      if (Jd == 0) {
        Jd = FLT_MIN;
      }


      ///-------------------- Robust Norm
      const double w = _NormTukey(y, norm_param_);

      // Uncertainties.
      const double depth_sigma = depth/20.0;

      // Error prop: NewSigma = J * Sigma * J_transpose
      const double depth_unc = Jd * (depth_sigma*depth_sigma) * Jd;

      // Try gradient as uncertainty. Makes more sense for ELAS.
      // Do finite differences on edge pixel to test all the way.
      const double inv_sigma = 1.0/((grey_sigma_*grey_sigma_)+depth_unc);

      hessian     += J.transpose() * w * inv_sigma * J;
      LHS         += J.transpose() * w * inv_sigma * J;
      RHS         += J.transpose() * w * inv_sigma * y;
      error       += y * y;
      num_obs++;
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  void join(const PoseRefine& other)
  {
    LHS         += other.LHS;
    RHS         += other.RHS;
    hessian     += other.hessian;
    error       += other.error;
    num_obs     += other.num_obs;
  }

private:
  ///////////////////////////////////////////////////////////////////////////
  inline double _NormTukey(double r,
                           double c)
  {
    const double roc    = r/c;
    const double omroc2 = 1.0f-roc*roc;

    return (fabs(r) <= c) ? omroc2*omroc2 : 0.0f;
  }


  ///
  ///////////////////////////////////////////////////////////////////////////

public:
  Eigen::Matrix6d   LHS;
  Eigen::Vector6d   RHS;
  Eigen::Matrix6d   hessian;
  double            error;
  unsigned int      num_obs;

private:
  cv::Mat           live_grey_;
  cv::Mat           ref_grey_;
  cv::Mat           ref_depth_;
  cv::Mat           grad_x_live_;
  cv::Mat           grad_y_live_;
  cv::Mat           grad_x_ref_;
  cv::Mat           grad_y_ref_;
  Eigen::Matrix3d   Klg_;
  Eigen::Matrix3d   Krg_;
  Eigen::Matrix3d   Krd_;
  Eigen::Matrix4d   Tgd_;
  Eigen::Matrix4d   Tlr_;
  Eigen::Matrix3x4d KlgTlr_;
  double            grey_sigma_;
  float             norm_param_;
  bool              discard_saturated_;
  float             min_depth_;
  float             max_depth_;
};
#endif



/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
DTrack::DTrack(unsigned int pyramid_levels) :
  kPyramidLevels(pyramid_levels)
#ifdef VIDTRACK_USE_CUDA
  , cu_dtrack_(nullptr)
#endif
#ifdef VIDTRACK_USE_TBB
  , tbb_scheduler_(tbb::task_scheduler_init::automatic)
#endif
{
#ifdef VIDTRACK_USE_TBB
  tbb_scheduler_.initialize();
#endif
}

///////////////////////////////////////////////////////////////////////////
DTrack::~DTrack()
{
#ifdef VIDTRACK_USE_CUDA
  if (cu_dtrack_ != nullptr) {
    free(cu_dtrack_);
  }
#endif
#ifdef VIDTRACK_USE_TBB
  tbb_scheduler_.terminate();
#endif
}

///////////////////////////////////////////////////////////////////////////
void DTrack::SetParams(
    const Eigen::Matrix3d&    live_grey_cmod,
    const Eigen::Matrix3d&    ref_grey_cmod,
    const Eigen::Matrix3d&    ref_depth_cmod,
    const Sophus::SE3d&       Tgd
    )
{
  // Store scaled camera models (to avoid recomputing).
  for (size_t ii = 0; ii < kPyramidLevels; ++ii) {
    live_grey_cam_model_.push_back(_ScaleCM(live_grey_cmod, ii));
    ref_grey_cam_model_.push_back(_ScaleCM(ref_grey_cmod, ii));
    ref_depth_cam_model_.push_back(_ScaleCM(ref_depth_cmod, ii));
  }

  // Copy reference camera's depth-grey transform.
  Tgd_ = Tgd;
  LOG(INFO) << "Tgd is: " << Tgd.log().transpose() << std::endl;

#ifdef VIDTRACK_USE_CUDA
  // Initialize cuDTrack if used.
  if (cu_dtrack_ != nullptr) {
    cu_dtrack_ = new cuDTrack(live_grey_cmod.Height(), live_grey_cmod.Width());
  }
#endif
}

///////////////////////////////////////////////////////////////////////////
void DTrack::SetKeyframe(
    const cv::Mat& ref_grey,  // Input: Reference image (float format, normalized).
    const cv::Mat& ref_depth  // Input: Reference depth (float format, meters).
    )
{
  // Build pyramids.
  cv::buildPyramid(ref_grey, ref_grey_pyramid_, kPyramidLevels);
  cv::buildPyramid(ref_depth, ref_depth_pyramid_, kPyramidLevels);

  // If semi-dense is used, run edge detector over pyramid.
  if (FLAGS_semi_dense) {
    ref_grey_edges_.clear();
    for (size_t pyramid_lvl = 0; pyramid_lvl < kPyramidLevels; ++pyramid_lvl) {
      cv::Mat detected_edges;
      cv::blur(ref_grey_pyramid_[pyramid_lvl], detected_edges, cv::Size(3,3));

      // Canny detector.
      const double kernel_size = 3;
      const double canny_threshold = 30;
      cv::Canny(detected_edges, detected_edges, canny_threshold,
                canny_threshold*3, kernel_size);

      // Store edge image. Contains 255 if edge, 0 otherwise.
      ref_grey_edges_.push_back(detected_edges);
    }
  }

}

#define DECIMATE 0

void DTrack::ComputeGradient(uint pyramid_lvl) {
  const cv::Mat& live_grey_img = live_grey_pyramid_[pyramid_lvl];
  const cv::Mat& ref_grey_img  = ref_grey_pyramid_[pyramid_lvl];

  // Pre-calculate gradients so we don't do it each iteration.
  gradient_x_live_.create(live_grey_img.rows, live_grey_img.cols, CV_32FC1);
  gradient_y_live_.create(live_grey_img.rows, live_grey_img.cols, CV_32FC1);
  gradient_x_ref_.create(live_grey_img.rows, live_grey_img.cols, CV_32FC1);
  gradient_y_ref_.create(live_grey_img.rows, live_grey_img.cols, CV_32FC1);
  _CalculateGradients(
        live_grey_img.data, live_grey_img.cols, live_grey_img.rows,
        reinterpret_cast<float*>(gradient_x_live_.data),
        reinterpret_cast<float*>(gradient_y_live_.data));
  _CalculateGradients(
        ref_grey_img.data, ref_grey_img.cols, ref_grey_img.rows,
        reinterpret_cast<float*>(gradient_x_ref_.data),
        reinterpret_cast<float*>(gradient_y_ref_.data));
}

void DTrack::BuildProblem(
    const Sophus::SE3d& Tlr,
    Eigen::Matrix6d&    LHS,
    Eigen::Vector6d&    RHS,
    double&             squared_error,
    double&             number_observations,
    uint                pyramid_lvl
    ) {
  // Options.
  const bool   discard_saturated = FLAGS_discard_saturated;
  const float  min_depth         = FLAGS_min_depth;
  const float  max_depth         = FLAGS_max_depth;
  const double norm_c            = FLAGS_norm_param;

  // Set pyramid norm parameter.
  const double norm_c_pyr = norm_c * (pyramid_lvl + 1);

  const cv::Mat& live_grey_img = live_grey_pyramid_[pyramid_lvl];
  const cv::Mat& ref_grey_img  = ref_grey_pyramid_[pyramid_lvl];
  const cv::Mat& ref_depth_img = ref_depth_pyramid_[pyramid_lvl];

  const Eigen::Matrix3d& Klg = ref_grey_cam_model_[pyramid_lvl];
  const Eigen::Matrix3d& Krg = ref_grey_cam_model_[pyramid_lvl];
  const Eigen::Matrix3d& Krd = ref_depth_cam_model_[pyramid_lvl];

  CHECK_EQ(gradient_x_live_.rows, live_grey_img.rows);

  // Inverse transform.
  const Eigen::Matrix3x4d KlgTlr = options_.optimize_wrt_depth_camera ?
        Klg * (Tgd_ * Tlr).matrix3x4() :
        Klg * Tlr.matrix3x4();

  for (int vv = 0; vv < ref_depth_img.rows; ++vv) {
    for (int uu = 0; uu < ref_depth_img.cols; ++uu) {

      // 2d point in reference depth camera.
      Eigen::Vector2d pr_d;
      pr_d << uu, vv;

      // Get depth.
      const double depth = ref_depth_img.at<float>(vv, uu);

      // Check if depth is NAN.
      if (depth != depth) {
        continue;
      }

      if (depth < min_depth || depth > max_depth) {
        continue;
      }

      // 3d point in reference depth camera.
      Eigen::Vector4d hPr_d;
      hPr_d(0) = depth * (pr_d(0)-Krd(0,2))/Krd(0,0);
      hPr_d(1) = depth * (pr_d(1)-Krd(1,2))/Krd(1,1);
      hPr_d(2) = depth;
      hPr_d(3) = 1;

      // 3d point in reference grey camera (homogenized).
      // If depth and grey cameras are aligned, Tgd_ = I4.
      const Eigen::Vector4d hPr_g = Tgd_.matrix() * hPr_d;

      // Project to reference grey camera's image coordinate.
      Eigen::Vector2d pr_g;
      pr_g(0) = (hPr_g(0)*Krg(0,0)/hPr_g(2)) + Krg(0,2);
      pr_g(1) = (hPr_g(1)*Krg(1,1)/hPr_g(2)) + Krg(1,2);

      // Check if point is out of bounds.
      if (pr_g(0) < 2 || pr_g(0) >= ref_grey_img.cols-3
         || pr_g(1) < 2 || pr_g(1) >= ref_grey_img.rows-3) {
        continue;
      }

      // For semi-dense: Check if point is not an edge.
      if (FLAGS_semi_dense) {
        const double edge =
            interp<unsigned char>(pr_g(0), pr_g(1),
                                  ref_grey_edges_[pyramid_lvl].data,
                                  ref_grey_edges_[pyramid_lvl].cols,
                                  ref_grey_edges_[pyramid_lvl].rows);
        if (edge == 0) {
          continue;
        }
      }

      // Homogenized 3d point in live grey camera.
      const Eigen::Vector4d hPl_g = Tlr.matrix() * hPr_g;

      // Project to live grey camera's image coordinate.
      Eigen::Vector2d pl_g;
      pl_g(0) = (hPl_g(0)*Klg(0,0)/hPl_g(2)) + Klg(0,2);
      pl_g(1) = (hPl_g(1)*Klg(1,1)/hPl_g(2)) + Klg(1,2);

      // Check if point is out of bounds.
      if (pl_g(0) < 2 || pl_g(0) >= live_grey_img.cols-3
         || pl_g(1) < 2 || pl_g(1) >= live_grey_img.rows-3) {
        continue;
      }

      // Get intensities.
      const double Il =
          interp<unsigned char>(pl_g(0), pl_g(1), live_grey_img.data,
                                live_grey_img.cols, live_grey_img.rows);
      const double Ir =
          interp<unsigned char>(pr_g(0), pr_g(1), ref_grey_img.data,
                                ref_grey_img.cols, ref_grey_img.rows);

      // Discard under/over-saturated pixels.
      if (discard_saturated) {
        if (Il == 0.0 || Il == 255.0 || Ir == 0.0 || Ir == 255.0) {
          continue;
        }
      }

      // Calculate error.
      const double y = Il-Ir;


      ///-------------------- Forward Compositional
      // Image derivative.
      Eigen::Matrix<double, 1, 2> dIl;
      dIl(0) = interp<float>(pl_g(0), pl_g(1),
                             reinterpret_cast<float*>(gradient_x_live_.data),
                             gradient_x_live_.cols, gradient_x_live_.rows);
      dIl(1) = interp<float>(pl_g(0), pl_g(1),
                             reinterpret_cast<float*>(gradient_y_live_.data),
                             gradient_y_live_.cols, gradient_y_live_.rows);


      ///-------------------- Inverse Compositional
      // Image derivative.
      Eigen::Matrix<double, 1, 2> dIr;
      dIr(0) = interp<float>(pr_g(0), pr_g(1),
                             reinterpret_cast<float*>(gradient_x_ref_.data),
                             gradient_x_ref_.cols, gradient_x_ref_.rows);
      dIr(1) = interp<float>(pr_g(0), pr_g(1),
                             reinterpret_cast<float*>(gradient_y_ref_.data),
                             gradient_y_ref_.cols, gradient_y_ref_.rows);


      // Projection & dehomogenization derivative.
      Eigen::Vector3d KlPl = Klg * hPl_g.head(3);

      Eigen::Matrix2x3d dPl;
      dPl  << 1.0/KlPl(2), 0, -KlPl(0)/(KlPl(2)*KlPl(2)),
          0, 1.0/KlPl(2), -KlPl(1)/(KlPl(2)*KlPl(2));

      const Eigen::Vector4d dIesm_dPl_KlgTlr = ((dIl+dIr)/2.0)*dPl*KlgTlr;

      // J = dIesm_dPl_KlgTlr * gen_i * Pr
      Eigen::Matrix<double, 1, 6> J;
      if (options_.optimize_wrt_depth_camera) {
        J << dIesm_dPl_KlgTlr(0),
             dIesm_dPl_KlgTlr(1),
             dIesm_dPl_KlgTlr(2),
            -dIesm_dPl_KlgTlr(1)*hPr_d(2) + dIesm_dPl_KlgTlr(2)*hPr_d(1),
            +dIesm_dPl_KlgTlr(0)*hPr_d(2) - dIesm_dPl_KlgTlr(2)*hPr_d(0),
            -dIesm_dPl_KlgTlr(0)*hPr_d(1) + dIesm_dPl_KlgTlr(1)*hPr_d(0);
      } else {
        J << dIesm_dPl_KlgTlr(0),
             dIesm_dPl_KlgTlr(1),
             dIesm_dPl_KlgTlr(2),
            -dIesm_dPl_KlgTlr(1)*hPr_g(2) + dIesm_dPl_KlgTlr(2)*hPr_g(1),
            +dIesm_dPl_KlgTlr(0)*hPr_g(2) - dIesm_dPl_KlgTlr(2)*hPr_g(0),
            -dIesm_dPl_KlgTlr(0)*hPr_g(1) + dIesm_dPl_KlgTlr(1)*hPr_g(0);
      }


      ///-------------------- Depth Derivative
      // Homogenization derivative.
      Eigen::Matrix<double, 4, 3> dPinv4;
      dPinv4 << 1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                0, 0, 0;

      // Homogenized depth pixel.
      Eigen::Vector3d hpr_d;
      hpr_d << pr_d(0), pr_d(1), 1;

      // Depth derivative on live image.
      // Jdl = dIl * dPl * Kl * Tlr * Tgd * dPinv * Kdinv * pr_d
      const double Jdl = dIl * dPl * KlgTlr * Tgd_.matrix() * dPinv4
                             * Krd.inverse() * hpr_d;

      // Depth derivative on reference image.
      // Projection & dehomogenization derivative.
      Eigen::Vector3d KrPr = Krg * hPr_g.head(3);

      Eigen::Matrix<double, 2, 3> dPr;
      dPr << 1.0/KrPr(2), 0, -KrPr(0)/(KrPr(2)*KrPr(2)),
          0, 1.0/KrPr(2), -KrPr(1)/(KrPr(2)*KrPr(2));

      // Jdr = dIr * dPr * Kr * Tgd * dPinv * Kdinv * pr_d
      const double Jdr = dIr * dPr * Krg * Tgd_.matrix3x4() * dPinv4
                         * Krd.inverse() * hpr_d;


      if (dIl(0) != 0 && dIl(1) != 0 && uu == 10 && vv == 10 && false) {
        std::cout << "----------------------------" << std::endl;
        std::cout << "Jd-a: " << Jdl - Jdr << std::endl;
        Eigen::Vector2d tmp = dPl * KlgTlr * Tgd_.matrix() * dPinv4
                              * Krd.inverse() * hpr_d;
//        std::cout << "Jd-a Pix: " << tmp.transpose() << std::endl;

        double epsilon = 1e-6;

        // Get depth.
        const double depthf = depth + epsilon;
        const double depthb = depth - epsilon;

        // 3d point in reference depth camera.
        Eigen::Vector4d hPr_df;
        hPr_df(0) = depthf * (pr_d(0)-Krd(0,2))/Krd(0,0);
        hPr_df(1) = depthf * (pr_d(1)-Krd(1,2))/Krd(1,1);
        hPr_df(2) = depthf;
        hPr_df(3) = 1;
        Eigen::Vector4d hPr_db;
        hPr_db(0) = depthb * (pr_d(0)-Krd(0,2))/Krd(0,0);
        hPr_db(1) = depthb * (pr_d(1)-Krd(1,2))/Krd(1,1);
        hPr_db(2) = depthb;
        hPr_db(3) = 1;

        // 3d point in reference grey camera (homogenized).
        // If depth and grey cameras are aligned, Tgd_ = I4.
        const Eigen::Vector4d hPr_gf = Tgd_.matrix() * hPr_df;
        const Eigen::Vector4d hPr_gb = Tgd_.matrix() * hPr_db;

        // Project to reference grey camera's image coordinate.
        Eigen::Vector2d pr_gf;
        pr_gf(0) = (hPr_gf(0)*Krg(0,0)/hPr_gf(2)) + Krg(0,2);
        pr_gf(1) = (hPr_gf(1)*Krg(1,1)/hPr_gf(2)) + Krg(1,2);
        Eigen::Vector2d pr_gb;
        pr_gb(0) = (hPr_gb(0)*Krg(0,0)/hPr_gb(2)) + Krg(0,2);
        pr_gb(1) = (hPr_gb(1)*Krg(1,1)/hPr_gb(2)) + Krg(1,2);

        // Homogenized 3d point in live grey camera.
        const Eigen::Vector4d hPl_gf = Tlr.matrix() * hPr_gf;
        const Eigen::Vector4d hPl_gb = Tlr.matrix() * hPr_gb;

        // Project to live grey camera's image coordinate.
        Eigen::Vector2d pl_gf;
        pl_gf(0) = (hPl_gf(0)*Klg(0,0)/hPl_gf(2)) + Klg(0,2);
        pl_gf(1) = (hPl_gf(1)*Klg(1,1)/hPl_gf(2)) + Klg(1,2);
        Eigen::Vector2d pl_gb;
        pl_gb(0) = (hPl_gb(0)*Klg(0,0)/hPl_gb(2)) + Klg(0,2);
        pl_gb(1) = (hPl_gb(1)*Klg(1,1)/hPl_gb(2)) + Klg(1,2);

        // Get intensities.
        const double Ilf =
            interp(pl_gf(0), pl_gf(1), live_grey_img.data,
                   live_grey_img.cols, live_grey_img.rows);
        const double Irf =
            interp(pr_gf(0), pr_gf(1), ref_grey_img.data,
                   ref_grey_img.cols, ref_grey_img.rows);
        const double Ilb =
            interp(pl_gb(0), pl_gb(1), live_grey_img.data,
                   live_grey_img.cols, live_grey_img.rows);
        const double Irb =
            interp(pr_gb(0), pr_gb(1), ref_grey_img.data,
                   ref_grey_img.cols, ref_grey_img.rows);


        std::cout << "Jd-fd: " << ((Ilf-Irf) - (Ilb-Irb))/(depthf-depthb) << std::endl;
//        std::cout << "Jd-fd Pix: " << ((pl_gf-pl_gb)/(depthf-depthb)).transpose() << std::endl;
      }


      // Final depth Jacobian: Jd = Jdl - Jdr
      double Jd = Jdl - Jdr;
      if (Jd == 0) {
        Jd = FLT_MIN;
      }


      ///-------------------- Robust Norm
      const double w = _NormTukey(y, norm_c_pyr);

      // Uncertainties.
//          const double depth_sigma = depth/20.0;
      const double depth_sigma = kDepthSigma;

      // Error prop: NewSigma = J * Sigma * J_transpose
      const double depth_unc = Jd * (depth_sigma*depth_sigma) * Jd;

      // Try gradient as uncertainty. Makes more sense for ELAS.
      // Do finite differences on edge pixel to test all the way.
      const double inv_sigma = 1.0/((kGreySigma*kGreySigma)+depth_unc);
//          const double inv_sigma = 1.0/(kGreySigma*kGreySigma);
//          const double inv_sigma = 1.0;

      LHS           += J.transpose() * w * inv_sigma * J;
      RHS           += J.transpose() * w * inv_sigma * y;
      squared_error += y * y;
      number_observations++;
    }
  }
}

///////////////////////////////////////////////////////////////////////////
double DTrack::Estimate(
    bool                      use_pyramid,
    const cv::Mat&            live_grey,
    Sophus::SE3d&             Trl,
    Eigen::Matrix6d&          covariance,
    unsigned int&             number_obs,
    unsigned int&             number_iters
  )
{
  // Reset output parameters.
  number_obs     = 0;
  number_iters   = 0;
  covariance.setZero();

  // Set pyramid max-iterations and full estimate mask. The pyramid is
  // constructed with the largest image first.

  // 0 is rotation only, 1 is both rotation and translation
  std::vector<bool>         vec_full_estimate  = {1, 1, 1, 0};
#if DECIMATE
  std::vector<unsigned int> vec_max_iterations = {0, 5, 5, 5};
#else
  std::vector<unsigned int> vec_max_iterations = {50, 50, 50, 50};
#endif

  if (use_pyramid == false) {
#if DECIMATE
    vec_max_iterations = {0, 5, 0, 0};
#else
    vec_max_iterations = {5, 0, 0, 0};
#endif
  }

  CHECK_GE(vec_full_estimate.size(), kPyramidLevels);
  CHECK_GE(vec_max_iterations.size(), kPyramidLevels);

  // Build live pyramid.
#if 1
  cv::Mat live_grey_copy = live_grey.clone();
  _BrightnessCorrectionImagePair(live_grey_copy.data,
                                 ref_grey_pyramid_[0].data,
                                 live_grey_copy.cols*live_grey_copy.rows);
  cv::buildPyramid(live_grey_copy, live_grey_pyramid_, kPyramidLevels);
#else
  cv::buildPyramid(live_grey, live_grey_pyramid_, kPyramidLevels);
#endif

  // Aux variables.
  Eigen::Matrix6d   LHS;
  Eigen::Vector6d   RHS;
  double            squared_error;
  double            number_observations;
  double            last_error = FLT_MAX;

  // Iterate through pyramid levels.
  for (int pyramid_lvl = kPyramidLevels-1; pyramid_lvl >= 0; pyramid_lvl--) {
    ComputeGradient(pyramid_lvl);
    // Reset error.
    last_error = FLT_MAX;

    for (unsigned int num_iters = 0;
         num_iters < vec_max_iterations[pyramid_lvl];
         ++num_iters) {
      // Reset.
      LHS.setZero();
      RHS.setZero();

      // Reset error.
      squared_error        = 0;
      number_observations  = 0;

      // Increase number of iterations.
      number_iters++;

      const Sophus::SE3d Tlr = Trl.inverse();

#if defined(VIDTRACK_USE_CUDA)
      cu_dtrack_->Estimate(live_grey_img, ref_grey_img, ref_depth_img, Klg,
                           Krg, Krd, Tgd_.matrix(), Tlr.matrix(), KlgTlr,
                           norm_c_pyr, discard_saturated, min_depth, max_depth);
#elif defined(VIDTRACK_USE_TBB)
      // Launch TBB.
      PoseRefine pose_ref(live_grey_img, ref_grey_img, ref_depth_img,
                          gradient_x_live, gradient_y_live,
                          gradient_x_ref, gradient_y_ref,
                          Klg, Krg, Krd, Tgd_.matrix(),
                          Tlr.matrix(), KlgTlr, kSigma, norm_c_pyr,
                          discard_saturated, min_depth, max_depth);

      tbb::parallel_reduce(tbb::blocked_range<size_t>(0,
                    ref_depth_img.cols*ref_depth_img.rows), pose_ref);

      LHS                  = pose_ref.LHS;
      RHS                  = pose_ref.RHS;
      squared_error        = pose_ref.error;
      number_observations  = pose_ref.num_obs;
      hessian              = pose_ref.hessian;

#else
      // Iterate through depth map.
      BuildProblem(Tlr, LHS, RHS, squared_error, number_observations,
                   pyramid_lvl);
#endif

      Eigen::Matrix6d hessian = LHS;

      // Solution.
      Eigen::Vector6d X;

      // Check if we are solving only for rotation, or full estimate.
      if (vec_full_estimate[pyramid_lvl]) {
        // Decompose matrix.
        Eigen::FullPivLU<Eigen::Matrix<double, 6, 6> > lu_JTJ(LHS);

        // Check degenerate system.
        if (lu_JTJ.rank() < 6) {
          LOG(WARNING) << "[@L:" << pyramid_lvl << " I:"
                       << num_iters << "] LS trashed. Rank deficient!";
        }

        X = -(lu_JTJ.solve(RHS));
      } else {
        // Extract rotation information only.
        Eigen::Matrix3d rLHS = LHS.block<3, 3>(3, 3);
        Eigen::Vector3d rRHS = RHS.tail(3);

        Eigen::FullPivLU<Eigen::Matrix<double, 3, 3> > lu_JTJ(rLHS);

        // Check degenerate system.
        if (lu_JTJ.rank() < 3) {
          LOG(WARNING) << "[@L:" << pyramid_lvl << " I:"
                       << num_iters << "] LS trashed. Rank deficient!";
        }

        Eigen::Vector3d rX;
        rX = -(lu_JTJ.solve(rRHS));

        // Pack solution.
        X.setZero();
        X.tail(3) = rX;
      }

//      std::cout << "-- LHS: " << std::endl << LHS << std::endl;
//      std::cout << "-- RHS: " << RHS.transpose() << std::endl;
//      std::cout << "-- Solved X: " << X.transpose() << std::endl;

      // Get RMSE.
      const double new_error = sqrt(squared_error/number_observations);

      if (new_error < last_error) {
        // Update error.
        last_error = new_error;

        // Update number of observations used in estimation.
        number_obs = number_observations;

        // Set covariance output.
        covariance = hessian.inverse();

        // Update Trl.
        Trl = (Tlr*Sophus::SE3Group<double>::exp(X)).inverse();

        if (X.norm() < 1e-5) {
          VLOG(1) << "[@L:" << pyramid_lvl << " I:"
                  << num_iters << "] Update is too small. Breaking early!";
          break;
        }
      } else {
        VLOG(1) << "[@L:" << pyramid_lvl << " I:"
                << num_iters << "] Error is increasing. Breaking early!";
        break;
      }
    }
  }

  return last_error;
}


///////////////////////////////////////////////////////////////////////////
Eigen::Matrix3d  DTrack::_ScaleCM(
    const Eigen::Matrix3d&    K,
    unsigned int              level
  )
{
  const double scale = 1.0f/(1 << level);

  Eigen::Matrix3d scaled_K = K;

  scaled_K(0,0) *= scale;
  scaled_K(1,1) *= scale;
  scaled_K(0,2) = scale*(scaled_K(0,2)+0.5) - 0.5;
  scaled_K(1,2) = scale*(scaled_K(1,2)+0.5) - 0.5;

  return scaled_K;
}


/////////////////////////////////////////////////////////////////////////////
void DTrack::_CalculateGradients(
    const unsigned char*      image_ptr,
    int                       image_width,
    int                       image_height,
    float*                    gradX_ptr,
    float*                    gradY_ptr
  )
{
#if 0
  for (int vv = 1; vv < image_height-1; ++vv) {
    for (int uu = 1; uu < image_width-1; ++uu) {
      unsigned char pix = image_ptr[uu + vv*image_width];
      unsigned char pix_left = image_ptr[uu-1 + vv*image_width];
      unsigned char pix_right = image_ptr[uu+1 + vv*image_width];
      unsigned char pix_above = image_ptr[uu + (vv-1)*image_width];
      unsigned char pix_below = image_ptr[uu + (vv+1)*image_width];
//      gradX_ptr[uu + vv*image_width] = ((pix_right-pix)+(pix-pix_left))/2.0;
//      gradY_ptr[uu + vv*image_width] = ((pix_below-pix)+(pix-pix_above))/2.0;
      gradX_ptr[uu + vv*image_width] = ((pix_right-pix_left))/2.0;
      gradY_ptr[uu + vv*image_width] = ((pix_below-pix_above))/2.0;
    }
  }
#else
  const int image_width_M1  = image_width - 1;
  const int image_height_M1 = image_height - 1;

  const unsigned char* row_ptr, *bottom_row_ptr, *top_row_ptr;

  row_ptr         = image_ptr;
  bottom_row_ptr  = image_ptr + image_width;
  float* rowX_ptr = gradX_ptr;
  float* rowY_ptr = gradY_ptr;

  // Work on the first row.
  rowX_ptr[0] = row_ptr[1] - row_ptr[0];
  rowY_ptr[0] = bottom_row_ptr[0] - row_ptr[0];
  for (int uu = 1; uu < image_width_M1; ++uu) {
    rowX_ptr[uu] = (row_ptr[uu+1] - row_ptr[uu-1])/2.0;
    rowY_ptr[uu] = bottom_row_ptr[uu] - row_ptr[uu];
  }
  rowX_ptr[image_width_M1] = row_ptr[image_width_M1] - row_ptr[image_width_M1-1];
  rowY_ptr[image_width_M1] = bottom_row_ptr[image_width_M1] - row_ptr[image_width_M1];

  row_ptr        = image_ptr + image_width;
  bottom_row_ptr = image_ptr + 2*image_width;
  top_row_ptr    = image_ptr;
  rowX_ptr       = gradX_ptr + image_width;
  rowY_ptr       = gradY_ptr + image_width;

  // Work from the second to the "last-1" row.
  for (int vv = 1; vv < image_height_M1; ++vv) {
    // First column.
    *rowX_ptr++ = row_ptr[1] - row_ptr[0];
    *rowY_ptr++ = (bottom_row_ptr[0] - top_row_ptr[0])/2.0;

    for (int uu = 1; uu < image_width_M1; ++uu) {
      *rowX_ptr++ = (row_ptr[uu+1] - row_ptr[uu-1])/2.0;
      *rowY_ptr++ = (bottom_row_ptr[uu] - top_row_ptr[uu])/2.0;
    }

    // Last column.
    *rowX_ptr++ = row_ptr[image_width_M1] - row_ptr[image_width_M1-1];
    *rowY_ptr++ = (bottom_row_ptr[image_width_M1] - top_row_ptr[image_width_M1])/2.0;

    // Move to next rows.
    row_ptr        += image_width;
    bottom_row_ptr += image_width;
    top_row_ptr    += image_width;
  }

  // Last row.
  top_row_ptr  = image_ptr + (image_height_M1 - 1) * image_width;
  row_ptr     = image_ptr + image_height_M1 * image_width;
  rowX_ptr    = gradX_ptr + image_height_M1 * image_width;
  rowY_ptr    = gradY_ptr + image_height_M1 * image_width;
  rowX_ptr[0] = row_ptr[1] - row_ptr[0];
  rowY_ptr[0] = row_ptr[0] - top_row_ptr[0];

  for (int uu = 1; uu < image_width_M1; ++uu) {
    rowX_ptr[uu] = (row_ptr[uu+1] - row_ptr[uu-1])/2.0;
    rowY_ptr[uu] = row_ptr[uu] - top_row_ptr[uu];
  }
  rowX_ptr[image_width_M1] = row_ptr[image_width_M1] - row_ptr[image_width_M1-1];
  rowY_ptr[image_width_M1] = row_ptr[image_width_M1] - top_row_ptr[image_width_M1];
#endif
}


///////////////////////////////////////////////////////////////////////////
inline double DTrack::_NormTukey(double r,
                                 double c)
{
  const double roc    = r/c;
  const double omroc2 = 1.0f-roc*roc;

  return (fabs(r) <= c) ? omroc2*omroc2 : 0.0f;
}


///////////////////////////////////////////////////////////////////////////
void DTrack::_BrightnessCorrectionImagePair(
    unsigned char*          img1_ptr,
    unsigned char*          img2_ptr,
    size_t                  image_size
  )
{
  // Save original ptr.
  unsigned char* img1_ptr_orig = img1_ptr;

  // Sampling variables.
  const size_t     sample_step = 1;
  size_t           num_samples = 0;

  // Compute mean.
  float mean1      = 0.0;
  float mean2      = 0.0;
  float mean1_sqrd = 0.0;
  float mean2_sqrd = 0.0;

  for (size_t ii = 0; ii < image_size;
       ii += sample_step, img1_ptr += sample_step, img2_ptr += sample_step) {
    mean1       += (*img1_ptr);
    mean1_sqrd  += (*img1_ptr) * (*img1_ptr);
    mean2       += (*img2_ptr);
    mean2_sqrd  += (*img2_ptr) * (*img2_ptr);
    num_samples++;
  }

  mean1       /= num_samples;
  mean2       /= num_samples;
  mean1_sqrd  /= num_samples;
  mean2_sqrd  /= num_samples;

  // Compute STD.
  float std1 = sqrt(mean1_sqrd - mean1*mean1);
  float std2 = sqrt(mean2_sqrd - mean2*mean2);

  // STD factor.
  float std_ratio = std2/std1;

  // Reset pointer.
  img1_ptr = img1_ptr_orig;

  // Integer mean.
  int imean1 = static_cast<int>(mean1);
  int imean2 = static_cast<int>(mean2);

  // Normalize image.
  float pix;
  for (size_t ii = 0; ii < image_size; ++ii) {
    pix = static_cast<float>(img1_ptr[ii] - imean1)*std_ratio + imean2;
    if(pix < 0.0)  pix = 0.0;
    if(pix > 255.0) pix = 255.0;
    img1_ptr[ii] = static_cast<unsigned char>(pix);
  }
}
