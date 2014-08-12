/*
 * Copyright (c) 2013  Juan M. Falquez,
 *                     Nima Keivan
 *                     George Washington University
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

#ifndef CERES_DENSE_BA_H_
#define CERES_DENSE_BA_H_

#include <ceres/ceres.h>
#include <Eigen/Eigen>
#include <sophus/sophus.hpp>
#include <opencv2/opencv.hpp>


/////////////////////////////////////////////////////////////////////////////
/// CERES Stuff.

namespace ceres {

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


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

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

#endif  // CERES_DENSE_BA_H_
