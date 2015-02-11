/*
 * Copyright (c) 2014  Juan M. Falquez,
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

#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include <sophus/se3.hpp>
#include <calibu/cam/CameraRig.h>


/////////////////////////////////////////////////////////////////////////////
namespace Eigen {
#define USING_VECTOR_ARRAY(size)                            \
  using Vector##size##tArray =                              \
  std::vector<Matrix<double, size, 1>,                      \
  Eigen::aligned_allocator<Matrix<double, size, 1>>>;

  USING_VECTOR_ARRAY(2);
  USING_VECTOR_ARRAY(3);
  USING_VECTOR_ARRAY(4);
  USING_VECTOR_ARRAY(5);
  USING_VECTOR_ARRAY(6);

#undef USING_VECTOR_ARRAY

  typedef Matrix<double, 2, 3> Matrix2x3d;
  typedef Matrix<double, 3, 4> Matrix3x4d;
  typedef Matrix<double, 6, 6> Matrix6d;
  typedef Matrix<double, 6, 1> Vector6d;

}  // namespace Eigen


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
class MUSE
{
  ///
  ///////////////////////////////////////////////////////////////////////////
  public:

    ///////////////////////////////////////////////////////////////////////////
    MUSE():
      is_init_(false)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    ~MUSE()
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    void Init(
        const calibu::CameraModelGeneric<double>& cmod
      )
    {
      // Store camera model.
      cam_model_ = cmod;

      //

      // Set initialize flag to true.
      is_init_ = true;
    }

  ///
  ///////////////////////////////////////////////////////////////////////////
  private:

    ///////////////////////////////////////////////////////////////////////////
    inline double _NormTukey(double r,
                             double c)
    {
      const double absr   = fabs(r);
      const double roc    = r/c;
      const double omroc2 = 1.0f-roc*roc;

      return (absr <= c) ? omroc2*omroc2 : 0.0f;
    }

  ///
  ///////////////////////////////////////////////////////////////////////////
  public:

  private:
    bool                                    is_init_;
    Eigen::Matrix3d                         F_;
    calibu::CameraModelGeneric<double>      cam_model_;
};
