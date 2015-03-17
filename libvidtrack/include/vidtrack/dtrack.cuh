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

#pragma once

#ifndef __CUDACC__

# ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Woverloaded-virtual"
# endif

# include <opencv2/opencv.hpp>

# ifdef __clang__
#   pragma clang diagnostic pop
# endif

# include <Eigen/Dense>
#endif


class cuDTrack {

public:
  cuDTrack(unsigned int max_height, unsigned int max_width);

  ~cuDTrack();

#ifndef __CUDACC__
  void Estimate(
      const cv::Mat&                      live_grey,
      const cv::Mat&                      ref_grey,
      const cv::Mat&                      ref_depth,
      const Eigen::Matrix3d&              Klg,
      const Eigen::Matrix3d&              Krg,
      const Eigen::Matrix3d&              Krd,
      const Eigen::Matrix4d&              Tgd,
      const Eigen::Matrix4d&              Tlr,
      const Eigen::Matrix<double, 3, 4>&  KlgTlr,
      float                               norm_param,
      bool                                discard_saturated,
      float                               min_depth,
      float                               max_depth
    )
  {
    // Pack all from structs (Eigen/OpenCV) to host pointers.
    _LaunchEstimate(live_grey.rows, live_grey.cols);
    // Unpack.
  }
#endif

private:
  void _LaunchEstimate(
      unsigned int image_height,
      unsigned int image_width
    );

  int _GCD(int a, int b);

  void _CheckErrors(const char* label);

  unsigned int _CheckMemory();


private:
  unsigned char*      d_ref_image_;
  float*              d_ref_depth_;
  unsigned char*      d_live_image_;
  float*              d_lss_;
};
