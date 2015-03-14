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
#include <Eigen/Dense>
#endif

class cuDTrack {

public:
  cuDTrack(unsigned int max_height, unsigned int max_width);

  ~cuDTrack();

#ifndef __CUDACC__
  void Estimate(unsigned int image_height, unsigned int image_width, Eigen::Matrix3d t)
  {

  }

#endif


#ifndef __CUDACC__
#if 0
    inline __host__ Mat() {
    }

    template<typename PF>
    inline __host__ Mat(const Eigen::Matrix<PF,R,C>& em) {
        for( size_t r=0; r<R; ++r )
            for( size_t c=0; c<C; ++c )
                m[r*C + c] = (P)em(r,c);
    }

    template<typename PT>
    inline __host__ operator Eigen::Matrix<PT,R,C>() const {
        Eigen::Matrix<PT,R,C> ret;
        for( size_t r=0; r<R; ++r )
            for( size_t c=0; c<C; ++c )
                ret(r,c) = (PT)m[r*C + c];
        return ret;
    }
#endif
#endif // EIGEN

private:
  void _LaunchEstimate(unsigned int image_height, unsigned int image_width);

  int _GCD(int a, int b);

  void _CheckErrors(const char* label);

  unsigned int _CheckMemory();


private:
  unsigned char*      ref_image_;
  float*              ref_depth_;
  unsigned char*      live_image_;
  float*              lss_;

#if 0
  struct LeastSquaresSystem
  {
    float*      LHS;
    float*      RHS;
    float*      squared_error;
    bool*       obs;
  };
  LeastSquaresSystem  lss_;
#endif
};
