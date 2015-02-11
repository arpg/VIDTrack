// Copyright (c) George Washington University, all rights reserved.  See the
// accompanying LICENSE file for more information.

#pragma once

#include <string>
#include <deque>
#include <vector>
#include <sstream>
#include <iomanip>

#include <SceneGraph/SceneGraph.h>

#include "GLSimpleObjects.h"

class GLVarHistory
{
 public:

  GLVarHistory() : max_history_length_(0), mean_val_(0), mean_num_(0) {}

  ~GLVarHistory() {}

  void InitReset(std::string sVarName, unsigned int nMaxLength) {
    var_name_          = sVarName;
    max_history_length_ = nMaxLength;
    mean_val_ = 0;
    mean_num_ = 0;
    history_.clear();
  }

  void Update(double dVarValue) {
    history_.push_back(dVarValue);

    if(history_.size() > max_history_length_)
      history_.pop_front();

    mean_val_ += dVarValue;
    mean_num_++;
  }

  void Draw(const float fWidth,
            const float fHeight,
            const float fOffsetX,
            const float fOffsetY,
            std::vector<double>& vfColor) {
    pangolin::GlState state;

    state.glDisable(GL_DEPTH_TEST);
    state.glEnable(GL_BLEND);

    // draw history
    int   nSteps  = (int)history_.size();
    float fTop    = fOffsetY;
    float fBottom = fOffsetY + fHeight;
    float fLeft   = fOffsetX;
    float fRight  = fOffsetX + fWidth;
    float fStepX  = fWidth / (max_history_length_ - 1);
    float fXStart = fRight - (nSteps - 1) * fStepX;
    float fScale  = 0.0;

    if(nSteps > 1)
    {
      for(int ii=0; ii < nSteps; ii++) {
        if(history_[ii] > fScale) {
          fScale = history_[ii];
        }
      }

      if(fScale > 0.0) {
        fScale = fHeight/fScale;
      }

      glColor4f(vfColor[0], vfColor[1], vfColor[2],0.6);
      GLfloat vertices[8];
      for(int ii=0; ii < nSteps-1; ++ii)
      {
        vertices[0] = fXStart + (ii+1) * fStepX;
        vertices[1] = fBottom;
        vertices[2] = fXStart + (ii+1) * fStepX;
        vertices[3] = fBottom - history_[ii+1]*fScale;
        vertices[4] = fXStart + ii * fStepX;
        vertices[5] = fBottom - history_[ii]*fScale;
        vertices[6] = fXStart + ii * fStepX;
        vertices[7] = fBottom;

        glDrawPolygon2d(vertices,4);
      }
    }

    // draw var name and last value
    glColor4f(1.0,1.0,1.0,1.0);
    std::stringstream ss;
    std::string sLine;

    ss << var_name_;
    if(nSteps > 0){
      ss  << ": " << std::setprecision(3) << history_.back();
    }

    sLine = ss.str();
    gl_text_.Draw(sLine, fLeft + 2.0, fBottom - 3.0);

    double mean = 0;
    ss.str("");
    ss << "m: ";
    if(nSteps > 0){
      mean = mean_val_ / mean_num_;
      ss << std::setprecision(2) << mean;
    }
    sLine = ss.str();
    gl_text_.Draw(sLine, fRight - fWidth/4.0, fBottom - 3.0);

    // draw median line
    glColor4f(229.0/255.0, 104.0/255.0, 63.0/255.0, 1.0);
    float fScaledMedian = (float)(fScale*mean);
    pangolin::glDrawLine(fLeft,
                         fBottom - fScaledMedian,
                         fRight,
                         fBottom - fScaledMedian);

    // draw globject bounding box
    glLineWidth(1.0);
    glColor4f(1.0,1.0,1.0,1.0);
    pangolin::glDrawRectPerimeter(fLeft,fTop,fRight,fBottom);
  }

 private:

  std::string         var_name_;
  unsigned int        max_history_length_;
  double              mean_val_;
  unsigned int        mean_num_;
  std::deque<double>  history_;
  SceneGraph::GLText  gl_text_;
};
