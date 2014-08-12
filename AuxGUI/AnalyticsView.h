// Copyright (c) George Washington University, all rights reserved.  See the
// accompanying LICENSE file for more information.

#pragma once

#include <string>
#include <map>
#include <mutex>
#include <vector>
#include <pangolin/pangolin.h>

#include "ColorPalette.h"
#include "GLVarHistory.h"

/////////////////////////////////////////////////////////////////////////////
class AnalyticsView : public pangolin::View {
 public:
  AnalyticsView();
  ~AnalyticsView();

  virtual void Resize(const pangolin::Viewport& parent);

  void InitReset(unsigned int uVarHistoryLength = 60);
  void Clear();
  void Render();
  void Update(std::map<std::string, float>& mData);

 private:
  unsigned int  m_uVarHistoryLength;
  float         m_fWidth;
  float         m_fHeight;
  float         m_fBorderOffset;
  ColorPalette  m_ColorPalette;

  // Data to display
  std::map< std::string, GLVarHistory* > m_mData;

  // Projection matrix
  pangolin::OpenGlMatrix m_Ortho;

  // mutex for blocking drawing while updating data
  std::mutex m_Mutex;
};
