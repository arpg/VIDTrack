// Copyright (c) George Washington University, all rights reserved.  See the
// accompanying LICENSE file for more information.

#include "AnalyticsView.h"
#include "GLVarHistory.h"

#include <string>
#include <map>

AnalyticsView::AnalyticsView():
  m_uVarHistoryLength(60),
  m_fWidth(0),
  m_fHeight(0),
  m_fBorderOffset(2.0)
{}

AnalyticsView::~AnalyticsView() {
  Clear();
}

void AnalyticsView::InitReset(unsigned int uVarHistoryLength) {
  Clear();
  m_uVarHistoryLength = uVarHistoryLength;
}

void AnalyticsView::Clear() {
  std::map< std::string, GLVarHistory* >::iterator itVar;

  for (itVar = m_mData.begin(); itVar != m_mData.end(); ++itVar) {
    // draw: (width,height, offset_x, offset_y)
    delete itVar->second;
  }

  m_mData.clear();
}

void AnalyticsView::Resize(const pangolin::Viewport& parent) {
  pangolin::View::Resize(parent);

  m_Ortho = pangolin::ProjectionMatrixOrthographic(0, v.w, v.h, 0.5, 0, 1E4);

  // recompute these for adjusting rendering
  m_fWidth  = static_cast<float>(v.w) - m_fBorderOffset*2;
  m_fHeight = static_cast<float>(v.h) - m_fBorderOffset*2;
}

void AnalyticsView::Render() {
  std::unique_lock<std::mutex> lock(m_Mutex);

  pangolin::GlState state;
  state.glDisable(GL_LIGHTING);
  state.glDisable(GL_DEPTH_TEST);

  // Activate viewport
  this->Activate();

  // Load orthographic projection matrix to match image
  glMatrixMode(GL_PROJECTION);
  m_Ortho.Load();

  // Reset ModelView matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  float fVerticalSpacing = 2.0;
  float fVarHeight = (m_fHeight - (fVerticalSpacing * m_mData.size() - 1)) /
      m_mData.size();
  float fVarWidth  = m_fWidth;

  Palette& colors = m_ColorPalette.GetPaletteRef(eOblivion);

  std::map< std::string, GLVarHistory* >::iterator itVar;

  int ii;
  for (ii = 0, itVar = m_mData.begin(); itVar != m_mData.end(); ++itVar, ++ii) {
    unsigned int cidx = ii % (colors.size()-1);
    // draw: (width,height, offset_x, offset_y)
    float fOffsetY = m_fBorderOffset + ii*(fVarHeight + fVerticalSpacing);
    itVar->second->Draw(fVarWidth, fVarHeight, m_fBorderOffset,
                        fOffsetY, colors[cidx]);
  }
}

void AnalyticsView::Update(std::map< std::string, float>& mData) {
  std::unique_lock<std::mutex> lock(m_Mutex);
  for (auto itInput = mData.begin(); itInput != mData.end(); ++itInput) {
    const auto it = m_mData.find(itInput->first);
    if (it == m_mData.end()) {
      // create variable
      m_mData[ itInput->first ] = new GLVarHistory;
      m_mData[ itInput->first ]->InitReset(itInput->first, m_uVarHistoryLength);
    }
    m_mData[ itInput->first]->Update(itInput->second);
  }
}
