#include <devil/tracker.h>

#include <miniglog/logging.h>


using namespace devil;

///////////////////////////////////////////////////////////////////////////
Tracker::Tracker(const calibu::CameraRig& rig, unsigned int window_size)
  : kWindowSize(window_size)
{
  rig_ = calibu::ToCoordinateConvention(rig, calibu::RdfRobotics);

  Sophus::SE3d M_rv;
  M_rv.so3() = calibu::RdfRobotics;
  for (calibu::CameraModelAndTransform& model : rig_.cameras) {
    model.T_wc = model.T_wc*M_rv;
  }

  LOG(INFO) << "Starting Tvs:" << std::endl << rig_.cameras[0].T_wc.matrix();

}

///////////////////////////////////////////////////////////////////////////
Tracker::~Tracker()
{

}

