#pragma once

#include <Eigen/Eigen>

#include <ba/Types.h>
#include <ba/InterpolationBuffer.h>


namespace devil {

class Tracker {

public:

  Tracker(const calibu::CameraRig& rig, unsigned int window_size = 5);

  ~Tracker();

  bool AddIMU(const Eigen::Vector3d&  accel,
              const Eigen::Vector3d&  gyro,
              double                  time
              );

public:
  const unsigned int kWindowSize;

private:
  calibu::CameraRig                                 rig_;

  typedef ba::ImuMeasurementT<double>     ImuMeasurement;
  ba::InterpolationBufferT<ImuMeasurement, double>  imu_buffer_;


};


} /* devil namespace */
