#pragma once

class iDTAM {
public:

#ifdef iDTAM_WINDOW_SIZE
  static const int kWindowSize = iDTAM_WINDOW_SIZE;
#else
  static const int kWindowSize = 5;  // DEFAULT
#endif

#ifdef iDTAM_CUDA_BLOCK_SIZE
  static const int kBlockSizeCUDA = iDTAM_CUDA_BLOCK_SIZE;
#else
  static const int kBlockSizeCUDA = 32;  // DEFAULT
#endif

  struct Parameters {
      float omega;              // Cost volume decay.
      float theta_start;        // Coupling term initial value.
      float theta_end;          // Coupling term final value.
      float beta;               // Theta step reduction parameter.
      float lambda;             // Initial cost-depth scaling parameter.
      float epsilon;            // Huber norm parameter.
      float g_alpha;            // Weight's L2 norm scaling parameter.
      float g_beta;             // Weight's L2 norm exponential parameter.
      float sig_d;              // Gradient descent time step.
      float sig_q;              // Gradient ascent time step.

      Parameters()
      {
        omega       = 0.5;
        theta_start = 1.0;
        theta_end   = 1e-4;
        beta        = 1e-4;
        lambda      = 1.0;
        epsilon     = 1e-4;
        g_alpha     = 100;
        g_beta      = 1.6;
        sig_d       = 0.7;
        sig_q       = 0.7;
      }
  }; /* options */

public:
  iDTAM();

  ~iDTAM();

  void Init(unsigned int width, unsigned int height,
            float kmin, float kmax, unsigned int levels,
            bool incremental_volume,
            float* K, float* Kinv, Parameters params);

  // Expecting normalized image intensities. Pose should be global, in column
  // major.
  void PushImage(float* host_image, float* host_pose);


  void AddLayerToCostVolume(float* Ir, unsigned char* Ir_mask, float* Im,
                            int id, float* Tmr, float omega);
  void OptimizeLayers();

  void GetDepth(float* host_buffer);
  void GetInverseDepth(float* host_buffer);
  void GetCostVolumeSlice(float* host_buffer, int n);




  void RescaleVolume(float threshold, float dz, float Kmin, float Kmax);


//private:
  void _AddToCostVolume(float* Ir, float* Im, float* Tmr, float omega);

  void _CalculateRelativePoses();
  size_t _CheckMemoryCUDA();
  bool _CheckNaN(char var);
  void _FreeAll();
  void _InitWeights();
  void _InitializeDAQ();
  void _Iterate(float theta);
  void _Optimize();
  void _UpdateQ();
  void _UpdateD(float theta);
  void _UpdateA(float theta);


private:
  unsigned int    width_;             // Image width.
  unsigned int    height_;            // Image height.
  float           kmin_;              // Min inverse depth.
  float           kmax_;              // Max inverse depth.
  unsigned int    levels_;            // Depth discretization levels.
  bool            incremental_vol_;   // True if incremental volume should be used.
  Parameters      params_;            // Optimization parameters.

  float*          K_;                 // K matrix (3x3).
  float*          Kinv_;              // K inverse matrix (3x3).

  float*          Iv_;                // Incremental volume's reference image.
  float*          Tvm_;               // Relative transform from volume.
  float*          Twv_;               // Global pose of incremental volume.

  unsigned char*  Ir_mask_;           // Reference image mask.
  float*          Ir_;                // Reference image.
  float*          Twr_;               // Global pose of reference image.
  bool            window_full_;       // Flag
  int             window_idx_;        // Tracks next available slot for comparison image.
  float*          Im_[kWindowSize];   // Support Images.
  float*          Twm_[kWindowSize];  // Global poses of support images (4x4 matrix).
  float*          Tmr_[kWindowSize];  // Temporal relative transforms.

  float*          diff_;              // Holds max-min differences (for cuadratic speedup).
  float*          weight_;            // Holds weight for huber norm regularization.
  float*          Q_;                 // Optimization auxiliary variable.
  float*          D_;                 // Optimization auxiliary variable.
  float*          A_;                 // Optimization auxiliary variable.

  float*          icost_;             // Incremental cost volume.
  float*          cost_;              // Auxiliary cost volume.
  float*          iframes_;           // Incremental cost volume.
  float*          frames_;            // Auxiliary cost volume.
  float*          scaled_cost_;       // Holds cost_ / frames_.

  float*          depth_;             // Output depth (in meters).
};
