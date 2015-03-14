/////////////////////////////////////////////////////////////////////////////
inline float Interpolate(
    __global __read_only unsigned char* pImage,
    const float                         x,
    const float                         y,
    const unsigned int                  ImageWidth,
    const unsigned int                  ImageHeight
    )
{
  const int   px = (int) x;  // top left corner
  const int   py = (int) y;
  const float ax = x - px;
  const float ay = y - py;
  const float ax1 = 1.0f - ax;
  const float ay1 = 1.0f - ay;

  const unsigned int index = (ImageWidth*py) + px;

  float p1 = (float)pImage[index];
  float p2 = (float)pImage[index+1];
  float p3 = (float)pImage[index+ImageWidth];
  float p4 = (float)pImage[index+ImageWidth+1];
  p1 *= ay1;
  p2 *= ay1;
  p3 *= ay;
  p4 *= ay;
  p1 += p3;
  p2 += p4;
  p1 *= ax1;
  p2 *= ax;

  return p1 + p2;
}

/////////////////////////////////////////////////////////////////////////////
inline float2 Project( __constant const float* K, const float4 P )
{
  float2 p;

  p.x = (P.x * K[0] / P.z) + K[2];
  p.y = (P.y * K[1] / P.z) + K[3];

  return p;
}

/////////////////////////////////////////////////////////////////////////////
inline float NormTukey( float r, float c )
{
  const float absr = fabs(r);
  const float roc = r / c;
  const float omroc2 = 1.0f - roc*roc;
  return (absr <= c ) ? omroc2*omroc2 : 0.0f;
}

/////////////////////////////////////////////////////////////////////////////
inline float NormL1( float r, float c )
{
  const float absr = fabs(r);
  return (absr == 0 ) ? 1.0f : 1.0f / absr;
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////


/*
 * IMPORTANT IMPORTANT IMPORTANT
 * TODO TODO TODO
 *
 * THIS CODE WILL NOT WORK IF THE IMAGES ARE DIFFERENT SIZES!!!!!
 * THE BOUNDS CHECK WILL BREAK!!!
 */

__kernel void DensePoseRefinement(
    __global __read_only unsigned char* LiveGreyImg,
    __global __read_only unsigned char* RefGreyImg,
    __global __read_only float*         RefDepthImg,
    unsigned int                        ImageWidth,
    __constant float*                   Klg,
    __constant float*                   Krg,
    __constant float*                   Krd,
    __constant float*                   Tgd,
    __constant float*                   Tlr,
    float                               NormC,
    __local float*                      Scratch,
    __global float*                     LHS,
    __global float*                     RHS,
    __global float*                     Error
    )
{
  // intrinsics are stored as a vector in this order: fu(0,0)  fv(1,1)  cu(0,2)  cv(1,2)
  // T transforms are stored as a 3x4 Matrix (since the last row is always '0 0 0 1') in row major.

  // pass them as arguments
  const unsigned int ImageHeight = get_global_size(0) / ImageWidth;

  // reset scratch memory
#pragma unroll
  for( int ii = 0; ii < 29; ++ii ) {
    Scratch[get_local_id(0)+(ii*get_local_size(0))] = 0;
  }

  // 2d point in reference depth camera
  int2 pr_d;
  pr_d.x = get_global_id(0) % ImageWidth;
  pr_d.y = get_global_id(0) / ImageWidth;

  // get depth
  const float fDepth = RefDepthImg[get_global_id(0)];

  if( fDepth != 0 ) {

    // 3d point from reference depth camera
    float4 Pr_d;
    Pr_d.x = fDepth * (pr_d.x - Krd[2]) / Krd[0];
    Pr_d.y = fDepth * (pr_d.y - Krd[3]) / Krd[1];
    Pr_d.z = fDepth;
    Pr_d.w = 1;


    // homogenized 3d point in reference grey camera
    float4 Pr_g;
    Pr_g.x = (Tgd[0] * Pr_d.x) + (Tgd[1] * Pr_d.y) + (Tgd[2] * Pr_d.z) + Tgd[3];
    Pr_g.y = (Tgd[4] * Pr_d.x) + (Tgd[5] * Pr_d.y) + (Tgd[6] * Pr_d.z) + Tgd[7];
    Pr_g.z = (Tgd[8] * Pr_d.x) + (Tgd[9] * Pr_d.y) + (Tgd[10] * Pr_d.z) + Tgd[11];
    Pr_g.w = 1;


    // project to reference grey camera's image coordinate
    float2 pr_g = Project( Krg, Pr_g );

    // check if point is out of bounds
    if( pr_g.x >= 2 && pr_g.x < ImageWidth-3 && pr_g.y >= 2 && pr_g.y < ImageHeight-3 ) {

      // homogenized 3d point in live grey camera
      float4 Pl_g;
      Pl_g.x = (Tlr[0] * Pr_g.x) + (Tlr[1] * Pr_g.y) + (Tlr[2] * Pr_g.z) + Tlr[3];
      Pl_g.y = (Tlr[4] * Pr_g.x) + (Tlr[5] * Pr_g.y) + (Tlr[6] * Pr_g.z) + Tlr[7];
      Pl_g.z = (Tlr[8] * Pr_g.x) + (Tlr[9] * Pr_g.y) + (Tlr[10] * Pr_g.z) + Tlr[11];
      Pl_g.w = 1;

      // project to live grey camera's image coordinate
      float2 pl_g = Project( Klg, Pl_g );


      // check if point is out of bounds
      if( pl_g.x >= 2 && pl_g.x < ImageWidth-3 && pl_g.y >= 2 && pl_g.y < ImageHeight-3 ) {

        // get intensities
        const float Il = Interpolate(LiveGreyImg, pl_g.x, pl_g.y, ImageWidth, ImageHeight);
        const float Ir = Interpolate(RefGreyImg, pr_g.x, pr_g.y, ImageWidth, ImageHeight);

        // TODO discard under/over-saturated pixels

        // calculate error
        const float y = Il - Ir;

        // image derivative
        const float Il_xr = Interpolate(LiveGreyImg, pl_g.x+1.0, pl_g.y, ImageWidth, ImageHeight);
        const float Il_xl = Interpolate(LiveGreyImg, pl_g.x-1.0, pl_g.y, ImageWidth, ImageHeight);
        const float Il_yu = Interpolate(LiveGreyImg, pl_g.x, pl_g.y-1.0, ImageWidth, ImageHeight);
        const float Il_yd = Interpolate(LiveGreyImg, pl_g.x, pl_g.y+1.0, ImageWidth, ImageHeight);

        float dIl[2];
        dIl[0] = (Il_xr - Il_xl)/2.0;
        dIl[1] = (Il_yd - Il_yu)/2.0;

        // projection & dehomogenization derivative
        float KlPl[3];
        KlPl[0] = (Klg[0] * Pl_g.x) + (Klg[2] * Pl_g.z);
        KlPl[1] = (Klg[1] * Pl_g.y) + (Klg[3] * Pl_g.z);
        KlPl[2] = Pl_g.z;

        float dPl[2][3];
        dPl[0][0] = 1.0/KlPl[2];
        dPl[0][1] = 0;
        dPl[0][2] = -KlPl[0]/(KlPl[2]*KlPl[2]);
        dPl[1][0] = 0;
        dPl[1][1] = 1.0/KlPl[2];
        dPl[1][2] = -KlPl[1]/(KlPl[2]*KlPl[2]);

        // derivative multiplication
        float dIl_dPl[3];
        dIl_dPl[0] = (dIl[0] * dPl[0][0]) + (dIl[1] * dPl[1][0]);
        dIl_dPl[1] = (dIl[0] * dPl[0][1]) + (dIl[1] * dPl[1][1]);
        dIl_dPl[2] = (dIl[0] * dPl[0][2]) + (dIl[1] * dPl[1][2]);


        float KlgTlr[3][4];
        KlgTlr[0][0] = (Klg[0] * Tlr[0]) + (Klg[2] * Tlr[8]);
        KlgTlr[0][1] = (Klg[0] * Tlr[1]) + (Klg[2] * Tlr[9]);
        KlgTlr[0][2] = (Klg[0] * Tlr[2]) + (Klg[2] * Tlr[10]);
        KlgTlr[0][3] = (Klg[0] * Tlr[3]) + (Klg[2] * Tlr[11]);
        KlgTlr[1][0] = (Klg[1] * Tlr[4]) + (Klg[3] * Tlr[8]);
        KlgTlr[1][1] = (Klg[1] * Tlr[5]) + (Klg[3] * Tlr[9]);
        KlgTlr[1][2] = (Klg[1] * Tlr[6]) + (Klg[3] * Tlr[10]);
        KlgTlr[1][3] = (Klg[1] * Tlr[7]) + (Klg[3] * Tlr[11]);
        KlgTlr[2][0] = Tlr[8];
        KlgTlr[2][1] = Tlr[9];
        KlgTlr[2][2] = Tlr[10];
        KlgTlr[2][3] = Tlr[11];


        // more derivative multiplications
        float dIl_dPl_KlgTlr[4];
        dIl_dPl_KlgTlr[0] = (dIl_dPl[0] * KlgTlr[0][0]) + (dIl_dPl[1] * KlgTlr[1][0]) + (dIl_dPl[2] * KlgTlr[2][0]);
        dIl_dPl_KlgTlr[1] = (dIl_dPl[0] * KlgTlr[0][1]) + (dIl_dPl[1] * KlgTlr[1][1]) + (dIl_dPl[2] * KlgTlr[2][1]);
        dIl_dPl_KlgTlr[2] = (dIl_dPl[0] * KlgTlr[0][2]) + (dIl_dPl[1] * KlgTlr[1][2]) + (dIl_dPl[2] * KlgTlr[2][2]);
        dIl_dPl_KlgTlr[3] = (dIl_dPl[0] * KlgTlr[0][3]) + (dIl_dPl[1] * KlgTlr[1][3]) + (dIl_dPl[2] * KlgTlr[2][3]);


        // J = dIl_dPl_KlgTlr * gen_i * Pr
        float J[6];
        J[0] = dIl_dPl_KlgTlr[0];
        J[1] = dIl_dPl_KlgTlr[1];
        J[2] = dIl_dPl_KlgTlr[2];
        J[3] = -dIl_dPl_KlgTlr[1]*Pr_g.z + dIl_dPl_KlgTlr[2]*Pr_g.y;
        J[4] = +dIl_dPl_KlgTlr[0]*Pr_g.z - dIl_dPl_KlgTlr[2]*Pr_g.x;
        J[5] = -dIl_dPl_KlgTlr[0]*Pr_g.y + dIl_dPl_KlgTlr[1]*Pr_g.x;


        // add robust norm here
//        const float w = 1.0;
        const float w = NormTukey(y, NormC);

        // sparse matrix format
        Scratch[get_local_id(0)+(0*get_local_size(0))] = J[0] * J[0] * w;
        Scratch[get_local_id(0)+(1*get_local_size(0))] = J[0] * J[1] * w;
        Scratch[get_local_id(0)+(2*get_local_size(0))] = J[0] * J[2] * w;
        Scratch[get_local_id(0)+(3*get_local_size(0))] = J[0] * J[3] * w;
        Scratch[get_local_id(0)+(4*get_local_size(0))] = J[0] * J[4] * w;
        Scratch[get_local_id(0)+(5*get_local_size(0))] = J[0] * J[5] * w;
        Scratch[get_local_id(0)+(6*get_local_size(0))] = J[1] * J[1] * w;
        Scratch[get_local_id(0)+(7*get_local_size(0))] = J[1] * J[2] * w;
        Scratch[get_local_id(0)+(8*get_local_size(0))] = J[1] * J[3] * w;
        Scratch[get_local_id(0)+(9*get_local_size(0))] = J[1] * J[4] * w;
        Scratch[get_local_id(0)+(10*get_local_size(0))] = J[1] * J[5] * w;
        Scratch[get_local_id(0)+(11*get_local_size(0))] = J[2] * J[2] * w;
        Scratch[get_local_id(0)+(12*get_local_size(0))] = J[2] * J[3] * w;
        Scratch[get_local_id(0)+(13*get_local_size(0))] = J[2] * J[4] * w;
        Scratch[get_local_id(0)+(14*get_local_size(0))] = J[2] * J[5] * w;
        Scratch[get_local_id(0)+(15*get_local_size(0))] = J[3] * J[3] * w;
        Scratch[get_local_id(0)+(16*get_local_size(0))] = J[3] * J[4] * w;
        Scratch[get_local_id(0)+(17*get_local_size(0))] = J[3] * J[5] * w;
        Scratch[get_local_id(0)+(18*get_local_size(0))] = J[4] * J[4] * w;
        Scratch[get_local_id(0)+(19*get_local_size(0))] = J[4] * J[5] * w;
        Scratch[get_local_id(0)+(20*get_local_size(0))] = J[5] * J[5] * w;

        // RHS
        Scratch[get_local_id(0)+(21*get_local_size(0))] = J[0] * y * w;
        Scratch[get_local_id(0)+(22*get_local_size(0))] = J[1] * y * w;
        Scratch[get_local_id(0)+(23*get_local_size(0))] = J[2] * y * w;
        Scratch[get_local_id(0)+(24*get_local_size(0))] = J[3] * y * w;
        Scratch[get_local_id(0)+(25*get_local_size(0))] = J[4] * y * w;
        Scratch[get_local_id(0)+(26*get_local_size(0))] = J[5] * y * w;

        // Error + Num Obs
        Scratch[get_local_id(0)+(27*get_local_size(0))] = y * y;
        Scratch[get_local_id(0)+(28*get_local_size(0))] = 1;

      }
    }
  }


  ///----- reduce code


  //    for( int ii = 0; ii < 29; ++ii ) {
  //        Scratch[get_local_id(0)+(ii*get_local_size(0))] = ii+1;
  //    }

  barrier(CLK_LOCAL_MEM_FENCE);

  for( unsigned int S=get_local_size(0)/2; S>0; S>>=1 ) {
    if( get_local_id(0) < S ) {
#pragma unroll
      for( int ii = 0; ii < 29; ++ii ) {
        unsigned int index = get_local_id(0)+(ii*get_local_size(0));
        Scratch[index] += Scratch[index+S];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // write out the result
  if(get_local_id(0) == 0) {
#pragma unroll
    for( int ii = 0; ii < 21; ++ii ) {
      LHS[(get_group_id(0)*21)+ii] = Scratch[ii*get_local_size(0)];
    }
#pragma unroll
    for( int ii = 0; ii < 6; ++ii ) {
      RHS[(get_group_id(0)*6)+ii] = Scratch[(ii+21)*get_local_size(0)];
    }
    Error[(get_group_id(0)*2)] = Scratch[27*get_local_size(0)];
    Error[(get_group_id(0)*2)+1] = Scratch[28*get_local_size(0)];
  }
}
