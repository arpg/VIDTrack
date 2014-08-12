#pragma once

#include <Eigen/Eigen>
#include <sophus/se3.hpp>

#include <SceneGraph/GLObject.h>

#define MAT4_COL_MAJOR_DATA(m) (Eigen::Matrix<float,4,4,Eigen::ColMajor>(m).data())


/////////////////////////////////////////////////////////////////////////////
// Code to render the vehicle path
class GLPath : public SceneGraph::GLObject
{
public:
    GLPath()
    {
        m_bInitGLComplete = false;
        m_fLineColor(0) = 1.0;
        m_fLineColor(1) = 1.0;
        m_fLineColor(2) = 0.0;
        m_fLineColor(3) = 1.0;
        m_fPointColor(0) = 1.0;
        m_fPointColor(1) = 0.0;
        m_fPointColor(2) = 0.0;
        m_fPointColor(3) = 1.0;
        m_fPointSize = 5.0;
        m_nPoseDisplay = 0;
        m_bDrawAxis = true;
        m_bDrawLines = true;
        m_bDrawPoints = true;
    }

    ~GLPath()
    {

    }

    // just draw the path
    void DrawCanonicalObject()
    {
        pangolin::GlState state;
//        state.glDisable( GL_LIGHTING );

        glLineWidth(1.0f);

        if( !m_vPath.empty() ) {

            if( m_bDrawAxis ) {
                int start = 0;
                if( m_nPoseDisplay != 0 ) {
                    if( m_vPath.size() > m_nPoseDisplay ) {
                        start = m_vPath.size() - m_nPoseDisplay;
                    }
                }
                glPushMatrix();
                for( size_t ii = start; ii < m_vPath.size(); ++ii ) {
                    Sophus::SE3d&    Pose = m_vPath[ii];
                    Eigen::Matrix4f  fPose = Pose.matrix().cast<float>();
                    glMultMatrixf( MAT4_COL_MAJOR_DATA( fPose ) );
                    glColor3f(1.0,0.0,0.0);
                    pangolin::glDrawLine(0,0,0,1,0,0);
                    glColor3f(0.0,1.0,0.0);
                    pangolin::glDrawLine(0,0,0,0,1,0);
                    glColor3f(0.0,0.0,1.0);
                    pangolin::glDrawLine(0,0,0,0,0,1);
                }
                glPopMatrix();
            }


            /*
            if( m_bDrawLines ) {
                glPushMatrix();
                glEnable( GL_LINE_SMOOTH );
                glLineWidth( 1 );
                glColor4f( m_fLineColor(0), m_fLineColor(1), m_fLineColor(2), m_fLineColor(3) );

                glBegin( GL_LINE_STRIP );
                for( unsigned int ii = start; ii < m_vPath.size(); ++ii ) {
                    Eigen::Matrix4f    Pose = m_vPath[ii].matrix();
                    glVertex3f( Pose(0,3), Pose(1,3), Pose(2,3) );
                }
                glEnd();
                glPopMatrix();
            }
            */


        }
    }

    std::vector<Sophus::SE3d>& GetPathRef()
    {
      return m_vPath;
    }

    void SetLineColor( float R, float G, float B, float A = 1.0 )
    {
        m_fLineColor(0) = R;
        m_fLineColor(1) = G;
        m_fLineColor(2) = B;
        m_fLineColor(3) = A;
    }

    void SetPointColor( float R, float G, float B, float A = 1.0 )
    {
        m_fPointColor(0) = R;
        m_fPointColor(1) = G;
        m_fPointColor(2) = B;
        m_fPointColor(3) = A;
    }

    void SetPoseDisplay( unsigned int Num )
    {
        m_nPoseDisplay = Num;
    }

    void SetPointSize( float Size )
    {
        m_fPointSize = Size;
    }

    void DrawLines( bool Val )
    {
        m_bDrawLines = Val;
    }

    void DrawPoints( bool Val )
    {
        m_bDrawPoints = Val;
    }

    void DrawAxis( bool Val )
    {
        m_bDrawAxis = Val;
    }


private:
    bool                            m_bDrawLines;
    bool                            m_bDrawAxis;
    bool                            m_bDrawPoints;
    GLuint                          m_nDrawListId;
    float                           m_fPointSize;
    unsigned int                    m_nPoseDisplay;
    bool                            m_bInitGLComplete;
    Eigen::Vector4f                 m_fLineColor;
    Eigen::Vector4f                 m_fPointColor;
    std::vector<Sophus::SE3d>       m_vPath;
};
