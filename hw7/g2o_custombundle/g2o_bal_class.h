#include <Eigen/Core>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"

#include "ceres/autodiff.h"

#include "tools/rotation.h"
#include "common/projection.h"


// create 顶点输入类型 for camera input (camera pose, focal_length, distortion_coeff)
class VertexCamInput 
{
public:
    VertexCamInput(){
        //exchange angle-axis and t
        Vector6d se3;
        se3.head<3>() = camera.block<3,1>(3,0);
        se3.tail<3>() = camera.head<3>();
        SE3_ = SE3::exp(se3);
        f_ = camera[6];
        k1_ = cmera[7];
        k2_ = cmera[8];
    }

public:
    SE3 SE3_;
    double f_;
    double k1_;
    double k2_;
};

// vertex class for camera's pose and intrinsics parameters.
class VertexCameraBAL : public g2o::BaseVertex<9, VertexCamInput>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        //Eigen::VectorXd::ConstMapType v ( update, VertexCameraBAL::Dimension );
        //_estimate += v;
        cout<<"1 VertexCameraBAL update:" 
        << update[0] << ", " << update[1] << ", " << update[2] << ", "
        << update[3] << ", " << update[4] << ", " << update[5] << endl;

        Vector6d update_se3;
        update_se3 << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate.SE3_ = SE3::exp(update_se3) * estimate().SE3_;
        _estimate.f_ += update[6]; 
        _estimate.k1_ += update[7];
        _estimate.k2_ += update[8];
    }

};

// vertex for 3d point
class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        Eigen::Vector3d::ConstMapType v ( update );
        _estimate += v;
    }
};

class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d,VertexCameraBAL, VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void computeError() override   // The virtual function comes from the Edge base class. Must define if you use edge.
    {
        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );

        //( *this ) ( cam->estimate().data(), point->estimate().data(), _error.data() );

        double predictions[2];
        CamProjectionWithDistortion(cam->estimate(), point->estimate(), predictions);
        _error[0] = double(measurement()(0)) - predictions[0];
        _error[1] = double(measurement()(1)) - predictions[1];

    }

    template<typename T>
    bool operator() ( const T* camera, const T* point, T* residuals ) const
    {
        T predictions[2];
        CamProjectionWithDistortion ( camera, point, predictions );
        residuals[0] = predictions[0] - T ( measurement() ( 0 ) );
        residuals[1] = predictions[1] - T ( measurement() ( 1 ) );

        return true;
    }


    virtual void linearizeOplus() override
    {
        // use numeric Jacobians
        // BaseBinaryEdge<2, Vector2d, VertexCameraBAL, VertexPointBAL>::linearizeOplus();
        // return;
        
        // using autodiff from ceres. Otherwise, the system will use g2o numerical diff for Jacobians

        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );


        /*
        Vector3d Pc;
        Pc = cam->estimate().SE3_ * point->estimate();
        //camera frame
        double xc = Pc[0];
        double yc = Pc[1];
        double zc = Pc[2];
        //normalized camera frame
        double xc1 = -xc/zc;
        double yc1 = -yc/zc;
        double zc1 = -1;
    */


        
        typedef ceres::internal::AutoDiff<EdgeObservationBAL, double, VertexCameraBAL::Dimension, VertexPointBAL::Dimension> BalAutoDiff;

        Eigen::Matrix<double, Dimension, VertexCameraBAL::Dimension, Eigen::RowMajor> dError_dCamera;
        Eigen::Matrix<double, Dimension, VertexPointBAL::Dimension, Eigen::RowMajor> dError_dPoint;
        double *parameters[] = { const_cast<double*> ( cam->estimate().data() ), const_cast<double*> ( point->estimate().data() ) };
        double *jacobians[] = { dError_dCamera.data(), dError_dPoint.data() };
        double value[Dimension];
        bool diffState = BalAutoDiff::Differentiate ( *this, parameters, Dimension, value, jacobians );

        // copy over the Jacobians (convert row-major -> column-major)
        if ( diffState )
        {
            _jacobianOplusXi = dError_dCamera;
            _jacobianOplusXj = dError_dPoint;
        }
        else
        {
            assert ( 0 && "Error while differentiating" );
            _jacobianOplusXi.setZero();
            _jacobianOplusXi.setZero();
        }
        
    }
};
