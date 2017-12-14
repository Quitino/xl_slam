#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
// Eigen 几何模块
#include <Eigen/Geometry>

/****************************
* 本程序演示了 Eigen 几何模块的使用方法
****************************/

int main ( int argc, char** argv )
{


     // p_c1 = [0.5,0,0.2]
    Eigen::Vector3d p_c1(0.5, 0, 0.2);

    // q1 = [0.35, 0.2, 0.3, 0.1], t1 = [0.3, 0.1, 0.1]
    Eigen::Quaterniond q1(0.35, 0.2, 0.3, 0.1);
    q1.normalize();

    Eigen::Isometry3d T_c1_w= Eigen::Isometry3d::Identity();
    T_c1_w.rotate(q1);
    T_c1_w.pretranslate(Eigen::Vector3d(0.3, 0.1, 0.1));

    // q2 = [-0.5, 0.4, -0.1, 0.2], t2 = [-0.1, 0.5, 0.3]

    Eigen::Quaterniond q2(-0.5, 0.4, -0.1, 0.2);
    q2.normalize();
    Eigen::Isometry3d T_c2_w= Eigen::Isometry3d::Identity();
    T_c2_w.rotate(q2);
    T_c2_w.pretranslate(Eigen::Vector3d(-0.1, 0.5, 0.3));   


    Eigen::Vector3d p_c2 = T_c2_w * T_c1_w.inverse() * p_c1;

    cout.precision(6);

    cout<<"result: " << p_c2.transpose()<<endl;


    return 0;
}
