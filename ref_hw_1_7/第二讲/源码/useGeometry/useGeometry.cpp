#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main()
{
cout<<"START..."<<endl;
Isometry3d T_C1_W = Isometry3d::Identity();
Quaterniond q_C1_W(0.55,0.3,0.2,0.2);
cout<<"q_C1_W:"<<q_C1_W.coeffs().transpose()<<endl;
T_C1_W.rotate(q_C1_W.normalized().toRotationMatrix());
T_C1_W.pretranslate(Vector3d(0.7,1.1,0.2));
cout<<"T_C1_W:\n"<<T_C1_W.matrix()<<endl;

Isometry3d T_C2_W = Isometry3d::Identity();
Quaterniond q_C2_W(-0.1,0.3,-0.7,0.2);
T_C2_W.rotate(q_C2_W.normalized().toRotationMatrix());
T_C2_W.pretranslate(Vector3d(-0.1,0.4,0.8));

Vector3d p_C1(0.5,-0.1,0.2);
Vector3d p_C2 = T_C2_W * T_C1_W.inverse() * p_C1;
cout<<"p_C2: "<<p_C2.transpose()<<endl;

return 1;

}
