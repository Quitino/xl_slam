#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

#include "sophus/se3.h"
using namespace Sophus;
string GTPath = "../data/groundtruth.txt";
string EsPath = "../data/estimated.txt";
ifstream ifsGT;
ifstream ifsEs;

SE3 T_W_Cg;
SE3 T_W_Ce;
Quaterniond q_g;
Quaterniond q_e;
Vector3d t_g;
Vector3d t_e;
double time_g;
double time_e;

SE3 T_Cg_Ce;
Matrix<double,6,1> kesi;
double err = 0;
double RMSE;
int main()
{
cout<<"traceError main ..."<<endl;
ifsGT.open(GTPath.c_str());
ifsEs.open(EsPath.c_str());
if(!ifsGT.is_open() || !ifsEs.is_open())
{
cerr<<"txt is not opened!"<<endl;
return -1;
}

int num = 0;
string sGTLine,sEsLine;
while(getline(ifsGT,sGTLine) && getline(ifsEs,sEsLine) 
	&&!sGTLine.empty() && !sEsLine.empty())
{
istringstream issGT(sGTLine);
istringstream issEs(sEsLine);
issGT >>time_g>>t_g[0]>>t_g[1]>>t_g[2]>>q_g.x()>>q_g.y()>>q_g.z()>>q_g.w();
issEs >>time_e>>t_e[0]>>t_e[1]>>t_e[2]>>q_e.x()>>q_e.y()>>q_e.z()>>q_e.w();
T_W_Cg = SE3(q_g, t_g);
T_W_Ce = SE3(q_e, t_e);
kesi = (T_W_Cg.inverse() * T_W_Ce).log();
err += kesi.transpose() * kesi;
num++;
}

RMSE = sqrt(err/num);
cout<<"RMSE: "<<RMSE<<endl;
return 1;
}
