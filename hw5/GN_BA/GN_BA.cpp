//
// Created by xiang on 12/21/17.
//

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "sophus/se3.h"

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d; // 3D points in world coordinate
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d; // 2D points in pixel coordinate at (input image)
typedef Matrix<double, 6, 1> Vector6d; // pose

// optimize one pose between reference image (identity matrix pose) and input image (pose to be optimized)
string p3d_file = "./p3d.txt";
string p2d_file = "./p2d.txt";

int main(int argc, char **argv) {

    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K; // calibration matrix
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // load points in to p3d and p2d 
    // START YOUR CODE HERE
    ifstream ifp3d, ifp2d;
    string sP3d, sP2d; // used to store each line of p3d.txt and p2d.txt

    ifp3d.open(p3d_file.c_str()); // open p3d.txt
    if(!ifp3d.is_open()) {
        cerr<<"ifp3d is not open! p3d_file: "<<p3d_file.c_str()<<endl;
        return -1;
    }
    ifp2d.open(p2d_file.c_str()); // open p2d.txt
    if(!ifp2d.is_open()) {
        cerr<<"ifp2d is not open! p2d_file: "<<p2d_file.c_str()<<endl;
        return -1;
    }

    while(getline(ifp3d,sP3d) && !sP3d.empty()) {
        istringstream issP3d(sP3d); // convert string (one line) into string stream (to read word by word)
        Vector3d vP3d;
        issP3d >> vP3d[0] >> vP3d[1] >> vP3d[2]; // read one 3D point's coordinates.
        p3d.push_back(vP3d); // push one 3D point's coordinate into vector p3d
    }
    cout<<"p3d size: "<<p3d.size()<<endl;

    while(getline(ifp2d, sP2d) && !sP2d.empty()) {
        istringstream issP2d(sP2d);
        Vector2d vP2d;
        issP2d >> vP2d[0] >> vP2d[1]; // read one 2D pixel's coordinates.
        p2d.push_back(vP2d); // push one 2D pixel's coordinate into vector p2d
    }
    cout<<"p2d size: "<<p2d.size()<<endl;
    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());

    int iterations = 100; // optimize in 100 times
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size(); // 76
    cout << "points: " << nPoints << endl;

    Sophus::SE3 T_esti; // estimated pose
    cout<<"T_esti start: \n" << T_esti.matrix()<<endl; // by default initialized as 4x4 identity matrix

    for (int iter = 0; iter < iterations; iter++) { // start the optimization

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero(); // initialize Hessian matrix as zeros matrix (6x6)
        Vector6d b = Vector6d::Zero();  // initialize bias vector (6-dim)
        Vector2d e; // measuremnt error (2-dim in pixel)
        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I]
            // START YOUR CODE HERE 
            Vector3d Pc = T_esti * p3d[i];
            // if (DEBUG) cout << "K * Pc : \n" << K*Pc << endl;
            Vector3d v3_e = Vector3d(p2d[i][0], p2d[i][1], 1) - K * Pc / Pc[2]; // more compact way by using homogenous coordinate
            e[0] = v3_e[0];
            e[1] = v3_e[1];
            // if (DEBUG) cout << "v3_e : " << v3_e.transpose() << " e: " << e.transpose() << " Pc: " << Pc.transpose() << endl;
            double x = Pc[0];
            double y = Pc[1];
            double z = Pc[2];
            double x2 = x*x; // used in computing Jacobian
            double y2 = y*y;
            double z2 = z*z;

	    // compute jacobian
            Matrix<double, 2, 6> J = Matrix<double, 2, 6>::Zero();
            J(0,0) = -fx / z;
            J(0,2) = fx * x / z2;
            J(0,3) = fx * x * y / z2;
            J(0,4) = -fx - fx * x2 / z2;
            J(0,5) = fx * y / z;
            J(1,1) = -fy / z;
            J(1,2) = fy * y /z2;
            J(1,3) = fy + fy * y2 / z2;
            J(1,4) = -fy * x * y / z2;
            J(1,5) = -fy * x / z;

            // END YOUR CODE HERE
            // if(DEBUG) cout<<"J: \n" << J <<endl;

            H += J.transpose() * J;
            b += -J.transpose() * e;

            cost += 0.5 * e.transpose()*e;
            //if(DEBUG) cout<<"cost: "<<cost<<endl;
        }

	    // solve dx
        Vector6d dx; // increments for pose

        // START YOUR CODE HERE 
        dx = H.ldlt().solve(b); // solve normal equation to get increments
        cout<<"iter: "<< iter <<", dx: "<<dx.transpose()<<endl;
        // END YOUR CODE HERE

        if (std::isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        //if (iter > 0 && cost >= lastCost) {
        if (iter > 0 && cost >= lastCost*1.1) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        // START YOUR CODE HERE 
        T_esti = Sophus::SE3::exp(dx) * T_esti;
        // END YOUR CODE HERE
        
        lastCost = cost;

        cout << "iteration " << iter << ", cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}
