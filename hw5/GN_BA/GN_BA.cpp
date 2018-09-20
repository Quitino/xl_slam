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

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "./p3d.txt";
string p2d_file = "./p2d.txt";

int main(int argc, char **argv) {

    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // load points in to p3d and p2d 
    // START YOUR CODE HERE

    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "points: " << nPoints << endl;

    Sophus::SE3 T_esti; // estimated pose
    cout<<"T_esti start: \n" << T_esti.matrix()<<endl;

    for (int iter = 0; iter < iterations; iter++) {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();
        Vector2d e;
        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I]
            // START YOUR CODE HERE 
            Vector3d Pc = T_esti * p3d[i];
            // if (DEBUG) cout << "K * Pc : \n" << K*Pc << endl;
            Vector3d v3_e = Vector3d(p2d[i][0], p2d[i][1], 1) - K * Pc / Pc[2];
            e[0] = v3_e[0];
            e[1] = v3_e[1];
            // if (DEBUG) cout << "v3_e : " << v3_e.transpose() << " e: " << e.transpose() << " Pc: " << Pc.transpose() << endl;
            double x = Pc[0];
            double y = Pc[1];
            double z = Pc[2];
            double x2 = x*x;
            double y2 = y*y;
            double z2 = z*z;
	    // END YOUR CODE HERE

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
            // if(DEBUG) cout<<"J: \n" << J <<endl;
            // START YOUR CODE HERE 

	    // END YOUR CODE HERE

            H += J.transpose() * J;
            b += -J.transpose() * e;

            cost += 0.5 * e.transpose()*e;
            //if(DEBUG) cout<<"cost: "<<cost<<endl;
        }

	    // solve dx
        Vector6d dx;

        // START YOUR CODE HERE 

        // END YOUR CODE HERE

        if (std::isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        // START YOUR CODE HERE 

        // END YOUR CODE HERE
        
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}
