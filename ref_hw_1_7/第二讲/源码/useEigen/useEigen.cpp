#include <iostream>
using namespace std;
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#define SIZE 100
int main()
{
cout<<"useEigen..."<<endl;
Eigen::Matrix<double, SIZE, SIZE>  A;
A = Eigen::MatrixXd::Random(SIZE, SIZE); 
//A = A.transpose() * A;

Eigen::Matrix<double, SIZE, 1> b;
b = Eigen::MatrixXd::Random(SIZE,1);


//inverse
Eigen::Matrix<double, SIZE, 1> x1;
x1  = A.inverse() * b;
cout<<"Direct Inverse Result:\n"<<x1.matrix().transpose()<<endl;

//QR
Eigen::Matrix<double, SIZE, 1> x2;
x2 = A.fullPivHouseholderQr().solve(b);
cout<<"QR Result:\n"<<x2.matrix().transpose()<<endl;

//Cholesky
Eigen::Matrix<double, SIZE, 1> x3;
x3 = A.ldlt().solve(b);
cout<<"Cholesky LDLT Result:\n"<<x3.matrix().transpose()<<endl;

return 1;
}
