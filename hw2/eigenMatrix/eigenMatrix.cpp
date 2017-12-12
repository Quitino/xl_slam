#include <iostream>
#include <ctime>
// Eigen 部分
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Dense>

#define MATRIX_SIZE 100

/****************************
* 本程序演示了 Eigen 基本类型的使用
****************************/
using namespace Eigen;
using namespace std;

int main( int argc, char** argv )
{

    // create random matrix and random vector
    MatrixXd A = MatrixXd::Random( MATRIX_SIZE, MATRIX_SIZE );
    MatrixXd b = MatrixXd::Random( MATRIX_SIZE,1 );
    MatrixXd x = Matrix<double, MATRIX_SIZE, 1>::Zero();

    clock_t time_stt = clock(); // 计时

    // QR decomposition
    cout << "Computing QR decomposition..." << endl;
    time_stt = clock();
    x = A.colPivHouseholderQr().solve(b);
    cout <<"time use in QR compsition is " <<1000*  (clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl<<endl;

    // Cholesky decomposition
    LLT<MatrixXd> llt;
    cout << "Computing Cholesky LLT decomposition..." << endl;
    time_stt = clock();
    llt.compute(A);
    x = llt.solve(b);
    cout <<"time use in Cholesky LLT compsition is " <<1000*  (clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;

    /*
    Eigen::Matrix< double, MATRIX_SIZE, MATRIX_SIZE > matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random( MATRIX_SIZE, MATRIX_SIZE );

    // create random vector
    Eigen::Matrix< double, MATRIX_SIZE,  1> v_Nd;
    v_Nd = Eigen::MatrixXd::Random( MATRIX_SIZE,1 );


    // 直接求逆
    Eigen::Matrix<double,MATRIX_SIZE,1> x = matrix_NN.inverse()*v_Nd;
    cout <<"time use in normal invers is " << 1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms"<< endl;
    
	// 通常用矩阵分解来求，例如QR分解，速度会快很多
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout <<"time use in Qr compsition is " <<1000*  (clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;

    // 通常用矩阵分解来求，例如cholesky分解，速度会快很多
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout <<"time use in Qr compsition is " <<1000*  (clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
*/
    return 0;
}
