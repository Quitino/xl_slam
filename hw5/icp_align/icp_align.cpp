#include <sophus/se3.h>
#include <string>
#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>

using namespace std;

// path to compared trajectory file
string trajectory_file = "./compare.txt"; // time_e , t_e , q_e , time_g , t_g , q_g

// function for plotting trajectory, don't edit this code
// start point is red and end point is blue
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g);

void DrawTrajectoryAligned(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e,
                           vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g,
                           Eigen::Matrix3d R, Eigen::Vector3d t);

double computeATE(vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> trans_e,
                  vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> trans_g);
/*void pose_estimation_3d3d (
        const vector<Eigen::Vector3d>& pts1,
        const vector<Eigen::Vector3d>& pts2,
        Mat& R, Mat& t
);
 */

int main(int argc, char **argv) {

    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e;
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g;

    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> trans_e;
    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> trans_g;

    /// implement pose reading code
    // start your code here (5~10 lines)
    
    // Read
    ifstream T_file(trajectory_file);
    if (!T_file) {
      cout<<"unable to read file!"<<endl;
	exit(1);
    }
    double data[16] = {0};
    while(!T_file.eof()) {
      for (auto &p:data)
	T_file >> p;
      Eigen::Quaterniond q_e(data[7], data[4], data[5], data[6]);
      Eigen::Vector3d t_e(data[1], data[2], data[3]);
      Sophus::SE3 pose_e(q_e,t_e);
      poses_e.push_back(pose_e);
      trans_e.push_back(t_e);

      Eigen::Quaterniond q_g(data[15], data[12], data[13], data[14]);
      Eigen::Vector3d t_g(data[9], data[10], data[11]);
      Sophus::SE3 pose_g(q_g,t_g);
      poses_g.push_back(pose_g);
      trans_g.push_back(t_g);
    }
    
    T_file.close();
    // end your code here

    // print size of estimated poses and GT poses
    cout<< "estimated poses: " << poses_e.size() << "個"<<endl;  // 612
    cout<< "GT poses: " << poses_g.size() << "個"<<endl;         // 612

    // 612 對 3D-3D pairs correspondence
    // Given t_e, t_g

    Eigen::Vector3d p1, p2;  // center of mass
    int N = trans_e.size(); // N=612
    for ( int i=0; i<N; i++ )
    {
        p1 = p1 + trans_e[i];
        p2 = p2 + trans_g[i];
    }
    p1 = p1 / N;
    p2 = p2 / N;

    cout << "Center of estimated poses: " << p1.transpose() << endl; // -1.26923  0.330327 -0.246748
    cout << "Center of GT poses: " << p2.transpose() << endl;        // 0.499393 0.0956568   1.45822


    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>  q1, q2; // remove the center, q1, and q2 are normalized points
    for ( int i=0; i<N; i++ )
    {
        q1.push_back(trans_e[i] - p1) ;
        q2.push_back(trans_g[i] - p2) ;
    }

    cout << "normalized estimated poses: " << q1[N-1].transpose() << endl;
    cout << "normalized GT poses: " << q2[N-1].transpose() << endl;

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += q1[i] * q2[i].transpose();
        //W += Eigen::Vector3d ( q1[i][0], q1[i][1], q1[i][2] ) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    cout<<"W="<<W<<endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout<<"U="<<U<<endl;
    cout<<"V="<<V<<endl;

    Eigen::Matrix3d R_ = U* ( V.transpose() );
    Eigen::Vector3d t_ = p1 - R_ * p2; //

    cout<<"Rotation matrix: " << R_ << endl;
    cout<<"translation vector: " << t_.transpose() << endl; // -1.60832  1.49821 0.706883
    // Before ICP alignment
    // draw trajectory in pangolin
    //DrawTrajectory(poses_e, poses_g);

    cout << "Compute error (ATE): "<< endl;
    //cout<<"Before ICP alignment, RMSE (translation error, ATE) is: "<<computeATE(trans_e, trans_g)<<endl;

    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> trans_g_transformed;
    for (auto i=0; i<trans_g.size(); i++) {
        trans_g_transformed.push_back(R_ * trans_g[i] + t_);
    }
    cout<<"After ICP alignment, RMSE (translation error, ATE) is: "<<computeATE(trans_e, trans_g_transformed)<<endl;
    // After ICP alignment
    DrawTrajectoryAligned(poses_e, poses_g, R_, t_);

    // Perform ICP alignment (compute the relative transformation between two trajectories)
    // use SVD from opencv api

    return 0;
}

double computeATE(const vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> trans_e,
                  const vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> trans_g) {
    vector<double> b;
    //Eigen::VectorXd a;
    double result2=0;
    for (auto i=0; i<trans_e.size(); i++) {

        Eigen::Vector3d tran_e = trans_e[i];
        Eigen::Vector3d tran_g = trans_g[i];
        Eigen::Matrix<double, 3, 1> err_trans; // err is 3-dim vector
        err_trans = tran_e - tran_g;
        b.push_back(err_trans.norm()); // get norm-2 (in double value)
    }

    // compute RMSE for err
    for (auto i=0; i<b.size(); i++) {
        result2 += b[i]*b[i]; // accumulated squared err
    }
    result2 = sqrt(result2/b.size()); // mean -> square root, obtain RMSE

    b.clear();
    return result2;
    // end your code here
}

void DrawTrajectoryAligned(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e,
                            vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g,
                            Eigen::Matrix3d R, Eigen::Vector3d t) {
    if (poses_e.empty() || poses_g.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses_e.size() - 1; i++) {
            glColor3f(1 - (float) i / poses_e.size(), 0.0f, (float) i / poses_e.size()); // start from red and end with blue color
            // start the drawing for estimated trajectory
            glBegin(GL_LINES);
            auto p1 = poses_e[i], p2 = poses_e[i + 1];
            // define two vertex, 1st is starting point, 2nd is ending point.
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
            // finish the drawing

            // start the drawing for GT trajectory,
            glColor3f(1 - (float) i / poses_g.size(), 1 - (float) i / poses_g.size(), (float) i / poses_g.size()); // start from yellow and end with blue color
            glBegin(GL_LINES);
            auto p3 = poses_g[i], p4 = poses_g[i + 1]; // p3, p4 is Sophus:SE3
            // define two vertex, 1st is starting point, 2nd is ending point.
            Eigen::Vector3d p3_transformed, p4_transformed;

            p3_transformed = R*p3.translation()+t;
            p4_transformed = R*p4.translation()+t;
            glVertex3d(p3_transformed[0], p3_transformed[1], p3_transformed[2]);
            glVertex3d(p4_transformed[0], p4_transformed[1], p4_transformed[2]);
            glEnd();
            // finish the drawing
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}



/*******************************************************************************************/
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g) {
    if (poses_e.empty() || poses_g.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses_e.size() - 1; i++) {
            glColor3f(1 - (float) i / poses_e.size(), 0.0f, (float) i / poses_e.size()); // start from red and end with blue color
            // start the drawing for estimated trajectory
            glBegin(GL_LINES);
            auto p1 = poses_e[i], p2 = poses_e[i + 1];
            // define two vertex, 1st is starting point, 2nd is ending point.
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
            // finish the drawing

            // start the drawing for GT trajectory
	    glColor3f(1 - (float) i / poses_g.size(), 1 - (float) i / poses_g.size(), (float) i / poses_g.size()); // start from yellow and end with blue color
            glBegin(GL_LINES);
            auto p3 = poses_g[i], p4 = poses_g[i + 1];
            // define two vertex, 1st is starting point, 2nd is ending point.
            glVertex3d(p3.translation()[0], p3.translation()[1], p3.translation()[2]);
            glVertex3d(p4.translation()[0], p4.translation()[1], p4.translation()[2]);
            glEnd();
            // finish the drawing
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}
