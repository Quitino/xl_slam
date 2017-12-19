#include <sophus/se3.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <Eigen/Dense>

// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>

using namespace std;

// path to trajectory file
//string trajectory_file = "./trajectory.txt";

// function for plotting trajectory, don't edit this code
// start point is red and end point is blue
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>);

int main(int argc, char **argv) {

    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e;
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g;
    
    /// implement pose reading code
    // start your code here (5~10 lines)
    
    // Read
    ifstream T_est("./estimated.txt");
    if (!T_est) {
      cout<<"unable to read estimated.txt file!"<<endl;
	exit(1);
    }

    ifstream T_gt("./groundtruth.txt");
    if (!T_gt) {
      cout<<"unable to read groundtruth.txt file!"<<endl;
	exit(1);
    }

    double data[8] = {0};
    while(!T_est.eof()) {
      for (auto &p:data)
	T_est >> p;
      Eigen::Quaterniond q(data[7], data[4], data[5], data[6]);
      Eigen::Vector3d t(data[1], data[2], data[3]);
      Sophus::SE3 pose(q,t);
      poses_e.push_back(pose);
    }
    
    while(!T_gt.eof()) {
      for (auto &p:data)
	T_gt >> p;
      Eigen::Quaterniond q(data[7], data[4], data[5], data[6]);
      Eigen::Vector3d t(data[1], data[2], data[3]);
      Sophus::SE3 pose(q,t);
      poses_g.push_back(pose);
      
    }

    vector<double> a;
    //Eigen::VectorXd a;
    double result=0;
    for (auto i=0; i<poses_e.size(); i++) {
      Sophus::SE3 e = poses_e[i];
      Sophus::SE3 g = poses_g[i];
      Eigen::Matrix<double, 6, 1> err;
      err = (g.inverse()*e).log();
      a.push_back(err.norm());
      //a << err.norm();
    }
   
   // compute RMSE for err
   //cout<<a<<endl;
   
   
    // compute RMSE for err
    for (auto i=0; i<a.size(); i++) {
      result += a[i]*a[i];
    }
    result = sqrt(result/a.size());
    cout<<"RMSE is: "<<result<<endl;
    // end your code here

    // draw trajectory in pangolin
    DrawTrajectory(poses_e);
    
    DrawTrajectory(poses_g);
    return 0;
}

/*******************************************************************************************/
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses) {
    if (poses.empty()) {
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
        for (size_t i = 0; i < poses.size() - 1; i++) {
            glColor3f(1 - (float) i / poses.size(), 0.0f, (float) i / poses.size());
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}
