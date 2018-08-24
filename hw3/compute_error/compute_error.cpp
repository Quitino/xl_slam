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
void DrawCompare(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e, 
		 vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g);

int main(int argc, char **argv) {

    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e;
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g;

    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> trans_e;
    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> trans_g;
    
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
      cout<<"unable to read groundtruth.2txt file!"<<endl;
	exit(1);
    }

    double data[8] = {0}; // timestep, tx, ty, tz, qx, qy, qz, qw
    while(!T_est.eof()) {
      for (auto &p:data)
          T_est >> p;
      Eigen::Quaterniond q(data[7], data[4], data[5], data[6]);
      Eigen::Vector3d t(data[1], data[2], data[3]);
      Sophus::SE3 pose(q,t); // pass to a Lie algebra variable "pose"
      poses_e.push_back(pose);

      trans_e.push_back(t);
    }
    
    while(!T_gt.eof()) {
      for (auto &p:data)
          T_gt >> p;
      Eigen::Quaterniond q(data[7], data[4], data[5], data[6]);
      Eigen::Vector3d t(data[1], data[2], data[3]);
      Sophus::SE3 pose(q,t);
      poses_g.push_back(pose);

      trans_g.push_back(t);
    }

    // print size of estimated poses and GT poses
    cout<< "estimated poses: " << poses_e.size() << "個"<<endl;  // 613
    cout<< "GT poses: " << poses_g.size() << "個"<<endl;         // 613

    vector<double> a,b;
    //Eigen::VectorXd a;
    double result=0;
    double result2=0;
    for (auto i=0; i<poses_e.size(); i++) {
      Sophus::SE3 e = poses_e[i];
      Sophus::SE3 g = poses_g[i];
      Eigen::Matrix<double, 6, 1> err; // err is 6-dim vector
      // use Sophus::SE3, can use it as Lie group. After log(), vee operator is implicitly executed, then return a 6-dim vector
      err = (g.inverse()*e).log();
      a.push_back(err.norm()); // get norm-2 (in double value)

      Eigen::Vector3d tran_e = trans_e[i];
      Eigen::Vector3d tran_g = trans_g[i];
      Eigen::Matrix<double, 3, 1> err_trans; // err is 3-dim vector
      err_trans = tran_e - tran_g;
      b.push_back(err_trans.norm()); // get norm-2 (in double value)
    }
   



    // compute RMSE for err
    for (auto i=0; i<a.size(); i++) {
      result += a[i]*a[i]; // accumulated squared err
      result2 += b[i]*b[i]; // accumulated squared err
    }
    result = sqrt(result/a.size()); // mean -> square root, obtain RMSE
    result2 = sqrt(result2/b.size()); // mean -> square root, obtain RMSE
    cout<<"RMSE (Lie algebra) is: "<<result<<endl;
    cout<<"RMSE (translation error) is: "<<result2<<endl;
    // end your code here

    // draw gt and est. trajectory in pangolin   
    DrawCompare(poses_e, poses_g);
    return 0;
}


void DrawCompare(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e, 
		 vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g) {
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

    // To draw
    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        //glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set background color as white
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);  // set background color as black

        glLineWidth(2);
        for (size_t i = 0; i < poses_e.size() - 1; i++) {
            glColor3f(1 - (float) i / poses_e.size(), 0.0f, (float) i / poses_e.size()); // from red to blue
            glBegin(GL_LINES);
            auto p1 = poses_e[i], p2 = poses_e[i + 1];
	        auto p3 = poses_g[i], p4 = poses_g[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]); // start to draw line for estimated poses
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
	        glEnd(); // finish draw segment for estimated poses
	    
            glColor3f(0.0f , 1.0f, 0.0f); // GT in green color
            glBegin(GL_LINES);	    
	        glVertex3d(p3.translation()[0], p3.translation()[1], p3.translation()[2]); // start to draw line for GT poses
            glVertex3d(p4.translation()[0], p4.translation()[1], p4.translation()[2]);
            glEnd(); // finish draw segment for GT poses
        }
        
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

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
