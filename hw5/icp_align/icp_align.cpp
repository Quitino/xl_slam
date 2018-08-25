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

    cout << "Center of estimated poses: " << p1.transpose() << endl;
    cout << "Center of GT poses: " << p2.transpose() << endl;

    // Before ICP alignment
    // draw trajectory in pangolin
    DrawTrajectory(poses_e, poses_g);


    // Perform ICP alignment (compute the relative transformation between two trajectories)
    // use SVD from opencv api

    return 0;
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
