//
// Created by xiang on 1/4/18.
// this program shows how to perform direct bundle adjustment
//
#include <iostream>

using namespace std;

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <Eigen/Core>
#include <sophus/se3.h>
#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>
#include <boost/format.hpp>

using namespace Sophus;
using namespace pangolin;
using namespace g2o;

typedef vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> VecSE3;  // a set of poses
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d; // a set of 3d points

// global variables
string pose_file = "./poses.txt";
string points_file = "./points.txt";

/* poses.txt -- total 7 images (0.png, ... 6.png), T_cw : store in quaternion format: tx,ty,tz,qx,qy,qz,qw
1.46323e+09 0.702775 0.084358 0.00503326 -0.0651624 0.112345 -0.160729 0.978416
1.46323e+09 1.0694 0.102769 -0.1487 0.00222945 0.0342752 -0.247501 0.968279
1.46323e+09 0.74723 0.168659 -0.341037 0.0646964 -0.0708158 -0.276756 0.956141
1.46323e+09 0.723157 0.148595 -0.16002 0.00764941 -0.0952651 -0.239383 0.96621
1.46323e+09 0.708933 0.153003 -0.0375575 -0.0474721 -0.0660129 -0.197027 0.97702
1.46323e+09 0.722901 0.171558 -0.0117945 -0.0554745 -0.0398198 -0.171559 0.982804
1.46323e+09 0.763371 0.172428 0.0192505 -0.0681163 -0.0208489 -0.16148 0.984302
 */

/* points.txt -- each line, first three float numbers denote [x,y,z], the last 16 numbers denote intensity value, total 4,118 points
-0.471274 -4.40961 3.76621 140.453 139.923 139.901 140.169 145.503 147.859 146.762 145.462 167.012 178.969 194.086 208.254 340.051 340.796 341.539 337.327
-0.459047 -4.45304 3.83186 147.859 146.762 145.462 147.12 178.969 194.086 208.254 220.14 340.796 341.539 337.327 337.38 339.113 338.191 337.683 338.611
0.221176 -3.60007 3.41384 308.475 325.507 333.023 338.321 299.138 316.006 328.791 336.053 290.544 302.37 320.362 331.31 285.574 291.719 306.631 322.094
 */

// camera intrinsics, the image is in 640x480 resolution.
float fx = 277.34;
float fy = 291.402;
float cx = 312.234;
float cy = 239.777;

// append a global function to check whether is out of border
inline bool bInImage(float u, float v, int w, int h)
{
    if(u>=0 && u<w && v>=0 && v<h) 
	   return true;
    else
	   return false;
}

// bilinear interpolation
// get the intensity value from pixel (x,y)
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // uchar : 8-bit无符号整形数据，范围为[0, 255]
    uchar *data = &img.data[int(y) * img.step + int(x)];
    // xx (yy) is offsets from floor value
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

// g2o vertex that use sophus::SE3 as pose, view as Lie algebra
class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSophus() {}

    ~VertexSophus() {}

    bool read(std::istream &is) {}

    bool write(std::ostream &os) const {}
    // initialize
    virtual void setToOriginImpl() {
        _estimate = Sophus::SE3();
    }

    virtual void oplusImpl(const double *update_) {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        // estimate() : return the current estimate of the vertex
        setEstimate(Sophus::SE3::exp(update) * estimate());
    }
};

// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch
typedef Eigen::Matrix<double,16,1> Vector16d;  // The dimention of error term is 16, type is Vector16d, connected vertex are "VertexSBAPointXYZ" & "VertexSophus"
class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16d, g2o::VertexSBAPointXYZ, VertexSophus> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeDirectProjection(float *color, cv::Mat &target) { // use constructor to set color and target image
        this->origColor = color; // origColor comes from 3d points (4x4=16 intensity value)
        this->targetImg = target;
        this->w = targetImg.cols;
		this->h = targetImg.rows;
    }

    ~EdgeDirectProjection() {}

    virtual void computeError() override {
        // TODO START YOUR CODE HERE
        // compute projection error ...
	    const VertexSBAPointXYZ* vertexPw = static_cast<const VertexSBAPointXYZ* >(vertex(0));
	    const VertexSophus* vertexTcw = static_cast<const VertexSophus* >(vertex(1));
	    Vector3d Pc = vertexTcw->estimate() * vertexPw->estimate();
	    float u = Pc[0]/Pc[2] * fx + cx; // 3Dto2D, see eqt. (5.5) in P.86
	    float v = Pc[1]/Pc[2] * fy + cy;
	    // check whether is near of out of border
	    if(!bInImage(u-3,v-3,w,h) || !bInImage(u+2,v+2,w,h)) {
	        this->setLevel(1); // //set the level of the edge (bad term)
	        for(int n=0;n<16;n++)
	    	_error[n] = 0; // if near of out of border, don't participate in _error
	    } else {
	        for(int i = -2; i<2; i++) {
	        	for(int j = -2; j<2; j++) {
	    	      int num = 4 * i + j + 10;
	    	      _error[num] = origColor[num] - GetPixelValue(targetImg, u+i, v+j); //compute projection error ...
	        	}
	        } // obtain _error[0], ... _error[15] , total 16 terms of projection error
	    }
             // END YOUR CODE HERE
    }    

    /*
    virtual void computeError() override {
        // TODO START YOUR CODE HERE
        // compute projection error ...
        const VertexSophus* pose = static_cast<const VertexSophus*> (vertex(1));
        g2o::VertexSBAPointXYZ* point = static_cast<g2o::VertexSBAPointXYZ*> (vertex(0));

        const Eigen::Vector3d pointe = point->estimate();
        const Sophus::SE3 posee = pose->estimate();

        auto Pw = posee*pointe;
        double X = Pw[0];
        double Y = Pw[1];
        double Z = Pw[2];
        double u = fx*X/Z + cx;
        double v = fy*Y/Z + cy;
        int n = -1;

        for (int i = -2; i<2; i++)
        {

            for (int j= -2; j<2; j++)
            {
                n++;
                _error[n] = _measurement[n] - GetPixelValue(targetImg, u+i, v+j);
            }
        }


        // END YOUR CODE HERE
    }*/

    // Let g2o compute jacobian for you, no need to override linearizeOplus() function

    // Here, we use analytic jacobian, so override it
    virtual void linearizeOplus() override {
	
        if(level()==1) {
	       _jacobianOplusXi = Matrix<double,16,3>::Zero();
	       _jacobianOplusXj = Matrix<double,16,6>::Zero();
	       return;
	    }
	    const VertexSBAPointXYZ* vertexPw = static_cast<const VertexSBAPointXYZ* >(vertex(0));
	    const VertexSophus* vertexTcw = static_cast<const VertexSophus* >(vertex(1));
	    Vector3d Pc = vertexTcw->estimate() * vertexPw->estimate();
	    float x = Pc[0];
	    float y = Pc[1];
	    float z = Pc[2];
	    float inv_z = 1.0/z;
	    float inv_z2 = inv_z * inv_z;
	    float u = x * inv_z * fx + cx;
	    float v = y * inv_z * fy + cy;
	    
	    Matrix<double,2,3> J_Puv_Pc;
	    J_Puv_Pc(0,0) = fx * inv_z;
	    J_Puv_Pc(0,1) = 0;
	    J_Puv_Pc(0,2) = -fx * x * inv_z2;
	    J_Puv_Pc(1,0) = 0;
	    J_Puv_Pc(1,1) = fy * inv_z;
	    J_Puv_Pc(1,2) = -fy * y * inv_z2;
	    //if(DEBUG) cout<<"J_Puv_PC:\n"<<J_Puv_Pc<<endl;
     
	    Matrix<double,3,6> J_Pc_kesi = Matrix<double,3,6>::Zero();
	    J_Pc_kesi(0,0) = 1;
	    J_Pc_kesi(0,4) = z;
	    J_Pc_kesi(0,5) = -y;
	    J_Pc_kesi(1,1) = 1;
	    J_Pc_kesi(1,3) = -z;
	    J_Pc_kesi(1,5) = x;
	    J_Pc_kesi(2,2) = 1;
	    J_Pc_kesi(2,3) = y;
	    J_Pc_kesi(2,4) = -x;	
	    //if(DEBUG) cout<<"J_Pc_kesi:\n"<<J_Pc_kesi<<endl;
     
	    Matrix<double,1,2> J_I_Puv;
	    for(int i = -2; i<2; i++)
	        for(int j = -2; j<2; j++) {
	    	int num = 4 * i + j + 10;
	    	J_I_Puv(0,0) = (GetPixelValue(targetImg,u+i+1,v+j) - GetPixelValue(targetImg,u+i-1,v+j))/2;
	    	J_I_Puv(0,1) = (GetPixelValue(targetImg,u+i,v+j+1) - GetPixelValue(targetImg,u+i,v+j-1))/2;
	    	_jacobianOplusXi.block<1,3>(num,0) = -J_I_Puv * J_Puv_Pc * vertexTcw->estimate().rotation_matrix(); 
	    	_jacobianOplusXj.block<1,6>(num,0) = -J_I_Puv * J_Puv_Pc * J_Pc_kesi;
	    }
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

private:
    cv::Mat targetImg;  // the target image
    float *origColor = nullptr;   // 16 floats, the color of this point
    int w;
    int h;
};

// plot the poses and points for you, need pangolin
void Draw(const VecSE3 &poses, const VecVec3d &points);

int main(int argc, char **argv) {

    // 1. read poses and points
    VecSE3 poses;
    VecVec3d points;
    ifstream fin(pose_file);

    while (!fin.eof()) {
        double timestamp = 0;
        fin >> timestamp;
        if (timestamp == 0) break;
        double data[7]; // temp variable
        for (auto &d: data) fin >> d; // 先順序讀出pose值: tx,ty,tz,qx,qy,qz,qw.
        // wrap for Sophus::SE3 format
        poses.push_back(Sophus::SE3(
                // Eigen::Quaterniond use real part as the first element
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
        ));
        if (!fin.good()) break;
    }
    fin.close();  // finish reading poses.

    // use color to store grapyscale value, store 16 pixel value for each point, for total 4,118 points
    vector<float *> color;
    fin.open(points_file);
    while (!fin.eof()) {
        double xyz[3] = {0};  // temp variable
        for (int i = 0; i < 3; i++) fin >> xyz[i];  // 讀出x, y, z值
        if (xyz[0] == 0) break;
        points.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2])); // wrap for Eigen::Vector3d format
        float *c = new float[16]; // temp variable, c is a 16-dim array
        for (int i = 0; i < 16; i++) 
        	fin >> c[i]; // c contains 16 value of intensities for 4x4 patch
        color.push_back(c); // color contains a set of c, for total 4,118 points

        if (fin.good() == false) break;
    }
    fin.close(); // finish reading points.

    cout << "poses: " << poses.size() << ", points: " << points.size() << endl;

    // 2. read images
    vector<cv::Mat> images; // images contains 7 images
    boost::format fmt("./%d.png");
    for (int i = 0; i < 7; i++) {
        images.push_back(cv::imread((fmt % i).str(), 0));
    } // finish reading images

    // Prior to optimization, draw the poses and points
    cout << "Prior to optimization, draw the poses and points" << endl;
    //Draw(poses, points);


    cout << "Start constructing graph" << endl;
    // 3. build optimization problem
    //  z - g(x,y) : x (SE3) in 6-dim, y (XYZ) in 3-dim
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> DirectBlock;  // 求解的向量是6＊1的, binary edge 對應的頂點， 一個是6維(pose), 一個是3維(point)
    // linearSolver : is the solver for equation Hx = -b, using dense solver
    DirectBlock::LinearSolverType *linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    // solver_ptr :  單個誤差項對應的參數塊, 參數塊 size = 6 x 3
    DirectBlock *solver_ptr = new DirectBlock(linearSolver);
    // construct the specific algorithm of gradient decent (L-M), for 單個誤差項對應的參數塊
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr); // L-M used
    g2o::SparseOptimizer optimizer; // create the "graph", including its elements (vertex & edges) and algorithm
    // set the algorithm (L-M)
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
    cout << "Finsh constructing graph" << endl;

    cout << "Start adding vertex into graph" << endl;
    // 4. TODO add vertices, edges into the graph optimizer
    // START YOUR CODE HERE
    // ADD vertices for VertexSBAPointXYZ (3D points)
    for(int i = 0; i < points.size(); i++) {
		VertexSBAPointXYZ* vertexPw = new VertexSBAPointXYZ();
		vertexPw->setEstimate(points[i]);
		vertexPw->setId(i);
		vertexPw->setMarginalized(true);
		optimizer.addVertex(vertexPw);
    }

    // ADD vertices for VertexSophus (camera_pose)
    for(int j = 0; j < poses.size(); j++) {
		VertexSophus* vertexTcw = new VertexSophus();
		vertexTcw->setEstimate(poses[j]);
		vertexTcw->setId(j + points.size()); // in case conflict with VertexSBAPointXYZ
		optimizer.addVertex(vertexTcw);
    }


    /*
    // ADD vertices
    // create pointer for VertexSophus, Camerap->camera_pose
    VertexSophus* Camerap = new VertexSophus(); // pointer, considered as a set of VertexSophus (camera_pose)
    for (int i=0; i<poses.size(); i++)
    // for (auto &pose: poses) {
    {
        Sophus::SE3 cam;
        cam = poses[i]; // poses are read from './poses.txt'
        Camerap->setId(i); // 

        if (i==0) {
            Camerap->setFixed(true); //第一個點固定為零點
        }
        //VertexSophus* Camerap = new VertexSophus();
        // 設定預設值（初始值）
        Camerap->setEstimate(cam);  //! set the estimate for the vertex also calls updateCache()

        optimizer.addVertex(Camerap); // add vertex
    }

    // ADD vertices
    // create pointer for VertexSBAPointXYZ
    g2o::VertexSBAPointXYZ* point3d = new g2o::VertexSBAPointXYZ();
    for (int i=0; i<points.size(); i++)
    {
        Eigen::Vector3d point;
        point = points[i]; // points are read from './points.txt'
        //g2o::VertexSBAPointXYZ* point3d = new g2o::VertexSBAPointXYZ();
        point3d->setId(7+i);
        // 設定預設值（初始值）
        point3d->setEstimate(point);
        // set as Marginalized
        point3d->setMarginalized(true);
        optimizer.addVertex(point3d); // add vertex
    }
    */
    cout << "Finsh adding vertex into graph" << endl;

    cout << "Start adding edges into graph" << endl;

    // ADD edges, first check whether near or out of border
    for(int c = 0; c < poses.size(); c++) 
		for(int p = 0; p < points.size(); p++) {
			// color[p] contains 16 intensity values, images[c] is a 640x480 grayscale image [0, 255.0]
			// edge : 3d point's 4x4 patch in intensity <--> grayscale image, this edge is binary edge
			// EdgeDirectProjection(float *color, cv::Mat &target) { // use constructor to set color (3D point's 16 intensity) and target image
		    EdgeDirectProjection* edge = new EdgeDirectProjection(color[p],images[c]); // By this constructor, we can easily compute _error terms
		    edge->setVertex(0,dynamic_cast<VertexSBAPointXYZ*>(optimizer.vertex(p))); // load vertex from optimizer (graph), index by id
		    edge->setVertex(1,dynamic_cast<VertexSophus*>(optimizer.vertex(c + points.size())));
		    edge->setInformation(Matrix<double,16,16>::Identity()); // set no correlation matrix
		    RobustKernelHuber* rk = new RobustKernelHuber; // select Huber kernel
		    rk->setDelta(1.0);
		    edge->setRobustKernel(rk); // add Huber kernel for each edge
		    optimizer.addEdge(edge); // finally, add the edge into optimizer (graph)    
		}
	/*
    // ADD edges, first check whether near or out of border
    int id=1;
    for (int i=0; i<poses.size(); i++) // loop through poses
    {
        for (int j=0; j<points.size(); j++) // loop through points
        {
            auto Pc = poses[i] * points[j]; // transform points[j] in world coordinate to point in camera (pose_i) coordinate
            double X = Pc[0];
            double Y = Pc[1];
            double Z = Pc[2];
            double m = fx*X/Z + cx;
            double n = fy*Y/Z + cy;
            // if out of boundary (point_j is invisible to frame_i), no need for optimization
            if (m-2<0 || n-2<0 || m+1 > images[i].cols || n+1 > images[i].rows)
            {
                continue;
            }
            // construct the binary edge "ba_edge", connecting color_j vertex with image_i vertex
            // each point_j corresponds to color_j
            // <g2o::VertexSBAPointXYZ, VertexSophus>
            EdgeDirectProjection *ba_edge = new EdgeDirectProjection(color[j], images[i]);
            //EdgeDirectProjection *ba_edge = new EdgeDirectProjection(points[j], poses[i]);
            //EdgeDirectProjection *ba_edge = new EdgeDirectProjection(color[j], poses[i]);
            // add Huber kernel
            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            rk->setDelta(1.0); // set delta params for huber kernel
            ba_edge->setRobustKernel(rk); // add kernel

            ba_edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(j+7))); //point
            ba_edge->setVertex(1, dynamic_cast<VertexSophus*>(optimizer.vertex(i))); //pose


            typedef Eigen::Matrix<double, 16, 1> Vector16d; // for 16 pixel intensities
            typedef Eigen::Matrix<double, 16, 16> Matrix16x16d; // for information matrix
            Vector16d colorvector;
            for (int w=0; w<16; m++) // loop inside the patch (4x4)
            {
                colorvector[w] = (color[j][w]);  // The j-th point.
            }
            // set measurement value, 16-dim Eigen vector
            ba_edge->setMeasurement(colorvector);
            // set information matrix
            ba_edge->setInformation(Matrix16x16d::Identity());
            ba_edge->setId ( id++ );
            optimizer.addEdge(ba_edge);
        }

    }*/

    cout << "Finsh adding edges into graph" << endl;

    cout<<"edges in graph: "<<optimizer.edges().size() <<endl;
 //   optimizer.setVerbose(true); //! verbose information during optimization
 //   optimizer.initializeOptimization();
//    optimizer.optimize ( 30 );
//    Tcw = pose->estimate();


    // END YOUR CODE HERE

    // 5. perform optimization
    cout << "START optimization ..." << endl;
    optimizer.initializeOptimization(0);  //Initializes the structures for optimizing the whole graph, in level 0, thus edges in level==1 will not participate in optimization.
    optimizer.optimize(200); // run optimization in 200 iterations
    cout << "FINISH optimization ..." << endl;
    // TODO fetch data from the optimizer
    // START YOUR CODE HERE

    /*
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();

    double* raw_cameras = bal_problem->mutable_cameras();
    for(int i = 0; i < num_cameras; ++i)
    {
        VertexCameraBAL* pCamera = dynamic_cast<VertexCameraBAL*>(optimizer->vertex(i));
        Eigen::VectorXd NewCameraVec = pCamera->estimate();
        memcpy(raw_cameras + i * camera_block_size, NewCameraVec.data(), sizeof(double) * camera_block_size);
    }

    double* raw_points = bal_problem->mutable_points();
    for(int j = 0; j < num_points; ++j)
    {
        VertexPointBAL* pPoint = dynamic_cast<VertexPointBAL*>(optimizer->vertex(j + num_cameras));
        Eigen::Vector3d NewPointVec = pPoint->estimate();
        memcpy(raw_points + j * point_block_size, NewPointVec.data(), sizeof(double) * point_block_size);
    }
     */


    /*
    for (int i=0; i<poses.size(); i++)
    {
        VertexSophus* pCamera = dynamic_cast<VertexSophus*>(optimizer.vertex(i));
        Sophus::SE3 NewCameraVec = pCamera->estimate();
        poses[i] = NewCameraVec;
    }

    for (int i=0; i<points.size(); i++)
    {
        g2o::VertexSBAPointXYZ* pPoint = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(7+i));
        Eigen::Vector3d NewPointVec = pPoint->estimate();
        points[i] = NewPointVec;
    }
	*/
    // END YOUR CODE HERE

    // plot the optimized points and poses
    Draw(poses, points);

    // delete color data
    for (auto &c: color) delete[] c;
    return 0;
} // end of main()


// use pangolin to draw camera poses and points.
// input:
//   poses:  a set of poses (T_cw : store in quaternion format: tx,ty,tz,qx,qy,qz,qw)
//   points: a set of 3d points (w.r.t. world coordinate)
void Draw(const VecSE3 &poses, const VecVec3d &points) {
    if (poses.empty() || points.empty()) {
        cerr << "parameter is empty!" << endl;
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
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1; // 0.1 scale in z direction
        int width = 640, height = 480;
        for (auto &Tcw: poses) {
            glPushMatrix();
            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>(); // Twc = Tcw.inverse() is suitable in plotting poses
            glMultMatrixf((GLfloat *) m.data());
            glColor3f(1, 0, 0); // draw the camera pose in red color
            glLineWidth(2);
            glBegin(GL_LINES); // start the drawing on lines for camera pose
            // totally, draw 8 lines for the camera, need 16 vertex
            glVertex3f(0, 0, 0); // the camera center
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz); // see book in P. 102, eq. (5.17')
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz); // finish 4 bevel edges

            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz); // finish 4 base edges
            glEnd(); // finsih this drawing
            glPopMatrix();
        }

        // draw points
        glPointSize(2);
        glBegin(GL_POINTS); // start the drawing on points
        for (size_t i = 0; i < points.size(); i++) {
            glColor3f(0.0, points[i][2]/4, 1.0-points[i][2]/4); // choose points' color, scale based on its z value (far: green, close: blue)
            													// assume the z value will not exceed 4m.
            glVertex3d(points[i][0], points[i][1], points[i][2]);
        }
        glEnd(); // finsih this drawing

        pangolin::FinishFrame(); // finish the entire drawing
        usleep(5000);   // sleep 5 ms
    }
}

