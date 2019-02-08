#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> VecSE3;  // a set of poses

// DEFINE GLOBAL VARIABLES
// Camera intrinsics
// 内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// 基线
double baseline = 0.573;
// paths
string left_file = "./left.png";
string disparity_file = "./disparity.png";
boost::format fmt_others("./%06d.png");    // use boost::format to read images in format
//boost::format fmt_others("/data/kitti-dataset/kitti_vo_grayscale_dataset/sequences/00/image_0/%06d.png"); // total 4541 images

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// TODO implement this function
/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1, // reference image
        const cv::Mat &img2, // current image
        const VecVector2d &px_ref, // pixel location in reference image
        const vector<double> depth_ref, // depth map in reference image, only use vector array to store them
        Sophus::SE3 &T21 // transformation matrix (relative pose) from img1(ref) to img2(curr)
);

// TODO implement this function
/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1, // reference image
        const cv::Mat &img2, // current image
        const VecVector2d &px_ref, // pixel location in reference image
        const vector<double> depth_ref, // depth map in reference image, only use vector array to store them
        Sophus::SE3 &T21 // transformation matrix (relative pose) from img1(ref) to img2(curr)
);

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)]; // typecast to int position, get [0..255] int value
    float xx = x - floor(x); // get shift value xx, used for later interpolation
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    ); // return [0..255] float value
}

// mainly return sampled pixel 
void UniformlySamplePts(
        const cv::Mat &left_img,
        const cv::Mat &disparity_img,
        VecVector2d &pixels_ref,
        vector<double> &depth_ref
) {

    // parameters:
    int nPoints = 1000; 
    int border = 40;     

    cv::RNG rng;
    // randomly sample pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) { // select 1000 points uniformly sampled from the img
        int x = rng.uniform(border, left_img.cols - border);  // don't pick pixels close to boarder
        int y = rng.uniform(border, left_img.rows - border);  // don't pick pixels close to boarder
        // x, y => (u,v) are pixel coordinates sampled uniformly
        int disparity = disparity_img.at<uchar>(y, x); // disparity only consumes several pixels, so use int type
        // fx, baseline are global variable
        double depth = fx * baseline / disparity; // you know this is disparity to depth. In book , P.91 eqt(5.16)
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }
}

// use pangolin to draw camera poses and points.
// input:
//   poses:  a set of poses (T_cw : store in quaternion format: tx,ty,tz,qx,qy,qz,qw)
//   points: a set of 3d points (w.r.t. world coordinate)
void Draw(const VecSE3 &poses) {
    if (poses.empty()) {
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
        //int width = 640, height = 480;
        int width = 1241, height = 376;
        for (auto &Twc: poses) {
            glPushMatrix();
            Sophus::Matrix4f m = Twc.matrix().cast<float>(); // Twc is suitable in plotting poses
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


        pangolin::FinishFrame(); // finish the entire drawing
        usleep(5000);   // sleep 5 ms
    }
}


int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0); // read in gray-scale (0-255)
    cv::Mat disparity_img = cv::imread(disparity_file, 0); // read in gray-scale (0-255), as disparity only consumes several pixels.

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 1000; // randomly choose 1000 points, original points: 1241x376=466616, selected points ratio = 0.2 %
    int boarder = 40; // border is 40 pixels
    VecVector2d pixels_ref; // pixels for comparison in reference image
    vector<double> depth_ref; // depth values in reference image

    // we can also set to pick pixels with sufficient image gradient, known as semi-dense
    // sd_pts = get_semi_dense_points(rgb_img);

    // randomly sample pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) { // select 1000 points uniformly sampled from the img
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        // x, y => (u,v) are pixel coordinates sampled uniformly
        int disparity = disparity_img.at<uchar>(y, x); // disparity only consumes several pixels, so use int type
        double depth = fx * baseline / disparity; // you know this is disparity to depth. In book , P.91 eqt(5.16)
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // UniformlySamplePts(left_img, disparity_img, pixels_ref, depth_ref);

    // estimates 01~05.png's pose using this information
    Sophus::SE3 T_cur_ref;
    Sophus::SE3 T_ref_cur; // used to store poses

    // the initialized value of T_cur_ref is eyes(4)
    cout << "the initialized T_cur_ref: \n" <<  T_cur_ref.matrix() << endl;
    for (int i = 1; i < 6; i++) {  // for each image
    //for (int i = 1; i <= 10; i++) { // 1~10
        cout << "Reading: " << (fmt_others % i).str() << "\n" << endl;
        cv::Mat img = cv::imread((fmt_others % i).str(), 0); // read in gray-scale
        // first you need to test single layer, it will return T_current_ref (T21)
        //DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);   // left_img is reference img, hence "000000.png",
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref); // for mult-layer (image pyramids)
        // print estimated poses:
        T_ref_cur = T_cur_ref.inverse();
        cout << "******* estimated poses w.r.t. reference frame(in 4x4 matrix): \n" << T_ref_cur.matrix() << endl;
        // save tracked poses T_ref_cur (quaternion format)
        // write tracked poses output to file : tracked_poses_result.txt
        std::ofstream myfile;
        myfile.open ("tracked_poses_result.txt", std::ofstream::app);
        myfile << std::setprecision(15);

        myfile << T_ref_cur.translation().transpose() << " " // tx, ty, tz
               << T_ref_cur.so3().unit_quaternion().x() << " " // qx
               << T_ref_cur.so3().unit_quaternion().y() << " " // qy
               << T_ref_cur.so3().unit_quaternion().z() << " " // qz
               << T_ref_cur.so3().unit_quaternion().w() << "\n"; // qw

        myfile.close();

    }

    // From tracked_poses_result.txt, draw the poses in pangolin
    // read poses
    VecSE3 poses;
   
    ifstream fin("tracked_poses_result.txt");

    while (!fin.eof()) {

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
    Draw(poses);
}

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21 //T_cur_ref
) {

    // parameters
    int half_patch_size = 4; // a 8x8 patch is used to compare photometric error
    int iterations = 100; // the maximum number of iterations in updating T_cur_ref

    double cost = 0, lastCost = 0;
    int nGood = 0;  // good projections // 以 good 标记出投影在内部的点, 有多少個能投影在內部
    VecVector2d goodProjection; // store projections (x,y) which does not run out of boundary, in current image

    for (int iter = 0; iter < iterations; iter++) {
        nGood = 0; // reset nGood
        goodProjection.clear(); // store good warped points (u,v)

        // Define Hessian and bias
        Matrix6d H = Matrix6d::Zero();  // 6x6 Hessian
        Vector6d b = Vector6d::Zero();  // 6x1 bias

        for (size_t i = 0; i < px_ref.size(); i++) { // px_ref: sampled pixel locations (1000 pixels) in reference image, loop through each pixel
                                                     // for each pixel, if warped points is good (succeed in obtaining warped points), 
                                                     // compute phtometric error within a 8x8 patch.
            // compute the projection in the second image
            // TODO START YOUR CODE HERE
            float u = 0, v = 0; // pixel location in current location
            double Xr = 0, Yr = 0, Zr = 0; // 3D position in reference image
            Eigen::Vector4d Pc; // 3D point in current camera frame, in homogenenous coordinate (4-by-1 dim)
            Xr = (px_ref[i][0] - cx)*depth_ref[i]/fx; // In book P.102 , eqt
            Yr = (px_ref[i][1] - cy)*depth_ref[i]/fy;
            Zr = depth_ref[i]; // finish back-project to obtain 3D point in reference image
            Pc = T21.matrix() * (Eigen::Vector4d(Xr, Yr, Zr, 1)); // T21.matrix() is 4-by-4 transformation matrix
            u = fx * Pc[0] / Pc[2] + cx;
            v = fy * Pc[1] / Pc[2] + cy;
            // Obtain the warped pixel coordinates u,v (pixel in reference image warped into current image).
            // cout<< "u,v : " << u << ", " << v << endl;

            //if ( u<half_patch_size || u>=img2.cols-half_patch_size || v<half_patch_size || v>=img2.rows-half_patch_size)
            //if ( u<0 || u>=img2.cols || v<0|| v>=img2.rows)
            // 判斷u,v是否跑出邊界
            if ( u<half_patch_size || u>=img2.cols-half_patch_size || v<half_patch_size || v>=img2.rows-half_patch_size)
                continue; // skip it
            nGood++; // 沒有跑出邊界, 記下累加, hence good warped points
            goodProjection.push_back(Eigen::Vector2d(u, v)); // add good warped points into good warped points.
            
            // and compute error and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++) // loop through small patch (8x8): x_displace=[-4,3]
                for (int y = -half_patch_size; y < half_patch_size; y++) { // y_displace=[-4,3]
                    // I_ref - I_current   ,       GetPixelValue() returns [0..255] float value     
                    double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) - GetPixelValue(img2, u+x, v+y);
                    // a 2x6 jacobian
                    Matrix26d J_pixel_xi;   // pixel to \xi in Lie algebra  (hence in book P.195, eqt (8.15))
                    J_pixel_xi << fx/Pc[2], 0, -fx*Pc[0]/(Pc[2]*Pc[2]), -fx*Pc[0]*Pc[1]/(Pc[2]*Pc[2]),fx+fx*Pc[0]*Pc[0]/(Pc[2]*Pc[2]), -fx*Pc[1]/Pc[2],
                                  0, fy/Pc[2], -fy*Pc[1]/(Pc[2]*Pc[2]), -fy-fy*Pc[1]*Pc[1]/(Pc[2]*Pc[2]), fy*Pc[0]*Pc[1]/(Pc[2]*Pc[2]), fy*Pc[0]/Pc[2];

                    Eigen::Vector2d J_img_pixel;    // image gradients (dI2/du), img2 is current image
                    J_img_pixel[0] = (GetPixelValue(img2, u+x+1, v+y) - GetPixelValue(img2, u+x-1, v+y)) / 2;
                    J_img_pixel[1] = (GetPixelValue(img2, u+x, v+y+1) - GetPixelValue(img2, u+x, v+y-1)) / 2;

                    // total jacobian
                    Vector6d J = -J_img_pixel.transpose() * J_pixel_xi;    // J is 1x6 dim, but J is defined as 6x1 dim.

                    // Compute H, b and set cost;
                    // (1x6) x (6x1) => scalar
                    H += J * J.transpose(); // however, J still in 6x1 dim, thus H is 6x6 dim
                    b += -error * J; // error is scalar, J is 6x1 dim, thus b is 6x1 dim
                    cost += error * error; // cost is scalar
                } 
            // obtain hessian H, bias b, and cost for one patch surrounding the i-th sampled pixel in ref_image    
            // END YOUR CODE HERE
        } // obtain hessian H, bias b, and cost for all sampled pixels (1000) in ref_image at each iteration. Then perform update.

        // solve update and put it into estimation
        // TODO START YOUR CODE HERE
        Vector6d update;
        update = H.ldlt().solve(b); // obtain update increment
        // T21 (T_current_ref) is Sophus::SE3, update is 6x1 vector
        T21 = Sophus::SE3::exp(update) * T21; // T21 is Lie group, left-multiply with increment transformation matrix (convert by exponential mapping)
        // END YOUR CODE HERE

        cost /= nGood; // average the cost

        if (isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) { // considered as diverged, we could keep 5 delayed comparisons for better result
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        lastCost = cost;
        cout << "Finish one iteration, cost = " << cost << ", good = " << nGood << endl; // finish one iteration
    } // finish all iteratins 
    cout << "Finish all iterations, good projection: " << nGood << endl;
    cout << "Finish all iterations, T21 = \n" << T21.matrix() << endl;

    // in order to help you debug, we plot the projected pixels here
    /*
    cv::Mat img1_show, img2_show;
    cv::cvtColor(img1, img1_show, CV_GRAY2BGR);
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    for (auto &px: px_ref) {
        cv::rectangle(img1_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
                      cv::Scalar(0, 250, 0));
    }
    for (auto &px: goodProjection) {
        cv::rectangle(img2_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
                      cv::Scalar(0, 250, 0));
    }
    cout<< "reference resolution: " << img1_show.cols << " , " << img1_show.rows << endl;
    cout<< "current resolution: " << img2_show.cols << " , " <<  img2_show.rows << endl;
    */
    //cv::imshow("reference", img1_show);
    //cv::imshow("current", img2_show);
    //cv::waitKey();
} // return T21 (T_current_ref)

void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref, // sampled pixels location in reference image
        const vector<double> depth_ref, // depth map in reference image, only use vector array to store them
        Sophus::SE3 &T21
) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125}; // level_0, level_1, level_2, level_3

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids, pyr1 for reference image, pyr2 for current image
    // TODO START YOUR CODE HERE
    for (int i=0; i<pyramids; i++) 
    {
        cv::Mat tmp1, tmp2; // 1:ref,   2:current
        cv::resize(img1, tmp1, cv::Size(img1.cols*scales[i], img1.rows*scales[i])); // use cv::resize() to create pyramid
        cv::resize(img2, tmp2, cv::Size(img2.cols*scales[i], img2.rows*scales[i]));

        pyr1.push_back(tmp1);
        pyr2.push_back(tmp2);
    }
    // END YOUR CODE HERE, obtain pyr1 & pyr2, starting from level0 to level3. level0 is original size, level3 is the smallest one.

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) { // from level_3(the toppest) to level_0
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) { // loop through 1000 sampled pixels
            px_ref_pyr.push_back(scales[level] * px); // 相應地, keypoints 也需要乘上pyramid level 的scale e.g. 0.125 * [100,100] = [12.5, 12.5]
        }

        // TODO START YOUR CODE HERE
        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];  // 相應地, fx, fy, cx, cy 也需要乘上pyramid level 的scale
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        // END YOUR CODE HERE
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21); // 每次執行時會以T21的當前值進行迭代
    } 
    // finish all pyramid levels
    cout << "Perform MultiLayer approach: \n";
    cout << "T21 = \n" << T21.matrix() << endl;
}
