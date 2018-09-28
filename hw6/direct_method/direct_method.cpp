#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// DEFINE GLOBAL VARIABLES
// Camera intrinsics
// 内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// 基线
double baseline = 0.573;
// paths
string left_file = "./left.png";
string disparity_file = "./disparity.png";
//boost::format fmt_others("./%06d.png");    // use boost::format to read images in format
boost::format fmt_others("/data/kitti-dataset/kitti_vo_grayscale_dataset/sequences/00/image_0/%06d.png"); // total 4541 images

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
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0); // read in gray-scale (0-255)
    cv::Mat disparity_img = cv::imread(disparity_file, 0); // read in gray-scale (0-255)

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 1000; // randomly choose 1000 points, original points: 1241x376=466616, selected points ratio = 0.2 %
    int boarder = 40; // border is 40 pixels
    VecVector2d pixels_ref; // pixels for comparison in reference image
    vector<double> depth_ref; // depth values in reference image

    // we can also set to pick pixels with sufficient image gradient, known as semi-dense

    // randomly sample pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x); // disparity only consumes several pixels, so use int type
        double depth = fx * baseline / disparity; // you know this is disparity to depth. In book , P.91 eqt(5.16)
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3 T_cur_ref;
    Sophus::SE3 T_ref_cur; // used to store poses

    // the initialized value of T_cur_ref is eyes(4)
    cout << "the initialized T_cur_ref: \n" <<  T_cur_ref.matrix() << endl;
    //for (int i = 1; i < 6; i++) {  // 1~10
    for (int i = 1; i <= 10; i++) { // 1~10
        cout << "Reading: " << (fmt_others % i).str() << "\n" << endl;
        cv::Mat img = cv::imread((fmt_others % i).str(), 0); // read in gray-scale
        // first you need to test single layer
        //DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);    // left_img is reference img, hence "000000.png",
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref); // for mult-layer (image pyramids)
        // print estimated poses:
        T_ref_cur = T_cur_ref.inverse();
        cout << "******* estimated poses w.r.t. reference frame(in 4x4 matrix): \n" << T_ref_cur.matrix() << endl;
        // save tracked poses T_ref_cur (quaternion format)
        // write tracked poses output to file : tracked_poses_result.txt
        std::ofstream myfile;
        myfile.open ("tracked_poses_result.txt", std::ofstream::app);
        myfile << std::setprecision(15);

        myfile << T_ref_cur.translation().transpose() << " "
               << T_ref_cur.so3().unit_quaternion().x() << " "
               << T_ref_cur.so3().unit_quaternion().y() << " "
               << T_ref_cur.so3().unit_quaternion().z() << " "
               << T_ref_cur.so3().unit_quaternion().w() << "\n";

        myfile.close();

    }

    // From tracked_poses_result.txt, draw the poses in pangolin

}

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21
) {

    // parameters
    int half_patch_size = 4;
    int iterations = 100;

    double cost = 0, lastCost = 0;
    int nGood = 0;  // good projections // 以 good 标记出投影在内部的点, 有多少個能投影在內部
    VecVector2d goodProjection; // store projections which does not run out of boundary, in current image

    for (int iter = 0; iter < iterations; iter++) {
        nGood = 0;
        goodProjection.clear();

        // Define Hessian and bias
        Matrix6d H = Matrix6d::Zero();  // 6x6 Hessian
        Vector6d b = Vector6d::Zero();  // 6x1 bias

        for (size_t i = 0; i < px_ref.size(); i++) { // px_ref: pixel location in reference image, loop through each pixel

            // compute the projection in the second image
            // TODO START YOUR CODE HERE
            float u = 0, v = 0; // pixel location in current location
            double Xr = 0, Yr = 0, Zr = 0; // 3D position in reference image
            Eigen::Vector4d Pc; // 3D point in current camera frame, in homogenenous coordinate (4-by-1 dim)
            Xr = (px_ref[i][0] - cx)*depth_ref[i]/fx; // In book P.102 , eqt
            Yr = (px_ref[i][1] - cy)*depth_ref[i]/fy;
            Zr = depth_ref[i];
            Pc = T21.matrix() * (Eigen::Vector4d(Xr, Yr, Zr, 1)); // T21.matrix() is 4-by-4 transformation matrix
            u = fx * Pc[0] / Pc[2] + cx;
            v = fy * Pc[1] / Pc[2] + cy;
            // cout<< "u,v : " << u << ", " << v << endl;

            //if ( u<half_patch_size || u>=img2.cols-half_patch_size || v<half_patch_size || v>=img2.rows-half_patch_size)
            //if ( u<0 || u>=img2.cols || v<0|| v>=img2.rows)
            // 判斷u,v是否跑出邊界
            if ( u<half_patch_size || u>=img2.cols-half_patch_size || v<half_patch_size || v>=img2.rows-half_patch_size)
                continue; // skip it
            nGood++; // 沒有跑出邊界, 記下累加
            goodProjection.push_back(Eigen::Vector2d(u, v));
            
            // and compute error and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++) // loop through small patch (4x4)
                for (int y = -half_patch_size; y < half_patch_size; y++) {

                    double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) - GetPixelValue(img2, u+x, v+y);

                    Matrix26d J_pixel_xi;   // pixel to \xi in Lie algebra  (hence in book P.195, eqt (8.15))
                    J_pixel_xi << fx/Pc[2], 0, -fx*Pc[0]/(Pc[2]*Pc[2]), -fx*Pc[0]*Pc[1]/(Pc[2]*Pc[2]),fx+fx*Pc[0]*Pc[0]/(Pc[2]*Pc[2]), -fx*Pc[1]/Pc[2],
                                  0, fy/Pc[2], -fy*Pc[1]/(Pc[2]*Pc[2]), -fy-fy*Pc[1]*Pc[1]/(Pc[2]*Pc[2]), fy*Pc[0]*Pc[1]/(Pc[2]*Pc[2]), fy*Pc[0]/Pc[2];

                    Eigen::Vector2d J_img_pixel;    // image gradients (dI2/du)
                    J_img_pixel[0] = (GetPixelValue(img2, u+x+1, v+y) - GetPixelValue(img2, u+x-1, v+y)) / 2;
                    J_img_pixel[1] = (GetPixelValue(img2, u+x, v+y+1) - GetPixelValue(img2, u+x, v+y-1)) / 2;

                    // total jacobian
                    Vector6d J = -J_img_pixel.transpose() * J_pixel_xi;    // J is 1x6 dim

                    // Compute H, b and set cost;

                    H += J * J.transpose(); // however, J still in 6x1 dim, thus H is 6x6 dim
                    b += -error * J; // b is 6x1 dim
                    cost += error * error;
                }
                
            // END YOUR CODE HERE
        }

        // solve update and put it into estimation
        // TODO START YOUR CODE HERE
        Vector6d update;
        update = H.ldlt().solve(b); // obtain update increment
        T21 = Sophus::SE3::exp(update) * T21; // T21 is Lie group, left-multiply with increment transformation matrix (convert by exponential mapping)
        // END YOUR CODE HERE

        cost /= nGood; // average the cost

        if (isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        lastCost = cost;
        cout << "cost = " << cost << ", good = " << nGood << endl;
    }
    cout << "good projection: " << nGood << endl;
    cout << "T21 = \n" << T21.matrix() << endl;

    // in order to help you debug, we plot the projected pixels here
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
    //cv::imshow("reference", img1_show);
    //cv::imshow("current", img2_show);
    //cv::waitKey();
}

void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref, // pixel location in reference image
        const vector<double> depth_ref, // depth map in reference image, only use vector array to store them
        Sophus::SE3 &T21
) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125}; // level_0, level_1, level_2, level_3

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    // TODO START YOUR CODE HERE
    for (int i=0; i<pyramids; i++) 
    {
        cv::Mat tmp1, tmp2;
        cv::resize(img1, tmp1, cv::Size(img1.cols*scales[i], img1.rows*scales[i])); // use cv::resize() to create pyramid
        cv::resize(img2, tmp2, cv::Size(img2.cols*scales[i], img2.rows*scales[i]));

        pyr1.push_back(tmp1);
        pyr2.push_back(tmp2);
    }
    // END YOUR CODE HERE

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) { // from level_3(the toppest) to level_0
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px); // 相應地, keypoints 也需要乘上pyramid level 的scale
        }

        // TODO START YOUR CODE HERE
        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];  // 相應地, fx 也需要乘上pyramid level 的scale
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        // END YOUR CODE HERE
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21); // 每次執行時會以T21的當前值進行迭代
    }
    cout << "Perform MultiLayer approach: \n";
    cout << "T21 = \n" << T21.matrix() << endl;
}
