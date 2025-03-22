#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../StereoImage/StereoImage.h"
#include "StereoRectification.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/matx.hpp>

namespace SensorFusion {
    void StereoRectification::initUndistortRectifyMap(
        const cv::Mat& intrinsics,
        const cv::Mat& distCoeffs,
        const cv::Size& size,
        const cv::Mat& newOptimalIntrinsics,
        cv::Mat& map_x,
        cv::Mat& map_y
    ){
        double cx = intrinsics.at<double>(0,2);
        double cy = intrinsics.at<double>(1,2);
        double fx = intrinsics.at<double>(0,0);
        double fy = intrinsics.at<double>(1,1);

        double k1 = distCoeffs.at<double>(0,0);
        double k2 = distCoeffs.at<double>(0,1);
        double p1 = distCoeffs.at<double>(0,2);
        double p2 = distCoeffs.at<double>(0,3);
        double k3 = distCoeffs.at<double>(0,4);

        map_x = cv::Mat::zeros(size.height, size.width, CV_32FC1);
        map_y = cv::Mat::zeros(size.height, size.width, CV_32FC1);

        for(int v{0}; v < size.height; v++){
            for(int u{0}; u < size.width; u++){
                double x = (u - cx) / fx;
                double y = (v - cy) / fy;
                double r2 = x*x + y*y;

                // u_radial + u_tangential
                double x_u = x * (1 + k1*pow(r2,1) + k2*pow(r2,2) + k3*pow(r2,3)) + 2*p1*x*y + p2*(r2 + 2*x*x);
                double y_u = y * (1 + k1*pow(r2,1) + k2*pow(r2,2) + k3*pow(r2,3)) + 2*p2*x*y + p1*(r2 + 2*y*y);

                map_x.at<float> (v, u) = static_cast<float>(x_u * fx + cx);
                map_y.at<float> (v, u) = static_cast<float>(y_u * fy + cy);
            }
        }
    }

    void StereoRectification::remap (
        const cv::Mat& srcImage,
        cv::Mat& dstImage,
        const cv::Mat& map_x,
        const cv::Mat& map_y,
        bool interpolate
    ){
        for(int v{0}; v < srcImage.rows; v++){
            for(int u{0}; u < srcImage.cols; u++){
                int x = map_x.at<float>(v, u);
                int y = map_y.at<float>(v, u);

                // Remap
                if (x >= 0 && x < dstImage.cols && y >= 0 && y < dstImage.rows) {

                    if(interpolate) {
                        // Bilinear interpolation
                        int x0 = floor(map_x.at<float>(v, u));
                        int x1 = x0+1 >= srcImage.cols ? srcImage.cols : x0+1 < 0 ? 0 : x0+1 ;

                        int y0 = floor(map_y.at<float>(v, u));
                        int y1 = y0+1 >= srcImage.rows ? srcImage.rows : y0+1 < 0 ? 0 : y0+1 ;

                        // Add weighted pixels and
                        // return wa*Ia + wb*Ib + wc*Ic + wd*Id

                        auto Ia = srcImage.at<cv::Vec3b>(y0, x0);
                        auto Ib = srcImage.at<cv::Vec3b>(y1, x0);
                        auto Ic = srcImage.at<cv::Vec3b>(y0, x1);
                        auto Id = srcImage.at<cv::Vec3b>(y1, x1);

                        auto wa = (x1-x) * (y1-y);
                        auto wb = (x1-x) * (y-y0);
                        auto wc = (x-x0) * (y1-y);
                        auto wd = (x-x0) * (y-y0);

                        // cv::Vec3b pixel = srcImage.at<cv::Vec3b>(y, x);
                        dstImage.at<cv::Vec3b> (v, u) = wa*Ia + wb*Ib + wc*Ic + wd*Id;
                    }
                    else {
                      dstImage.at<cv::Vec3b> (v, u) = srcImage.at<cv::Vec3b>(y, x);
                    }
                }
            }
        }
    }

    cv::Mat StereoRectification::undistort(
            const cv::Mat& image,
            const cv::Mat& intrinsics,
            const cv::Mat& distCoeffs,
            bool interpolate
    ){
        cv::Size size = image.size(); // Assuming resolution is defined
        cv::Mat uImage, map_x, map_y;
        uImage.create( image.size(), image.type() );

        // My implementation
        initUndistortRectifyMap(intrinsics, distCoeffs, size, intrinsics, map_x, map_y);
        remap(image, uImage, map_x, map_y, interpolate);

        // hconcat(uImage, image, uImage);
        // cv::imshow("Undistorted [L] --------- Original[R]", uImage);
        // cv::waitKey();
        return uImage;
    }

    cv::Mat StereoRectification::computeRectificationRotation(cv::Mat epipole) {

        double Tx = epipole.at<double>(0);
        double Ty = epipole.at<double>(1);
        cv::Mat r1 = epipole / cv::norm(epipole);
        cv::Mat r2 = (cv::Mat_<double>(3,1) << -Ty, Tx, 1) / sqrt(Tx*Tx + Ty*Ty);
        cv::Mat r3 = r1.cross(r2);
        std::vector<cv::Mat> rs = {r1.t(), r2.t(), r3.t()};
        cv::Mat R_rect;
        cv::vconcat(rs, R_rect);
        return R_rect;
    }

    void StereoRectification::applyRectification(const StereoImage& stereoImage, cv::Mat R_rect) {

        cv::Mat H_1 = stereoImage.get_intrinsic_left() * R_rect;
        cv::Mat H_2 = stereoImage.get_intrinsic_right() ;//* stereoImage.get_relative_cam_transformation().first * R_rect;

        cv::Mat rectified_image_left, rectified_image_right;
        cv::Size img_size = stereoImage.get_left_image().size();
        auto img_type = stereoImage.get_left_image().type();

        rectified_image_left.create( img_size, img_type);
        rectified_image_right.create(img_size, img_type);
        auto image_left = stereoImage.get_left_image();
        auto image_right = stereoImage.get_right_image();

        for(int v{0}; v < img_size.height; v++){
            for(int u{0}; u < img_size.width; u++) {

                cv::Mat left_pt = (cv::Mat_<double>(3,1) << u, v, 1);
                cv::Mat right_pt = (cv::Mat_<double>(3,1) << u, v, 1);



    //            cv::Mat pixel = (cv::Mat_<double>(3,1) << u, v, 1);
    //            pixel = H_1 * pixel;
    //            int u_new = pixel.at<double>(0);
    //            int v_new = pixel.at<double>(1);
    //
    //            if (u_new >= 0 && u_new < img_size.width && v_new >= 0 && v_new < img_size.height) {
    //                rectified_image_left.at<cv::Vec3d>(v_new, u_new) = image_left.at<cv::Vec3d>(v, u);
    //            }
            }
        }
        // cv::imshow("Undistorted [L] --------- Original[R]", rectified_image_left);
        // cv::waitKey();
    }

}


