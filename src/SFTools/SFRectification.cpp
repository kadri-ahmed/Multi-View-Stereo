//
// Created by Ahmed Kadri on 22.01.24.
//
#include "SFRectification.h"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

void SensorFusion::Rectification::initUndistortRectifyMap(
    const cv::Mat &intrinsics,
    const cv::Mat &distCoeffs,
    const cv::Mat& R,
    const cv::Size &size,
    cv::Mat &newOptimalIntrinsics,
    int m1type,
    cv::Mat &map_x, cv::Mat &map_y)
{

  if(newOptimalIntrinsics.cols == 4) {
      // cv::Mat homographyMatrix = projectionMatrix(cv::Rect(0, 0, 3, 3));
      newOptimalIntrinsics = newOptimalIntrinsics(cv::Rect(0,0,3,3));
  }
  double cx = newOptimalIntrinsics.at<double>(0,2);
  double cy = newOptimalIntrinsics.at<double>(1,2);
  double fx = newOptimalIntrinsics.at<double>(0,0);
  double fy = newOptimalIntrinsics.at<double>(1,1);

  double k1 = distCoeffs.at<double>(0,0);
  double k2 = distCoeffs.at<double>(0,1);
  double p1 = distCoeffs.at<double>(0,2);
  double p2 = distCoeffs.at<double>(0,3);
  double k3 = distCoeffs.at<double>(0,4);

  map_x = cv::Mat::zeros(size.height, size.width, m1type);
  map_y = cv::Mat::zeros(size.height, size.width, m1type);

  cv::Mat invR = R.inv();

  for(int v{0}; v < size.height; v++){
    for(int u{0}; u < size.width; u++){
      double x = (u - cx) / fx;
      double y = (v - cy) / fy;

      cv::Vec3f X_imagePlane( x, y, 1);
      cv::Mat X_camCoords = invR * X_imagePlane;
      x = X_camCoords.at<float>(0,0) / X_camCoords.at<float>(2,0);
      y = X_camCoords.at<float>(1,0) / X_camCoords.at<float>(2,0);
      double r2 = x*x + y*y;

      // u_radial + u_tangential
      double x_u = x * (1 + k1*pow(r2,1) + k2*pow(r2,2) + k3*pow(r2,3)) + 2*p1*x*y + p2*(r2 + 2*x*x);
      double y_u = y * (1 + k1*pow(r2,1) + k2*pow(r2,2) + k3*pow(r2,3)) + 2*p2*x*y + p1*(r2 + 2*y*y);

      map_x.at<float> (v, u) = static_cast<float>(x_u * intrinsics.at<double>(0,0) + cx);
      map_y.at<float> (v, u) = static_cast<float>(y_u * intrinsics.at<double>(1,1) + cy);
    }
  }
}


void SensorFusion::Rectification::remap(
    const cv::Mat &srcImage,
    cv::Mat &dstImage,
    const cv::Mat &map_x,
    const cv::Mat &map_y,
    bool interpolate)
{
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

void SensorFusion::Rectification::rectifyImagePair(StereoImage &stereoImage, cv::Mat R, cv::Mat t, cv::Mat P_l, cv::Mat P_r) {
    //////////////////////////////////////////////////////////
    //// Get rectification mapX and mapY
    //////////////////////////////////////////////////////////
    cv::Mat mapX_l, mapY_l, mapX_r, mapY_r;

    initUndistortRectifyMap(
        stereoImage.get_intrinsic_left(),
        stereoImage.get_distCoeffs_left(),
            cv::Mat::eye(3,3, CV_32FC1),
        stereoImage.get_left_image().size(),
        P_l,
        CV_32FC1,
        mapX_l,
        mapY_l
    );

    initUndistortRectifyMap(
            stereoImage.get_intrinsic_right(),
            stereoImage.get_distCoeffs_right(),
            cv::Mat::eye(3,3, CV_32FC1),
            stereoImage.get_right_image().size(),
            P_r,
            CV_32FC1,
            mapX_r,
            mapY_r
    );

    //////////////////////////////////////////////////////////
    //// Apply remap
    //////////////////////////////////////////////////////////
    cv::Mat leftImageRectified, rightImageRectified;
    leftImageRectified.create(stereoImage.get_left_image().size(), stereoImage.get_left_image().type());
    rightImageRectified.create(stereoImage.get_right_image().size(), stereoImage.get_right_image().type());

    remap(
            stereoImage.get_left_image(),
            leftImageRectified,
            mapX_l,
            mapY_l
    );

    remap(
            stereoImage.get_right_image(),
            rightImageRectified,
            mapX_r,
            mapY_r
    );

    stereoImage.set_left_image(leftImageRectified);
    stereoImage.set_right_image(rightImageRectified);
}
cv::Mat SensorFusion::Rectification::drawlines(cv::Mat img1, cv::Mat img2, cv::Mat lines, std::vector<cv::Point2f> pts1, std::vector<cv::Point2f> pts2) {
    // Convert the images to color if they are not already
    if (img1.channels() == 1)
        cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);
    if (img2.channels() == 1)
        cv::cvtColor(img2, img2, cv::COLOR_GRAY2BGR);

    // Draw the epilines and the corresponding points
    for (int r = 0; r < lines.rows; r++)
    {
        cv::RNG rng;
        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));

        float a = lines.at<float>(r,0);
        float b = lines.at<float>(r,1);
        float c = lines.at<float>(r,2);

        // Find two points on the epiline to define the line
        cv::Point pt1(0,-c/b), pt2(img1.cols, -(c+a*img1.cols)/b);

        // Draw the line between the points
        cv::line(img2, pt1, pt2, color, 1);

        // Draw the points on the images
        cv::circle(img1, pts1[r], 5, color, -1);
        cv::circle(img2, pts2[r], 5, color, -1);
    }

    return img1, img2;
}
