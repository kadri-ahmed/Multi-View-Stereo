//
// Created by Ahmed Kadri on 22.01.24.
//

#ifndef SENSOR_FUSION_SFRECTIFICATION_H
#define SENSOR_FUSION_SFRECTIFICATION_H

#include "../StereoImage/StereoImage.h"
#include <opencv2/core.hpp>

namespace SensorFusion {

    class Rectification {
    public:

      static void rectifyImagePair(StereoImage& stereoImage, cv::Mat R, cv::Mat t, cv::Mat P_l, cv::Mat P_r);
      static cv::Mat drawlines(cv::Mat img1, cv::Mat img2, cv::Mat lines, std::vector<cv::Point2f> pts1, std::vector<cv::Point2f> pts2);

    private:

      /**
       * @brief Compute the joint undistortion and rectification transformation
       * and saved them in the mapX, mapY for remap
       * @param intrinsics
       * @param distCoeffs
       * @param size
       * @param newOptimalIntrinsics
       * @param map_x
       * @param map_y
       */
      static void initUndistortRectifyMap(
            const cv::Mat& intrinsics,
            const cv::Mat& distCoeffs,
            const cv::Mat& R,
            const cv::Size& size,
            cv::Mat& newOptimalIntrinsics,
            int m1type,
            cv::Mat& map_x ,
            cv::Mat& map_y
        );

        static void remap(
            const cv::Mat& srcImage,
            cv::Mat& dstImage,
            const cv::Mat& map_x,
            const cv::Mat& map_y,
            bool interpolate = true
        );

        static cv::Mat getOptimalNewCameraMatrix(
            const cv::Mat& intrinsicsMatrix,
            const cv::Mat& distCoeffs,
            const cv::Size& imageSize,
            float alpha,
            const cv::Size& newImageSize,
            cv::Rect& ROI
        );
    };

}

#endif // SENSOR_FUSION_SFRECTIFICATION_H
