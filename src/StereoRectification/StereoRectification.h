#ifndef STEREO_RECTIFICATION_H
#define STEREO_RECTIFICATION_H

#include <opencv2/features2d.hpp>

#include "../StereoImage/StereoImage.h"
#include "../SparseMatching/SparseMatching.h"

namespace SensorFusion {
    class StereoRectification final
    {
    public:
        StereoRectification() = default;

        static cv::Mat undistort(
            const cv::Mat& image,
            const cv::Mat& intrinsics,
            const cv::Mat& distCoeffs,
            bool interpolate = false
        );

        /**
         *
         * @param epipole - translation vector between the two camera centers
         * @return rectification rotation R_rect
         */
        static cv::Mat computeRectificationRotation(cv::Mat epipole);

        static void applyRectification(const StereoImage& stereoImage, cv::Mat R_rect);


    private:
        static void initUndistortRectifyMap(
            const cv::Mat& intrinsics,
            const cv::Mat& distCoeffs,
            const cv::Size& size,
            const cv::Mat& newOptimalIntrinsics,
            cv::Mat& map_x ,
            cv::Mat& map_y
        );

        static void remap(
            const cv::Mat& srcImage,
            cv::Mat& dstImage,
            const cv::Mat& map_x,
            const cv::Mat& map_y,
            bool interpolate = false
        );

    };
}


#endif