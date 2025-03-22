//
// Created by Ahmed Kadri on 21.01.24.
//

#ifndef SENSOR_FUSION_SFSPARSEMATCHING_H
#define SENSOR_FUSION_SFSPARSEMATCHING_H

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include "../StereoImage/StereoImage.h"
#include "../SparseMatching/SparseMatching.h"

template<class T> using ImageKeypoints = std::vector<T>;

namespace SensorFusion {

    struct NormalizedPoints {
        cv::Mat Tl;
        cv::Mat Tr;
        ImageKeypoints<cv::Point2d> normalizedPointsL;
        ImageKeypoints<cv::Point2d> normalizedPointsR;
    };

    class SparseMatching {
    public:
        /**
        * @brief Computes the keypoint matches between left image and right image pair
        * and stores them in:
        * @note points_1 -> Left Image
        * @note points_2 -> Right Image
        * @param stereo_image
        * @return points_1 in m_left_image, points_2 in m_right_image
        */
        static std::pair<ImageKeypoints<cv::Point2d>, ImageKeypoints<cv::Point2d>> computeKeyPointMatches(
                const StereoImage &stereoImage,
                bool show = false);

        /**
         *
         * @param pointsL
         * @param pointsR
         * @param stereoImage
         * @param confidence
         * @param threshold
         * @return F, E, inliers
         */
        static std::tuple<cv::Mat, cv::Mat, std::vector<size_t>> findFundamentAndEssentialMatrices(
                ImageKeypoints<cv::Point2d> pointsL,
                ImageKeypoints<cv::Point2d> pointsR,
                const StereoImage &stereoImage,
                float confidence = 0.99f,
                float threshold = 1.f);

        /**
         * @brief Compute left and right projection matrices
         * and store relative camera transformation (R,t)
         * from left to right in stereoImage object
         * @param essential_matrix
         * @param stereoImage
         * @param point_left
         * @param point_right
         * @return pair of projection matrices
         */
        static std::pair<cv::Mat, cv::Mat> computeProjections(
                const cv::Mat &essential_matrix,
                StereoImage &stereoImage,
                const cv::Point2d &point_left,
                const cv::Point2d &point_right);

    private:
        static SensorFusion::NormalizedPoints normalizePoints(
                ImageKeypoints<cv::Point2d> pointsL,
                ImageKeypoints<cv::Point2d> pointsR);

        static cv::Mat apply8PointMatchingAlgorithm(
                const ImageKeypoints<cv::Point2d> &pointsL,
                const ImageKeypoints<cv::Point2d> &pointsR);

        static int getRansacIterationNumber(
                const int &pointNumber,
                const int &inlierNumber,
                const int &sampleSize,
                const double &confidence);

        static cv::Mat evaluateNormalizationEffect(
                const ImageKeypoints<cv::Point2d> &pointsL,
                const ImageKeypoints<cv::Point2d> &pointsR,
                const NormalizedPoints normalizedPoints,
                const std::vector<size_t> &inliers);
    };
}

#endif //SENSOR_FUSION_SFSPARSEMATCHING_H
