//
// Created by Ahmed Kadri on 21.01.24.
//

#ifndef SENSOR_FUSION_OPENCVHELPERS_H
#define SENSOR_FUSION_OPENCVHELPERS_H

#include <opencv2/core.hpp>

#include "../SFTools/SFSparseMatching.h"
#include "../StereoImage/StereoImage.h"

class OpenCVHelpers {
public:
    static std::pair<ImageKeypoints<cv::Point2f>, ImageKeypoints<cv::Point2f>> computeKeyPointMatches(
            const StereoImage& stereoImage,
            bool show = false);

    static std::tuple<cv::Mat,cv::Mat,cv::Mat> findFundamentAndEssentialMatrices(
            ImageKeypoints<cv::Point2f> pointsL,
            ImageKeypoints<cv::Point2f> pointsR,
            const StereoImage& stereoImage
    );

    static void rectifyImagePair(StereoImage& stereoImage);
};


#endif //SENSOR_FUSION_OPENCVHELPERS_H
