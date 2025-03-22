//
// Created by Ahmed Kadri on 21.01.24.
//
#include <iostream>

// #include <opencv2/sfm/fundamental.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "OpenCVHelpers.h"

std::pair<ImageKeypoints<cv::Point2f>, ImageKeypoints<cv::Point2f>> OpenCVHelpers::computeKeyPointMatches(const StereoImage& stereoImage, bool show) {

    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << "STEP 2.1: Computing keypoint matches... ⏳" <<std::endl;
    //-- Convert to grayscale for better feature detection
    cv::Mat leftImageG, rightImageG;
    cv::cvtColor(stereoImage.get_left_image(), leftImageG, cv::COLOR_RGB2GRAY);
    cv::cvtColor(stereoImage.get_right_image(), rightImageG, cv::COLOR_RGB2GRAY);


    auto detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keyPointsL, keyPointsR;
    cv::Mat descriptorsL, descriptorsR;
    detector->detectAndCompute(leftImageG,cv::noArray(), keyPointsL, descriptorsL);
    detector->detectAndCompute(rightImageG,cv::noArray(), keyPointsR, descriptorsR);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( descriptorsL, descriptorsR, knn_matches, 2 );

    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch> good_matches;
    for (auto & knn_match : knn_matches)
    {
        if (knn_match[0].distance < ratio_thresh * knn_match[1].distance)
        {
            good_matches.push_back(knn_match[0]);
        }
    }

    std::vector<cv::Point2f> pointsL, pointsR;
    pointsL.reserve(good_matches.size());
    pointsR.reserve(good_matches.size());
    for (auto& match : good_matches)
    {
        pointsL.push_back(keyPointsL[match.queryIdx].pt);
        pointsR.push_back(keyPointsR[match.trainIdx].pt);
    }

    // -- Draw matches
    if(show){
        cv::Mat img_matches;
        cv::drawMatches(
                stereoImage.get_left_image(),
                keyPointsL,
                stereoImage.get_right_image(),
                keyPointsR,
                good_matches,
                img_matches);
        cv::imshow("matched_keypoints",img_matches);
        cv::waitKey();
    }

    std::cout << "Computing keypoint matches completed! ✅" <<std::endl;

    return std::make_pair(pointsL, pointsR);
}

std::tuple<cv::Mat, cv::Mat, cv::Mat>OpenCVHelpers::findFundamentAndEssentialMatrices(ImageKeypoints<cv::Point2f> pointsL,
                                                                                     ImageKeypoints<cv::Point2f> pointsR,
                                                                                     const StereoImage &stereoImage)
{
    cv::Mat inliers;
    auto F = cv::findFundamentalMat(
            pointsL,
            pointsR,
            inliers);
    cv::Mat E;
    // cv::sfm::essentialFromFundamental(F, stereoImage.get_intrinsic_left(), stereoImage.get_intrinsic_right(), E);

    int firstInlierIdx = 0;
    while(firstInlierIdx < inliers.size().height){
      if(inliers.at<char>(firstInlierIdx) == 1)
        break;
      firstInlierIdx++;
    }
    auto xL = pointsL[inliers.at<char>(firstInlierIdx)];
    auto xR = pointsR[inliers.at<char>(firstInlierIdx)];
    cv::Mat pL = (cv::Mat_<double>(3,1) << xL.x, xL.y, 1);
    cv::Mat pR = (cv::Mat_<double>(3,1) << xR.x, xR.y, 1);

    std::cout << "[OpenCV] Checking Epipolar Constraint: \nx.T * F * x' = " << pL.t() * F * pR << std::endl;

    return std::make_tuple(F,E,inliers);
}

void OpenCVHelpers::rectifyImagePair(StereoImage &stereoImage) {
    cv::Rect ROI_L = cv::Rect();
    cv::Rect ROI_R = cv::Rect();

    auto K_l = cv::getOptimalNewCameraMatrix(
        stereoImage.get_intrinsic_left(),
        stereoImage.get_distCoeffs_left(),
        stereoImage.get_left_image().size(),
        1,
        stereoImage.get_left_image().size(),
        &ROI_L
    );

    auto K_r = cv::getOptimalNewCameraMatrix(
        stereoImage.get_intrinsic_left(),
        stereoImage.get_distCoeffs_left(),
        stereoImage.get_left_image().size(),
        1,
        stereoImage.get_left_image().size(),
        &ROI_R
    );

    cv::Mat mapX_l, mapY_l;
    cv::Mat uLeftImage;
    cv::Mat mapX_r, mapY_r;
    cv::Mat uRightImage;

    //-- Using RectifyMap & Remap

    cv::initUndistortRectifyMap(
        stereoImage.get_intrinsic_left(),
        stereoImage.get_distCoeffs_left(),
        cv::Mat::eye(3,3, CV_32FC1),
        K_l,
        stereoImage.get_left_image().size(),
        CV_32FC1,
        mapX_l,
        mapY_l
    );
    cv::remap(
        stereoImage.get_left_image(),
        uLeftImage,
        mapX_l,
        mapY_l,
        cv::INTER_LINEAR
    );

    cv::initUndistortRectifyMap(
        stereoImage.get_intrinsic_right(),
        stereoImage.get_distCoeffs_right(),
        cv::Mat::eye(3,3, CV_32FC1),
        K_r,
        stereoImage.get_right_image().size(),
        CV_32FC1,
        mapX_r,
        mapY_r
    );
    cv::remap(
        stereoImage.get_right_image(),
        uRightImage,
        mapX_r,
        mapY_r,
        cv::INTER_LINEAR
    );

    //-- Using undistort()
//            cv::undistort(
//                    stereoImage.get_left_image(),
//                    uLeftImage,
//                    stereoImage.get_intrinsic_left(),
//                    stereoImage.get_distCoeffs_left(),
//                    K_l
//            );
    //
    //        cv::undistort(
    //                stereo_image.get_right_image(),
    //                uRightImage,
    //                stereo_image.get_intrinsic_right(),
    //                stereo_image.get_distCoeffs_right(),
    //                K_l
    //        );

    uLeftImage = uLeftImage(ROI_L);
    uRightImage = uRightImage(ROI_L);

    stereoImage.set_left_image(uLeftImage);
    stereoImage.set_right_image(uRightImage);
}
