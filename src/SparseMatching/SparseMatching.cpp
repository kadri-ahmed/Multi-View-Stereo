#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include "SparseMatching.h"
#include "../StereoImage/StereoImage.h"
#include "../ReconstructionUtils/ReconstructionUtils.h"

SparseMatcher::SparseMatcher() : m_feature_detector(cv::SIFT::create(SparseMatcher::MIN_HESSIAN))
{
}

SparseMatcher::SparseMatcher(const cv::Ptr<cv::Feature2D>& feature_detector) : m_feature_detector(feature_detector)
{
    // Note: const cv::Ptr marks the actual cv::Feature2D as constant (unlike std::unique_ptr)
}

/**
 * @brief Computes the keypoint matches between m_left_image and m_right_image image pair
 * and stores them in:
 * @note points_1 -> Left Image
 * @note points_2 -> Right Image
 * @param stereo_image
 * @return points_1 in m_left_image, points_2 in m_right_image, matches_pair
 */
StereoReconstruction::MatchedKeypoints SparseMatcher::computeKeyPointMatches(const StereoImage& stereo_image) const
{
    //-- Step 1: Detect the keypoints using SURF feature_detector, compute the descriptors
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    this->m_feature_detector->detectAndCompute( stereo_image.get_left_image(), cv::noArray(), keypoints_1, descriptors_1 );
    this->m_feature_detector->detectAndCompute( stereo_image.get_right_image(), cv::noArray(), keypoints_2, descriptors_2 );

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( descriptors_1, descriptors_2, knn_matches, 2 );

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // -- Draw matches
    cv::Mat img_matches;
    cv::drawMatches(
        stereo_image.get_left_image(),
        keypoints_1,
        stereo_image.get_right_image(),
        keypoints_2,
        good_matches,
        img_matches);
    cv::imshow("matched_keypoints",img_matches);
    cv::waitKey();

    std::vector<cv::KeyPoint> points_1, points_2;
    points_1.reserve(good_matches.size());
    points_2.reserve(good_matches.size());
    for (auto& match : good_matches)
    {
        points_1.push_back(keypoints_1[match.queryIdx]);
        points_2.push_back(keypoints_2[match.trainIdx]);
        match.queryIdx = points_1.end()- points_1.begin() - 1; // Index of last element
        match.trainIdx = points_2.end()- points_2.begin() - 1; // Index of last element
    }

    return StereoReconstruction::MatchedKeypoints{std::move(points_1), std::move(points_2), std::move(good_matches)};
}

/**
 *
 * @param matched_keypoints - matches keypoint pairs from m_left_image and m_right_image images
 * @param camera_matrix_left - intrinsics m_left_image
 * @param confidence
 * @param threshold
 * @return
 */
std::tuple<cv::Mat, cv::Mat, std::vector<size_t>> SparseMatcher::compute_fundamental_essential_inliers(
        const StereoReconstruction::MatchedKeypoints& matched_keypoints,
        const cv::Mat& camera_matrix_left,
        const cv::Mat& camera_matrix_right,
        const double& confidence,
        const double& threshold)
{
    const SparseMatcher::NormalizedPoints& normalized_points = SparseMatcher::normalize_points(matched_keypoints);

    // The so-far-the-best fundamental matrix
    cv::Mat best_fundamental_matrix;
    std::vector<size_t> best_inliers;

    // The number of correspondences
    const size_t point_number = matched_keypoints.keypoints_1.size();

    // Initializing the index pool from which the minimal samples are selected
    std::vector<size_t> index_pool(point_number);
    for (size_t i = 0; i < point_number; ++i)
        index_pool[i] = i;

    // The size of a minimal sample
    constexpr size_t sample_size = 8;
    // The minimal sample
    std::vector<size_t> mss(sample_size);

    size_t maximum_iterations = std::numeric_limits<int>::max(), // The maximum number of iterations set adaptively when a new best model is found
        iteration_limit = 5000, // A strict iteration limit which mustn't be exceeded
        iteration = 0; // The current iteration number

    std::vector<cv::Point2d> source_points(sample_size),
        destination_points(sample_size);

    while (iteration++ < MIN(iteration_limit, maximum_iterations))
    {

        for (auto sample_idx = 0; sample_idx < sample_size; ++sample_idx)
        {
            // Select a random index from the pool
            const size_t idx = round((rand() / (double)RAND_MAX) * (index_pool.size() - 1));
            mss[sample_idx] = index_pool[idx];
            index_pool.erase(index_pool.begin() + idx);

            // Put the selected correspondences into the point containers
            const size_t point_idx = mss[sample_idx];
            source_points[sample_idx] = normalized_points.normalized_points_1[point_idx];
            destination_points[sample_idx] = normalized_points.normalized_points_2[point_idx];
        }

        // Estimate fundamental matrix
        cv::Mat fundamental_matrix = fundamental_least_squares(source_points, destination_points);
        fundamental_matrix = normalized_points.T2.t() * fundamental_matrix * normalized_points.T1; // Denormalize the fundamental matrix

        // Count the inliers
        std::vector<size_t> inliers;
        for (int i = 0; i < matched_keypoints.keypoints_1.size(); ++i)
        {
            // Symmetric epipolar distance
            cv::Mat pt1 = (cv::Mat_<double>(3, 1) << matched_keypoints.keypoints_1[i].pt.x, matched_keypoints.keypoints_1[i].pt.y, 1);
            cv::Mat pt2 = (cv::Mat_<double>(3, 1) << matched_keypoints.keypoints_2[i].pt.x, matched_keypoints.keypoints_2[i].pt.y, 1);

            // Calculate the error
            cv::Mat lL = fundamental_matrix.t() * pt2;
            cv::Mat lR = fundamental_matrix * pt1;

            // Calculate the distance of point pt1 from lL
            const double
                & aL = lL.at<double>(0),
                & bL = lL.at<double>(1),
                & cL = lL.at<double>(2);

            double tL = abs(aL * matched_keypoints.keypoints_1[i].pt.x + bL * matched_keypoints.keypoints_1[i].pt.y + cL);
            double dL = sqrt(aL * aL + bL * bL);
            double distanceL = tL / dL;

            // Calculate the distance of point pt2 from lR
            const double
                & aR = lR.at<double>(0),
                & bR = lR.at<double>(1),
                & cR = lR.at<double>(2);

            double tR = abs(aR * matched_keypoints.keypoints_2[i].pt.x + bR * matched_keypoints.keypoints_2[i].pt.y + cR);
            double dR = sqrt(aR * aR + bR * bR);
            double distanceR = tR / dR;

            double dist = 0.5 * (distanceL + distanceR);

            if (dist < threshold)
                inliers.push_back(i);
        }

        // Update if the new model is better than the previous so-far-the-best.
        if (best_inliers.size() < inliers.size())
        {
            // Update the set of inliers
            best_inliers.swap(inliers);
            inliers.clear();
            inliers.resize(0);
            // Update the model parameters
            best_fundamental_matrix = fundamental_matrix;
            // Update the iteration number
            maximum_iterations = fundamental_ransac_iteration_number(point_number,
                best_inliers.size(),
                sample_size,
                confidence);
        }

        // Put back the selected points to the pool
        for (size_t i = 0; i < sample_size; ++i)
            index_pool.push_back(mss[i]);
    }
    best_fundamental_matrix = SparseMatcher::evaluate_normalization_effect(matched_keypoints, normalized_points, best_inliers);
    cv::Mat E = camera_matrix_left.t() * best_fundamental_matrix * camera_matrix_right;
    return {std::move(best_fundamental_matrix), std::move(E), std::move(best_inliers)};
}

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
std::pair<cv::Mat, cv::Mat> SparseMatcher::compute_projections(
        const cv::Mat &essential_matrix,
        StereoImage &stereoImage,
        const cv::Point2d &point_left,
        const cv::Point2d &point_right)
{
    // Calculate the projection matrix of the first camera
    cv::Mat projection_1 = stereoImage.get_intrinsic_left() * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat projection_2;

    // Calculate the projection matrix of the second camera
    // 1st step - Decompose the essential matrix
    cv::Mat rotation_1, rotation_2, translation;
    cv::SVD svd(essential_matrix, cv::SVD::FULL_UV);

    // It gives matrices U D Vt
    // U and V are rotation matrices, D is a scaling matrix
    if (cv::determinant(svd.u) < 0) {
        svd.u.col(2) *= -1;
    }
    if (cv::determinant(svd.vt) < 0) {
        svd.vt.row(2) *= -1;
    }

    cv::Mat w = (cv::Mat_<double>(3, 3) << 0, -1, 0,
                                           1, 0, 0,
                                           0, 0, 1);

    rotation_1 = svd.u * w * svd.vt;
    rotation_2 = svd.u * w.t() * svd.vt;
    translation = svd.u.col(2) / cv::norm(svd.u.col(2));

    // The possible solutions:
    // rotation1 with translation
    // rotation2 with translation
    // rotation1 with -translation
    // rotation2 with -translation
    std::vector<std::pair<cv::Mat, cv::Mat>> possible_transformations;
    possible_transformations.push_back(std::make_pair(rotation_1, translation));
    possible_transformations.push_back(std::make_pair(rotation_2, translation));
    possible_transformations.push_back(std::make_pair(rotation_1, -translation));
    possible_transformations.push_back(std::make_pair(rotation_2, -translation));

    /**
     * Choose the right rotation and translation based on
     * which projection matrix delivers the minimal
     * re-projection error
     */
//	std::vector <const cv::Mat*> Ps = { &P21, &P22, &P23, &P24 };
    double minDistance = std::numeric_limits<double>::max();
    for (const auto& [R, t] : possible_transformations) {
        const cv::Mat& P1 = projection_1;
        cv::Mat P2;
        cv::hconcat(R,t,P2);
        P2 = stereoImage.get_intrinsic_right() * P2;

        // Estimate the 3D coordinates of a point correspondance
        cv::Mat point3d (StereoReconstruction::linear_triangulation(
            P1,
            P2,
            point_left,
            point_right));

        point3d.push_back(1.0);
        cv::Mat projection1 =
            P1 * point3d;
        cv::Mat projection2 =
            P2 * point3d;

        if (projection1.at<double>(2) < 0 ||
            projection2.at<double>(2) < 0)
            continue;
        projection1 = projection1 / projection1.at<double>(2);
        projection2 = projection2 / projection2.at<double>(2);

        double dx1 = projection1.at<double>(0) - point_left.x;
        double dy1 = projection1.at<double>(1) - point_left.y;
        double squaredDist1 = dx1 * dx1 + dy1 * dy1;

        double dx2 = projection2.at<double>(0) - point_right.x;
        double dy2 = projection2.at<double>(1) - point_right.y;
        double squaredDist2 = dx2 * dx2 + dy2 * dy2;

        if (squaredDist1 + squaredDist2 < minDistance) {
            minDistance = squaredDist1 + squaredDist2;
            projection_2 = P2;
            //stereoImage.set_relative_cam_transformation(R, t);
        }
    }
    // std::cout << "projection_1 " << projection_1 <<std::endl;
    // std::cout << "projection_2 " << projection_2 <<std::endl;
    // std::cout << "rotation_1 " << rotation_1 <<std::endl;
    // std::cout << "rotation_2 " << rotation_2 <<std::endl;
    // std::cout << "translation " << translation <<std::endl;
    return {std::move(projection_1), std::move(projection_2)};;
}


//
//  Private Member Functions
//


cv::Mat SparseMatcher::evaluate_normalization_effect(const StereoReconstruction::MatchedKeypoints& matched_keypoints,
                                            const SparseMatcher::NormalizedPoints& normalized_points,
                                            const std::vector<size_t>& inliers)
{
    std::vector<cv::Point2d> source_inliers;
    std::vector<cv::Point2d> destination_inliers;
    std::vector<cv::Point2d> normalized_source_inliers;
    std::vector<cv::Point2d> normalized_destination_inliers;

    for (const auto& inlierIdx : inliers) {
        source_inliers.emplace_back(matched_keypoints.keypoints_1[inlierIdx].pt);
        destination_inliers.emplace_back(matched_keypoints.keypoints_2[inlierIdx].pt);
        normalized_source_inliers.emplace_back(normalized_points.normalized_points_1[inlierIdx]);
        normalized_destination_inliers.emplace_back(normalized_points.normalized_points_2[inlierIdx]);
    }

    // Estimate the fundamental matrix from the original points
    cv::Mat unnormalized_fundamental_matrix = fundamental_least_squares(source_inliers, destination_inliers);

    // Estimate the fundamental matrix from the normalized original points
    cv::Mat normalized_fundamental_matrix = fundamental_least_squares(normalized_source_inliers, normalized_destination_inliers);

    normalized_fundamental_matrix = normalized_points.T2.t() * normalized_fundamental_matrix * normalized_points.T1; // Denormalize the fundamental matrix

    // Calculating the error of unnormalized and normalized fundamental matrices
    double error1 = 0.0, error2 = 0.0;
    for (size_t i = 0; i < inliers.size(); ++i) {
        // Symmetric epipolar distance
        cv::Mat pt1 = (cv::Mat_<double>(3, 1) << source_inliers[i].x, source_inliers[i].y, 1);
        cv::Mat pt2 = (cv::Mat_<double>(3, 1) << destination_inliers[i].x, destination_inliers[i].y, 1);

        // Calculate the error
        cv::Mat lL = unnormalized_fundamental_matrix.t() * pt2;
        cv::Mat lR = unnormalized_fundamental_matrix * pt1;

        // Calculate the distance of point pt1 from lL
        const double
            & aL = lL.at<double>(0),
            & bL = lL.at<double>(1),
            & cL = lL.at<double>(2);

        double tL = abs(aL * source_inliers[i].x + bL * source_inliers[i].y + cL);
        double dL = sqrt(aL * aL + bL * bL);
        double distanceL = tL / dL;

        // Calculate the distance of point pt2 from lR
        const double
            & aR = lR.at<double>(0),
            & bR = lR.at<double>(1),
            & cR = lR.at<double>(2);

        double tR = abs(aR * destination_inliers[i].x + bR * destination_inliers[i].y + cR);
        double dR = sqrt(aR * aR + bR * bR);
        double distanceR = tR / dR;

        double dist = 0.5 * (distanceL + distanceR);
        error1 += dist;
    }

    for (size_t i = 0; i < inliers.size(); ++i) {
        // Symmetric epipolar distance
        cv::Mat pt1 = (cv::Mat_<double>(3, 1) << source_inliers[i].x, source_inliers[i].y, 1);
        cv::Mat pt2 = (cv::Mat_<double>(3, 1) << destination_inliers[i].x, destination_inliers[i].y, 1);

        // Calculate the error
        cv::Mat lL = normalized_fundamental_matrix.t() * pt2;
        cv::Mat lR = normalized_fundamental_matrix * pt1;

        // Calculate the distance of point pt1 from lL
        const double
            & aL = lL.at<double>(0),
            & bL = lL.at<double>(1),
            & cL = lL.at<double>(2);

        double tL = abs(aL * source_inliers[i].x + bL * source_inliers[i].y + cL);
        double dL = sqrt(aL * aL + bL * bL);
        double distanceL = tL / dL;

        // Calculate the distance of point pt2 from lR
        const double
            & aR = lR.at<double>(0),
            & bR = lR.at<double>(1),
            & cR = lR.at<double>(2);

        double tR = abs(aR * destination_inliers[i].x + bR * destination_inliers[i].y + cR);
        double dR = sqrt(aR * aR + bR * bR);
        double distanceR = tR / dR;

        double dist = 0.5 * (distanceL + distanceR);
        error2 += dist;
    }
    error1 = error1 / inliers.size();
    error2 = error2 / inliers.size();

    std::cout<<"Error of the unnormalized fundamental matrix is "<<error1<<" px."<<std::endl;
    std::cout<<"Error of the normalized fundamental matrix is "<<error2<<" px."<<std::endl;

    return normalized_fundamental_matrix;
}


/**
 * @brief Apply the 8 point matching algorithm and solve for F
 * @param points_1 - matched points from m_left_image camera
 * @param points_2 - matched points from m_right_image camera
 * @return F - fundamental matrix
 */
cv::Mat SparseMatcher::fundamental_least_squares(const std::vector<cv::Point2d> &points_1, const std::vector<cv::Point2d> &points_2)
{
    const size_t pointNumber = points_1.size();
    cv::Mat A(pointNumber, 9, CV_64F);

    for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
    {
        const double
            &x1 = points_1[pointIdx].x,
            &y1 = points_1[pointIdx].y,
            &x2 = points_2[pointIdx].x,
            &y2 = points_2[pointIdx].y;

        A.at<double>(pointIdx, 0) = x1 * x2;
        A.at<double>(pointIdx, 1) = x2 * y1;
        A.at<double>(pointIdx, 2) = x2;
        A.at<double>(pointIdx, 3) = y2 * x1;
        A.at<double>(pointIdx, 4) = y2 * y1;
        A.at<double>(pointIdx, 5) = y2;
        A.at<double>(pointIdx, 6) = x1;
        A.at<double>(pointIdx, 7) = y1;
        A.at<double>(pointIdx, 8) = 1;
    }

    cv::Mat evals, evecs;
    cv::Mat AtA = A.t() * A;
    cv::eigen(AtA, evals, evecs);

    cv::Mat x = evecs.row(evecs.rows - 1); // x = [f1 f2 f3 f4 f5 f6 f7 f8 f9]
    x = x.reshape(0, 3);
    return x;
}

int SparseMatcher::fundamental_ransac_iteration_number(const int& point_number,
                                                const int& inlier_number,
                                                const int& sample_size,
                                                const double& confidence)
{
    const double inlier_ratio = static_cast<float>(inlier_number) / point_number;

    static const double log1 = log(1.0 - confidence);
    const double log2 = log(1.0 - pow(inlier_ratio, sample_size));

    const int k = log1 / log2;
    if (k < 0)
        return INT_MAX;
    return k;
}

SparseMatcher::NormalizedPoints SparseMatcher::normalize_points(const StereoReconstruction::MatchedKeypoints& matched_keypoints)
{
    cv::Mat T1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat T2 = cv::Mat::eye(3, 3, CV_64F);
    std::vector<cv::Point2d> normalized_points_1;
    std::vector<cv::Point2d> normalized_points_2;

    const size_t pointNumber = matched_keypoints.keypoints_1.size();

    normalized_points_1.resize(pointNumber);
    normalized_points_2.resize(pointNumber);

    // Calculate the mass points
    cv::Point2f mass1(0, 0), mass2(0, 0);

    for (auto i = 0; i < pointNumber; ++i)
    {
        mass1 = mass1 + matched_keypoints.keypoints_1[i].pt;
        mass2 = mass2 + matched_keypoints.keypoints_2[i].pt;
    }
    mass1 = mass1 * (1.0 / pointNumber);
    mass2 = mass2 * (1.0 / pointNumber);

    // Translate the point clouds to origin
    for (auto i = 0; i < pointNumber; ++i)
    {
        normalized_points_1[i] = matched_keypoints.keypoints_1[i].pt - mass1;
        normalized_points_2[i] = matched_keypoints.keypoints_2[i].pt - mass2;
    }

    // Calculate the average distances of the points from the origin
    double avgDistance1 = 0.0,
        avgDistance2 = 0.0;
    for (auto i = 0; i < pointNumber; ++i)
    {
        avgDistance1 += cv::norm(normalized_points_1[i]);
        avgDistance2 += cv::norm(normalized_points_2[i]);
    }

    avgDistance1 /= pointNumber;
    avgDistance2 /= pointNumber;

    const double multiplier1 =
        sqrt(2) / avgDistance1;
    const double multiplier2 =
        sqrt(2) / avgDistance2;

    for (auto i = 0; i < pointNumber; ++i)
    {
        normalized_points_1[i] *= multiplier1;
        normalized_points_2[i] *= multiplier2;
    }

    T1.at<double>(0, 0) = multiplier1;
    T1.at<double>(1, 1) = multiplier1;
    T1.at<double>(0, 2) = -multiplier1 * mass1.x;
    T1.at<double>(1, 2) = -multiplier1 * mass1.y;

    T2.at<double>(0, 0) = multiplier2;
    T2.at<double>(1, 1) = multiplier2;
    T2.at<double>(0, 2) = -multiplier2 * mass2.x;
    T2.at<double>(1, 2) = -multiplier2 * mass2.y;

    return SparseMatcher::NormalizedPoints{std::move(T1), std::move(T2), std::move(normalized_points_1), std::move(normalized_points_2)};
}


