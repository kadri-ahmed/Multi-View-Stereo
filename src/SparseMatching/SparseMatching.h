#ifndef SPARSE_MATCHING_H
#define SPARSE_MATCHING_H
#include <opencv2/features2d.hpp>

#include "../StereoImage/StereoImage.h"
#include "../ReconstructionUtils/ReconstructionUtils.h"

class SparseMatcher final
{
public:
    SparseMatcher();
    SparseMatcher(const cv::Ptr<cv::Feature2D>& feature_detector);

    StereoReconstruction::MatchedKeypoints computeKeyPointMatches(const StereoImage& stereo_image) const;
    static std::tuple<cv::Mat, cv::Mat, std::vector<size_t>> compute_fundamental_essential_inliers(
                                                            const StereoReconstruction::MatchedKeypoints& matched_keypoints,
                                                            const cv::Mat& camera_matrix_left,
                                                            const cv::Mat& camera_matrix_right,
                                                            const double& confidence,
                                                            const double& threshold);

    static std::pair<cv::Mat, cv::Mat>
    compute_projections(
            const cv::Mat &essential_matrix,
            StereoImage &stereoImage,
            const cv::Point2d &point_left,
            const cv::Point2d &point_right);



private:

    struct NormalizedPoints
    {
        cv::Mat T1;
        cv::Mat T2;
        std::vector<cv::Point2d> normalized_points_1;
        std::vector<cv::Point2d> normalized_points_2;
    };

    static NormalizedPoints normalize_points(const StereoReconstruction::MatchedKeypoints& matched_keypoints);
    static int fundamental_ransac_iteration_number(const int& point_number,
                                                const int& inlier_number,
                                                const int& sample_size,
                                                const double& confidence);
    static cv::Mat fundamental_least_squares(const std::vector<cv::Point2d> &points_1, const std::vector<cv::Point2d> &points_2);
    static cv::Mat evaluate_normalization_effect(const StereoReconstruction::MatchedKeypoints& matched_keypoints,
                                                const NormalizedPoints& normalized_points,
                                                const std::vector<size_t>& inliers);

    static constexpr int MIN_HESSIAN = 1200;
    const cv::Ptr<cv::Feature2D> m_feature_detector;
};




#endif