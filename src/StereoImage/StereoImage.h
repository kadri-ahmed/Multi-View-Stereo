#ifndef IMAGE_H
#define IMAGE_H
#include <unordered_set>
#include <filesystem>

#include <opencv2/core.hpp>

#include "../ReconstructionUtils/ReconstructionUtils.h"

namespace fs = std::filesystem;

struct Camera
{
    std::vector<double> matrix;
};

struct Calibration
{
    std::map<std::string, Camera> cameras;
    double doffs;
    double baseline;
    int width;
    int height;
    int ndisp;
    int vmin;
    int vmax;
};

class StereoImage
{
public:
    StereoImage(const std::string &path_left, const std::string &path_right, const std::string &intrinsic_path, const std::string &distCoeffs_path, const bool &read_GT = false);

    StereoImage(
        const std::filesystem::path &scenePath,
        const Calibration &calibration,
        const std::string &distCoeffs_path,
        const bool &read_GT = false);

    bool read_images_from_files(const std::string &path_left, const std::string &path_right);

    std::vector<std::pair<cv::Point3d, cv::Vec3b>> traingulate_matching_points(const cv::Mat &projection_left,
                                                                               const cv::Mat &projection_right,
                                                                               const std::vector<cv::Point2d> &points_left,
                                                                               const std::vector<cv::Point2d> &points_right) const;

    static Calibration parseCalibration(const std::string &filename);

    static void parseMatrix(std::istringstream &iss, std::vector<double> &matrix);

    static void printCalibration(const Calibration &calib);

    const cv::Mat &get_left_image() const;
    const cv::Mat &get_right_image() const;

    void set_left_image(cv::Mat &image);
    void set_right_image(cv::Mat &image);

    const cv::Mat &get_intrinsic_left() const;
    const cv::Mat &get_intrinsic_right() const;

    const cv::Mat &get_gt_disparity_left() const;
    const cv::Mat &get_gt_disparity_right() const;

    const cv::Mat &get_distCoeffs_left() const;
    const cv::Mat &get_distCoeffs_right() const;

    const std::pair<cv::Mat, cv::Mat> &get_relative_cam_transformation() const;
    void set_relative_cam_transformation(cv::Mat R, cv::Mat t);

    static void depth_points_to_file(const std::string &filepath, const std::vector<std::pair<cv::Point3d, cv::Vec3b>> &depth_points);

    cv::Mat calculate_depth(const cv::Mat &disparity, const int &max_depth) const;
    cv::Mat calculate_depth(const int &max_depth) const;

    cv::Mat calculate_filtered_disparity(const unsigned &block_size, const unsigned &max_pixel_shift) const;
    cv::Mat calculate_disparity_left(const unsigned &block_size, const unsigned &max_pixel_shift, const double &scaling_factor = 1) const;
    cv::Mat calculate_disparity_right(const unsigned &block_size, const unsigned &max_pixel_shift, const double &scaling_factor = 1) const;

    template <typename T>
    static double calculateMSE(const cv::Mat &mat1, const cv::Mat &mat2);

    static cv::Mat logical_AND(const cv::Mat &mat, const cv::Mat &reference);

    void depth_To_PT_cloud(const cv::Mat &gt_left_depth, const cv::Mat &image, std::string &ply_filename, bool write_mesh) const;

    cv::Mat draw_matched_keypoints(const StereoReconstruction::MatchedKeypoints &matched_points);

private:
    bool intrinsic_from_file(const std::string &path);
    bool distCoeffs_from_file(const std::string &path);

    // Members
    cv::Mat m_distCoeffs_left;
    cv::Mat m_distCoeffs_right;

    cv::Mat m_intrinsic_left;
    cv::Mat m_intrinsic_right;

    double doffs;
    double baseline;

    cv::Mat m_left_image;
    cv::Mat m_right_image;

    // relative transformation from left to right
    // pair = [R t]
    std::pair<cv::Mat, cv::Mat> m_relative_cam_transformation;

    bool m_hasGT{false};
    cv::Mat gt_disparity_left;
    cv::Mat gt_disparity_right;
};

#endif