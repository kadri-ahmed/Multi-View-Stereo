#ifndef RECONSTRUCTION_UTILS_H
#define RECONSTRUCTION_UTILS_H

#include <vector>
#include <opencv2/core.hpp>

#include "Eigen.h"

// Forward declarations for open3D
namespace open3d
{
    namespace geometry
    {
        class PointCloud;
    }
    namespace pipelines
    {
        namespace registration
        {
            class Feature;
        }
    }
}

namespace StereoReconstruction
{
    struct MatchedKeypoints
    {
        std::vector<cv::KeyPoint> keypoints_1;
        std::vector<cv::KeyPoint> keypoints_2;
        std::vector<cv::DMatch> matches;
    };

    struct Vertex
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // position stored as 4 floats (4th component is supposed to be 1.0)
        Vector4f position;
        // color stored as 4 unsigned char
        Vector4uc color;
    };

    cv::Mat load_middleburry_groundtruth(const std::string &path);

    cv::Point3d linear_triangulation(const cv::Mat &projection_left,
                                     const cv::Mat &projection_right,
                                     const cv::Point2d &point_left,
                                     const cv::Point2d &point_right);

    void visualize_disparity_map(const cv::Mat &disparity);

    bool WriteMesh(Vertex *vertices, unsigned int width, unsigned int height, const std::string &filename);
    float CalculateEdgeLength(const Vertex &v1, const Vertex &v2);

    cv::Mat float_depth_to_int_depth(const cv::Mat &float_depth);

    // O3D methods
    std::shared_ptr<open3d::geometry::PointCloud> depth_To_O3D_PT_cloud(const cv::Mat &depth, const cv::Mat &image, cv::Mat intrinsics, const double &max_depth = 5.0);

	std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::shared_ptr<open3d::pipelines::registration::Feature>> preprocess_pointcloud(const std::shared_ptr<open3d::geometry::PointCloud> &pcd, const double &voxel_size);
	std::shared_ptr<open3d::geometry::PointCloud> merge_pointclouds(const std::shared_ptr<open3d::geometry::PointCloud> &source, const std::shared_ptr<open3d::geometry::PointCloud> &target);
} // namespace Sparse_Matching

#endif