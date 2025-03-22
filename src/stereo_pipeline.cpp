#include <stdio.h>
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <filesystem>

#include <yaml-cpp/yaml.h>
#include <matplot/matplot.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "./StereoImage/StereoImage.h"
#include "./SparseMatching/SparseMatching.h"

#include <open3d/Open3D.h>

#include "./OpenCVTools/OpenCVHelpers.h"
#include "./SFTools/SFSparseMatching.h"
#include "./SFTools/SFRectification.h"


namespace SF = SensorFusion;

int main(int argc, char *argv[])
{
    const cv::String keys = "{help h usage ? |              | print this message   }"
                            "{path          |     ./       | path to config.yaml file }";

    cv::CommandLineParser parser(argc, argv, keys);
    std::string yaml_path{parser.get<std::string>("path")};

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    YAML::Node config;
    try
    {
        config = YAML::LoadFile(yaml_path);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
    bool do_undistort_rectify = config["apply_undistort_rectify"].as<bool>();
    bool do_pointcloud = config["generate_pointcloud"].as<bool>();
    bool do_mesh = config["generate_mesh"].as<bool>();
    bool use_GT = config["use_groundtruth"].as<bool>();

    int max_depth = config["max_depth"].as<int>();
    int max_disparity = config["max_disparity"].as<int>();
    int block_size = config["block_size"].as<int>();

    /////////////////////////////////////////////////////////////////////////////////////////////
    ///// STEP 1: Load Stereo-Images from given path
    /////////////////////////////////////////////////////////////////////////////////////////////
    std::unordered_map<std::string, StereoImage> image_map;
    std::cout << "Loading images..." << std::endl;
    fs::path dataset{config["dataset"].as<std::string>()};
    for (const auto &scenePath : fs::directory_iterator(dataset))
    {
        auto calibration_path = scenePath.path() / "calib.txt";
        if (exists(calibration_path))
        {
            auto calibration = StereoImage::parseCalibration(calibration_path.string());
            fs::path distortion_coeff_path;
            if (do_undistort_rectify)
            {
                distortion_coeff_path = std::string(distortion_coeff_path / "dist_coeffs.txt");
            }
            else
            {
                distortion_coeff_path = "";
            }
            image_map.emplace(scenePath.path()/"", StereoImage(scenePath, calibration, distortion_coeff_path, use_GT));
        }
        else
        {
            std::cout << "calib.txt wasn't found in the path: " << calibration_path << std::endl;
        }
    }
    std::cout << "Loaded " << image_map.size() << " stereo image pair(s)!" << std::endl;

    std::shared_ptr<open3d::geometry::PointCloud> combined_pointcloud;

    SparseMatcher sparse_matcher;
    for (auto &[path, stereo_image] : image_map)
    {
        if (do_undistort_rectify)
        {
            /////////////////////////////////////////////////////////////////////////////////////////////
            ///// STEP 2: Compute Sparse Keypoint Matching for each image
            /////////////////////////////////////////////////////////////////////////////////////////////

            const auto &[pointsL1, pointsR1] = OpenCVHelpers::computeKeyPointMatches(stereo_image);
            const auto &[pointsL2, pointsR2] = SF::SparseMatching::computeKeyPointMatches(stereo_image);

            /////////////////////////////////////////////////////////////////////////////////////////////
            ///// STEP 3.2: Compute Fundamental & Essential matrices
            /////////////////////////////////////////////////////////////////////////////////////////////
            // TODO: evaluate how small should the epipolar constraint be close to zero

            std::cout << "----------------------------------------------------------------------" << std::endl;
            std::cout << "STEP 2.2: Computing F and E matrices... ⏳" << std::endl;
            const auto &[F1, E1, inliers1] = OpenCVHelpers::findFundamentAndEssentialMatrices(pointsL1, pointsR1, stereo_image);
            const auto &[F2, E2, inliers2] = SF::SparseMatching::findFundamentAndEssentialMatrices(pointsL2, pointsR2, stereo_image);

            cv::Mat R_1, t_1;
            cv::recoverPose(
                E1,
                pointsL1,
                pointsR1,
                stereo_image.get_intrinsic_left(),
                R_1, t_1);

            const auto &[P_l, P_r] = SF::SparseMatching::computeProjections(E2, stereo_image, pointsL2.at(0), pointsR2.at(0));
            const auto &[R_2, t_2] = stereo_image.get_relative_cam_transformation();

            double errorR = cv::norm(R_1 - R_2);
            double errorT = cv::norm(t_1 - t_2);
            std::cout << "Error estimating R = " << errorR << std::endl;
            std::cout << "Error estimating t = " << errorT << std::endl;

            std::cout << "Computing F and E matrices completed! ✅" << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////////
            ///// STEP 3: Rectify Image Pairs
            /////////////////////////////////////////////////////////////////////////////////////////////

            std::cout << "----------------------------------------------------------------------" << std::endl;

            std::cout << "STEP 3: Rectifying Image Pairs... ⏳" << std::endl;

            StereoImage stereo_image_cv = stereo_image;

            OpenCVHelpers::rectifyImagePair(stereo_image_cv);
            std::cout << "Raw image shape: " << stereo_image.get_left_image().size << std::endl;
            SF::Rectification::rectifyImagePair(stereo_image, R_2, t_2, P_l, P_r);
            std::cout << "Rectified image shape: " << stereo_image.get_left_image().size << std::endl;

            auto result = stereo_image.get_right_image();
            cv::vconcat(result, stereo_image.get_left_image(), result);

            std::cout << "STEP 3: Undistorting images completed! ✅" << std::endl;
        }
        /////////////////////////////////////////////////////////////////////////////////////////////
        ///// STEP 4: Create Disparity Map
        /////////////////////////////////////////////////////////////////////////////////////////////

        std::cout << "----------------------------------------------------------------------" << std::endl;
        std::cout << "STEP 4: Creating Disparity Map... ⏳" << std::endl;

        cv::Mat disparity;
        disparity = stereo_image.calculate_filtered_disparity(block_size, max_disparity);

        StereoReconstruction::visualize_disparity_map(disparity);

        if (use_GT)
        {
            auto gt = stereo_image.get_gt_disparity_left();
            std::cout << "MSE own-to-GT: " << StereoImage::calculateMSE<float>(disparity, gt) << std::endl;
        }

        std::cout << "STEP 4: Creating Disparity Map completed! ✅" << std::endl;

        /////////////////////////////////////////////////////////////////////////////////////////////
        ///// STEP 5: Create Depth-Map and Pointcloud
        /////////////////////////////////////////////////////////////////////////////////////////////

        cv::Mat depth_image = stereo_image.calculate_depth(disparity, max_depth);
        cv::imwrite(path + "depth.png", StereoReconstruction::float_depth_to_int_depth(depth_image));

        if (use_GT)
        {
            cv::Mat gt_depth_image = stereo_image.calculate_depth(stereo_image.get_gt_disparity_left(), max_depth);
            cv::imwrite(path + "GT_depth.png", StereoReconstruction::float_depth_to_int_depth(gt_depth_image));

            if (do_pointcloud)
            {
                std::string gt_ply_filename = path + "GT_ptCloud.ply";

                stereo_image.depth_To_PT_cloud(gt_depth_image, stereo_image.get_left_image(), gt_ply_filename, do_mesh);

                if (combined_pointcloud)
                {
                    auto current_pointcloud = StereoReconstruction::depth_To_O3D_PT_cloud(StereoReconstruction::float_depth_to_int_depth(gt_depth_image), stereo_image.get_left_image(), stereo_image.get_intrinsic_left(), max_depth);
                    combined_pointcloud = StereoReconstruction::merge_pointclouds(current_pointcloud, combined_pointcloud);
                    open3d::visualization::DrawGeometries({combined_pointcloud});
                }
                else
                {
                    combined_pointcloud = StereoReconstruction::depth_To_O3D_PT_cloud(StereoReconstruction::float_depth_to_int_depth(gt_depth_image), stereo_image.get_left_image(), stereo_image.get_intrinsic_left(), max_depth);
                }
            }
        }

        if (do_pointcloud)
        {
            std::cout << "----------------------------------------------------------------------" << std::endl;
            std::cout << "STEP 5: Creating Pointcloud... ⏳" << std::endl;

            std::string ply_filename = path + "ptCloud.ply";
            stereo_image.depth_To_PT_cloud(depth_image, stereo_image.get_left_image(), ply_filename, do_mesh);
            std::cout << "Successfully written pointcloud to file: " << ply_filename << std::endl;
            std::cout << "STEP 5: Creating Pointcloud completed! ✅" << std::endl;
        }

        if (do_pointcloud && combined_pointcloud)
        {
            open3d::io::WritePointCloud(path + "Multiview_ICP_ptCloud.ply", *combined_pointcloud);
        }
    }

    return 0;
}
