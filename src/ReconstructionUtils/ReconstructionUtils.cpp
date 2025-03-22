#include <iostream>
#include <fstream>
#include <fstream>
#include <utility>
#include <tuple>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <open3d/Open3D.h>

#include "Eigen.h"
#include "ReconstructionUtils.h"

namespace StereoReconstruction
{
	std::shared_ptr<open3d::geometry::PointCloud> depth_To_O3D_PT_cloud(const cv::Mat &depth, const cv::Mat &image, cv::Mat intrinsics, const double &max_depth)
	{
		float fx = intrinsics.at<double>(0);
		float fy = intrinsics.at<double>(4);
		float cx = intrinsics.at<double>(2);
		float cy = intrinsics.at<double>(5);
		unsigned width = image.cols;
		unsigned height = image.rows;
		open3d::camera::PinholeCameraIntrinsic o3d_intrinsics(width, height, fx, fy, cx, cy);
		open3d::geometry::Image rgb_o3d;
		open3d::geometry::Image depth_o3d;

		// Needs conversion, as opencv stores colors in bgr, while o3d uses rgb
		cv::Mat rgb_image;
		cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

		// alloc buffer
		rgb_o3d.Prepare(width, height, 3, 1);
		depth_o3d.Prepare(width, height, 1, 2);

		// copy
		memcpy(rgb_o3d.data_.data(), rgb_image.data, rgb_o3d.data_.size());
		memcpy(depth_o3d.data_.data(), depth.data, depth_o3d.data_.size());

		auto rgbd_image = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
			rgb_o3d,
			depth_o3d,
			1000,
			max_depth/1000,
			false);
		auto pcd = open3d::geometry::PointCloud::CreateFromRGBDImage(*rgbd_image, o3d_intrinsics);
		return pcd;
	}

	cv::Mat float_depth_to_int_depth(const cv::Mat &float_depth)
	{
		cv::Mat finite_float_depth;
		cv::Mat integer_depth_image;
		
		float_depth.convertTo(integer_depth_image, CV_16U);

		for (unsigned i = 0; i < float_depth.cols * float_depth.rows; i++)
		{
			if (!std::isfinite(float_depth.at<float>(i)))
			{
				integer_depth_image.at<ushort>(i) = 0;
			}
		}
		
		return integer_depth_image;
	}

	void visualize_disparity_map(const cv::Mat &disparity)
	{
		// Normalization for proper display
		cv::Mat disp_norm = disparity;
		for (unsigned i = 0; i < disparity.cols * disparity.rows; i++)
		{
			if (!std::isfinite(disp_norm.at<float>(i)))
			{
				disp_norm.at<float>(i) = 0;
			}
		}
		disparity.convertTo(disp_norm, CV_32S);

		cv::normalize(disp_norm, disp_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
		cv::imshow("Disparity map", disp_norm);
		cv::waitKey();
	}

	cv::Mat load_middleburry_groundtruth(const std::string &path)
	{
		std::ifstream in_stream(path, std::ios::binary);

		if (!in_stream.is_open())
		{
			throw std::runtime_error(std::string("Groundtruth-file: \"") + path + std::string("\" not found!"));
		}

		std::string line;
		std::getline(in_stream, line);
		bool color;
		if (line == "PF")
		{
			color = true;
		}
		else if (line == "Pf")
		{
			color = false;
		}
		else
		{
			throw std::runtime_error(std::string("Groundtruth-file: \"") + path + std::string("\" is not a PFM file!"));
		}
		int cols;
		int rows;
		float scale;

		std::getline(in_stream, line);
		int split_index = line.find(' ');
		cols = std::stoi(line.substr(0, split_index));
		rows = std::stoi(line.substr(split_index, line.length()));

		std::getline(in_stream, line);
		scale = std::stof(line);
		std::vector<char> buffer(cols * rows * sizeof(float));

		in_stream.read(buffer.data(), cols * rows * sizeof(float));
		cv::Mat gt(rows, cols, CV_32F);

		for (unsigned i = 0; i < rows * cols; ++i)
		{
			// This should be probably done different
			gt.at<float>(i) = *((float *)&buffer.at(i * sizeof(float)));
		}
		cv::flip(gt, gt, 0); // Flip images to correct buffer

		return gt;
	}

	cv::Point3d linear_triangulation(const cv::Mat &projection_left,
									 const cv::Mat &projection_right,
									 const cv::Point2d &point_left,
									 const cv::Point2d &point_right)
	{
		cv::Mat A(4, 3, CV_64F);
		cv::Mat b(4, 1, CV_64F);

		{
			const double
				&px = point_left.x,
				&py = point_left.y,
				&p1 = projection_left.at<double>(0, 0),
				&p2 = projection_left.at<double>(0, 1),
				&p3 = projection_left.at<double>(0, 2),
				&p4 = projection_left.at<double>(0, 3),
				&p5 = projection_left.at<double>(1, 0),
				&p6 = projection_left.at<double>(1, 1),
				&p7 = projection_left.at<double>(1, 2),
				&p8 = projection_left.at<double>(1, 3),
				&p9 = projection_left.at<double>(2, 0),
				&p10 = projection_left.at<double>(2, 1),
				&p11 = projection_left.at<double>(2, 2),
				&p12 = projection_left.at<double>(2, 3);

			A.at<double>(0, 0) = px * p9 - p1;
			A.at<double>(0, 1) = px * p10 - p2;
			A.at<double>(0, 2) = px * p11 - p3;
			A.at<double>(1, 0) = py * p9 - p5;
			A.at<double>(1, 1) = py * p10 - p6;
			A.at<double>(1, 2) = py * p11 - p7;

			b.at<double>(0) = p4 - px * p12;
			b.at<double>(1) = p8 - py * p12;
		}

		{
			const double
				&px = point_right.x,
				&py = point_right.y,
				&p1 = projection_right.at<double>(0, 0),
				&p2 = projection_right.at<double>(0, 1),
				&p3 = projection_right.at<double>(0, 2),
				&p4 = projection_right.at<double>(0, 3),
				&p5 = projection_right.at<double>(1, 0),
				&p6 = projection_right.at<double>(1, 1),
				&p7 = projection_right.at<double>(1, 2),
				&p8 = projection_right.at<double>(1, 3),
				&p9 = projection_right.at<double>(2, 0),
				&p10 = projection_right.at<double>(2, 1),
				&p11 = projection_right.at<double>(2, 2),
				&p12 = projection_right.at<double>(2, 3);

			A.at<double>(2, 0) = px * p9 - p1;
			A.at<double>(2, 1) = px * p10 - p2;
			A.at<double>(2, 2) = px * p11 - p3;
			A.at<double>(3, 0) = py * p9 - p5;
			A.at<double>(3, 1) = py * p10 - p6;
			A.at<double>(3, 2) = py * p11 - p7;

			b.at<double>(2) = p4 - px * p12;
			b.at<double>(3) = p8 - py * p12;
		}

		// cv::Mat x = (A.t() * A).inv() * A.t() * b; // numerically unstable

		// opencv doesn't allow for conversion between cv::MatExpr and cv::Point3d ...
		cv::Mat x = A.inv(cv::DECOMP_SVD) * b;
		return cv::Point3d(x);
	}

	// Code copied from Code copied from 3D-Scanning & Motion Capture Exercise 1
	bool WriteMesh(Vertex *vertices, unsigned int width, unsigned int height, const std::string &filename)
	{
		float edgeThreshold = 0.01f; // 1cm

		unsigned int nVertices = width * height;

		unsigned nFaces = 0;
		for (int y = 0; y < height - 1; ++y)
		{
			for (int x = 0; x < width - 1; ++x)
			{
				const Vertex &v0 = vertices[y * width + x];
				const Vertex &v1 = vertices[y * width + (x + 1)];
				const Vertex &v2 = vertices[(y + 1) * width + x];
				const Vertex &v3 = vertices[(y + 1) * width + (x + 1)];

				float edge01 = CalculateEdgeLength(v0, v1);
				float edge02 = CalculateEdgeLength(v0, v2);
				float edge30 = CalculateEdgeLength(v3, v0);
				float edge12 = CalculateEdgeLength(v1, v2);
				float edge23 = CalculateEdgeLength(v2, v3);
				float edge13 = CalculateEdgeLength(v1, v3);
				if (v0.position.x() != MINF && v1.position.x() != MINF && v2.position.x() != MINF && v3.position.x() != MINF)
				{
					if (edge01 < edgeThreshold && edge12 < edgeThreshold && edge02 < edgeThreshold)
					{
						nFaces++;
					}
					if (edge12 < edgeThreshold && edge23 < edgeThreshold && edge13 < edgeThreshold)
					{
						nFaces++;
					}
				}
			}
		}

		// Write off file
		std::ofstream outFile(filename);
		if (!outFile.is_open())
			return false;

		// write header
		outFile << "COFF" << std::endl;

		outFile << "# numVertices numFaces numEdges" << std::endl;

		outFile << nVertices << " " << nFaces << " 0" << std::endl;

		outFile << "# list of vertices" << std::endl;
		outFile << "# X Y Z R G B A" << std::endl;

		for (int j = 0; j < width * height; j++)
		{

			if (vertices[j].position[0] != MINF) //&& currentVertex->position[2] < edgeThreshold)
			{
				outFile << vertices[j].position.x() << " " << vertices[j].position.y() << " " << vertices[j].position.z() << " "; //<< currentVertex->color[0]<< " " <<currentVertex->color[1]<< " " <<currentVertex->color[2]<< " " <<currentVertex->color[3]<< std::endl;
			}
			else
			{
				outFile << "0 0 0";
			}
			outFile << static_cast<unsigned char>(vertices[j].color[0]) % 256 << " " << static_cast<unsigned char>(vertices[j].color[1]) % 256 << " ";
			outFile << static_cast<unsigned char>(vertices[j].color[2]) % 256 << std::endl;
		}

		outFile << "# list of faces" << std::endl;

		// Loop through the grid and create triangles
		for (int y = 0; y < height - 1; ++y)
		{
			for (int x = 0; x < width - 1; ++x)
			{
				const Vertex &v0 = vertices[y * width + x];
				const Vertex &v1 = vertices[y * width + (x + 1)];
				const Vertex &v2 = vertices[(y + 1) * width + x];
				const Vertex &v3 = vertices[(y + 1) * width + (x + 1)];

				float edge01 = CalculateEdgeLength(v0, v1);
				float edge02 = CalculateEdgeLength(v0, v2);

				float edge12 = CalculateEdgeLength(v1, v2);

				float edge23 = CalculateEdgeLength(v2, v3);
				float edge13 = CalculateEdgeLength(v1, v3);
				if (v0.position.x() != MINF && v1.position.x() != MINF && v2.position.x() != MINF && v3.position.x() != MINF)
				{
					if (edge01 < edgeThreshold && edge12 < edgeThreshold && edge02 < edgeThreshold)
					{
						outFile << 3 << " " << y * width + x << " " << (y + 1) * width + x << " " << y * width + (x + 1) << std::endl;
					}

					if (edge12 < edgeThreshold && edge23 < edgeThreshold && edge13 < edgeThreshold)
					{
						outFile << 3 << " " << (y + 1) * width + x << " " << (y + 1) * width + (x + 1) << " " << y * width + (x + 1) << std::endl;
					}
				}
			}
		}
		// close file
		outFile.close();

		return true;
	}

	float CalculateEdgeLength(const Vertex &v1, const Vertex &v2)
	{
		float dx = v2.position[0] - v1.position[0];
		float dy = v2.position[1] - v1.position[1];
		float dz = v2.position[2] - v1.position[2];

		// Calculate the square of the Euclidean distance (for efficiency)
		float distanceSquared = dx * dx + dy * dy + dz * dz;
		// Take the square root to get the actual distance
		float distance = sqrt(distanceSquared);
		return distance;
	}

	std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::shared_ptr<open3d::pipelines::registration::Feature>> preprocess_pointcloud(const std::shared_ptr<open3d::geometry::PointCloud> &pcd, const double &voxel_size)
	{
		std::shared_ptr<open3d::geometry::PointCloud> pcd_downsampled = (*pcd).VoxelDownSample(voxel_size);
		double radius_normal = voxel_size * 2;
		pcd_downsampled->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(radius_normal, 30));

		double radius_feature = voxel_size * 5;
		std::shared_ptr<open3d::pipelines::registration::Feature> feature = open3d::pipelines::registration::ComputeFPFHFeature(*pcd_downsampled, open3d::geometry::KDTreeSearchParamHybrid(radius_feature, 100));

		return {pcd_downsampled, feature};
	}

	std::shared_ptr<open3d::geometry::PointCloud> merge_pointclouds(const std::shared_ptr<open3d::geometry::PointCloud> &source, const std::shared_ptr<open3d::geometry::PointCloud> &target)
	{
		double voxel_size = 0.05;
		// fast global registration
		double distance_threshold = voxel_size * 0.5;

		auto [source_down, source_fpfh] = preprocess_pointcloud(source, voxel_size);
		auto [target_down, target_fpfh] = preprocess_pointcloud(target, voxel_size);

		auto result = open3d::pipelines::registration::FastGlobalRegistrationBasedOnFeatureMatching(*source_down, *target_down, *source_fpfh, *target_fpfh, open3d::pipelines::registration::FastGlobalRegistrationOption(1.4, false, true, distance_threshold));

		const std::vector<double> voxel_radius{0.04, 0.01, 0.008};
		const std::vector<int> max_iterations{80, 20, 15};

		Eigen::Matrix4d current_transformation = result.transformation_;

		for (unsigned i = 0; i < voxel_radius.size(); i++)
		{
			auto [source_down, source_fpfh] = preprocess_pointcloud(source, voxel_radius.at(i));
			auto [target_down, target_fpfh] = preprocess_pointcloud(target, voxel_radius.at(i));

			auto result_icp = open3d::pipelines::registration::RegistrationColoredICP(*source_down,
																					  *target_down,
																					  voxel_radius.at(i),
																					  current_transformation,
																					  open3d::pipelines::registration::TransformationEstimationForColoredICP(),
																					  open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, max_iterations.at(i)));

			current_transformation = result_icp.transformation_;
		}

		std::shared_ptr<open3d::geometry::PointCloud> transformed_source = std::make_shared<open3d::geometry::PointCloud>(*source);
		(*transformed_source).Transform(current_transformation);
		*transformed_source = *transformed_source + *target;
		return transformed_source;
	}
} // namespace StereoReconstruction