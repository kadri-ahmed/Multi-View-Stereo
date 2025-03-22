#include <iostream>
#include <stdexcept>
#include <fstream>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/highgui.hpp>

#include <open3d/Open3D.h>

#include "../ReconstructionUtils/Eigen.h"
#include "../ReconstructionUtils/ReconstructionUtils.h"
#include "../SFTools/ProgressBar.h"
#include "StereoImage.h"

/*
	Public methods
*/

StereoImage::StereoImage(const std::string &path_left, const std::string &path_right, const std::string &intrinsic_path, const std::string &distCoeffs_path, const bool &read_GT) {
    if (!read_images_from_files(path_left, path_right) || !intrinsic_from_file(intrinsic_path) || (distCoeffs_path != "" && !distCoeffs_from_file(distCoeffs_path))) {
        throw std::runtime_error(std::string("Image not found!"));
    }

    if (read_GT) {
        this->gt_disparity_left = StereoReconstruction::load_middleburry_groundtruth(path_left.substr(0, path_left.find_last_of('.')) + ".pfm");
        this->gt_disparity_right = StereoReconstruction::load_middleburry_groundtruth(path_right.substr(0, path_right.find_last_of('.')) + ".pfm");
    }
}
StereoImage::StereoImage(
    const std::filesystem::path &scenePath,
    const Calibration &calibration,
	const std::string &distCoeffs_path,
    const bool &read_GT): m_hasGT{read_GT}
{
    fs::path path_left{fs::path(scenePath / "im0.png")}, path_right{fs::path(scenePath / "im1.png")};
    if (!read_images_from_files(path_left.string(), path_right.string()) || (distCoeffs_path != "" && !distCoeffs_from_file(distCoeffs_path))){
        throw std::runtime_error(std::string("Either images, intrinsics or distortion coefficients not found!"));
    }

    cv::Mat intrinsic_left(3, 3, cv::DataType<double>::type);
    for (unsigned row = 0; row < 3; row++)
    {
        for (unsigned col = 0; col < 3; col++)
        {
            intrinsic_left.at<double>(row, col) = calibration.cameras.at("cam0").matrix[row*3+col];
        }
    }
    m_intrinsic_left = intrinsic_left;

    cv::Mat intrinsic_right(3, 3, cv::DataType<double>::type);
    for (unsigned row = 0; row < 3; row++)
    {
        for (unsigned col = 0; col < 3; col++)
        {
            intrinsic_left.at<double>(row, col) = calibration.cameras.at("cam1").matrix[row*3+col];
        }
    }
    m_intrinsic_right = intrinsic_right;
	doffs = calibration.doffs;
	baseline = calibration.baseline;

    if(read_GT){
        gt_disparity_left= StereoReconstruction::load_middleburry_groundtruth(scenePath / "disp0.pfm");
        gt_disparity_right = StereoReconstruction::load_middleburry_groundtruth(scenePath / "disp1.pfm");
    }
}

bool StereoImage::read_images_from_files(const std::string &path_left, const std::string &path_right)
{
	this->m_left_image = cv::imread(path_left, cv::IMREAD_UNCHANGED);
	this->m_right_image = cv::imread(path_right, cv::IMREAD_UNCHANGED);

	return !(this->m_left_image.empty() || this->m_right_image.empty());
}

bool StereoImage::distCoeffs_from_file(const std::string &path)
{
	std::ifstream distCoeffs_file(path);

	if (!distCoeffs_file.is_open())
	{
		std::cerr << "Error while opening distortion coefficients file \"" << path << "\"" << std::endl;
		return false;
	}

	// Read distortion coefficients for m_left_image image
	std::string input;
	cv::Mat distCoeffs_left(1, 5, cv::DataType<double>::type);
	for (unsigned row = 0; row < distCoeffs_left.rows; row++)
	{
		for (unsigned col = 0; col < distCoeffs_left.cols; col++)
		{
			input = "";
			distCoeffs_file >> input;
			distCoeffs_left.at<double>(row, col) = std::stod(input);
		}
	}
	this->m_distCoeffs_left = distCoeffs_left;
	std::cout << "Read coeffs m_left_image \n"
			  << distCoeffs_left << std::endl;

	// Read distCoeffs for m_right_image image
	cv::Mat distCoeffs_right(1, 5, cv::DataType<double>::type);
	for (unsigned row = 0; row < distCoeffs_right.rows; row++)
	{
		for (unsigned col = 0; col < distCoeffs_right.cols; col++)
		{
			input = "";
			distCoeffs_file >> input;
			distCoeffs_right.at<double>(row, col) = std::stod(input);
		}
	}
	this->m_distCoeffs_right = distCoeffs_right;
	std::cout << "Read coeffs m_right_image\n"
			  << distCoeffs_right << std::endl;

	return true;
}

const cv::Mat &StereoImage::get_left_image() const
{
	return this->m_left_image;
}

const cv::Mat &StereoImage::get_right_image() const
{
	return this->m_right_image;
}

void StereoImage::set_left_image(cv::Mat &image)
{
	this->m_left_image = image;
}

void StereoImage::set_right_image(cv::Mat &image)
{
	this->m_right_image = image;
}

const cv::Mat &StereoImage::get_intrinsic_left() const
{
	return this->m_intrinsic_left;
}

const cv::Mat &StereoImage::get_intrinsic_right() const
{
	return this->m_intrinsic_right;
}

const cv::Mat &StereoImage::get_distCoeffs_left() const
{
	return this->m_distCoeffs_left;
}

const cv::Mat &StereoImage::get_distCoeffs_right() const
{
	return this->m_distCoeffs_right;
}

const cv::Mat &StereoImage::get_gt_disparity_left() const
{
	return this->gt_disparity_left;
}

const cv::Mat &StereoImage::get_gt_disparity_right() const
{
	return this->gt_disparity_right;
}

void StereoImage::set_relative_cam_transformation(cv::Mat R, cv::Mat t)
{
	m_relative_cam_transformation.first = std::move(R);
	m_relative_cam_transformation.second = std::move(t);
}

const std::pair<cv::Mat, cv::Mat> &StereoImage::get_relative_cam_transformation() const
{
	return m_relative_cam_transformation;
}

cv::Mat StereoImage::calculate_depth(const int &max_depth) const
{
	cv::Mat disparity = calculate_filtered_disparity(5, 300);
	return calculate_depth(disparity, max_depth);
}

cv::Mat StereoImage::calculate_depth(const cv::Mat &disparity, const int &max_depth) const
{
	cv::Mat depth = cv::Mat::zeros(cv::Size(disparity.cols, disparity.rows), CV_32FC1);
	
	for (unsigned i = 0; i < depth.rows * depth.cols; ++i)
	{
		float f = this->m_intrinsic_left.at<double>(0);
		depth.at<float>(i) = baseline * (f / (disparity.at<float>(i) + this->doffs));

		if (std::fabs(depth.at<float>(i)) > max_depth || depth.at<float>(i) <= 0)
		{
			depth.at<float>(i) = std::numeric_limits<float>::infinity();
		}
	}
	return depth;
}

cv::Mat StereoImage::calculate_filtered_disparity(const unsigned &block_size, const unsigned &max_pixel_shift) const
{
	cv::Mat left_disparity = calculate_disparity_left(block_size, max_pixel_shift);
	cv::Mat right_disparity = -1 * calculate_disparity_right(block_size, max_pixel_shift);

	cv::Mat result;

	// Keep in mind: https://github.com/opencv/opencv_contrib/issues/2974
	auto wsl_filter = cv::ximgproc::createDisparityWLSFilterGeneric(true);
	wsl_filter->filter(left_disparity, this->m_left_image, result, right_disparity, cv::Rect(), this->m_right_image);
	// wsl_filter->filter(left_disparity, left, result);

	return StereoImage::logical_AND(result, this->m_left_image);
}

cv::Mat StereoImage::logical_AND(const cv::Mat &mat, const cv::Mat &reference)
{
	// Conversion to double prec. needed to compare and not loose data in the process
	cv::Mat double_mat;
	mat.convertTo(double_mat, CV_64F);
	cv::Mat double_ref;
	reference.convertTo(double_ref, CV_64F);

	if (mat.cols != reference.cols || mat.rows != reference.rows)
	{
		throw std::runtime_error("Error: Logical AND can not be calculated for matrices of different size.");
	}

	cv::Mat result = cv::Mat::zeros(cv::Size(mat.cols, mat.rows), CV_64F);
	for (unsigned row = 0; row < mat.rows; ++row)
	{
		for (unsigned col = 0; col < mat.cols; ++col)
		{
			if (double_ref.at<double>(row, col) != 0 && std::isfinite(double_ref.at<double>(row, col)))
			{
				result.at<double>(row, col) = double_mat.at<double>(row, col);
			}
			else
			{
				result.at<double>(row, col) = 0;
			}
		}
	}

	// convert back to original type (lossless)
	result.convertTo(result, mat.type());
	return result;
}

template <typename T>
double StereoImage::calculateMSE(const cv::Mat &mat1, const cv::Mat &mat2)
{
	if (mat1.cols != mat2.cols || mat1.rows != mat2.rows)
	{
		throw std::runtime_error("Error: MSE can not be calculated for matrices of different size.");
	}

	double sq_sum = 0;
	for (unsigned row = 0; row < mat1.rows; ++row)
	{
		for (unsigned col = 0; col < mat1.cols; ++col)
		{
			double local_err = 0;
			if (!std::isfinite(mat1.at<T>(row, col)))
			{
				if (std::isfinite(mat2.at<T>(row, col)))
				{
					local_err = mat2.at<T>(row, col);
				}
			}
			else if (!std::isfinite(mat2.at<T>(row, col))) // mat1 is finite, mat2 is not
			{
				local_err = mat1.at<T>(row, col);
			}
			else // Both are finit
			{
				local_err = mat1.at<T>(row, col) - mat2.at<T>(row, col);
			}
			sq_sum += local_err * local_err;
		}
	}
	return sq_sum / (mat1.cols * mat1.rows);
}

cv::Mat StereoImage::calculate_disparity_right(const unsigned &block_size, const unsigned &max_pixel_shift, const double &scaling_factor) const
{
	cv::Mat gray_left, gray_right;
	cv::cvtColor(this->m_left_image, gray_left, cv::COLOR_BGR2GRAY);
	cv::cvtColor(this->m_right_image, gray_right, cv::COLOR_BGR2GRAY);
	cv::resize(gray_left, gray_left, cv::Size(gray_left.cols * scaling_factor, gray_left.rows * scaling_factor));
	cv::resize(gray_right, gray_right, cv::Size(gray_right.cols * scaling_factor, gray_right.rows * scaling_factor));

	const int half_window = (block_size - 1) / 2;
	const unsigned rows = gray_left.rows;
	const unsigned cols = gray_left.cols;

	cv::Mat disparity = cv::Mat::zeros(cv::Size(cols, rows), CV_32F);

    std::cerr << "Disparity Right: ";
    Progressbar progressbar(rows - 2 * half_window);
    progressbar.set_done_char("█");
    progressbar.set_opening_bracket_char("{");
    progressbar.set_closing_bracket_char("}");

	// Epipolar rows of both images
	for (unsigned y = half_window; y < rows - half_window; ++y)
	{
		// Columns of right image
		for (unsigned x_im_R = half_window; x_im_R < cols - half_window; ++x_im_R)
		{
			double ssd_min = std::numeric_limits<double>::infinity();
			float disparity_value = 0;
			bool has_second_minimum = false;

			const unsigned upper_search_bound = (x_im_R + max_pixel_shift) > (cols - half_window) ? cols - half_window : x_im_R + max_pixel_shift;
			// Columns of left image -> Only match upward as downward results in negative depth
			for (unsigned x_im_L = x_im_R; x_im_L < upper_search_bound; ++x_im_L)
			{
				double ssd_curr = 0;
				// Sum of squared differences over current block
				for (int y_block = -1 * half_window; y_block <= half_window; ++y_block)
				{
					for (int x_block = -1 * half_window; x_block <= half_window; ++x_block)
					{
						int pixel_diff = (int)gray_left.at<unsigned char>(y + y_block, x_im_L + x_block) - (int)gray_right.at<unsigned char>(y + y_block, x_im_R + x_block);
						ssd_curr += pixel_diff * pixel_diff;
					}
				}
				if (ssd_curr < ssd_min)
				{
					// Found new best match
					has_second_minimum = false;
					disparity_value = x_im_L - x_im_R;
					ssd_min = ssd_curr;
				}
				else if (ssd_curr == ssd_min)
				{
					// Matched disparity invalid if second correspondence exists
					has_second_minimum = true;
				}
			}
			if (!has_second_minimum)
			{
				disparity.at<float>(y, x_im_R, 0) = disparity_value;
			}
		}
	    progressbar.update();
	}
    std::cerr << std::endl;

	cv::resize(disparity, disparity, cv::Size(gray_right.cols / scaling_factor, gray_right.rows / scaling_factor));

	return disparity;
}

cv::Mat StereoImage::calculate_disparity_left(const unsigned &block_size, const unsigned &max_pixel_shift, const double &scaling_factor) const
{
	cv::Mat gray_left, gray_right;
	cv::cvtColor(this->m_left_image, gray_left, cv::COLOR_BGR2GRAY);
	cv::cvtColor(this->m_right_image, gray_right, cv::COLOR_BGR2GRAY);
	cv::resize(gray_left, gray_left, cv::Size(gray_left.cols * scaling_factor, gray_left.rows * scaling_factor));
	cv::resize(gray_right, gray_right, cv::Size(gray_right.cols * scaling_factor, gray_right.rows * scaling_factor));

	const int half_window = (block_size - 1) / 2;
	const unsigned rows = gray_left.rows;
	const unsigned cols = gray_left.cols;

	cv::Mat disparity = cv::Mat::zeros(cv::Size(cols, rows), CV_32F);

    std::cerr << "Disparity Left: ";
    Progressbar progressbar(rows - 2 * half_window);
    progressbar.set_done_char("█");
    progressbar.set_opening_bracket_char("{");
    progressbar.set_closing_bracket_char("}");

	// Epipolar rows of both images
	for (unsigned y = half_window; y < rows - half_window; ++y)
	{
		// Columns of left image
		for (unsigned x_im_L = half_window; x_im_L < cols - half_window; ++x_im_L)
		{
			double ssd_min = std::numeric_limits<double>::infinity();
			float disparity_value = 0;
			bool has_second_minimum = false;

			const unsigned lower_search_bound = ((int)x_im_L - (int)max_pixel_shift) < half_window - 1 ? half_window - 1 : x_im_L - max_pixel_shift;
			// Columns of right image -> Only match downward as upward results in negative depth
			for (unsigned x_im_R = x_im_L; x_im_R > lower_search_bound; --x_im_R)
			{
				double ssd_curr = 0;
				// Sum of squared differences over current block
				for (int y_block = -1 * half_window; y_block <= half_window; ++y_block)
				{
					for (int x_block = -1 * half_window; x_block <= half_window; ++x_block)
					{
						int pixel_diff = (int)gray_left.at<unsigned char>(y + y_block, x_im_L + x_block) - (int)gray_right.at<unsigned char>(y + y_block, x_im_R + x_block);
						ssd_curr += pixel_diff * pixel_diff;
					}
				}
				if (ssd_curr < ssd_min)
				{
					// Found new best match
					has_second_minimum = false;
					disparity_value = (x_im_L - x_im_R);
					ssd_min = ssd_curr;
				}
				else if (ssd_curr == ssd_min)
				{
					// Matched disparity invalid if second correspondence exists
					has_second_minimum = true;
				}
			}
			if (!has_second_minimum)
			{
				disparity.at<float>(y, x_im_L, 0) = disparity_value;
			}
		}
	    progressbar.update();
	}
    std::cerr << std::endl;

	cv::resize(disparity, disparity, cv::Size(gray_right.cols / scaling_factor, gray_right.rows / scaling_factor));

	return disparity;
}

std::vector<std::pair<cv::Point3d, cv::Vec3b>> StereoImage::traingulate_matching_points(const cv::Mat &projection_left,
																						const cv::Mat &projection_right,
																						const std::vector<cv::Point2d> &points_left,
																						const std::vector<cv::Point2d> &points_right) const
{

	if (points_left.size() != points_right.size())
	{
		std::cout << "Triangulation of matching points not possible. Number of matched points is not equal." << std::endl;
		return {};
	}

	std::vector<std::pair<cv::Point3d, cv::Vec3b>> depth_points;
	depth_points.reserve(points_left.size());

	for (size_t i = 0; i < points_left.size(); i++)
	{
		// Triangulation
		cv::Point3d depth_point = StereoReconstruction::linear_triangulation(projection_left, projection_right, points_left[i], points_right[i]);

		const int xi1 = round(points_left[i].x);
		const int yi1 = round(points_left[i].y);
		const int xi2 = round(points_right[i].x);
		const int yi2 = round(points_right[i].y);

		const auto &color1 = this->m_left_image.at<cv::Vec3b>(yi1, xi1);
		const auto &color2 = this->m_right_image.at<cv::Vec3b>(yi2, xi2);

		// Average color between m_left_image and m_right_image image
		const cv::Vec3b color = 0.5 * (color1 + color2);
		depth_points.push_back({std::move(depth_point), color});
	}

	return depth_points;
}

void StereoImage::depth_points_to_file(const std::string &filepath, const std::vector<std::pair<cv::Point3d, cv::Vec3b>> &depth_points)
{
	std::ofstream out_file(filepath, std::ofstream::out);
	// Check if the file is open
	if (!out_file.is_open())
	{
		std::cerr << "Error opening the file!" << std::endl;
		return;
	}

	for (const auto &[point, color] : depth_points)
	{
		out_file << point.x << " "
				 << point.y << " "
				 << point.z << " "
				 << (int)color[0] << " "
				 << (int)color[1] << " "
				 << (int)color[2] << "\n";
	}

	out_file.close();

	std::cout << "Successfully written " << depth_points.size() << " points to file \"" << filepath << "\"." << std::endl;
}

//
//	Private member functions
//

bool StereoImage::intrinsic_from_file(const std::string &path)
{
	std::ifstream intrinsic_file(path);

	if (!intrinsic_file.is_open())
	{
		std::cerr << "Error while opening intrinsic file \"" << path << "\"" << std::endl;
		return false;
	}

	// Read intrinsic for m_left_image image
	std::string input;
	cv::Mat intrinsic_left(3, 3, cv::DataType<double>::type);
	for (unsigned row = 0; row < 3; row++)
	{
		for (unsigned col = 0; col < 3; col++)
		{
			input = "";
			intrinsic_file >> input;
			intrinsic_left.at<double>(row, col) = std::stod(input);
		}
	}
	this->m_intrinsic_left = intrinsic_left;

	// Read intrinsic for left image
	cv::Mat intrinsic_right(3, 3, cv::DataType<double>::type);
	for (unsigned row = 0; row < 3; row++)
	{
		for (unsigned col = 0; col < 3; col++)
		{
			input = "";
			intrinsic_file >> input;
			intrinsic_right.at<double>(row, col) = std::stod(input);
		}
	}
	this->m_intrinsic_right = intrinsic_right;

	intrinsic_file >> this->doffs;
	intrinsic_file >> this->baseline;

	return true;
}

void StereoImage::depth_To_PT_cloud(const cv::Mat &gt_left_depth, const cv::Mat &image, std::string &ply_filename, bool write_mesh) const
{
	// IMREAD_COLOR loads the image in the BGR 8-bit format.

	int H = gt_left_depth.rows;
	int W = gt_left_depth.cols;

	float f = this->m_intrinsic_left.at<double>(0);
	float cx = this->m_intrinsic_left.at<double>(2);
	float cy = this->m_intrinsic_left.at<double>(5);
	float f_inv = 1.0f / f;

	std::vector<cv::Vec3b> colors;
	std::vector<cv::Point3f> points;
	StereoReconstruction::Vertex *vertices = new StereoReconstruction::Vertex[H * W];
	int curr = 0;
	for (int v = 0; v < gt_left_depth.rows; v += 1)
	{
		for (int u = 0; u < gt_left_depth.cols; u += 1)
		{
			cv::Vec3d point;

			float z = gt_left_depth.at<float>(v, u) * 0.001f; // millimeters to meters;
			float x = (static_cast<float>(u) - cx) * z * f_inv;
			float y = (static_cast<float>(v) - cy) * z * f_inv;

			if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z))
			{
				points.push_back(cv::Point3f(x, y, z));
				float one = {1.0f};
				Vector4f worldCoo{x, y, z, one};
				vertices[curr].position = worldCoo;

				cv::Vec3b intensity = image.at<cv::Vec3b>(v, u);
				uchar b = intensity.val[0];
				uchar g = intensity.val[1];
				uchar r = intensity.val[2];
				colors.push_back(cv::Vec3b(r, g, b));
				vertices[curr].color = Vector4uc(r, g, b, 0);				
			}
			curr++;
		}
	}

	std::ofstream ply_file(ply_filename, std::ios::out);
	if (!ply_file.is_open())
	{
		std::cerr << "Error opening PLY file." << std::endl;
		exit(EXIT_FAILURE);
	}
	unsigned int N = points.size();

	// Write PLY header with the number of vertices
	ply_file << "ply\n";
	ply_file << "format ascii 1.0\n";
	ply_file << "element vertex " << N << "\n";
	ply_file << "property float x\n";
	ply_file << "property float y\n";
	ply_file << "property float z\n";
	ply_file << "property uchar red\n";
	ply_file << "property uchar green\n";
	ply_file << "property uchar blue\n";
	ply_file << "end_header\n";

	for (int v = 0; v < N; ++v)
	{
		cv::Vec3b color = colors[v];
		cv::Point3f point = points[v];
		if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
		{
			ply_file << point.x << " "
					 << point.y << " "
					 << point.z << " "
					 << static_cast<int>(color[0]) << " "
					 << static_cast<int>(color[1]) << " "
					 << static_cast<int>(color[2]) << "\n";
		}
	}

	ply_file.close();

	if (write_mesh)
	{
		std::stringstream ss;
		ss << ply_filename.substr(0, ply_filename.length() - 4)+".off";
		if (!WriteMesh(vertices, W, H, ss.str()))
		{
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
		}
	}

	delete[] vertices;
}

cv::Mat StereoImage::draw_matched_keypoints(const StereoReconstruction::MatchedKeypoints &matched_points)
{
	//-- Draw matches
	cv::Mat img_matches;
	cv::drawMatches(this->get_left_image(),
					matched_points.keypoints_1,
					this->get_right_image(),
					matched_points.keypoints_2,
					matched_points.matches,
					img_matches,
					cv::Scalar::all(-1),
					cv::Scalar::all(-1),
					std::vector<char>(),
					cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	return img_matches;
}

/**
 * \brief
 * \param filename calib.txt from middlebury dataset
 * \return a calibration object containing the parsed parameters
 */
Calibration StereoImage::parseCalibration(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    Calibration calib;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '=')) {
            if (key == "cam0") {
                parseMatrix(iss, calib.cameras["cam0"].matrix);
            } else if (key == "cam1") {
                parseMatrix(iss, calib.cameras["cam1"].matrix);
            } else if (key == "doffs") {
                iss >> calib.doffs;
            } else if (key == "baseline") {
                iss >> calib.baseline;
            } else if (key == "width") {
                iss >> calib.width;
            } else if (key == "height") {
                iss >> calib.height;
            } else if (key == "ndisp") {
                iss >> calib.ndisp;
            } else if (key == "vmin") {
                iss >> calib.vmin;
            } else if (key == "vmax") {
                iss >> calib.vmax;
            }
        }
    }
    return calib;
}

void StereoImage::parseMatrix(std::istringstream& iss, std::vector<double>& matrix) {
    std::string row;
    while (std::getline(iss, row, ';')) {
        if(row[0] = '[')
            row = row.substr(1,row.size());
        if(row[row.size()-1] = ']')
            row = row.substr(0,row.size()-1);
        std::istringstream row_iss(row);
        double value;
        while (row_iss >> value) {
            matrix.push_back(value);
        }
    }
}



void StereoImage::printCalibration(const Calibration& calib) {
    for (const auto& pair : calib.cameras) {
        std::cout << pair.first << "=[";
        for (const auto& val : pair.second.matrix) {
            std::cout << val << " ";
        }
        std::cout << "]\n";
    }
    std::cout << "doffs=" << calib.doffs << "\n";
    std::cout << "baseline=" << calib.baseline << "\n";
    std::cout << "width=" << calib.width << "\n";
    std::cout << "height=" << calib.height << "\n";
    std::cout << "ndisp=" << calib.ndisp << "\n";
}


// Explicit Template instantiation
template double StereoImage::calculateMSE<unsigned char>(const cv::Mat &mat1, const cv::Mat &mat2);
template double StereoImage::calculateMSE<int>(const cv::Mat &mat1, const cv::Mat &mat2);
template double StereoImage::calculateMSE<float>(const cv::Mat &mat1, const cv::Mat &mat2);
template double StereoImage::calculateMSE<double>(const cv::Mat &mat1, const cv::Mat &mat2);
