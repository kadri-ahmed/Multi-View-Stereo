//
// Created by Ahmed Kadri on 21.01.24.
//

#include "./SFSparseMatching.h"

namespace SensorFusion {

    std::pair<ImageKeypoints<cv::Point2d>, ImageKeypoints<cv::Point2d>> SparseMatching::computeKeyPointMatches(const StereoImage& stereoImage, bool show)
    {
        //-- Step 1: Detect the keypoints using SURF feature_detector, compute the descriptors
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        cv::Mat descriptors_1, descriptors_2;
        auto featureDetector = cv::SIFT::create();

        //-- Convert to grayscale for better feature detection
        cv::Mat leftImageG, rightImageG;
        cv::cvtColor(stereoImage.get_left_image(), leftImageG, cv::COLOR_RGB2GRAY);
        cv::cvtColor(stereoImage.get_right_image(), rightImageG, cv::COLOR_RGB2GRAY);

        featureDetector->detectAndCompute(leftImageG, cv::noArray(), keypoints_1, descriptors_1 );
        featureDetector->detectAndCompute(rightImageG, cv::noArray(), keypoints_2, descriptors_2 );

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
        if(show){
            cv::Mat img_matches;
            cv::drawMatches(
                    stereoImage.get_left_image(),
                    keypoints_1,
                    stereoImage.get_right_image(),
                    keypoints_2,
                    good_matches,
                    img_matches);
            cv::imshow("Matched Keypoints",img_matches);
            cv::waitKey();
        }

        ImageKeypoints<cv::Point2d> pointsL, pointsR;
        pointsL.reserve(good_matches.size());
        pointsR.reserve(good_matches.size());
        for (auto& match : good_matches)
        {
            pointsL.push_back(keypoints_1[match.queryIdx].pt);
            pointsR.push_back(keypoints_2[match.trainIdx].pt);
        }

        return std::make_pair(pointsL, pointsR);
    }

    std::tuple<cv::Mat,cv::Mat,std::vector<size_t>> SparseMatching::findFundamentAndEssentialMatrices(
            ImageKeypoints<cv::Point2d> pointsL,
            ImageKeypoints<cv::Point2d> pointsR,
            const StereoImage& stereoImage,
            float confidence,
            float threshold
    ){
        //-- Normalize Points
        auto normalizedPoints = normalizePoints(pointsL, pointsR);

        // The so-far-the-best fundamental matrix
        cv::Mat bestFundamentalMatrix;
        std::vector<size_t> bestInliers;
        
        // The number of correspondences
        const size_t pointNumber = pointsL.size();

        // Initializing the index pool from which the minimal samples are selected
        std::vector<size_t> indexPool(pointNumber);
        for (size_t i = 0; i < pointNumber; ++i)
            indexPool[i] = i;

        // The size of a minimal sample
        constexpr size_t sampleSize = 8;
        // The minimal sample
        std::vector<size_t> mss(sampleSize);
        
        size_t maximumIterations = std::numeric_limits<int>::max(), // The maximum number of iterations set adaptively when a new best model is found
            iterationLimit = 5000, // A strict iteration limit which mustn't be exceeded
            iteration = 0; // The current iteration number

        ImageKeypoints<cv::Point2d> srcPoints(sampleSize),
                dstPoints(sampleSize);

        while (iteration++ < MIN(iterationLimit, maximumIterations))
        {
            for (auto sampleIdx = 0; sampleIdx < sampleSize; ++sampleIdx)
            {
                // Select a random index from the pool
                const size_t idx = round((rand() / (double)RAND_MAX) * (indexPool.size() - 1));
                mss[sampleIdx] = indexPool[idx];
                indexPool.erase(indexPool.begin() + idx);

                // Put the selected correspondences into the point containers
                const size_t point_idx = mss[sampleIdx];
                srcPoints[sampleIdx] = normalizedPoints.normalizedPointsL[point_idx];
                dstPoints[sampleIdx] = normalizedPoints.normalizedPointsR[point_idx];
            }

            // Estimate fundamental matrix
            cv::Mat fundamental_matrix = apply8PointMatchingAlgorithm(srcPoints, dstPoints);
            fundamental_matrix = normalizedPoints.Tl.t() * fundamental_matrix * normalizedPoints.Tr; // Denormalize the fundamental matrix

            // Count the inliers
            std::vector<size_t> inliers;
            for (int i = 0; i < pointsL.size(); ++i)
            {
                // Symmetric epipolar distance
                cv::Mat ptL = (cv::Mat_<double>(3, 1) << pointsL[i].x, pointsL[i].y, 1);
                cv::Mat ptR = (cv::Mat_<double>(3, 1) << pointsR[i].x, pointsR[i].y, 1);

                // Calculate the error
                cv::Mat lL = fundamental_matrix.t() * ptR;
                cv::Mat lR = fundamental_matrix * ptL;

                // Calculate the distance of point pt1 from lL
                const double
                        & aL = lL.at<double>(0),
                        & bL = lL.at<double>(1),
                        & cL = lL.at<double>(2);

                double tL = abs(aL * pointsL[i].x + bL * pointsL[i].y + cL);
                double dL = sqrt(aL * aL + bL * bL);
                double distanceL = tL / dL;

                // Calculate the distance of point pt2 from lR
                const double
                        & aR = lR.at<double>(0),
                        & bR = lR.at<double>(1),
                        & cR = lR.at<double>(2);

                double tR = abs(aR * pointsR[i].x + bR * pointsR[i].y + cR);
                double dR = sqrt(aR * aR + bR * bR);
                double distanceR = tR / dR;

                double dist = 0.5 * (distanceL + distanceR);

                if (dist < threshold)
                    inliers.push_back(i);
            }

            // Update if the new model is better than the previous so-far-the-best.
            if (bestInliers.size() < inliers.size())
            {
                // Update the set of inliers
                bestInliers.swap(inliers);
                inliers.clear();
                inliers.resize(0);
                // Update the model parameters
                bestFundamentalMatrix = fundamental_matrix;
                // Update the iteration number
                maximumIterations = getRansacIterationNumber(
            pointNumber,
            bestInliers.size(),
             sampleSize,
                        confidence);
            }

            // Put back the selected points to the pool
            for (size_t i = 0; i < sampleSize; ++i)
                indexPool.push_back(mss[i]);
        }
        bestFundamentalMatrix = evaluateNormalizationEffect(
                pointsL,
                pointsR,
                normalizedPoints,
                bestInliers);
        cv::Mat E = stereoImage.get_intrinsic_left().t() * bestFundamentalMatrix * stereoImage.get_intrinsic_right();



        auto xL = pointsL[bestInliers[0]];
        auto xR = pointsR[bestInliers[0]];
        cv::Mat pL = (cv::Mat_<double>(3,1) << xL.x, xL.y, 1);
        cv::Mat pR = (cv::Mat_<double>(3,1) << xR.x, xR.y, 1);

        std::cout << "[SensorFusion] Checking Epipolar Constraint: \nx.T * F * x' = " << pL.t() * bestFundamentalMatrix * pR << std::endl;

        return std::make_tuple<cv::Mat,cv::Mat,std::vector<size_t>>(std::move(bestFundamentalMatrix), std::move(E), std::move(bestInliers));
    }
    
    SensorFusion::NormalizedPoints SparseMatching::normalizePoints(ImageKeypoints<cv::Point2d> pointsL, ImageKeypoints<cv::Point2d> pointsR) {
        
        cv::Mat Tl = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat Tr = cv::Mat::eye(3, 3, CV_64F);
        std::vector<cv::Point2d> normalizedPointsL;
        std::vector<cv::Point2d> normalizedPointsR;

        const size_t pointNumber = pointsL.size();
        normalizedPointsL.resize(pointNumber);
        normalizedPointsR.resize(pointNumber);
        
        // Calculate the mass points
        cv::Point2d massL(0, 0), massR(0, 0);

        for (auto i = 0; i < pointNumber; ++i)
        {
            massL = massL + pointsL[i];
            massR = massR + pointsR[i];
        }
        massL = massL * (1.0 / pointNumber);
        massR = massR * (1.0 / pointNumber);

        // Translate the point clouds to origin
        for (auto i = 0; i < pointNumber; ++i)
        {
            normalizedPointsL[i] = pointsL[i] - massL;
            normalizedPointsR[i] = pointsR[i] - massR;
        }

        // Calculate the average distances of the points from the origin
        double avgDistanceL = 0.0, avgDistanceR = 0.0;
        for (auto i = 0; i < pointNumber; ++i)
        {
            avgDistanceL += cv::norm(normalizedPointsL[i]);
            avgDistanceR += cv::norm(normalizedPointsR[i]);
        }

        avgDistanceL /= pointNumber;
        avgDistanceR /= pointNumber;

        const double multiplier1 = sqrt(2) / avgDistanceL;
        const double multiplier2 = sqrt(2) / avgDistanceR;

        for (auto i = 0; i < pointNumber; ++i)
        {
            normalizedPointsL[i] *= multiplier1;
            normalizedPointsR[i] *= multiplier2;
        }

        Tl.at<double>(0, 0) = multiplier1;
        Tl.at<double>(1, 1) = multiplier1;
        Tl.at<double>(0, 2) = -multiplier1 * massL.x;
        Tl.at<double>(1, 2) = -multiplier1 * massL.y;

        Tr.at<double>(0, 0) = multiplier2;
        Tr.at<double>(1, 1) = multiplier2;
        Tr.at<double>(0, 2) = -multiplier2 * massR.x;
        Tr.at<double>(1, 2) = -multiplier2 * massR.y;

        return NormalizedPoints{std::move(Tl), std::move(Tr), std::move(normalizedPointsL), std::move(normalizedPointsR)};
    }

    cv::Mat SparseMatching::apply8PointMatchingAlgorithm(const ImageKeypoints<cv::Point2d> &pointsL, const ImageKeypoints<cv::Point2d> &pointsR) {
        const size_t pointNumber = pointsL.size();
        cv::Mat A(pointNumber, 9, CV_64F);

        for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
        {
            const double
                    &x1 = pointsL[pointIdx].x,
                    &y1 = pointsL[pointIdx].y,
                    &x2 = pointsR[pointIdx].x,
                    &y2 = pointsR[pointIdx].y;

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

        cv::Mat eigenValues, eigenVectors;
        cv::Mat AtA = A.t() * A;
        cv::eigen(AtA, eigenValues, eigenVectors);

        cv::Mat F = eigenVectors.row(eigenVectors.rows - 1); // x = [f1 f2 f3 f4 f5 f6 f7 f8 f9]
        F = F.reshape(0, 3);
        return F;
    }

    int SparseMatching::getRansacIterationNumber(
            const int& pointNumber,
            const int& inlierNumber,
            const int& sampleSize,
            const double& confidence
    ) {
        const double inlierRatio = static_cast<float>(inlierNumber) / pointNumber;

        static const double log1 = log(1.0 - confidence);
        const double log2 = log(1.0 - pow(inlierRatio, sampleSize));

        const int k = log1 / log2;
        if (k < 0)
            return INT_MAX;
        return k;
    }

    cv::Mat SparseMatching::evaluateNormalizationEffect(const ImageKeypoints<cv::Point2d> &pointsL, const ImageKeypoints<cv::Point2d> &pointsR,
                                                        const NormalizedPoints normalizedPoints,
                                                        const std::vector<size_t> &inliers) {
        ImageKeypoints<cv::Point2d> sourceInliers;
        ImageKeypoints<cv::Point2d> destinationInliers;
        ImageKeypoints<cv::Point2d> normalizedSourceInliers;
        ImageKeypoints<cv::Point2d> normalizedDestinationInliers;

        for (const auto& inlierIdx : inliers) {
            sourceInliers.emplace_back(pointsL[inlierIdx]);
            destinationInliers.emplace_back(pointsR[inlierIdx]);
            normalizedSourceInliers.emplace_back(normalizedPoints.normalizedPointsL[inlierIdx]);
            normalizedDestinationInliers.emplace_back(normalizedPoints.normalizedPointsR[inlierIdx]);
        }

        // Estimate the fundamental matrix from the original points
        cv::Mat unnormalizedFundamentalMatrix = apply8PointMatchingAlgorithm(sourceInliers, destinationInliers);

        // Estimate the fundamental matrix from the normalized original points
        cv::Mat normalizedFundamentalMatrix = apply8PointMatchingAlgorithm(normalizedSourceInliers, normalizedDestinationInliers);

        normalizedFundamentalMatrix = normalizedPoints.Tr.t() * normalizedFundamentalMatrix * normalizedPoints.Tl; // Denormalize the fundamental matrix

        // Calculating the error of unnormalized and normalized fundamental matrices
        double errorUnNormalizedF = 0.0, errorNormalizedF = 0.0;
        for (size_t i = 0; i < inliers.size(); ++i) {
            // Symmetric epipolar distance
            cv::Mat ptL = (cv::Mat_<double>(3, 1) << sourceInliers[i].x, sourceInliers[i].y, 1);
            cv::Mat ptR = (cv::Mat_<double>(3, 1) << destinationInliers[i].x, destinationInliers[i].y, 1);

            // Calculate the error
            cv::Mat lL = unnormalizedFundamentalMatrix.t() * ptR;
            cv::Mat lR = unnormalizedFundamentalMatrix * ptL;

            // Calculate the distance of point pt1 from lL
            const double
                    & aL = lL.at<double>(0),
                    & bL = lL.at<double>(1),
                    & cL = lL.at<double>(2);

            double tL = abs(aL * sourceInliers[i].x + bL * sourceInliers[i].y + cL);
            double dL = sqrt(aL * aL + bL * bL);
            double distanceL = tL / dL;

            // Calculate the distance of point pt2 from lR
            const double
                    & aR = lR.at<double>(0),
                    & bR = lR.at<double>(1),
                    & cR = lR.at<double>(2);

            double tR = abs(aR * destinationInliers[i].x + bR * destinationInliers[i].y + cR);
            double dR = sqrt(aR * aR + bR * bR);
            double distanceR = tR / dR;

            double dist = 0.5 * (distanceL + distanceR);
            errorUnNormalizedF += dist;
        }

        for (size_t i = 0; i < inliers.size(); ++i) {
            // Symmetric epipolar distance
            cv::Mat ptL = (cv::Mat_<double>(3, 1) << sourceInliers[i].x, sourceInliers[i].y, 1);
            cv::Mat ptR = (cv::Mat_<double>(3, 1) << destinationInliers[i].x, destinationInliers[i].y, 1);

            // Calculate the error
            cv::Mat lL = normalizedFundamentalMatrix.t() * ptR;
            cv::Mat lR = normalizedFundamentalMatrix * ptL;

            // Calculate the distance of point pt1 from lL
            const double
                    & aL = lL.at<double>(0),
                    & bL = lL.at<double>(1),
                    & cL = lL.at<double>(2);

            double tL = abs(aL * sourceInliers[i].x + bL * sourceInliers[i].y + cL);
            double dL = sqrt(aL * aL + bL * bL);
            double distanceL = tL / dL;

            // Calculate the distance of point pt2 from lR
            const double
                    & aR = lR.at<double>(0),
                    & bR = lR.at<double>(1),
                    & cR = lR.at<double>(2);

            double tR = abs(aR * destinationInliers[i].x + bR * destinationInliers[i].y + cR);
            double dR = sqrt(aR * aR + bR * bR);
            double distanceR = tR / dR;

            double dist = 0.5 * (distanceL + distanceR);
            errorNormalizedF += dist;
        }
        errorUnNormalizedF = errorUnNormalizedF / inliers.size();
        errorNormalizedF = errorNormalizedF / inliers.size();

//        std::cout << "Error of the unnormalized fundamental matrix is "<< errorUnNormalizedF <<" px."<< std::endl;
//        std::cout << "Error of the normalized fundamental matrix is "<< errorNormalizedF <<" px."<< std::endl;

        return normalizedFundamentalMatrix;
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
    std::pair<cv::Mat, cv::Mat> SparseMatching::computeProjections(
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
                stereoImage.set_relative_cam_transformation(R, t);
            }
        }
        // std::cout << "projection_1 " << projection_1 <<std::endl;
        // std::cout << "projection_2 " << projection_2 <<std::endl;
        // std::cout << "rotation_1 " << rotation_1 <<std::endl;
        // std::cout << "rotation_2 " << rotation_2 <<std::endl;
        // std::cout << "translation " << translation <<std::endl;
        return {std::move(projection_1), std::move(projection_2)};;
    }
}


