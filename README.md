# Multi View Stereo
An implementation of a stereo-pipeline based on block-matching.
Multiview-pointcloud registration uses Open3D implementations for ICP and coarse alignment.

Code was developed and tested under Ubuntu 22.04 LTS. 
## Dependencies:
* OpenCV with contrib modules as per: https://github.com/opencv/opencv_contrib
* Open3D library for C++: https://www.open3d.org/docs/release/compilation.html
* cmake
<br><br>
Auto-install via cmake:
* Yaml-cpp is used for nice user IO, but installation is automated within cmake file
https://github.com/jbeder/yaml-cpp
## Compile:
* mkdir -p build && cd build
* cmake ..
* make

## Run:
### Images
* Download desired images from the Midleburry dataset: https://vision.middlebury.edu/stereo/data/scenes2021/ <br>
(Or use own dataset and introduce the same structure.) <br>
Point-cloud registration for multiple images is only available for ground-truth data (*.pfm)
* An example structure would be: 
```
./data/artroom 
		|-- artroom1 
			|-- calib.txt 
			|-- im0.png 
			|-- im1.png 
			|-- disp0.pfm (Optional -> Ground-Truth)
			|-- disp1.pfm (Optional Ground-Truth)
			|-- dist_coeffs.txt
		|-- artroom2
			|-- calib.txt
			|-- im0.png
			|-- im1.png
			|-- disp0.pfm (Optional -> Ground-Truth)
			|-- disp1.pfm (Optional Ground-Truth)
			|-- dist_coeffs.txt (Optional -> Undistort/Rectify)
```
Make sure to not add any other files to the filestructure as this may result in errors while parsing it.
The outputfiles will be located in the same folder as the original scenery.
The resulting depth image always corresponds to the <u>LEFT</u> stereo image.

* Edit config.yaml file to suit individual purpose

* cd build <br>
* ./stereo_fusion --path=../config.yaml

## References
### Rectification follows:
https://people.scs.carleton.ca/~c_shu/Courses/comp4900d/notes/rectification.pdf
https://www.cs.cmu.edu/~16385/s17/Slides/13.1_Stereo_Rectification.pdf

### Corresponding OpenCV methods to our own implementations are implemented for comparison:
https://opencv.org/

### PointCloud registration is implemented using Open3D:
https://www.open3d.org/docs/release/index.html
