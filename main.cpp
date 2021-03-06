/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <memory>
#include <time.h>
#include <sys/stat.h>
#include <signal.h>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

#include "src/libsgm.h"
#include "stereo_camera.hpp"

#define ASSERT_MSG(expr, msg)          \
    if (!(expr))                       \
    {                                  \
        std::cerr << msg << std::endl; \
        std::exit(EXIT_FAILURE);       \
    }

struct device_buffer
{
    device_buffer() : data(nullptr) {}
    device_buffer(size_t count) { allocate(count); }
    void allocate(size_t count) { cudaMalloc(&data, count); }
    ~device_buffer() { cudaFree(data); }
    void *data;
};

static bool is_streaming = true;
static void sig_handler(int sig)
{
    is_streaming = false;
}

static cv::CommandLineParser getConfig(int argc, char **argv)
{
    const char *params = "{ help           | false              | print usage          }"
                         "{ fps            | 30                 | (int) Frame rate }"
                         "{ width          | 1280               | (int) Image width }"
                         "{ height         | 720                | (int) Image height }"
                         "{ disp_size      | 128                | (int) Maximum disparity }"
                         "{ subpixel       | true               | Compute subpixel accuracy }"
                         "{ num_paths      | 4                  | (int) Num path to optimize, 4 or 8 }"
                         "{ out_depth      | 16                 | (int) Disparity image's bits per pixel}";

    cv::CommandLineParser config(argc, argv, params);
    if (config.get<bool>("help"))
    {
        config.printMessage();
        exit(0);
    }

    return config;
}

int main(int argc, char *argv[])
{
    // handle signal by user
    struct sigaction act;
    act.sa_handler = sig_handler;
    sigaction(SIGINT, &act, NULL);

    // parse config from cmd
    cv::CommandLineParser config = getConfig(argc, argv);

    StereoCameraConfig camConfig;
    camConfig.fps = config.get<int>("fps");
    camConfig.width = config.get<int>("width");
    camConfig.height = config.get<int>("height");

    StereoCamera::Ptr camera = std::make_shared<StereoCamera>(camConfig);
    if (!camera->checkCameraStarted())
    {
        std::cout << "Camera open fail..." << std::endl;
        return 0;
    }

    // Stereo Camera
    cv::Mat frame_0, frame_1, frame_0_rect, frame_1_rect;
    cv::Mat disp16, disp32;
    StereoCameraData camDataCamera;

    cv::FileStorage fs("ocams_calibration_720p.xml", cv::FileStorage::READ);

    cv::Mat D_L, K_L, D_R, K_R;
    cv::Mat Rect_L, Proj_L, Rect_R, Proj_R, Q;
    cv::Mat baseline;
    cv::Mat Rotation, Translation;

    fs["D_L"] >> D_L;
    fs["K_L"] >> K_L;
    fs["D_R"] >> D_R;
    fs["K_R"] >> K_R;
    fs["baseline"] >> baseline;
    fs["Rotation"] >> Rotation;
    fs["Translation"] >> Translation;

    // Code to calculate Rotation matrix and Projection matrix for each camera
    cv::Vec3d Translation_2((double *)Translation.data);

    cv::stereoRectify(K_L, D_L, K_R, D_R, cv::Size(camConfig.width, camConfig.height), Rotation, Translation_2,
                      Rect_L, Rect_R, Proj_L, Proj_R, Q, cv::CALIB_ZERO_DISPARITY);

    cv::Mat map11, map12, map21, map22;

    cv::initUndistortRectifyMap(K_L, D_L, Rect_L, Proj_L, cv::Size(camConfig.width, camConfig.height), CV_32FC1, map11, map12);
    cv::initUndistortRectifyMap(K_R, D_R, Rect_R, Proj_R, cv::Size(camConfig.width, camConfig.height), CV_32FC1, map21, map22);

    const int disp_size = config.get<int>("disp_size");
    const int out_depth = config.get<int>("out_depth");
    const bool subpixel = config.get<bool>("subpixel");
    const int num_paths = config.get<int>("num_paths");

    ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");
    if (subpixel)
    {
        ASSERT_MSG(out_depth == 16, "output depth bits must be 16 if subpixel option is enabled.");
    }
    else
    {
        ASSERT_MSG(out_depth == 8 || out_depth == 16, "output depth bits must be 8 or 16");
    }
    ASSERT_MSG(num_paths == 4 || num_paths == 8, "number of scanlines must be 4 or 8");

    while(1)
    {
        if (camera->getCamData(camDataCamera))
        {
            // read the next frames
            frame_0 = camDataCamera.frame_0;
            frame_1 = camDataCamera.frame_1;
            break;
        }
    }

	const int input_depth = frame_0.type() == CV_8U ? 8 : 16;
	const int input_bytes = input_depth * camConfig.width * camConfig.height / 8;
	const int output_depth = disp_size < 256 ? 8 : 16;
	const int output_bytes = output_depth * camConfig.width * camConfig.height / 8;

	sgm::StereoSGM sgm(camConfig.width, camConfig.height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

	const int invalid_disp = output_depth == 8
			? static_cast< uint8_t>(sgm.get_invalid_disparity())
			: static_cast<uint16_t>(sgm.get_invalid_disparity());

	cv::Mat disparity(camConfig.height, camConfig.width, output_depth == 8 ? CV_8U : CV_16U);
	cv::Mat disparity_8u, disparity_color;

	device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);

    while (1)
    {
        if (!is_streaming)
        {
            std::cout << "Exit by user signal" << std::endl;
            break;
        }
        if (camera->getCamData(camDataCamera))
        {
            // read the next frames
            frame_0 = camDataCamera.frame_0;
            frame_1 = camDataCamera.frame_1;

            cv::remap(frame_0, frame_0_rect, map11, map12, cv::INTER_LINEAR);
            cv::remap(frame_1, frame_1_rect, map21, map22, cv::INTER_LINEAR);

            cudaMemcpy(d_I1.data, frame_0_rect.data, input_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_I2.data, frame_1_rect.data, input_bytes, cudaMemcpyHostToDevice);

            auto t = std::chrono::steady_clock::now();

            sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
            cudaDeviceSynchronize();

            std::chrono::milliseconds dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t);
            const double fps = 1000 / dur.count();

            cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);

            cv::Mat disparity_8u, disparity_color;
            disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);
            cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
            disparity_color.setTo(cv::Scalar(0, 0, 0), static_cast< uint8_t>(sgm.get_invalid_disparity()));

            cv::putText(disparity_color, "sgm execution time: " + std::to_string(dur.count()) + "[msec] " + std::to_string(fps) + "[FPS]",
                        cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));


            cv::imshow("disparity", disparity_color);
            const char c = cv::waitKey(1);
            if (c == 27) // ESC
                break;
        }
    }

    return 0;
}
