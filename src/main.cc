// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov5_seg.h"
#include "image_utils.h"
// #include "file_utils.h"
// #include "image_drawing.h"
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <iostream>
#include <functional>
#include <boost/bind.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <numeric>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <deque>

struct Image_And_Odresults{
    cv::Mat img;
    object_detect_result_list od_results;
};

std::deque<cv::Mat> d_input_images;
std::mutex input_image_mtx;
std::condition_variable input_image_cond;

std::deque<Image_And_Odresults> d_inference_images_and_od_results;
std::mutex inference_image_mtx;
std::condition_variable inference_image_cond;

// 全局退出标志
std::atomic<bool> g_exit_flag(false);

/*-------------------------------------------
            yolov5_seg Function
-------------------------------------------*/

void run_yolov5_seg(cv::Mat& src_img, rknn_app_context_t& rknn_app_ctx, object_detect_result_list& od_results){
    int ret = 0;
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);

    ret = inference_yolov5_seg_model_syy(&rknn_app_ctx, src_img, &od_results);
    if (ret != 0)
    {
        printf("init_yolov5_seg_model fail! ret=%d\n", ret);
    }

    cv::cvtColor(src_img, src_img, cv::COLOR_RGB2BGR);

    Image_And_Odresults image_and_odresults;
    image_and_odresults.img = src_img;
    image_and_odresults.od_results = od_results;

    std::unique_lock<std::mutex> u_lock(inference_image_mtx);
    {
        d_inference_images_and_od_results.emplace_back(std::move(image_and_odresults));
    }
    inference_image_cond.notify_one();
}

/*-------------------------------------------
             image_msg callback
-------------------------------------------*/

void imageCallback(const sensor_msgs::ImageConstPtr& msg){
    try{
        // 转换ROS图像消息为OpenCV格式
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        cv::Mat frame = cv_ptr->image;

        std::unique_lock<std::mutex> u_lock(input_image_mtx);
        {
            d_input_images.emplace_back(std::move(frame));
        }
        input_image_cond.notify_one();
        
        // run_yolov5_seg(frame, rknn_app_ctx, od_results);
    }
    catch(cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void run_yolov5_seg_loop(rknn_app_context_t& rknn_app_ctx, object_detect_result_list& od_results){
    cv::Mat frame;
    while(!g_exit_flag){
        std::unique_lock<std::mutex> u_lock(input_image_mtx);
        {
            // 设置条件变量等待
            input_image_cond.wait(u_lock, [](){
                return !d_input_images.empty() || g_exit_flag;
            });
            if(g_exit_flag)
                break;

            frame = std::move(d_input_images.front());
            d_input_images.pop_front();
        }
        run_yolov5_seg(frame, rknn_app_ctx, od_results);
    }
}

void get_frame_from_ros(ros::NodeHandle nh){
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub = 
        it.subscribe("video/image", 1, boost::bind(imageCallback, _1));

    // 使用ROS自带的多线程工具
    ros::AsyncSpinner spinner(1);
    spinner.start();

    while(ros::ok() && !g_exit_flag){
        ros::spinOnce();
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    spinner.stop();
    
    // ros::spin();
}

void draw_mask_and_box_loop(){
    Image_And_Odresults image_and_odresults;
    cv::Mat frame;
    object_detect_result_list od_results;

    while(!g_exit_flag){
        struct timeval start_time, stop_time;
        gettimeofday(&start_time, NULL);

        std::unique_lock<std::mutex> u_lock(inference_image_mtx);
        inference_image_cond.wait(u_lock, [](){
            return !d_inference_images_and_od_results.empty();
        });

        {
            image_and_odresults = std::move(d_inference_images_and_od_results.front());
            d_inference_images_and_od_results.pop_front();
        }

        frame = image_and_odresults.img;
        od_results = std::move(image_and_odresults.od_results);
        draw_mask_and_box(frame, od_results, start_time, "video");
        // draw_mask_and_box_test(frame, *od_results, start_time);
    }
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    std::cout << " ROS cv and lidar " << std::endl;

    const char *model_path = argv[1];

    // rknn init
    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolov5_seg_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov5_seg_model fail! ret=%d model_path=%s\n", ret, model_path);
    }

    object_detect_result_list od_results;

    // ros init
    ros::init(argc, argv, "rknn_yolov5_demo");
	ros::NodeHandle nh;

    std::thread get_frame_from_ros_thread(get_frame_from_ros, nh);
    std::thread run_yolov5_seg_loop_thread(run_yolov5_seg_loop, std::ref(rknn_app_ctx), std::ref(od_results));
    std::thread draw_mask_and_box_loop_thread(draw_mask_and_box_loop);

    // 等待ROS关闭
    ros::waitForShutdown();

    // 通知所有线程退出
    g_exit_flag = true;
    input_image_cond.notify_all();
    inference_image_cond.notify_all();

    // 等待所有线程结束
    get_frame_from_ros_thread.join();
    run_yolov5_seg_loop_thread.join();
    draw_mask_and_box_loop_thread.join();


    // image_transport::ImageTransport it(nh);
    // image_transport::Subscriber image_sub = 
    //     it.subscribe("video/image", 1, boost::bind(imageCallback, _1, rknn_app_ctx, od_results));

    // ros::spin();

    // run_yolov5_seg(src_img, rknn_app_ctx, od_results);

    // 清理资源
    deinit_post_process();

    ret = release_yolov5_seg_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov5_seg_model fail! ret=%d\n", ret);
    }

    return 0;
}
