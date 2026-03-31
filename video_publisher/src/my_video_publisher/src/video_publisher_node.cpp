#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]){
    ros::init(argc, argv, "video_publisher_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("video/image", 1);

    std::string video_path;
    nh.param<std::string>("video_path", video_path, "/home/orangepi/yolov5_seg_ros/videos/rong2_long.mp4");

    cv::VideoCapture cap(video_path);
    if(!cap.isOpened()){
        ROS_ERROR("Could not open video file: %s", video_path.c_str());
        return -1;
    }

    cv::Mat frame;
    sensor_msgs::ImagePtr msg;

    // 尝试以30fps发布
    ros::Rate loop_rate(30);

    while(ros::ok()){
        cap >> frame;
        if(frame.empty()){
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // 将OpenCV图像转换为ROS图像消息，编码类型为BGR8
        try{
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
            pub.publish(msg);
        }

        catch(cv_bridge::Exception& e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }

        loop_rate.sleep();
    }

    cap.release();
    return 0;
}
