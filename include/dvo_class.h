//
// Created by hide on 18/09/06.
//

#ifndef SIMPLE_DVO_DVO_CLASS_H
#define SIMPLE_DVO_DVO_CLASS_H

#include "ros/ros.h"
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_alignment.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Path.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <pcl/common/transforms.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Geometry>

/**
 * @class DVO
 * @brief Class for Direct Visual Odometry
 */
class DVO {
private:
    ros::NodeHandle nh;
    Eigen::Matrix4f accumulated_transform = Eigen::Matrix4f::Identity();
    cv::Mat img_prev, depth_prev;
    Eigen::Matrix3f K;
    ros::Publisher pub_pointcloud;
    tf::TransformBroadcaster br;

public:
    DVO(ros::NodeHandle nh_input){
        nh = nh_input;
    }

    std::vector<geometry_msgs::PoseStamped> cam_pose_vec_;

    ros::Publisher cam_odom_pub;
    ros::Publisher cam_path_pub;

    void makePointCloud( const cv::Mat& img_rgb, const cv::Mat& img_depth, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud);

    void callback(const sensor_msgs::ImageConstPtr& image_rgb, const sensor_msgs::ImageConstPtr& image_depth, const sensor_msgs::CameraInfoConstPtr& info);
};


/**
 * @brief Generate colored pointcloud from rgb/depth image.
 */
void DVO::makePointCloud( const cv::Mat& img_rgb, const cv::Mat& img_depth, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud) {
    cloud->header.frame_id = "/cam_origin";
    cloud->is_dense = true;
    cloud->height = img_depth.rows;
    cloud->width = img_depth.cols;
    cloud->points.resize( cloud->height*cloud->width );

    float fxi = 1.f / K(0,0), fyi = 1.f / K(1,1);
    float cx = K(0,2), cy = K(1,2);

    int idx = 0;
    float* depthdata = (float*)( &img_depth.data[0] );
    unsigned char* colordata = &img_rgb.data[0];

    for(int y = 0; y < img_depth.rows; y++ ) {
        for(int x = 0; x < img_depth.cols; x++ ) {
            pcl::PointXYZRGB& p = cloud->points[idx];
            p.z = (float)(*depthdata);
            p.x = (x - cx) * p.z * fxi;
            p.y = (y - cy) * p.z * fyi;
            depthdata++;

            int b = (*colordata++);
            int g = (*colordata++);
            int r = (*colordata++);
            int rgb = (r << 16) + (g << 8) + b;
            p.rgb = *((float*)(&rgb));

            idx++;
        }
    }

    return;

}


/**
 * @brief Subscribe images, run direct image alignment, and publish pointcloud and camera pose.
 */
void DVO::callback(const sensor_msgs::ImageConstPtr& image_rgb, const sensor_msgs::ImageConstPtr& image_depth, const sensor_msgs::CameraInfoConstPtr& info){

    //Initialization.
    K <<    info->K[0], 0.0, info->K[2],
            0.0, info->K[4], info->K[5],
            0.0, 0.0, 1.0;

    cv_bridge::CvImageConstPtr img_rgb_cv_ptr = cv_bridge::toCvShare( image_rgb, "bgr8" );
    cv_bridge::CvImageConstPtr img_depth_cv_ptr = cv_bridge::toCvShare( image_depth, "32FC1" );

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    cv::Mat img_curInt, img_cur, depth_cur;
    cv::cvtColor( img_rgb_cv_ptr->image.clone(), img_curInt, CV_BGR2GRAY);
    img_curInt.convertTo(img_cur, CV_32FC1, 1.f/255.f);
    depth_cur = img_depth_cv_ptr->image.clone();

    //Run image alignment.
    DirectImageAlignment dia;
    if( !img_prev.empty() )
        dia.doAlignment( transform, img_prev, depth_prev, img_cur, depth_cur, K );

    //Update variables.
    accumulated_transform = accumulated_transform * transform.inverse();
    img_prev = img_cur.clone();
    depth_prev = depth_cur.clone();

    //Publish pointcloud.
    ros::Time timestamp = ros::Time::now();
    pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB > );
    makePointCloud( img_rgb_cv_ptr->image, img_depth_cv_ptr->image, cloud );
    pcl::transformPointCloud( *cloud, *cloud, accumulated_transform );
    pub_pointcloud = nh.advertise< pcl::PointCloud< pcl::PointXYZRGB >>( "pointcloud", 1 );
    pub_pointcloud.publish( *cloud );

    //Publish camera pose.
    tf::Transform tform;
    tform.setOrigin( tf::Vector3(accumulated_transform(0,3), accumulated_transform(1,3), accumulated_transform(2,3)));
    tf::Matrix3x3 rotation;
    rotation.setValue(
                accumulated_transform(0,0), accumulated_transform(0,1), accumulated_transform(0,2),
                accumulated_transform(1,0), accumulated_transform(1,1), accumulated_transform(1,2),
                accumulated_transform(2,0), accumulated_transform(2,1), accumulated_transform(2,2)
                );
    tform.setBasis(rotation);
    br.sendTransform(tf::StampedTransform(tform, timestamp, "cam_origin", "camera"));

    tf::Quaternion tf_odom_quat;
    rotation.getRotation(tf_odom_quat);

    geometry_msgs::PoseStamped odom_msg;
    odom_msg.header.stamp       = ros::Time::now();
    odom_msg.header.frame_id    = "cam_origin";
    odom_msg.pose.position.x    = accumulated_transform(0,3);
    odom_msg.pose.position.y    = accumulated_transform(1,3);
    odom_msg.pose.position.z    = accumulated_transform(2,3);
    odom_msg.pose.orientation.x = tf_odom_quat[0];
    odom_msg.pose.orientation.y = tf_odom_quat[1];
    odom_msg.pose.orientation.z = tf_odom_quat[2];
    odom_msg.pose.orientation.w = tf_odom_quat[3];


    cam_odom_pub.publish(odom_msg);

    cam_pose_vec_.push_back(odom_msg);

    nav_msgs::Path odom_path;
    odom_path.header.stamp  = ros::Time::now();
    odom_path.header.frame_id = "cam_origin";
    odom_path.poses = cam_pose_vec_;
    cam_path_pub.publish(odom_path);

    return;
}


#endif //SIMPLE_DVO_DVO_CLASS_H
