#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigen>

#include <opencv2/opencv.hpp>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;

void LoadImage(const string folderImage, const int imageNumber);

class Kmeans
{
public:
    Kmeans() {  }


    static void clustering(const vector<Eigen::Vector3d>& point_cloud, int k, int max_iters, double epsilon_th, vector<int>& classes)
    {
        if(point_cloud.size() < k)
        {
            cout<<"point cloud number is later cluster classes!"<<endl;
            return;
        }

        classes = vector<int>(point_cloud.size(), 0);  // 每个点的类别标签

        // 1.随机选择k个中心点
        vector<Eigen::Vector3d> k_means;
        k_means.reserve(k);  // 预留k个内存空间

        cv::RNG rng(cv::getTickCount());  // 初始化随机种子
        set<int> selected_ids;
        while(selected_ids.size() < k)
        {
            int id = rng.uniform(0, point_cloud.size());
            if( !selected_ids.count(id) )  // set中不包含该元素
            {
                selected_ids.insert(id);
                k_means.push_back( point_cloud.at(id) );
            }
        }

        for(size_t iter = 0; iter < max_iters; ++iter)
        {
            // 2.将每个点分配到对应的聚类中心点
            for(size_t i = 0; i < point_cloud.size(); ++i)
            {
                const Eigen::Vector3d& pt = point_cloud.at( i );

                // 标记每个点属于哪个聚类中心点
                int classes_id = -1;
                double min_dist = std::numeric_limits<double>::max();

                for(size_t j = 0; j < k_means.size(); ++j)
                {
                    const Eigen::Vector3d& mean = k_means.at( j );
                    double dist = (pt - mean).norm();  // 计算L2欧式距离
                    if(dist < min_dist)  // 标记欧式聚类最小点的序号，将其划分为本类别
                    {
                        min_dist = dist;
                        classes_id = j;
                    }
                }
                classes.at(i) = classes_id;  // 给当前点分类到对应的聚类中心点
            }

            // 3.重新计算聚类中心
            vector<Eigen::Vector3d> new_mean(k, Eigen::Vector3d(0., 0., 0.));  // 每个中心点区域的所有点之和
            vector<int> new_classes(k, 0);  //每个中心点包含的点数量
            for (size_t i = 0; i < point_cloud.size(); ++i)
            {
                const Eigen::Vector3d& pt = point_cloud.at( i );
                const int class_id = classes.at( i );

                new_mean.at( class_id ) += pt;
                new_classes.at( class_id )++; 
            }

            for(int j = 0; j < k; ++j)
            {
                new_mean.at( j ) /= (double)(new_classes.at( j ));  // 重新计算每个聚类中心
            }

            // 4.检查聚类是否满足条件，满足则直接跳出（聚类中心不在变化）
            double delta_mean = 0.0;
            for(int i = 0; i < k; ++i)
            {
                delta_mean += (new_mean.at(i) - k_means.at(i)).norm();
            }

            if(delta_mean < epsilon_th)
                break;
            
            k_means = new_mean;  // 更新聚类中心
        }

    }

};


int main(int argc, char** argv)
{

    // 1.TUM数据集测试（彩色和深度图）
    if(argc != 3)
    {
        cerr << "Usage: ./testKmeans  path_to_image  image_number" << endl;
        return 0;
    }



    string folderImage = argv[1];

    LoadImage(folderImage, atoi(argv[2]));

    return 0;
}

void LoadImage(const string folderImage, const int imageNumber)
{
    // 相机内参
    const double fx = 530.395;
    const double fy = 530.758;
    const double cx = 319.659;
    const double cy = 239.168;
    const double depth_factor = 5000.0;
    const int imageWidth  = 640;
    const int imageHeight = 480; 
    const std::vector<uint8_t> colors = {213,0,0,197,17,98,170,0,255,98,0,234,48,79,254,41,98,255,0,145,234,0,184,212,0,191,165,0,200,83,100,221,23,174,234,0,255,214,0,255,171,0,255,109,0,221,44,0,62,39,35,33,33,33,38,50,56,144,164,174,224,224,224,161,136,127,255,112,67,255,152,0,255,193,7,255,235,59,192,202,51,139,195,74,67,160,71,0,150,136,0,172,193,3,169,244,100,181,246,63,81,181,103,58,183,171,71,188,236,64,122,239,83,80, 213,0,0,197,17,98,170,0,255,98,0,234,48,79,254,41,98,255,0,145,234,0,184,212,0,191,165,0,200,83,100,221,23,174,234,0,255,214,0,255,171,0,255,109,0,221,44,0,62,39,35,33,33,33,38,50,56,144,164,174,224,224,224,161,136,127,255,112,67,255,152,0,255,193,7,255,235,59,192,202,51,139,195,74,67,160,71,0,150,136,0,172,193,3,169,244,100,181,246,63,81,181,103,58,183,171,71,188,236,64,122,239,83,80};

    // 创建点云
    vector<Eigen::Vector3d> pts_cloud;
    pts_cloud.reserve(imageWidth * imageHeight);

    // pcl显示设置
    pcl::visualization::CloudViewer viewer ("Clusters");
    
    for(int i = 1; i < imageNumber; ++i)
    {
        string folderRGB = folderImage + "/rgb/rgb";
        string folderDepth = folderImage + "/depth/depth";
        cv::Mat imageRGB = cv::imread(folderRGB + to_string(i) + ".png", CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imageDepth = cv::imread(folderDepth + to_string(i) + ".png", CV_LOAD_IMAGE_UNCHANGED);

        if(imageRGB.empty())
        {
            cerr << "image is empty!!!" << endl;
            return ;
        }

        cv::imshow("imageRGB", imageRGB);
        cv::waitKey(1);

        pts_cloud.clear();
        for(int v = 0; v < imageDepth.rows; ++v)
        {
            for(int u = 0; u < imageDepth.cols; ++u)
            {
                unsigned int d = imageDepth.ptr<unsigned short>(v)[u];
                if(d == 0)
                {
                    continue;
                }

                double z = d * depth_factor;
                double x = (u - cx) * z / fx;
                double y = (v - cy) * z / fy;

                pts_cloud.push_back(Eigen::Vector3d(x, y, z));
            }
        }

        vector<int> classes;
        Kmeans::clustering(pts_cloud, 10, 50, 1e-6, classes);  // 点云，聚类中心数量，迭代次数，类别

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        for(size_t i = 0; i < pts_cloud.size(); ++i)
        {
            const Eigen::Vector3d& pts = pts_cloud.at( i );
            const int& class_id = classes.at(i);
            pcl::PointXYZRGB pt;
            pt.x = pts.x();
            pt.y = pts.y();
            pt.z = pts.z();

            pt.r = colors.at(class_id*3);
            pt.g = colors.at(class_id*3+1);
            pt.b = colors.at(class_id*3+2);

            cloud->points.push_back(pt);
        }
        viewer.showCloud(cloud);

    }
}