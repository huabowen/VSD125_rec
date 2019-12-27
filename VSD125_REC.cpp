#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <string>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/common/transforms.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/features/board.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/pfh.h>
#include <thread>
#include <mutex>
//NOMINMAX
//_ARM64_
using namespace std;  // ���Լ��� std �������ռ�

#define Feature PFHSignature125
#define H histogram

mutex m;
struct i_p_t {
	int i = 0;
	float percent = 0.0f;
	Eigen::Matrix4f trans;
};
struct c_k_f {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr key;
	pcl::PointCloud<pcl::Feature>::Ptr feature;
};



///////////////////��ʾ//////////////////////////////////////////////////////////////////////////////
void show_key_scene(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr key) {
	// ��ʼ�����ƿ��ӻ�����
	pcl::visualization::PCLVisualizer view("3D Viewer");
	view.setBackgroundColor(255, 255, 255);  //��ɫ����

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(cloud, 0, 0, 255);//BLUE
	view.addPointCloud<pcl::PointXYZ>(cloud, color_cloud, "1");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key(key, 255, 0, 0);//�ؼ���
	view.addPointCloud<pcl::PointXYZ>(key, color_key, "2");
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "2");

	// �ȴ�ֱ�����ӻ����ڹر�
	while (!view.wasStopped())
	{
		view.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_key_model(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr key) {
	// ��ʼ�����ƿ��ӻ�����
	pcl::visualization::PCLVisualizer view("3D Viewer");
	view.setBackgroundColor(255, 255, 255);  //��ɫ����

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(cloud, 0, 255, 0);//GREEN
	view.addPointCloud<pcl::PointXYZ>(cloud, color_cloud, "1");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key(key, 255, 0, 0);//�ؼ���
	view.addPointCloud<pcl::PointXYZ>(key, color_key, "2");
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "2");

	// �ȴ�ֱ�����ӻ����ڹر�
	while (!view.wasStopped())
	{
		view.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_source_target_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target) {
	// ��ʼ�����ƿ��ӻ�����
	pcl::visualization::PCLVisualizer view("3D Viewer");
	view.setBackgroundColor(255, 255, 255);  //��ɫ����

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_source(cloud_source, 0, 255, 0);//��ɫ����
	view.addPointCloud<pcl::PointXYZ>(cloud_source, color_source, "1");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_target(cloud_target, 0, 0, 255);//��ɫ����
	view.addPointCloud<pcl::PointXYZ>(cloud_target, color_target, "2");

	// �ȴ�ֱ�����ӻ����ڹر�
	while (!view.wasStopped())
	{
		view.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_point_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	// ��ʼ�����ƿ��ӻ�����
	pcl::visualization::PCLVisualizer view("3D Viewer");
	view.setBackgroundColor(255, 255, 255); //��ɫ����
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(cloud, 0, 255, 0);//��ɫ����
	view.addPointCloud<pcl::PointXYZ>(cloud, color_cloud, "1");
	// �ȴ�ֱ�����ӻ����ڹر�
	while (!view.wasStopped())
	{
		view.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_point_clouds(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds) {
	// ��ʼ\�����ƿ��ӻ�����
	pcl::visualization::PCLVisualizer viewer_final("3D Viewer");
	viewer_final.setBackgroundColor(255, 255, 255);   //��ɫ����
	for (int i = 0; i < clouds.size(); i++) {
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(clouds[i], rand() % 255, rand() % 255, rand() % 255);
		viewer_final.addPointCloud<pcl::PointXYZ>(clouds[i], color, std::to_string(i));
	}
	// �ȴ�ֱ�����ӻ����ڹر�
	while (!viewer_final.wasStopped())
	{
		viewer_final.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_coor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scenes,
	pcl::PointCloud<pcl::PointXYZ> keypoints_model, pcl::PointCloud<pcl::PointXYZ> keypoints_scenes,
	pcl::PointCloud<pcl::Feature>::Ptr features_model, pcl::PointCloud<pcl::Feature>::Ptr features_scenes,
	pcl::CorrespondencesPtr& corr) {

	for (int i = 0; i < corr->size(); i++) {
		cout << corr->at(i).index_query << "---" << corr->at(i).index_match << "---" << corr->at(i).distance << endl;
		pcl::visualization::PCLPlotter plotter;
		plotter.addFeatureHistogram<pcl::Feature>(*features_model, "pfh", corr->at(i).index_query);
		plotter.addFeatureHistogram<pcl::Feature>(*features_scenes, "pfh", corr->at(i).index_match);
		std::cout << features_model->points[corr->at(i).index_query] << endl;
		std::cout << features_scenes->points[corr->at(i).index_match] << endl;
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_model(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_scenes(new pcl::PointCloud<pcl::PointXYZ>());
		keypoints_ptr_model->push_back(keypoints_model.points[corr->at(i).index_query]);
		keypoints_ptr_scenes->push_back(keypoints_scenes.points[corr->at(i).index_match]);
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

		int v1(0);
		viewer->createViewPort(0.5, 0.0, 1, 1.0, v1);  //4�������ֱ���X�����Сֵ�����ֵ��Y�����Сֵ�����ֵ��ȡֵ0-1��v1�Ǳ�ʶ
		viewer->setBackgroundColor(255, 255, 255, v1);    //�����ӿڵı�����ɫ
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_model(keypoints_ptr_model, 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(keypoints_ptr_model, color_key_model, "color_key_model", v1);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "color_key_model");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_model(cloud_model, 0, 255, 0);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_model, color_cloud_model, "cloud_model", v1);
		int v2(0);
		viewer->createViewPort(0, 0.0, 0.5, 1.0, v2);  //4�������ֱ���X�����Сֵ�����ֵ��Y�����Сֵ�����ֵ��ȡֵ0-1��v1�Ǳ�ʶ
		viewer->setBackgroundColor(255, 255, 255, v2);    //�����ӿڵı�����ɫ
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_scenes(keypoints_ptr_scenes, 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(keypoints_ptr_scenes, color_key_scenes, "color_key_scenes", v2);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "color_key_scenes");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_scenes(cloud_scenes, 0, 0, 255);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_scenes, color_cloud_scenes, "cloud_scenes", v2);

		//plotter.plot();
		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}
	}
}
void show_line(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target,
	pcl::CorrespondencesPtr& corr, float& leaf_size) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_target(new pcl::PointCloud<pcl::PointXYZ>());
	for (int i = 0; i < corr->size(); i++) {
		new_key_source->push_back(key_source->points[corr->at(i).index_query]);
		new_key_target->push_back(key_target->points[corr->at(i).index_match]);
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	*new_cloud_source = *cloud_source;
	for (int i = 0; i < cloud_source->size(); i++) {
		new_cloud_source->points[i].x -= 200.0f* leaf_size;
	}
	for (int i = 0; i < new_key_source->size(); i++) {
		new_key_source->points[i].x -= 200.0f* leaf_size;
	}
	pcl::visualization::PCLVisualizer line("line");
	line.setBackgroundColor(255, 255, 255);
	line.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0, 255, 0), "cloud_target");
	line.addPointCloud(new_cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new_cloud_source, 0, 0, 255), "cloud_source");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>color_new_key_target(new_key_target, 255, 0, 0);
	line.addPointCloud<pcl::PointXYZ>(new_key_target, color_new_key_target, "new_key_target");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_target");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_new_key_source(new_key_source, 255, 0, 0);
	line.addPointCloud<pcl::PointXYZ>(new_key_source, color_new_key_source, "new_key_source");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_source");

	for (int i = 0; i < new_key_source->size(); i++)
	{
		pcl::PointXYZ source_point = new_key_source->points[i];
		pcl::PointXYZ target_point = new_key_target->points[i];
		line.addLine(source_point, target_point, 0, 0, 0, std::to_string(i));
		line.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, std::to_string(i));
	}
	line.spin();
}

void show_line(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target,
	std::vector<int>& corr, float& leaf_size) {

	//////////////////������ƽ�ƣ�������ʾ////////////////////////////////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_target(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < corr.size(); i++) {
		new_key_source->push_back(key_source->points[corr[i]]);
		new_key_target->push_back(key_target->points[corr[i]]);
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	*new_cloud_source = *cloud_source;
	for (int i = 0; i < new_cloud_source->size(); i++) {
		new_cloud_source->points[i].x += 300.0f* leaf_size;
		//new_cloud_source->points[i].y += 300.0f* leaf_size;
	}
	for (int i = 0; i < new_key_source->size(); i++) {
		new_key_source->points[i].x += 300.0f* leaf_size;
		//new_key_source->points[i].y += 300.0f* leaf_size;
	}
	////////////////////��ʾ��Ӧ������//////////////////////////////////////////////////////////////////////
	pcl::visualization::PCLVisualizer line("line");
	line.setBackgroundColor(255, 255, 255);
	line.addPointCloud(new_cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new_cloud_source, 0, 0, 255), "new_cloud_source");
	line.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0, 255, 0), "cloud_target");

	line.addPointCloud<pcl::PointXYZ>(new_key_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new_key_source, 255, 0, 0), "new_key_source");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_source");

	line.addPointCloud<pcl::PointXYZ>(new_key_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new_key_target, 255, 0, 0), "new_key_target");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_target");

	for (int i = 0; i < new_key_source->size(); i++)
	{
		line.addLine(new_key_source->points[i], new_key_target->points[i], 0, 0, 0, std::to_string(i));
		line.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, std::to_string(i));
	}
	line.spin();
}
void show_key_corr(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source,
	pcl::PointXYZ ps0, pcl::PointXYZ ps1, pcl::PointXYZ ps2,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target,
	pcl::PointXYZ pt0, pcl::PointXYZ pt1, pcl::PointXYZ pt2) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr key_ps0(new pcl::PointCloud<pcl::PointXYZ>);
	key_ps0->push_back(ps0);
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_ps(new pcl::PointCloud<pcl::PointXYZ>);
	key_ps->push_back(ps1);
	key_ps->push_back(ps2);
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_pt0(new pcl::PointCloud<pcl::PointXYZ>);
	key_pt0->push_back(pt0);
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_pt(new pcl::PointCloud<pcl::PointXYZ>);
	key_pt->push_back(pt1);
	key_pt->push_back(pt2);
	pcl::visualization::PCLVisualizer view("3D Viewer");
	int v1(0);
	int v2(1);
	view.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	view.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	view.setBackgroundColor(255, 255, 255, v1);
	view.setBackgroundColor(255, 255, 255, v2);
	view.addPointCloud<pcl::PointXYZ>(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0, 0, 0), "cloud_source", v1);
	view.addPointCloud<pcl::PointXYZ>(key_ps0, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(key_ps0, 255, 0, 0), "key_ps0", v1);
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_ps0", v1);
	view.addPointCloud<pcl::PointXYZ>(key_ps, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(key_ps, 0, 255, 0), "key_ps", v1);
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_ps", v1);
	view.addPointCloud<pcl::PointXYZ>(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0, 0, 0), "cloud_target", v2);
	view.addPointCloud<pcl::PointXYZ>(key_pt0, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(key_pt0, 255, 0, 0), "key_pt0", v2);
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_pt0", v2);
	view.addPointCloud<pcl::PointXYZ>(key_pt, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(key_pt, 0, 255, 0), "key_pt", v2);
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_pt", v2);
	while (!view.wasStopped()) {
		view.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_point_clouds_and_trans_models(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scenes,
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> models, std::vector<i_p_t> result_final) {

	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> models_results;
	pcl::visualization::PCLVisualizer viewer_final("3D Viewer");
	viewer_final.setBackgroundColor(255, 255, 255);   //��ɫ����
	for (int i = 0; i < scenes.size(); i++) {
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(scenes[i], rand() % 255, rand() % 255, rand() % 255);
		viewer_final.addPointCloud<pcl::PointXYZ>(scenes[i], color, "scenes" + to_string(i));
		viewer_final.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scenes" + to_string(i));
		pcl::PointCloud<pcl::PointXYZ>::Ptr models_result(new pcl::PointCloud<pcl::PointXYZ>);
		*models_result = *models[result_final[i].i];
		pcl::transformPointCloud(*models_result, *models_result, result_final[i].trans);
		models_results.push_back(models_result);
		if (result_final[i].percent < 0.8f)
			continue;
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_model(models_results[i], rand() % 255, rand() % 255, rand() % 255);
		viewer_final.addPointCloud<pcl::PointXYZ>(models_results[i], color_model, "models_results" + to_string(i));
	}
	while (!viewer_final.wasStopped()) {
		viewer_final.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}


/////////////////ѭ���˲�///////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::PointCloud<pcl::Normal>::Ptr normal_estimation_OMP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float radius) {
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	ne.setNumberOfThreads(10);
	ne.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(radius);
	//ne.setKSearch(k);
	ne.compute(*normals);
	return normals;
}

float com_leaf(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);//Acloud��Bcloud�н�������
	int K = 2;
	std::vector<int> pointIdxNKNSearch(K);//���������
	std::vector<float> pointNKNSquaredDistance(K);//��������
	float leaf_size = 0;
	for (int i = 0; i < cloud->size(); i++) {
		kdtree.nearestKSearch(cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance);
		leaf_size = leaf_size + sqrt(pointNKNSquaredDistance[1]);
		pointIdxNKNSearch.clear();
		pointNKNSquaredDistance.clear();
	}
	leaf_size = (float)leaf_size / (float)(cloud->size());
	//std::cout << "ƽ�����룺" << leaf_size << "���Ƶ�����" << cloud->size() << std::endl;
	return leaf_size;
}

float get_leaf_size_by_leaf_size(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size, float target_leaf_size) {
	return target_leaf_size + 0.2f*(target_leaf_size - leaf_size);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size) {
	//�����˲�
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;  //�����˲�����
	sor.setInputCloud(cloud);            //������Ҫ���˵ĵ��Ƹ��˲�����
	sor.setLeafSize(leaf_size, leaf_size, leaf_size);  //�����˲�ʱ�������������
	sor.filter(*cloud_filtered);           //ִ���˲������洢���	
	return cloud_filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr filter_to_leaf_size(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float target_leaf_size) {
	float leaf_size = com_leaf(cloud);
	if (target_leaf_size > leaf_size) {
		*cloud = *voxel_grid(cloud, target_leaf_size);
	}
	else {
		return cloud;
	}
	leaf_size = com_leaf(cloud);
	int num = 0;
	while (target_leaf_size > leaf_size + leaf_size * 0.02f) {
		num = cloud->size();
		leaf_size = get_leaf_size_by_leaf_size(cloud, leaf_size, target_leaf_size);
		*cloud = *voxel_grid(cloud, leaf_size);
		leaf_size = com_leaf(cloud);
		if (cloud->size() == num) {
			break;
		}
	}
	return cloud;
}

///////////////////////�ؼ������//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//bool is_max(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<float> avg_angle, int i, vector<bool>& pre_key,
//	pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, float max_rad = 5) {
//	vector<int> id;
//	vector<float> dis;
//	kdtree.radiusSearch(cloud->points[i], max_rad, id, dis);
//	for (int j = 1; j < id.size(); j++) {
//		if (avg_angle[id[0]] > avg_angle[id[j]])
//			pre_key[id[j]] = false;
//		else if (avg_angle[id[0]] < avg_angle[id[j]])
//			pre_key[id[0]] = false;
//	}
//	return pre_key[id[0]];
//}

bool is_max(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<float> avg_angle, int i, vector<bool>& pre_key,
	pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, float max_rad = 5) {
	vector<int> id;
	vector<float> dis;
	kdtree.radiusSearch(cloud->points[i], max_rad, id, dis);
	//float num = 0;
	//for (int j = 0; j < id.size(); j++) {
	//	if (avg_angle[id[0]] >= avg_angle[id[j]])
	//		num +=1.0f;
	//}
	//if (num >= 0.8*id.size())
	//	pre_key[id[0]] = true;
	//else
	//	pre_key[id[0]] = false;
	for (int i = 1; i < id.size(); i++) {
		if (pre_key[id[i]]) {
			if (avg_angle[id[0]] > avg_angle[id[i]])
				pre_key[id[i]] = false;
			else if (avg_angle[id[0]] < avg_angle[id[i]])
				pre_key[id[0]] = false;
		}
	}

	return pre_key[id[0]];
}

float com_angle(float cx, float cy, float cz, float nx, float ny, float nz) {
	if (isnan(nx) || isnan(ny) || isnan(nz) || isnan(cx) || isnan(cy) || isnan(cz) || (cx == nx && cy == ny && cz == nz))
		return 0;
	float cos_angle = (nx*cx + ny * cy + nz * cz) / (sqrtf(pow(nx, 2) + pow(ny, 2) + pow(nz, 2))*sqrtf(pow(cx, 2) + pow(cy, 2) + pow(cz, 2)));

	return acos(cos_angle)*180.0f / 3.1415926;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr key_detect(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::PointCloud<pcl::Normal>::Ptr normal, float avg_rad, float max_rad, vector<float>& l) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>);
	vector<bool> pre_key(cloud->size(), true);

	vector<float> avg_angle;

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	int num_pre_key = 0;
	for (int i = 0; i < cloud->size(); i++) {
		if (normal->points[i].curvature > 0.01) {
			num_pre_key++;
			std::vector<int> id;//���������
			std::vector<float> dis;//��������
			kdtree.radiusSearch(cloud->points[i], avg_rad, id, dis);
			if (id.size() < 10) {
				avg_angle.push_back(0);
				continue;
			}
			pcl::Normal avg_normal;
			for (int j = 0; j < id.size(); j++) {
				avg_normal.normal_x += normal->points[id[j]].normal_x;
				avg_normal.normal_y += normal->points[id[j]].normal_y;
				avg_normal.normal_z += normal->points[id[j]].normal_z;
			}
			avg_normal.normal_x = avg_normal.normal_x / float(id.size());
			avg_normal.normal_y = avg_normal.normal_y / float(id.size());
			avg_normal.normal_z = avg_normal.normal_z / float(id.size());
			float angle = 0;
			for (int j = 0; j < id.size(); j++) {
				angle += com_angle(normal->points[i].normal_x, normal->points[i].normal_y, normal->points[i].normal_z,
					normal->points[id[j]].normal_x, normal->points[id[j]].normal_y, normal->points[id[j]].normal_z);

				//angle += pow(normal->points[i].normal_x - avg_normal.normal_x, 2) +
				//	pow(normal->points[i].normal_y - avg_normal.normal_y, 2) +
				//	pow(normal->points[i].normal_z - avg_normal.normal_z, 2);
			}
			angle = angle / (float)id.size();
			avg_angle.push_back(angle);

		}
		else {
			avg_angle.push_back(0);
			pre_key[i] = false;
		}
	}
	cout << "Ԥ�ؼ���������" << num_pre_key << endl;
	for (int i = 0; i < cloud->size(); i++) {
		if (pre_key[i]) {
			if (is_max(cloud, avg_angle, i, pre_key, kdtree, max_rad) && avg_angle[i] != 0) {//�˴��뾶Ϊ�������ʺ͵İ뾶
				key->push_back(cloud->points[i]);
				l.push_back(avg_angle[i]);
			}
		}
	}
	return key;
}

void finall_key(pcl::PointCloud<pcl::PointXYZ>::Ptr key_source, vector<float> ls,
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_target, vector<float> lt,int num) {
	if (ls.size() < num || lt.size() < num)
		return;
	vector<float> ls_order = ls;
	vector<float> lt_order = lt;
	sort(ls_order.begin(), ls_order.end());
	sort(lt_order.begin(), lt_order.end());
	float lsmin, lsmax, ltmin, ltmax, lmin, lmax;
	lsmax = ls_order[ls.size() - 1];
	ltmax = lt_order[lt.size() - 1];
	lsmin = ls_order[ls.size() - num];
	ltmin = lt_order[lt.size() - num];
	if (lsmax< ltmin || lsmin>ltmax)
		return;
	lmin = max(lsmin, ltmin);
	lmax = min(lsmax, ltmax);

	pcl::PointCloud<pcl::PointXYZ>::Ptr finall_key_source(new pcl::PointCloud<pcl::PointXYZ>);
	*finall_key_source = *key_source;
	key_source->clear();
	pcl::PointCloud<pcl::PointXYZ>::Ptr finall_key_target(new pcl::PointCloud<pcl::PointXYZ>);
	*finall_key_target = *key_target;
	key_target->clear();

	for (int i = 0; i < finall_key_source->size(); i++) {
		if (ls[i] >= lmin && ls[i] <= lmax) {
			key_source->push_back(finall_key_source->points[i]);
		}
	}
	for (int i = 0; i < finall_key_target->size(); i++) {
		if (lt[i] >= lmin && lt[i] <= lmax) {
			key_target->push_back(finall_key_target->points[i]);
		}
	}
	return;
}


/////////////////////��������/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::PointCloud<pcl::PFHSignature125>::Ptr com_vsd125_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr key, pcl::PointCloud<pcl::Normal>::Ptr normal,
	float leaf_size, float m = 5.0f) {

	pcl::PointCloud<pcl::ReferenceFrame>::Ptr rf(new pcl::PointCloud<pcl::ReferenceFrame>());
	pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZ, pcl::Normal, pcl::ReferenceFrame> rf_est;
	rf_est.setFindHoles(true);
	rf_est.setRadiusSearch(5.0f*leaf_size);
	rf_est.setInputCloud(key);
	rf_est.setInputNormals(normal);
	rf_est.setSearchSurface(cloud);
	rf_est.compute(*rf);

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);

	pcl::PointCloud<pcl::PFHSignature125>::Ptr features(new pcl::PointCloud<pcl::PFHSignature125>);
	for (int i = 0; i < key->size(); i++) {

		pcl::PFHSignature125 feature;
		for (int i = 0; i < feature.descriptorSize(); i++) {
			feature.histogram[i] = 0;
		}

		if (isnan(rf->points[i].x_axis[0])) {
			features->push_back(feature);
			continue;
		}

		std::vector<int> id;//���������
		std::vector<float> dis;//��������
		kdtree.radiusSearch(key->points[i], m*2.5f*sqrt(3.0f)*leaf_size, id, dis);
		int num = 0;
		//pcl::PointCloud<pcl::PointXYZ>::Ptr before_trans_key_source(new pcl::PointCloud<pcl::PointXYZ>);
		//pcl::PointCloud<pcl::PointXYZ>::Ptr after_trans_key_source(new pcl::PointCloud<pcl::PointXYZ>);
		//pcl::PointCloud<pcl::PointXYZ>::Ptr key_id(new pcl::PointCloud<pcl::PointXYZ>);
		//key_id->push_back(key->points[i]);
		for (int j = 0; j < id.size(); j++) {

			pcl::PointXYZ point0;
			point0.x = cloud->points[id[j]].x - key->points[i].x;
			point0.y = cloud->points[id[j]].y - key->points[i].y;
			point0.z = cloud->points[id[j]].z - key->points[i].z;
			pcl::PointXYZ point;
			point.x = rf->points[i].x_axis[0] * point0.x
				+ rf->points[i].y_axis[0] * point0.y
				+ rf->points[i].z_axis[0] * point0.z;
			if (abs(point.x) >= m * 2.5f*leaf_size)
				continue;
			point.y = rf->points[i].x_axis[1] * point0.x
				+ rf->points[i].y_axis[1] * point0.y
				+ rf->points[i].z_axis[1] * point0.z;
			if (abs(point.y) >= m * 2.5f*leaf_size)
				continue;
			point.z = rf->points[i].x_axis[2] * point0.x
				+ rf->points[i].y_axis[2] * point0.y
				+ rf->points[i].z_axis[2] * point0.z;
			if (abs(point.z) >= m * 2.5f*leaf_size)
				continue;
			int x, y, z = 0;
			num += 1;
			x = (point.x + m * 2.5f* leaf_size) / (m*leaf_size);
			y = (point.y + m * 2.5f* leaf_size) / (m*leaf_size);
			z = (point.z + m * 2.5f* leaf_size) / (m*leaf_size);
			feature.histogram[x + 5 * y + 25 * z] += 1;
			//before_trans_key_source->push_back(cloud->points[id[j]]);
			//after_trans_key_source->push_back(point);
		}
		//cout << rf_source->points[i].x_axis[0]<<"    " << rf_source->points[i].x_axis[1] << "    "<< rf_source->points[i].x_axis[2] << endl;
		//cout << rf_source->points[i].y_axis[0] << "    " << rf_source->points[i].y_axis[1] << "    " << rf_source->points[i].y_axis[2] << endl;
		//cout << rf_source->points[i].z_axis[0] << "    " << rf_source->points[i].z_axis[1] << "    " << rf_source->points[i].z_axis[2] << endl;

		//show_one_key_cloud(cloud, before_trans_key_source, after_trans_key_source,key_id);
		for (int k = 0; k < feature.descriptorSize(); k++) {
			feature.histogram[k] = feature.histogram[k] / float(num);
		}
		features->push_back(feature);
	}

	return features;
}


pcl::CorrespondencesPtr com_first_corr(pcl::PointCloud<pcl::PFHSignature125>::Ptr feature_source,
	pcl::PointCloud<pcl::PFHSignature125>::Ptr feature_target, float dis_f) {

	pcl::CorrespondencesPtr corrs(new pcl::Correspondences());
	pcl::KdTreeFLANN<pcl::PFHSignature125> kdtree;   //������׼�ķ���
	kdtree.setInputCloud(feature_target);  //����ģ����Ƶ�������
	for (size_t i = 0; i < feature_source->size(); ++i)
	{
		std::vector<int> id(1);   //��������ڵ������
		std::vector<float> dis(1); //���������ƽ������ֵ
		int flag = kdtree.nearestKSearch(feature_source->at(i), 1, id, dis);
		if (flag == 1 && dis[0] < dis_f) // �������������ٽ����ƽ������С��0.25�����������ٽ��ľ�����һ����0��1֮�䣩�����ƥ��
		{
			//neigh_indices[0]�����㣬  i  ����׼��  neigh_sqr_dists[0]���ٽ����ƽ������
			pcl::Correspondence corr(static_cast<int> (i), id[0], dis[0]);
			corrs->push_back(corr);   //����׼�ĵ�洢��������
		}
	}
	return corrs;
}

/////////////////////����ͶƱ/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void hough_vote(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model, pcl::PointCloud<pcl::PointXYZ>::Ptr key_model,
	pcl::PointCloud<pcl::Normal>::Ptr normal_model,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scene, pcl::PointCloud<pcl::PointXYZ>::Ptr key_scene,
	pcl::PointCloud<pcl::Normal>::Ptr normal_scene,
	pcl::CorrespondencesPtr first_corr,
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& RT, std::vector<pcl::Correspondences>& corr,
	float rf_radius, float cg_size, float cg_thresh) {

	//
//  Compute (Keypoints) Reference Frames only for Hough
//
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr rf_model(new pcl::PointCloud<pcl::ReferenceFrame>());
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr rf_scene(new pcl::PointCloud<pcl::ReferenceFrame>());

	pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZ, pcl::Normal, pcl::ReferenceFrame> rf_est;
	rf_est.setFindHoles(true);
	rf_est.setRadiusSearch(rf_radius);

	rf_est.setInputCloud(key_model);
	rf_est.setInputNormals(normal_model);
	rf_est.setSearchSurface(cloud_model);
	rf_est.compute(*rf_model);

	rf_est.setInputCloud(key_scene);
	rf_est.setInputNormals(normal_scene);
	rf_est.setSearchSurface(cloud_scene);
	rf_est.compute(*rf_scene);

	//  Clustering
	pcl::Hough3DGrouping<pcl::PointXYZ, pcl::PointXYZ, pcl::ReferenceFrame, pcl::ReferenceFrame> HG;
	HG.setHoughBinSize(cg_size);
	HG.setHoughThreshold(cg_thresh);
	HG.setUseInterpolation(true);
	HG.setUseDistanceWeight(false);

	HG.setInputCloud(key_model);
	HG.setInputRf(rf_model);
	HG.setSceneCloud(key_scene);
	HG.setSceneRf(rf_scene);
	HG.setModelSceneCorrespondences(first_corr);

	//clusterer.cluster (clustered_corrs);
	HG.recognize(RT, corr);
}


///////////////////���Ƽ�����////////////////////////////////////
pcl::PointCloud<pcl::PointXYZ>::Ptr add_gaussian_noise(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float m) {
	float leaf_size = 0;
	leaf_size = com_leaf(cloud);
	//��Ӹ�˹����
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudfiltered(new pcl::PointCloud<pcl::PointXYZ>());
	cloudfiltered->points.resize(cloud->points.size());//�����Ƶ�cloud��size��ֵ������
	cloudfiltered->header = cloud->header;
	cloudfiltered->width = cloud->width;
	cloudfiltered->height = cloud->height;
	boost::mt19937 rng;
	rng.seed(static_cast<unsigned int>(time(0)));
	boost::normal_distribution<> nd(0, m*leaf_size);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> var_nor(rng, nd);
	//�������
	for (size_t point_i = 0; point_i < cloud->points.size(); ++point_i)
	{
		//cloudfiltered->points[point_i].x = cloud->points[point_i].x + static_cast<float> (var_nor());
		//cloudfiltered->points[point_i].y = cloud->points[point_i].y + static_cast<float> (var_nor());
		//cloudfiltered->points[point_i].z = cloud->points[point_i].z + static_cast<float> (var_nor());
		cloudfiltered->points[point_i].x = cloud->points[point_i].x + static_cast<float> (var_nor());
		cloudfiltered->points[point_i].y = cloud->points[point_i].y;
		cloudfiltered->points[point_i].z = cloud->points[point_i].z;
	}
	return cloudfiltered;
}

//////////////////ŷʽ�ָ�//////////////////////////
vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> euclidean_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	int tolerance = 4, int min = 1000, int max = 50000) {
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;   //ŷʽ�������
	ec.setClusterTolerance(tolerance);                     // ���ý��������������뾶Ϊ2cm
	ec.setMinClusterSize(min);                 //����һ��������Ҫ�����ٵĵ���ĿΪ100
	ec.setMaxClusterSize(max);               //����һ��������Ҫ��������ĿΪ25000
	ec.setSearchMethod(tree);                    //���õ��Ƶ���������
	ec.setInputCloud(cloud);
	ec.extract(cluster_indices);           //�ӵ�������ȡ���࣬������������������cluster_indices��
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
			cloud_cluster->points.push_back(cloud->points[*pit]);
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;
		clouds.push_back(cloud_cluster);
	}
	return clouds;
}

//////////////////�������һ����///////////////////////////
std::vector<int> ransac(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source,
	pcl::PointCloud<pcl::Feature>::Ptr feature_source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target,
	pcl::PointCloud<pcl::Feature>::Ptr feature_target,
	float leaf_size, Eigen::Matrix4f& trans) {

	////////////////////////�������һ����//////////////////////////////////////////////////////////////////////
	pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::Feature> align;
	align.setInputSource(key_source);
	align.setSourceFeatures(feature_source);
	align.setInputTarget(key_target);
	align.setTargetFeatures(feature_target);
	align.setMaximumIterations(10000); // Number of RANSAC iterations
	align.setNumberOfSamples(3); // Number of points to sample for generating/prerejecting a pose
	align.setCorrespondenceRandomness(3); // Number of nearest features to use
	align.setSimilarityThreshold(0.9f); // Polygonal edge length similarity threshold
	align.setMaxCorrespondenceDistance(2.5f*leaf_size); // Inlier threshold
	//align.setRANSACOutlierRejectionThreshold(5.0f * leaf_size);
	align.setInlierFraction(0.25f); // Required inlier fraction for accepting a pose hypothesis
	align.align(*key_source);
	//std::cout << "������ " << align.getFitnessScore(5.0f*leaf_size) << std::endl;
	trans = align.getFinalTransformation();
	pcl::transformPointCloud(*cloud_source, *cloud_source, trans);
	std::vector<int> corr;
	corr = align.getInliers();
	return corr;

}

///////////////////ICP/////////////////////////////////////////////
void icp(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, float& leaf_size, Eigen::Matrix4f& trans) {
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(cloud_source);
	icp.setInputTarget(cloud_target);
	icp.setTransformationEpsilon(0.1*leaf_size);
	icp.setMaxCorrespondenceDistance(5.0f * leaf_size);
	icp.setMaximumIterations(50000);
	icp.align(*cloud_source);
	//std::cout << "icp������ " << icp.getFitnessScore(1.0f*leaf_size) << std::endl;
	Eigen::Matrix4f TR = icp.getFinalTransformation();
	pcl::transformPointCloud(*cloud_source, *cloud_source, TR);
	trans = trans * TR;
}


//////////////////��׼����//////////////////////////////////////////////////////////////////
void my_align(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scene, pcl::PointCloud<pcl::PointXYZ>::Ptr key_scene,
	pcl::PointCloud<pcl::Normal>::Ptr normal_scene, pcl::PointCloud<pcl::Feature>::Ptr feature_scene,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model, pcl::PointCloud<pcl::PointXYZ>::Ptr key_model,
	pcl::PointCloud<pcl::Normal>::Ptr normal_model, pcl::PointCloud<pcl::Feature>::Ptr feature_model,
	float leaf_size) {

	////////////////////��ʼ��Ӧ��ϵ����////////////////////////////////////////////////////////////////////////
	pcl::CorrespondencesPtr first_corr(new pcl::Correspondences());
	first_corr = com_first_corr(feature_model, feature_scene, 0.015);
	std::cout << "first corr found: " << first_corr->size() << std::endl;
	if (first_corr->size() == 0)
		return;
	show_line(cloud_model,cloud_scene, key_model,key_scene , first_corr, leaf_size);
	//show_coor(cloud_model, cloud_scene, *key_model, *key_scene, feature_model, feature_scene, first_corr);
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > RT;
	std::vector<pcl::Correspondences> corr;
	hough_vote(cloud_model, key_model, normal_model, cloud_scene, key_scene, normal_scene,
		first_corr, RT, corr, 5.0f*leaf_size, 20, 5);
	std::cout << "Model instances found: " << RT.size() << std::endl;
	if (corr.size() == 0)
		return;
	//for (std::size_t i = 0; i < RT.size(); ++i)
	//{
	//	std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
	//	std::cout << "        Correspondences belonging to this instance: " << corr[i].size() << std::endl;
	//	// Print the rotation matrix and translation vector
	//	Eigen::Matrix3f rotation = RT[i].block<3, 3>(0, 0);
	//	Eigen::Vector3f translation = RT[i].block<3, 1>(0, 3);
	//	printf("\n");
	//	printf("            | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
	//	printf("        R = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
	//	printf("            | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
	//	printf("\n");
	//	printf("        t = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));
	//}
	pcl::visualization::PCLVisualizer viewer("Correspondence Grouping");
	viewer.setBackgroundColor(255, 255, 255);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> off_scene_color_handler(cloud_scene, 0, 0, 255);
	viewer.addPointCloud(cloud_scene, off_scene_color_handler, "scene_cloud");
	pcl::PointCloud<pcl::PointXYZ>::Ptr off_model(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr off_model_keypoints(new pcl::PointCloud<pcl::PointXYZ>());
	//  We are translating the model so that it doesn't end in the middle of the scene representation
	pcl::transformPointCloud(*cloud_model, *off_model, Eigen::Vector3f(200, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
	pcl::transformPointCloud(*key_model, *off_model_keypoints, Eigen::Vector3f(200, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> off_scene_model_color_handler(off_model, 0, 255, 0);
	viewer.addPointCloud(off_model, off_scene_model_color_handler, "off_scene_model");
	if (corr.size() == 0)
		return;
	int res = 0, max_size = 0;
	for (int i = 0; i < corr.size(); i++) {
		if (corr[i].size() > max_size) {
			res = i;
			max_size = corr[i].size();
		}
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_model(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::transformPointCloud(*cloud_model, *rotated_model, RT[res]);

	for (std::size_t i = 0; i < corr[res].size(); i++)
	{
		pcl::PointXYZ& model_point = off_model_keypoints->at(corr[res][i].index_query);
		pcl::PointXYZ& scene_point = key_scene->at(corr[res][i].index_match);
		viewer.addLine<pcl::PointXYZ, pcl::PointXYZ>(model_point, scene_point, 0, 0, 0, std::to_string(i));
		viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, std::to_string(i));
	}
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
	return;
}

int main() {
	string model_name, scene_name;
	float leaf_size = 1.0f;
	while (cin >> scene_name >> model_name) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scene(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::io::loadPLYFile("D:/PCD/ʶ����ƽǶ�����/scene/filter/" + scene_name + ".ply", *cloud_scene);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::io::loadPLYFile("D:/PCD/ʶ����ƽǶ�����/model/filter/" + model_name + ".ply", *cloud_model);
		double start, end;
		start = GetTickCount();
		pcl::PointCloud<pcl::Normal>::Ptr normal_scene(new pcl::PointCloud<pcl::Normal>);		
		*normal_scene = *normal_estimation_OMP(cloud_scene, 5.0f*leaf_size);
		pcl::PointCloud<pcl::PointXYZ>::Ptr key_scene(new pcl::PointCloud<pcl::PointXYZ>);
		vector<float> ls;
		*key_scene = *key_detect(cloud_scene, normal_scene, 5.0f*leaf_size, 5.0f * leaf_size, ls);
		cout << "�����ؼ��㣺" << key_scene->size() << endl;
		//show_key_cloud(cloud_scene, key_scene);
		pcl::PointCloud<pcl::Normal>::Ptr normal_model(new pcl::PointCloud<pcl::Normal>);
		*normal_model = *normal_estimation_OMP(cloud_model, 5.0f*leaf_size);
		pcl::PointCloud<pcl::PointXYZ>::Ptr key_model(new pcl::PointCloud<pcl::PointXYZ>);
		vector<float> lm;
		*key_model = *key_detect(cloud_model, normal_model, 5.0f*leaf_size, 5.0f * leaf_size, lm);
		cout << "ģ�͹ؼ��㣺" << key_model->size() << endl;
		//show_key_cloud(cloud_model, key_model);
		finall_key(key_scene, ls, key_model, lm,500);
		cout << "��������Ӧ�ؼ��㣺" << key_scene->size() << endl;
		cout << "ģ�ͻ���Ӧ�ؼ��㣺" << key_model->size() << endl;
		show_key_scene(cloud_scene, key_scene);
		show_key_model(cloud_model, key_model);

		pcl::PointCloud<pcl::PFHSignature125>::Ptr feature_scene(new pcl::PointCloud<pcl::PFHSignature125>);
		*feature_scene = *com_vsd125_feature(cloud_scene, key_scene, normal_scene, leaf_size, 5.0f);

		pcl::PointCloud<pcl::PFHSignature125>::Ptr feature_model(new pcl::PointCloud<pcl::PFHSignature125>);
		*feature_model = *com_vsd125_feature(cloud_model, key_model, normal_model, leaf_size, 5.0f);

		my_align(cloud_scene, key_scene, normal_scene, feature_scene, cloud_model, key_model, normal_model, feature_model, leaf_size);
		end = GetTickCount();
		cout << "ʶ��ʱ�䣺" << end - start << "ms" << endl;
	}
	return 0;
}


/////////////////////filter/////////////////////////////
//int main() {
//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//		"ganesha","gorilla","gun","horse","lioness","para" ,"trex","victoria","wolf" };
//	for (int i = 0; i < names.size(); i++) {
//		pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::io::loadPLYFile("D:/PCD/ʶ����ƽǶ�����/scene/scene/" + names[i] + ".ply", *scene);
//		*scene = *filter_to_leaf_size(scene, 1.0f);
//		pcl::io::savePLYFile("D:/PCD/ʶ����ƽǶ�����/scene/filter/" + names[i] + ".ply", *scene);
//
//		pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::io::loadPLYFile("D:/PCD/ʶ����ƽǶ�����/model/model/" + names[i] + ".ply", *model);
//		*model = *filter_to_leaf_size(model, 1.0f);
//		pcl::io::savePLYFile("D:/PCD/ʶ����ƽǶ�����/model/filter/" + names[i] + ".ply", *model);
//	}
//	return 0;
//}

/////////////////////���߼���/////////////////////////////////////////////////////////////
//int main(int argc, char** argv) {
//	string road = "D:/code/PCD/ʶ�����/model/";
//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//		"ganesha","gorilla","gun","horse","lioness","para" ,"trex","victoria","wolf" };
//
//	for (int i = 0; i < names.size(); i++) {
//		string name = "D:/PCD/ʶ����ƽǶ�����/model/filter/" + names[i] + ".ply";
//		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::io::loadPLYFile(name, *cloud);
//		float leaf_size = 1;
//
//		pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());
//		normal = normal_estimation_OMP(cloud, leaf_size*5.0f);
//		pcl::PointCloud<pcl::Feature>::Ptr feature(new pcl::PointCloud<pcl::Feature>());
//		pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>());
//		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
//		tree->setInputCloud(cloud);
//		*key = *key_detect(cloud, normal, 5.0f * leaf_size, 3.0f * leaf_size);
//		cout << names[i] << ": " << key->size() << endl;
//		*feature = *com_lsfh33_feature(cloud, normal, key, 10.0f * leaf_size);
//		pcl::io::savePLYFile("D:/PCD/ʶ����ƽǶ�����/model/key/" + names[i] + "_key.ply", *key);
//		pcl::io::savePLYFile("D:/PCD/ʶ����ƽǶ�����/model/feature/" + names[i] + "_feature.ply", *feature);
//	}
//	return 0;
//}

////////////////////////ʶ��///////////////////////////////////////////////////////////////////////////////////////////
//int main(int argc, char** argv) {
//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//	"ganesha","gorilla","horse","para" ,"trex","wolf" };
//
//	//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//	//		"ganesha","gorilla","gun","horse","lioness","para" ,"trex","victoria","wolf" };
//
//	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> models;
//	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> keys;
//	vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> features;
//	for (int i = 0; i < names.size(); i++) {
//		pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature(new pcl::PointCloud<pcl::FPFHSignature33>);
//		pcl::io::loadPLYFile("D:/PCD/ʶ����ƽǶ�����/model/filter/" + names[i] + ".ply", *model);
//		pcl::io::loadPLYFile("D:/PCD/ʶ����ƽǶ�����/model/key/" + names[i] + "_key.ply", *key);
//		pcl::io::loadPLYFile("D:/PCD/ʶ����ƽǶ�����/model/feature/" + names[i] + "_feature.ply", *feature);
//		models.push_back(model);
//		keys.push_back(key);
//		features.push_back(feature);
//	}
//	string name;
//	while (cin >> name) {
//		pcl::PointCloud<pcl::PointXYZ>::Ptr scenes(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::io::loadPLYFile("D:/PCD/ʶ����ƽǶ�����/scene/filter/" + name + ".ply", *scenes);
//		vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
//		float leaf_size = 1.0f;
//		clouds = euclidean_cluster(scenes, 10.0f*leaf_size);
//		vector<i_p_t> result_final;
//		com_k_f(clouds, leaf_size);
//
//		for (int i = 0; i < clouds.size(); i++) {
//			result_final.push_back(predict(clouds_keys_features[i].cloud, clouds_keys_features[i].key,
//				clouds_keys_features[i].feature, names, models, keys, features));
//			if (result_final[i].percent > 0.8)
//				cout << names[result_final[i].i] << endl;
//			else
//				cout << "nnnnnnnnnn" << endl;
//		}
//		show_point_clouds(clouds);
//		show_point_clouds_and_trans_models(clouds, models, result_final);
//	}
//
//	return 0;
//}