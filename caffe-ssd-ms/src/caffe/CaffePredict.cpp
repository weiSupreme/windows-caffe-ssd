#include<iostream>
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>

#include "caffe/caffe.hpp"

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <utility>
#include <vector>
#include <io.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace caffe;

class CaffePredict
{
public:
	CaffePredict(const std::string& model_file,
		const std::string& weights_file,


		//const string& mean_file,
		//const string& mean_value,
		const int gpu_id);

	std::vector<float> Predict(const cv::Mat& img);

private:
	//void SetMean(const string& mean_file, const string& mean_value);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	boost::shared_ptr<caffe::Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	//cv::Mat mean_;

};

CaffePredict::CaffePredict(const string& model_file,
							const string& weights_file,
							//const string& mean_file,
							//const string& mean_value,
							const int gpu_id) {
	if (gpu_id < 0)
		Caffe::set_mode(Caffe::CPU);
	else
	{
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(gpu_id);
	}

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(weights_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	//SetMean(mean_file, mean_value);
}


std::vector<float> CaffePredict::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];//output_layerָ��������������ݣ��洢����������ݵ�blob�Ĺ����(1,c,1,1)
	const float* begin = output_layer->cpu_data();//beginָ���������ݶ�Ӧ�ĵ�һ��ĸ���
	const float* end = begin + output_layer->channels();//endָ���������ݶ�Ӧ�����һ��ĸ���
	return std::vector<float>(begin, end);//�����������ݾ�������ǰ����������Ķ�Ӧ�ڸ�����ķ���
}

void CaffePredict::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void CaffePredict::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	//cv::Mat sample_normalized;
	//cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_float, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}