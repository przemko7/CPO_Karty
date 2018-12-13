#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <string>
using namespace std;

struct Card_Corners {
	cv::Point2f left_bottom;
	cv::Point2f left_top;
	cv::Point2f right_bottom;
	cv::Point2f right_top;
};

struct Card {
	Card_Corners corners_on_image;
	cv::Mat image;
	int number;
	char symbol;

	vector<cv::Vec4i>						hierarchy;
	std::vector<std::vector<cv::Point> >	contours;
	vector<int>								card_contour_index;
};

struct Images {
	string file_path;
	cv::Mat image;
	cv::Mat preprocessed_image;
	cv::Mat binary;

	vector<Card>		card;
	vector<int>			card_contour_index;
	vector<cv::Vec4i>	hierarchy;
	std::vector<std::vector<cv::Point> > contours;
	vector<cv::RotatedRect> min_rect;
};


void OpenFileImagesPaths(fstream& file, string path) {
	file.open(path, ios::in | ios::out);
	try {
		if (!file.is_open()) {
			throw "File is missing";
		}
	}
	catch (string error) {
		cerr << error << endl;
		cerr << "Type correct path to file with paths to images";
		string t_path;
		cin >> t_path;
		OpenFileImagesPaths(file, t_path);
	}
	return;
}

void LoadImage(vector<Images> &vector, fstream &file) {
	Images temp;
	getline(file, temp.file_path);
	temp.image = cv::imread(temp.file_path, 0);
	if (temp.image.empty()) {
		cout << "Path " << temp.file_path << " has no file" << endl;
	}
	vector.push_back(temp);
	return;
}

void Preprocessing(vector<Images>& images_files) {
	for (int i = 0; i < images_files.size(); i++) {
		cv::Mat temp = images_files[i].image.clone();
		cv::medianBlur(temp, temp, 5);
		cv::medianBlur(temp, temp, 5);
		images_files[i].preprocessed_image = temp.clone();
	}
}

void CreateBinaryImage(cv::Mat& input, cv::Mat& output) {
	cv::threshold(input, output, -1, 255, CV_THRESH_OTSU);
}

void OpenOperation(cv::Mat& image, int level) {
	for (int i = 0; i < level; i++) {
		cv::dilate(image, image, cv::Mat());
	}

	for (int i = 0; i < level; i++) {
		cv::erode(image, image, cv::Mat());
	}
}

void FindContoursImages(vector<Images>& images_files) {
	for (int i = 0; i < images_files.size(); i++) {
		cv::findContours(images_files[i].binary,
						 images_files[i].contours,
						 images_files[i].hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
	}
}

void FindContoursCard(Card& card) {
	cv::findContours(card.image,
		card.contours,
		card.hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
}

void SetHierarchyCardIndex(vector<int>&				card_contour_index,
							  const vector<cv::Vec4i>&	hierarchy) 
{
	for (int i = 0; i < hierarchy.size(); i++) {
		if (hierarchy[i][3] == -1) {
			card_contour_index.push_back(i);
		}
	}
}

void SetImagesCardsIndex(vector<Images>& images_files) {
	for (int i = 0; i < images_files.size(); i++) {
		SetHierarchyCardIndex(images_files[i].card_contour_index, images_files[i].hierarchy);
	}
}

void CreateCardsStructs(vector<Images>& images) {
	for (int i = 0; i < images.size(); i++) {
		for (int j = 0; j < images[i].card_contour_index.size(); j++) {
			Card temp;
			images[i].card.push_back(temp);
		}
	}
}

void CreateMinRect(vector<Images>& images) {
	for (int j = 0; j < images.size(); j++) {
		vector<cv::RotatedRect> min_rect(images[j].contours.size());
		for (size_t i = 0; i < images[j].contours.size(); i++)
		{
			min_rect[i] = minAreaRect(cv::Mat(images[j].contours[i]));
		}
		images[j].min_rect = min_rect;
	}
}

void SetCardsRectanglesCornerPoints(Card_Corners&					card_corners_positions,
								    const vector<cv::RotatedRect>&	min_rect,
								    const vector<int>&				card_contour_index,
									int j) 
{
	Card_Corners temp;
	cv::Point2f t_corners[4];
	min_rect[card_contour_index[j]].points(t_corners);
	temp.left_bottom = t_corners[0];
	temp.left_top = t_corners[1];
	temp.right_top = t_corners[2];
	temp.right_bottom = t_corners[3];
	card_corners_positions = temp;
}

void SetImagesCardsRectanglesCornerPoints(vector<Images>& images) {
	for (int i = 0; i < images.size(); i++) {
		for (int j = 0; j < images[i].card.size(); j++) {
			SetCardsRectanglesCornerPoints(images[i].card[j].corners_on_image,
										   images[i].min_rect,
										   images[i].card_contour_index,
										   j);
		}
	}
}

void SplitCards(Card& card,
				cv::Mat& source, 
				vector<cv::Point2f>& output_quad) 
{
	vector<cv::Point2f> input_quad = { card.corners_on_image.left_top,
									   card.corners_on_image.right_top,
									   card.corners_on_image.right_bottom,
									   card.corners_on_image.left_bottom };

	cv::Mat lambda = cv::Mat::zeros(source.rows, source.cols, source.type());
	lambda = cv::getPerspectiveTransform(input_quad, output_quad);
	card.image = cv::Mat(600, 400, source.type());
	cv::warpPerspective(source, card.image, lambda, card.image.size());
}

void SplitCardsOnImages(vector<Images> &images) {
	vector<cv::Point2f> output_quad = {
		cv::Point2f(0, 0),
		cv::Point2f(400, 0),
		cv::Point2f(400, 600),
		cv::Point2f(0, 600)
	};

	for (int i = 0; i < images.size(); i++) {
		for (int j = 0; j < images[i].card_contour_index.size(); j++) {
			SplitCards(images[i].card[j], images[i].preprocessed_image, output_quad);
		}
	}

	
}

void SetCardNumber(Card& card) {
	int t_number=0;
	for (int i = 0; i < card.hierarchy.size(); i++) {
		if (card.hierarchy[i][3] == 0) t_number++;
	}

	if (t_number > 10) {
		card.number = 10;
	}
	else {
		card.number = t_number;
	}
}

int main() {
	fstream file_images_paths;
	vector<Images> images_files;

	OpenFileImagesPaths(file_images_paths, "images_path.txt");
	while (!file_images_paths.eof()) {
		LoadImage(images_files, file_images_paths);
	}

	Preprocessing(images_files);
	for (int i = 0; i < images_files.size(); i++) {
		CreateBinaryImage(images_files[i].preprocessed_image, images_files[i].binary);
		OpenOperation(images_files[i].binary, 1);
	}

	FindContoursImages(images_files);

	SetImagesCardsIndex(images_files);

	CreateCardsStructs(images_files);

	CreateMinRect(images_files);

	SetImagesCardsRectanglesCornerPoints(images_files);

	SplitCardsOnImages(images_files);

	//TODO: symbole i ulepszyć preprocessing

	for (int i = 0; i < images_files.size(); i++) {
		for (int j = 0; j < images_files[i].card.size(); j++) {
			CreateBinaryImage(images_files[i].card[j].image, images_files[i].card[j].image);
			FindContoursCard(images_files[i].card[j]);
			SetCardNumber(images_files[i].card[j]);
		}
	}
	

	/*cv::Mat drawing = cv::Mat::zeros(binary_rotated.size(), CV_8UC3);
	cv::RNG rng(12345);
	for (int i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, cv::Point());
		cv::Point2f rect_points[4];
		minRect[i].points(rect_points);
		for (int j = 0; j < 4; j++)
			line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
	}

	cv::namedWindow("Contours", CV_WINDOW_NORMAL);
	cvResizeWindow("Contours", 1000, 750);
	imshow("Contours", drawing);

	cv::waitKey(1);*/

	file_images_paths.close();
	return  0;
}