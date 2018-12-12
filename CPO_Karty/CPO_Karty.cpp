#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
using namespace std;

struct Card_Corners {
	cv::Point2i left_bottom;
	cv::Point2i left_top;
	cv::Point2i right_bottom;
	cv::Point2i right_top;
};

struct Card {
	cv::Mat contour;
	int number;
	char symbol;
};

struct Images {
	string file_path;
	cv::Mat image;
	vector<Card> card;
};


void open_file_images_path(fstream& file, string path) {
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
		open_file_images_path(file, t_path);
	}
	return;
}


void load_image(vector<Images> &vector, fstream &file) {
	Images temp;
	getline(file, temp.file_path);
	temp.image = cv::imread(temp.file_path, 0);
	if (temp.image.empty()) {
		cout << "Path " << temp.file_path << " has no file" << endl;
	}
	vector.push_back(temp);
	return;
}


void set_hierarchy_card_index(vector<int>&				card_contour_index,
							  const vector<cv::Vec4i>&	hierarchy) 
{
	for (int i = 0; i < hierarchy.size(); i++) {
		if (hierarchy[i][3] == -1) {
			card_contour_index.push_back(i);
		}
	}
}


void set_cards_extreme_points(vector<Card_Corners>&						 card_corners_positions,
							  const vector<cv::RotatedRect>&			 minRect,
							  const vector<int>&						 card_contour_index) 
	{
		for (int i = 0; i < card_contour_index.size(); i++) {
			Card_Corners temp;
			cv::Point2f t_corners[4];
			minRect[card_contour_index[i]].points(t_corners);
			temp.left_bottom = t_corners[0];
			temp.left_top = t_corners[1];
			temp.right_top = t_corners[2];
			temp.right_bottom = t_corners[3];
			card_corners_positions.push_back(temp);
	}
}

int main() {
	fstream file_images_paths;
	vector<Images> images_files;

	open_file_images_path(file_images_paths, "images_path.txt");
	load_image(images_files, file_images_paths);
	/*while (!file_images_paths.eof()) {
		load_image(images_files, file_images_paths);
	}*/
	cv::Mat clone, clone_rotated, binary, binary_rotated;
	clone = images_files[0].image.clone();
	std::vector<std::vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;

	cv::Mat rotation = cv::getRotationMatrix2D(cv::Point2f(binary.cols / 2, binary.rows / 2), -5, 1);

	cv::medianBlur(clone, images_files[0].image, 3);
	cv::medianBlur(clone, images_files[0].image, 3);
	cv::warpAffine(clone, clone_rotated, rotation, binary.size());

	cv::threshold(clone_rotated, binary_rotated, 100, 255, CV_THRESH_OTSU);

	cv::erode(binary_rotated, binary_rotated, cv::Mat());
	cv::dilate(binary_rotated, binary_rotated, cv::Mat());
	cv::dilate(binary_rotated, binary_rotated, cv::Mat());
	cv::erode(binary_rotated, binary_rotated, cv::Mat());

	cv::findContours(binary_rotated, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

	vector<cv::RotatedRect> minRect(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		minRect[i] = minAreaRect(cv::Mat(contours[i]));
	}


	vector<int> card_contour_index;
	card_contour_index.clear();
	set_hierarchy_card_index(card_contour_index, hierarchy);

	vector<Card_Corners> card_positions;
	card_positions.clear();
	set_cards_extreme_points(card_positions, minRect, card_contour_index);


	Card temp;
	cv::Point2f input_quad[4] = { card_positions[0].left_top,
								  card_positions[0].right_top,
								  card_positions[0].right_bottom,
								  card_positions[0].left_bottom };

	cv::Point2f output_quad[4];
	output_quad[0] = { 0, 0 };
	output_quad[1] = { 400, 0 };
	output_quad[2] = { 400, 600 };
	output_quad[3] = { 0, 600 };

	cv::Mat lambda = cv::Mat::zeros(clone_rotated.rows, clone_rotated.cols, clone_rotated.type());

	lambda = getPerspectiveTransform(input_quad, output_quad);
	temp.contour = cv::Mat(600, 400, clone_rotated.type());
	cv::warpPerspective(clone_rotated, temp.contour, lambda, temp.contour.size());

	cv::Mat drawing = cv::Mat::zeros(binary_rotated.size(), CV_8UC3);
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

	cv::waitKey(1);

	file_images_paths.close();
	return  0;
}