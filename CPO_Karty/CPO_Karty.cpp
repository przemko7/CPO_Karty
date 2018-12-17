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
	int red_number;
	bool red;
	vector<cv::Moments> figures_moments;
	vector<vector<double>> hu_moments;

	vector<cv::Vec4i>						hierarchy;
	std::vector<std::vector<cv::Point> >	contours;
	vector<int>								card_contour_index;
};

struct Images {
	string file_path;
	cv::Mat image;
	cv::Mat preprocessed_image;
	cv::Mat binary;
	bool divideable_cards;

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

void CheckCornersPointsOrientation(Card_Corners& temp) {
	double x_mean, y_mean;
	x_mean = (temp.left_bottom.x + temp.right_top.x) / 2;
	y_mean = (temp.left_bottom.y + temp.right_top.y) / 2;
	while (temp.left_bottom.x > x_mean ||
		temp.left_top.x > x_mean ||
		temp.right_top.x < x_mean ||
		temp.right_bottom.x < x_mean) {
		cv::Point2f t_point;
		t_point = temp.left_bottom;
		temp.left_bottom = temp.right_bottom;
		temp.right_bottom = temp.right_top;
		temp.right_top = temp.left_top;
		temp.left_top = t_point;
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
	CheckCornersPointsOrientation(temp);
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
	int t_number=-4;
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

bool CheckRed(cv::Mat& image) {
	int t_counter = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.at<unsigned char>(i,j) > 25 && image.at<unsigned char>(i,j) < 75) {
				t_counter++;
			}
		}
	}
	if (t_counter > 4000) {
		return 1;
	}
	else {
		return 0;
	}
}

void CountMoments(Card& card) {
	for (int i = 0; i < card.contours.size(); i++) {
		cv::Moments mu;
		mu = cv::moments(card.contours[i]);
		card.figures_moments.push_back(mu);
	}
}

void CountHuMoments(Card& card) {
	card.hu_moments.clear();
	vector<double> temp(7);
	card.hu_moments.assign(card.figures_moments.size(), temp);
	for (int i = 0; i < card.figures_moments.size(); i++){
		vector<double> temp(7);
		cv::HuMoments(card.figures_moments[i], temp);
		card.hu_moments[i]=temp;
	}
}

void SetSymbol(Card& card) {
	int S_counter = 0, D_counter=0, C_counter=0, H_counter=0;
	for (int i = 0; i < card.figures_moments.size(); i++) {
		if (card.figures_moments[i].m00 > 1500 &&
			card.figures_moments[i].m00 < 5000) {
			if (card.hu_moments[i][2] > 0.0009){
				H_counter++;
			}
			if (card.hu_moments[i][2] > 0.00001 &&
				card.hu_moments[i][2] < 0.0005) {
				C_counter++;
			}
		}
	}
	if (H_counter > 0) {
		card.symbol = 'H';
		return ;
	}
	if (C_counter > 0) {
		card.symbol = 'C';
		return ;
	}
	if (card.red == 1) {
		card.symbol = 'D';
		return;
	}
	else {
		card.symbol = 'S';
		return;
	}

}


bool CheckDivideable(Images& image) {
	for (int i = 0; i < image.card.size(); i++) {
		for (int j = 0; j < image.card.size(); j++) {
			if (image.card[i].number % 
				image.card[j].number == 0 &&
				i != j) {
				return true;
			}
		}
	}
	return false;
}

void SetDivideableCardsAttribute(vector<Images>& images) {
	for (int i = 0; i < images.size(); i++) {
		images[i].divideable_cards = CheckDivideable(images[i]);
	}
}



void WriteCards(vector<Images>& images)
 {
	for (int i = 0; i < images.size(); i++) {
		cout << images[i].file_path<<"\t\t";
		for (int j = 0; j < images[i].card_contour_index.size(); j++) {
			cout << images[i].card[j].number << images[i].card[j].symbol << "\t";
		}
		if (images[i].divideable_cards) {
			cout << "Yes, number of one card can divide another";
		}
		else {
			cout << "No, there are no divideable cards";

		}
		cout << "\n";
	}
}

void WriteCardMoments(Card& card) {
	for (int i = 0; i < card.figures_moments.size(); i++) {
		if (card.figures_moments[i].m00 > 1500 &&
			card.figures_moments[i].m00 < 5000) {
			for (int j = 0; j < card.hu_moments[i].size(); j++) {
				cout << card.hu_moments[i][j] << "\t";
			}
		}
		cout << "\n";
	}
	cout << "\n\n";
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

	//TODO: ulepszyć preprocessing

	for (int i = 0; i < images_files.size(); i++) {
		for (int j = 0; j < images_files[i].card.size(); j++) {
			images_files[i].card[j].red=CheckRed(images_files[i].card[j].image);
			CreateBinaryImage(images_files[i].card[j].image, images_files[i].card[j].image);
			FindContoursCard(images_files[i].card[j]);
			SetCardNumber(images_files[i].card[j]);
			CountMoments(images_files[i].card[j]);
			CountHuMoments(images_files[i].card[j]);
			SetSymbol(images_files[i].card[j]);
		}
	}
	
	SetDivideableCardsAttribute(images_files);

	/*WriteCardMoments(images_files[0].card[3]);
	WriteCardMoments(images_files[12].card[0]);
	WriteCardMoments(images_files[12].card[2]);
	WriteCardMoments(images_files[12].card[3]);*/

	WriteCards(images_files);

	file_images_paths.close();
	return  0;
}



