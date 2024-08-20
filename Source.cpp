#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <queue>

#include <zxing/LuminanceSource.h>
#include <zxing/MultiFormatReader.h>
#include <zxing/oned/OneDReader.h>
#include <zxing/oned/EAN8Reader.h>
#include <zxing/oned/EAN13Reader.h>
#include <zxing/oned/Code128Reader.h>
#include <zxing/datamatrix/DataMatrixReader.h>
#include <zxing/qrcode/QRCodeReader.h>
#include <zxing/aztec/AztecReader.h>
#include <zxing/common/GlobalHistogramBinarizer.h>
#include <zxing/Exception.h>

using namespace cv;
using namespace std;

using namespace zxing;
using namespace qrcode;
using namespace oned;
using namespace datamatrix;
using namespace aztec;

const int CV_QR_NORTH = 0;
const int CV_QR_EAST = 1;
const int CV_QR_SOUTH = 2;
const int CV_QR_WEST = 3;

float cv_distance(Point2f P, Point2f Q);					
float cv_lineEquation(Point2f L, Point2f M, Point2f J);		
float cv_lineSlope(Point2f L, Point2f M, int& alignement);	
void cv_getVertices(vector<vector<Point> > contours, int c_id, float slope, vector<Point2f>& X);
void cv_updateCorner(Point2f P, Point2f ref, float& baseline, Point2f& corner);
void cv_updateCornerOr(int orientation, vector<Point2f> IN, vector<Point2f> &OUT);
bool getIntersectionPoint(Point2f a1, Point2f a2, Point2f b1, Point2f b2, Point2f& intersection);
float cross(Point2f v1, Point2f v2);

typedef struct t_color_node {
	cv::Mat       mean;       // The mean of this node
	cv::Mat       cov;
	uchar         classid;    // The class ID

	t_color_node  *left;
	t_color_node  *right;
} t_color_node;

cv::Mat get_dominant_palette(std::vector<cv::Vec3b> colors) {
	const int tile_size = 64;
	cv::Mat ret = cv::Mat(tile_size, tile_size*colors.size(), CV_8UC3, cv::Scalar(0));

	for (int i = 0; i<colors.size(); i++) {
		cv::Rect rect(i*tile_size, 0, tile_size, tile_size);
		cv::rectangle(ret, rect, cv::Scalar(colors[i][0], colors[i][1], colors[i][2]), CV_FILLED);
	}

	return ret;
}

std::vector<t_color_node*> get_leaves(t_color_node *root) {
	std::vector<t_color_node*> ret;
	std::queue<t_color_node*> queue;
	queue.push(root);

	while (queue.size() > 0) {
		t_color_node *current = queue.front();
		queue.pop();

		if (current->left && current->right) {
			queue.push(current->left);
			queue.push(current->right);
			continue;
		}

		ret.push_back(current);
	}
	return ret;
}

std::vector<cv::Vec3b> get_dominant_colors(t_color_node *root) {
	std::vector<t_color_node*> leaves = get_leaves(root);
	std::vector<cv::Vec3b> ret;

	for (int i = 0; i<leaves.size(); i++) {
		cv::Mat mean = leaves[i]->mean;
		ret.push_back(cv::Vec3b(mean.at<double>(0)*255.0f,
			mean.at<double>(1)*255.0f,
			mean.at<double>(2)*255.0f));
	}

	return ret;
}

int get_next_classid(t_color_node *root) {
	int maxid = 0;
	std::queue<t_color_node*> queue;
	queue.push(root);

	while (queue.size() > 0) {
		t_color_node* current = queue.front();
		queue.pop();

		if (current->classid > maxid)
			maxid = current->classid;

		if (current->left != NULL)
			queue.push(current->left);

		if (current->right)
			queue.push(current->right);
	}

	return maxid + 1;
}

void get_class_mean_cov(cv::Mat img, cv::Mat classes, t_color_node *node) {
	const int width = img.cols;
	const int height = img.rows;
	const uchar classid = node->classid;

	cv::Mat mean = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
	cv::Mat cov = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0));

	// We start out with the average color
	double pixcount = 0;
	for (int y = 0; y<height; y++) {
		cv::Vec3b* ptr = img.ptr<cv::Vec3b>(y);
		uchar* ptrClass = classes.ptr<uchar>(y);
		for (int x = 0; x<width; x++) {
			if (ptrClass[x] != classid)
				continue;

			cv::Vec3b color = ptr[x];
			cv::Mat scaled = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
			scaled.at<double>(0) = color[0] / 255.0f;
			scaled.at<double>(1) = color[1] / 255.0f;
			scaled.at<double>(2) = color[2] / 255.0f;

			mean += scaled;
			cov = cov + (scaled * scaled.t());

			pixcount++;
		}
	}

	cov = cov - (mean * mean.t()) / pixcount;
	mean = mean / pixcount;

	// The node mean and covariance
	node->mean = mean.clone();
	node->cov = cov.clone();

	return;
}

void partition_class(cv::Mat img, cv::Mat classes, uchar nextid, t_color_node *node) {
	const int width = img.cols;
	const int height = img.rows;
	const int classid = node->classid;

	const uchar newidleft = nextid;
	const uchar newidright = nextid + 1;

	cv::Mat mean = node->mean;
	cv::Mat cov = node->cov;
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(cov, eigenvalues, eigenvectors);

	cv::Mat eig = eigenvectors.row(0);
	cv::Mat comparison_value = eig * mean;

	node->left = new t_color_node();
	node->right = new t_color_node();

	node->left->classid = newidleft;
	node->right->classid = newidright;

	// We start out with the average color
	for (int y = 0; y<height; y++) {
		cv::Vec3b* ptr = img.ptr<cv::Vec3b>(y);
		uchar* ptrClass = classes.ptr<uchar>(y);
		for (int x = 0; x<width; x++) {
			if (ptrClass[x] != classid)
				continue;

			cv::Vec3b color = ptr[x];
			cv::Mat scaled = cv::Mat(3, 1,
				CV_64FC1,
				cv::Scalar(0));

			scaled.at<double>(0) = color[0] / 255.0f;
			scaled.at<double>(1) = color[1] / 255.0f;
			scaled.at<double>(2) = color[2] / 255.0f;

			cv::Mat this_value = eig * scaled;

			if (this_value.at<double>(0, 0) <= comparison_value.at<double>(0, 0)) {
				ptrClass[x] = newidleft;
			}
			else {
				ptrClass[x] = newidright;
			}
		}
	}
	return;
}

cv::Mat get_quantized_image(cv::Mat classes, t_color_node *root) {
	std::vector<t_color_node*> leaves = get_leaves(root);

	const int height = classes.rows;
	const int width = classes.cols;
	cv::Mat ret(height, width, CV_8UC3, cv::Scalar(0));

	for (int y = 0; y<height; y++) {
		uchar *ptrClass = classes.ptr<uchar>(y);
		cv::Vec3b *ptr = ret.ptr<cv::Vec3b>(y);
		for (int x = 0; x<width; x++) {
			uchar pixel_class = ptrClass[x];
			for (int i = 0; i<leaves.size(); i++) {
				if (leaves[i]->classid == pixel_class) {
					ptr[x] = cv::Vec3b(leaves[i]->mean.at<double>(0) * 255,
						leaves[i]->mean.at<double>(1) * 255,
						leaves[i]->mean.at<double>(2) * 255);
				}
			}
		}
	}

	return ret;
}

cv::Mat get_viewable_image(cv::Mat classes) {
	const int height = classes.rows;
	const int width = classes.cols;

	const int max_color_count = 12;
	cv::Vec3b *palette = new cv::Vec3b[max_color_count];
	palette[0] = cv::Vec3b(0, 0, 0);
	palette[1] = cv::Vec3b(255, 0, 0);
	palette[2] = cv::Vec3b(0, 255, 0);
	palette[3] = cv::Vec3b(0, 0, 255);
	palette[4] = cv::Vec3b(255, 255, 0);
	palette[5] = cv::Vec3b(0, 255, 255);
	palette[6] = cv::Vec3b(255, 0, 255);
	palette[7] = cv::Vec3b(128, 128, 128);
	palette[8] = cv::Vec3b(128, 255, 128);
	palette[9] = cv::Vec3b(32, 32, 32);
	palette[10] = cv::Vec3b(255, 128, 128);
	palette[11] = cv::Vec3b(128, 128, 255);

	cv::Mat ret = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
	for (int y = 0; y<height; y++) {
		cv::Vec3b *ptr = ret.ptr<cv::Vec3b>(y);
		uchar *ptrClass = classes.ptr<uchar>(y);
		for (int x = 0; x<width; x++) {
			int color = ptrClass[x];
			if (color >= max_color_count) {
				printf("You should increase the number of predefined colors!\n");
				continue;
			}
			ptr[x] = palette[color];
		}
	}

	return ret;
}

t_color_node* get_max_eigenvalue_node(t_color_node *current) {
	double max_eigen = -1;
	cv::Mat eigenvalues, eigenvectors;

	std::queue<t_color_node*> queue;
	queue.push(current);

	t_color_node *ret = current;
	if (!current->left && !current->right)
		return current;

	while (queue.size() > 0) {
		t_color_node *node = queue.front();
		queue.pop();

		if (node->left && node->right) {
			queue.push(node->left);
			queue.push(node->right);
			continue;
		}

		cv::eigen(node->cov, eigenvalues, eigenvectors);
		double val = eigenvalues.at<double>(0);
		if (val > max_eigen) {
			max_eigen = val;
			ret = node;
		}
	}

	return ret;
}

std::vector<cv::Vec3b> find_dominant_colors(cv::Mat img, int count) {
	const int width = img.cols;
	const int height = img.rows;

	cv::Mat classes = cv::Mat(height, width, CV_8UC1, cv::Scalar(1));
	t_color_node *root = new t_color_node();

	root->classid = 1;
	root->left = NULL;
	root->right = NULL;

	t_color_node *next = root;
	get_class_mean_cov(img, classes, root);
	for (int i = 0; i<count - 1; i++) {
		next = get_max_eigenvalue_node(root);
		partition_class(img, classes, get_next_classid(root), next);
		get_class_mean_cov(img, classes, next->left);
		get_class_mean_cov(img, classes, next->right);
	}

	std::vector<cv::Vec3b> colors = get_dominant_colors(root);

	cv::Mat quantized = get_quantized_image(classes, root);
	cv::Mat viewable = get_viewable_image(classes);
	cv::Mat dom = get_dominant_palette(colors);

	cv::imwrite("./classification.png", viewable);
	cv::imwrite("./quantized.png", quantized);
	cv::imwrite("./palette.png", dom);

	return colors;
}

class OpenCVBitmapSource : public LuminanceSource
{
private:
	cv::Mat m_pImage;

public:
	OpenCVBitmapSource(cv::Mat &image)
		: LuminanceSource(image.cols, image.rows)
	{
		m_pImage = image.clone();
	}

	~OpenCVBitmapSource()
	{
	}

	int getWidth() const { return m_pImage.cols; }
	int getHeight() const { return m_pImage.rows; }

	ArrayRef<char> getRow(int y, ArrayRef<char> row) const
	{
		int width_ = getWidth();
		if (!row)
			row = ArrayRef<char>(width_);
		const char *p = m_pImage.ptr<char>(y);
		for (int x = 0; x<width_; ++x, ++p)
			row[x] = *p;
		return row;
	}

	ArrayRef<char> getMatrix() const
	{
		int width_ = getWidth();
		int height_ = getHeight();
		ArrayRef<char> matrix = ArrayRef<char>(width_*height_);
		for (int y = 0; y < height_; ++y)
		{
			const char *p = m_pImage.ptr<char>(y);
			for (int x = 0; x < width_; ++x, ++p)
			{
				matrix[y*width_ + x] = *p;
			}
		}
		return matrix;
	}
};

void decode_image(Reader *reader, cv::Mat &image)
{
	try
	{
		Ref<OpenCVBitmapSource> source(new OpenCVBitmapSource(image));
		Ref<Binarizer> binarizer(new GlobalHistogramBinarizer(source));
		Ref<BinaryBitmap> bitmap(new BinaryBitmap(binarizer));
		Ref<Result> result(reader->decode(bitmap, DecodeHints(DecodeHints::TRYHARDER_HINT)));
		cout << "\n" << result->getText()->getText() << endl;
	}
	catch (zxing::Exception& e)
	{
		
			//cerr << "Error: " << e.what() << endl;
	}
}

int main(int argc, char **argv)
{

	setlocale(LC_ALL, "Russian");

	VideoCapture capture(0);

	Mat image;

	if (!capture.isOpened()) {
		cerr << " VIDEO READING ERROR!" << endl;
		return -1;
	}


	capture >> image;
	if (image.empty()) {
		cerr << "VIDEO PROBLEMS!" << endl;
		return -1;
	}


	Mat gray(image.size(), CV_MAKETYPE(image.depth(), 1));
	Mat edges(image.size(), CV_MAKETYPE(image.depth(), 1));
	Mat traces(image.size(), CV_8UC3);
	Mat qr, qr_raw, qr_gray, qr_thres;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Point> pointsseq;

	QRCodeReader cr;

	int mark, A, B, C, top, right, bottom, median1, median2, outlier;
	float AB, BC, CA, dist, slope, areat, arear, areab, large, padding;

	int align, orientation;

	int DBG = 1;	// debug mode					

	int key = 0;

	if (DBG == 1)
		printf("Glasses DEBUG MODE");

	int mode = 0;
	cin >> mode;
	if (mode == 0)
	{
		while (key != 'q')				// main
		{

			traces = Scalar(0, 0, 0);
			qr_raw = Mat::zeros(100, 100, CV_8UC3);
			qr = Mat::zeros(100, 100, CV_8UC3);
			qr_gray = Mat::zeros(100, 100, CV_8UC1);
			qr_thres = Mat::zeros(100, 100, CV_8UC1);

			capture >> image;

			cvtColor(image, gray, CV_RGB2GRAY);
			Canny(gray, edges, 100, 200, 3);


			findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

			mark = 0;


			vector<Moments> mu(contours.size());
			vector<Point2f> mc(contours.size());

			for (int i = 0; i < contours.size(); i++)
			{
				mu[i] = moments(contours[i], false);
				mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
			}





			for (int i = 0; i < contours.size(); i++)
			{
				approxPolyDP(contours[i], pointsseq, arcLength(contours[i], true)*0.02, true);
				if (pointsseq.size() == 4)
				{
					int k = i;
					int c = 0;

					while (hierarchy[k][2] != -1)
					{
						k = hierarchy[k][2];
						c = c + 1;
					}
					if (hierarchy[k][2] != -1)
						c = c + 1;

					if (c >= 5)
					{
						if (mark == 0)		A = i;
						else if (mark == 1)	B = i;
						else if (mark == 2)	C = i;
						mark = mark + 1;
					}
				}
			}


			if (mark >= 3)
			{
				AB = cv_distance(mc[A], mc[B]);
				BC = cv_distance(mc[B], mc[C]);
				CA = cv_distance(mc[C], mc[A]);

				if (AB > BC && AB > CA)
				{
					outlier = C; median1 = A; median2 = B;
				}
				else if (CA > AB && CA > BC)
				{
					outlier = B; median1 = A; median2 = C;
				}
				else if (BC > AB && BC > CA)
				{
					outlier = A;  median1 = B; median2 = C;
				}

				top = outlier;

				dist = cv_lineEquation(mc[median1], mc[median2], mc[outlier]);
				slope = cv_lineSlope(mc[median1], mc[median2], align);

				if (align == 0)
				{
					bottom = median1;
					right = median2;
				}
				else if (slope < 0 && dist < 0)
				{
					bottom = median1;
					right = median2;
					orientation = CV_QR_NORTH;
				}
				else if (slope > 0 && dist < 0)
				{
					right = median1;
					bottom = median2;
					orientation = CV_QR_EAST;
				}
				else if (slope < 0 && dist > 0)
				{
					right = median1;
					bottom = median2;
					orientation = CV_QR_SOUTH;
				}

				else if (slope > 0 && dist > 0)
				{
					bottom = median1;
					right = median2;
					orientation = CV_QR_WEST;
				}


				float area_top, area_right, area_bottom;

				if (top < contours.size() && right < contours.size() && bottom < contours.size() && contourArea(contours[top]) > 10 && contourArea(contours[right]) > 10 && contourArea(contours[bottom]) > 10)
				{

					vector<Point2f> L, M, O, tempL, tempM, tempO;
					Point2f N;

					vector<Point2f> src, dst;

					Mat warp_matrix;

					cv_getVertices(contours, top, slope, tempL);
					cv_getVertices(contours, right, slope, tempM);
					cv_getVertices(contours, bottom, slope, tempO);

					cv_updateCornerOr(orientation, tempL, L);
					cv_updateCornerOr(orientation, tempM, M);
					cv_updateCornerOr(orientation, tempO, O);

					int iflag = getIntersectionPoint(M[1], M[2], O[3], O[2], N);


					src.push_back(L[0]);
					src.push_back(M[1]);
					src.push_back(N);
					src.push_back(O[3]);

					dst.push_back(Point2f(0, 0));
					dst.push_back(Point2f(qr.cols, 0));
					dst.push_back(Point2f(qr.cols, qr.rows));
					dst.push_back(Point2f(0, qr.rows));

					if (src.size() == 4 && dst.size() == 4)
					{
						warp_matrix = getPerspectiveTransform(src, dst);
						warpPerspective(image, qr_raw, warp_matrix, Size(qr.cols, qr.rows));
						copyMakeBorder(qr_raw, qr, 10, 10, 10, 10, BORDER_CONSTANT, Scalar(255, 255, 255));

						cvtColor(qr, qr_gray, CV_RGB2GRAY);
						threshold(qr_gray, qr_thres, 127, 255, CV_THRESH_BINARY);
						//imwrite("qrcode.png", qr_thres);
						//printf("QR CODE IMAGE SAVED...");
						decode_image(&cr, qr_thres);

					}

					drawContours(image, contours, top, Scalar(255, 200, 0), 2, 8, hierarchy, 0);
					drawContours(image, contours, right, Scalar(0, 0, 255), 2, 8, hierarchy, 0);
					drawContours(image, contours, bottom, Scalar(255, 0, 100), 2, 8, hierarchy, 0);

					//DEBUG
					if (DBG == 1)
					{
						if (slope > 5)
							circle(traces, Point(10, 20), 5, Scalar(0, 0, 255), -1, 8, 0);
						else if (slope < -5)
							circle(traces, Point(10, 20), 5, Scalar(255, 255, 255), -1, 8, 0);

						drawContours(traces, contours, top, Scalar(255, 0, 100), 1, 8, hierarchy, 0);
						drawContours(traces, contours, right, Scalar(255, 0, 100), 1, 8, hierarchy, 0);
						drawContours(traces, contours, bottom, Scalar(255, 0, 100), 1, 8, hierarchy, 0);
						circle(traces, L[0], 2, Scalar(255, 255, 0), -1, 8, 0);
						circle(traces, L[1], 2, Scalar(0, 255, 0), -1, 8, 0);
						circle(traces, L[2], 2, Scalar(0, 0, 255), -1, 8, 0);
						circle(traces, L[3], 2, Scalar(128, 128, 128), -1, 8, 0);

						circle(traces, M[0], 2, Scalar(255, 255, 0), -1, 8, 0);
						circle(traces, M[1], 2, Scalar(0, 255, 0), -1, 8, 0);
						circle(traces, M[2], 2, Scalar(0, 0, 255), -1, 8, 0);
						circle(traces, M[3], 2, Scalar(128, 128, 128), -1, 8, 0);

						circle(traces, O[0], 2, Scalar(255, 255, 0), -1, 8, 0);
						circle(traces, O[1], 2, Scalar(0, 255, 0), -1, 8, 0);
						circle(traces, O[2], 2, Scalar(0, 0, 255), -1, 8, 0);
						circle(traces, O[3], 2, Scalar(128, 128, 128), -1, 8, 0);
						circle(traces, N, 2, Scalar(255, 255, 255), -1, 8, 0);

						line(traces, M[1], N, Scalar(0, 0, 255), 1, 8, 0);
						line(traces, O[3], N, Scalar(0, 0, 255), 1, 8, 0);


						int fontFace = FONT_HERSHEY_PLAIN;

						if (orientation == CV_QR_NORTH)
						{
							putText(traces, "NORTH", Point(20, 30), fontFace, 1, Scalar(0, 255, 0), 1, 8);
						}
						else if (orientation == CV_QR_EAST)
						{
							putText(traces, "EAST", Point(20, 30), fontFace, 1, Scalar(0, 255, 0), 1, 8);
						}
						else if (orientation == CV_QR_SOUTH)
						{
							putText(traces, "SOUTH", Point(20, 30), fontFace, 1, Scalar(0, 255, 0), 1, 8);
						}
						else if (orientation == CV_QR_WEST)
						{
							putText(traces, "WEST", Point(20, 30), fontFace, 1, Scalar(0, 255, 0), 1, 8);
						}

					}

				}
			}

			imshow("Image", image);
			imshow("Traces", traces);
			imshow("QR code", qr_thres);



			key = waitKey(1);

		}

		return 0;
	}
	else if (mode == 1)
	{
		
		while (key != 'w') {
			capture >> image;
			imshow("image", image);
			key = waitKey(1);
		}
		/*Mat3b hsv;
		cvtColor(image, hsv, COLOR_BGR2HSV);
		Mat1b mask1, mask2;

		inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
		inRange(hsv, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);

		Mat1b mask = mask1 | mask2;

		imshow("Mask", mask);*/
			int count = 5;
			vector<cv::Vec3b> colors = find_dominant_colors(image, count);
			

			
		}

		return 0;
	}


float cv_distance(Point2f P, Point2f Q)
{
	return sqrt(pow(abs(P.x - Q.x), 2) + pow(abs(P.y - Q.y), 2));
}



float cv_lineEquation(Point2f L, Point2f M, Point2f J)
{
	float a, b, c, pdist;

	a = -((M.y - L.y) / (M.x - L.x));
	b = 1.0;
	c = (((M.y - L.y) / (M.x - L.x)) * L.x) - L.y;


	pdist = (a * J.x + (b * J.y) + c) / sqrt((a * a) + (b * b));
	return pdist;
}


float cv_lineSlope(Point2f L, Point2f M, int& alignement)
{
	float dx, dy;
	dx = M.x - L.x;
	dy = M.y - L.y;

	if (dy != 0)
	{
		alignement = 1;
		return (dy / dx);
	}
	else				
	{
		alignement = 0;
		return 0.0;
	}
}



void cv_getVertices(vector<vector<Point> > contours, int c_id, float slope, vector<Point2f>& quad)
{
	Rect box;
	box = boundingRect(contours[c_id]);

	Point2f M0, M1, M2, M3;
	Point2f A, B, C, D, W, X, Y, Z;

	A = box.tl();
	B.x = box.br().x;
	B.y = box.tl().y;
	C = box.br();
	D.x = box.tl().x;
	D.y = box.br().y;


	W.x = (A.x + B.x) / 2;
	W.y = A.y;

	X.x = B.x;
	X.y = (B.y + C.y) / 2;

	Y.x = (C.x + D.x) / 2;
	Y.y = C.y;

	Z.x = D.x;
	Z.y = (D.y + A.y) / 2;

	float dmax[4];
	dmax[0] = 0.0;
	dmax[1] = 0.0;
	dmax[2] = 0.0;
	dmax[3] = 0.0;

	float pd1 = 0.0;
	float pd2 = 0.0;

	if (slope > 5 || slope < -5)
	{

		for (int i = 0; i < contours[c_id].size(); i++)
		{
			pd1 = cv_lineEquation(C, A, contours[c_id][i]);
			pd2 = cv_lineEquation(B, D, contours[c_id][i]);

			if ((pd1 >= 0.0) && (pd2 > 0.0))
			{
				cv_updateCorner(contours[c_id][i], W, dmax[1], M1);
			}
			else if ((pd1 > 0.0) && (pd2 <= 0.0))
			{
				cv_updateCorner(contours[c_id][i], X, dmax[2], M2);
			}
			else if ((pd1 <= 0.0) && (pd2 < 0.0))
			{
				cv_updateCorner(contours[c_id][i], Y, dmax[3], M3);
			}
			else if ((pd1 < 0.0) && (pd2 >= 0.0))
			{
				cv_updateCorner(contours[c_id][i], Z, dmax[0], M0);
			}
			else
				continue;
		}
	}
	else
	{
		int halfx = (A.x + B.x) / 2;
		int halfy = (A.y + D.y) / 2;

		for (int i = 0; i < contours[c_id].size(); i++)
		{
			if ((contours[c_id][i].x < halfx) && (contours[c_id][i].y <= halfy))
			{
				cv_updateCorner(contours[c_id][i], C, dmax[2], M0);
			}
			else if ((contours[c_id][i].x >= halfx) && (contours[c_id][i].y < halfy))
			{
				cv_updateCorner(contours[c_id][i], D, dmax[3], M1);
			}
			else if ((contours[c_id][i].x > halfx) && (contours[c_id][i].y >= halfy))
			{
				cv_updateCorner(contours[c_id][i], A, dmax[0], M2);
			}
			else if ((contours[c_id][i].x <= halfx) && (contours[c_id][i].y > halfy))
			{
				cv_updateCorner(contours[c_id][i], B, dmax[1], M3);
			}
		}
	}

	quad.push_back(M0);
	quad.push_back(M1);
	quad.push_back(M2);
	quad.push_back(M3);

}

void cv_updateCorner(Point2f P, Point2f ref, float& baseline, Point2f& corner)
{
	float temp_dist;
	temp_dist = cv_distance(P, ref);

	if (temp_dist > baseline)
	{
		baseline = temp_dist;			
		corner = P;						
	}

}

void cv_updateCornerOr(int orientation, vector<Point2f> IN, vector<Point2f> &OUT)
{
	Point2f M0, M1, M2, M3;
	if (orientation == CV_QR_NORTH)
	{
		M0 = IN[0];
		M1 = IN[1];
		M2 = IN[2];
		M3 = IN[3];
	}
	else if (orientation == CV_QR_EAST)
	{
		M0 = IN[1];
		M1 = IN[2];
		M2 = IN[3];
		M3 = IN[0];
	}
	else if (orientation == CV_QR_SOUTH)
	{
		M0 = IN[2];
		M1 = IN[3];
		M2 = IN[0];
		M3 = IN[1];
	}
	else if (orientation == CV_QR_WEST)
	{
		M0 = IN[3];
		M1 = IN[0];
		M2 = IN[1];
		M3 = IN[2];
	}

	OUT.push_back(M0);
	OUT.push_back(M1);
	OUT.push_back(M2);
	OUT.push_back(M3);
}

bool getIntersectionPoint(Point2f a1, Point2f a2, Point2f b1, Point2f b2, Point2f& intersection)
{
	Point2f p = a1;
	Point2f q = b1;
	Point2f r(a2 - a1);
	Point2f s(b2 - b1);

	if (cross(r, s) == 0) { return false; }

	float t = cross(q - p, s) / cross(r, s);

	intersection = p + t*r;
	return true;
}

float cross(Point2f v1, Point2f v2)
{
	return v1.x*v2.y - v1.y*v2.x;
}



