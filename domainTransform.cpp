#include <iostream>
#include <vector>
using namespace std;
 
#include "opencv2/opencv.hpp"
using namespace cv;

void create2Matrix(vector<Mat>& Ix, vector<Mat>& Iy, int windowSize, int width, int height){

	// This command creates a vector of empty matrixes
	for( int i = 0; i <= windowSize; i++){
		if( i == 0 ){
			Mat temp = Mat::zeros(height,width,CV_64FC1);
			Ix.push_back(temp);
			Mat temp1 = Mat::zeros(height,width,CV_64FC1);
			Iy.push_back(temp1);
		}
		else{
			Mat temp = Mat(height, width, CV_64FC1);
			Ix.push_back(temp);
			Mat temp1 = Mat(height, width, CV_64FC1);
			Iy.push_back(temp1);
		}
	}
}

void domainTransform(Mat& output,Mat& I, int windowSize, double sigma_r, double sigma_s, double numIter){

	int width = I.cols;
	int height = I.rows;
	int numChannels = I.channels();
	output  = I.clone(); 
	I.copyTo(output);

	
	vector<Mat> IxForward;
	vector<Mat> IyForward;

	vector<Mat> IxBackward;
	vector<Mat> IyBackward;
	
	// create vectors of matrices
	create2Matrix(IxForward, IyForward, windowSize, width, height);
	create2Matrix(IxBackward, IyBackward, windowSize, width, height);

	// Creating IxForward
	for( int i =1; i <= windowSize; i++){
		for(int y=0; y<height; y++) {
			for(int x=0; x < width-(windowSize + 1); x++) {
					double sum = 0.0;
					for(int k=0; k<numChannels; k++) {
						sum += abs(I.at<double>(y, (x+(i+1))*numChannels+k) - I.at<double>(y, (x + i)*numChannels+k));
					}
					// equation (12) in paper
					if( i == 1)
						IxForward[i].at<double>(y, x) = 1.0 + (sigma_s / sigma_r) * sum;
					else
						IxForward[i].at<double>(y, x) = IxForward[i-1].at<double>(y, x) + 1.0 + (sigma_s / sigma_r) * sum;
			}
		}
	}
	// Creating IxBackward
	for( int i = 1; i <= windowSize; i++){
		for(int y = 0; y < height; y++) {
			for(int x = width - 1; x > windowSize+1; x--) {
					double sum = 0.0;
					for(int k=0; k<numChannels; k++) {
						sum += abs(I.at<double>(y, (x-(i-1))*numChannels+k) - I.at<double>(y, (x - i)*numChannels+k));
					}
					// equation (12) in paper
					if( i == 1)
						IxBackward[i].at<double>(y, x) = 1.0 + (sigma_s / sigma_r) * sum;
					else
						IxBackward[i].at<double>(y, x) = IxBackward[i-1].at<double>(y, x) + 1.0 + (sigma_s / sigma_r) * sum;
			}
		}
	}
	// Creating IyForward
	for( int i = 1; i <= windowSize; i++){
		for(int x=0; x < width; x++) {
			for(int y = 0; y < height-windowSize-1; y++) {
			
				double sum = 0.0;
				for(int k=0; k<numChannels; k++) {
					sum += abs(I.at<double>(y + (i+1), x*numChannels+k) - I.at<double>(y + i, x*numChannels+k));
				}
				// equation (12) in paper
				if( i == 1)
					IyForward[i].at<double>(y, x) = 1.0 + (sigma_s / sigma_r) * sum;
				else
					IyForward[i].at<double>(y, x) = IyForward[i-1].at<double>(y,x) + 1.0 + (sigma_s / sigma_r) * sum;
			}
		}
	}
	// Creating IyBackward
	for( int i = 1; i <= windowSize; i++){
		for(int x=0; x < width; x++) {
			for(int y = height - 1; y> windowSize+1; y--) {
				
				double sum = 0.0;
				for(int k=0; k<numChannels; k++) {
					sum += abs(I.at<double>(y - (i-1), x*numChannels+k) - I.at<double>(y-i, x*numChannels+k));
				}
				// equation (12) in paper
				if( i == 1)
					IyBackward[i].at<double>(y, x) = 1.0 + (sigma_s / sigma_r) * sum;
				else
					IyBackward[i].at<double>(y, x) = IyBackward[i-1].at<double>(y,x) + 1.0 + (sigma_s / sigma_r) * sum;
			}
		}
	}
	// Main part of program, we do the filtering here
	for (int iter = 0; iter < numIter; iter++){
		cout << "Current itteration = " << iter + 1 << endl;
		// Eqution (14) from paper, I've realised that there was a mistake in paper at this place
		double sigmaH = sigma_s * sqrt(3.0) * pow(2.0, numIter - iter - 1) / sqrt(pow(4.0, numIter) - 1.0);
		// Feedback coeff from (21) from paper
		double a = exp(-sqrt(2.0) / sigmaH);
		vector<Mat> expIxForward;
		vector<Mat> expIyForward;

		vector<Mat> expIxBackward;
		vector<Mat> expIyBackward;
		
		create2Matrix(expIxForward, expIyForward, windowSize, width, height);
		create2Matrix(expIxBackward, expIyBackward, windowSize, width, height);

		// expIx
		for(int y=0; y<height; y++) {
			for(int x=0; x<width - (windowSize + 1); x++) {
				double sumForward = 0.0;
				double sumBackward = 0.0;
				for( int i = 0; i <= windowSize ; i++){
					expIxForward[i].at<double>(y,x) = pow(a, IxForward[i].at<double>(y,x));
					expIxBackward[i].at<double>(y,x+windowSize+1) = pow(a, IxBackward[i].at<double>(y,x+windowSize+1));
					sumForward += expIxForward[i].at<double>(y,x);
					sumBackward += expIxBackward[i].at<double>(y,x+windowSize+1);
				}
				for( int i = 0; i <= windowSize; i ++ ){
					expIxForward[i].at<double>(y,x) /= sumForward;
					expIxBackward[i].at<double>(y,x+windowSize+1) /= sumBackward;
				}
			}
		}
		// expIy
		for(int x=0; x<width; x++) {
			for(int y=0; y<height-windowSize-1; y++) {
				double sumForward = 0.0;
				double sumBackward = 0.0;
				for( int i = 0; i <= windowSize ; i++){
					expIyForward[i].at<double>(y,x) = pow(a, IyForward[i].at<double>(y,x));
					expIyBackward[i].at<double>(y+windowSize+1,x) = pow(a , IyBackward[i].at<double>(y+windowSize+1,x));
					sumForward += expIyForward[i].at<double>(y,x);
					sumBackward += expIyBackward[i].at<double>(y+windowSize+1,x);
				}
				for( int i = 0; i <= windowSize; i ++ ){
					expIyForward[i].at<double>(y,x) /= sumForward;
					expIyBackward[i].at<double>(y+windowSize+1,x) /= sumBackward;
				}
			}
		}
		// Filter Forward Vertical
		for( int x = 0 ; x < width - (windowSize + 1) ; x ++){
			for( int y = 0 ; y < height; y ++){
				for( int k = 0 ; k < numChannels; k++){
					output.at<double>(y,x * numChannels + k) = output.at<double>(y,x * numChannels + k) * expIxForward[0].at<double>(y,x);
					for(int i = 1 ; i <= windowSize; i++){
						output.at<double>(y,x*numChannels + k) += output.at<double>(y,(x + i)*numChannels + k) * expIxForward[i].at<double>(y,x);
					}

				}
			}
		}
		// Filter Backward Vertical
		for( int x = width - 1 ; x > windowSize  ; x--){
			for( int y = 0 ; y < height; y ++){
				for( int k = 0 ; k < numChannels; k++){
					output.at<double>(y,x * numChannels + k) = output.at<double>(y,x * numChannels + k) * expIxBackward[0].at<double>(y,x);
					for(int i = 1 ; i <= windowSize; i++){
						output.at<double>(y,x*numChannels + k) += output.at<double>(y,(x - i)*numChannels + k) * expIxBackward[i].at<double>(y,x);
					}

				}
			}
		}
		// Filter Forward Horizontal
		for( int x = 0 ; x < width  ; x ++){
			for( int y = 0 ; y < height - (windowSize + 1); y++){
				for( int k = 0 ; k < numChannels; k++){
					output.at<double>(y,x * numChannels + k) = output.at<double>(y,x * numChannels + k) * expIyForward[0].at<double>(y,x);
					for(int i = 1 ; i <= windowSize; i++){
						output.at<double>(y,x*numChannels + k) += output.at<double>(y + i,(x)*numChannels + k) * expIyForward[i].at<double>(y,x);
					}

				}
			}
		}
		// Filter Backward Horizontal
		for( int x = 0 ; x < width  ; x ++){
			for( int y = height -1  ; y > windowSize ; y--){
				for( int k = 0 ; k < numChannels; k++){
					output.at<double>(y,x * numChannels + k) = output.at<double>(y,x * numChannels + k) * expIyBackward[0].at<double>(y,x);
					for(int i = 1 ; i <= windowSize; i++){
						output.at<double>(y,x*numChannels + k) += output.at<double>(y - i,(x)*numChannels + k) * expIyBackward[i].at<double>(y,x);
					}

				}
			}
		}
	}
}

int main(int argc, char** argv){

	//Check whether there are enough input arguments
	if(argc < 2){
		cout << "you need to specify the image file" << endl << flush;
		cout << "usage: *.exe image_file [sigma_s] [sigma_r] [numberOfIterations] [windowSize]" << endl << flush;
		return -1;
	}
	Mat img;
	img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(! img.data ) {
		cout << "failed to load image, check the path" << endl << flush;
	}
	double sigma_s = argc <= 2 ? 25  : atof(argv[2]);
	double sigma_r = argc <= 3 ? 1   : atof(argv[3]);
	double numIter = argc <= 4 ? 10  : atof(argv[4]);
	int windowSize = argc <= 5 ? 1   : atoi(argv[5]);

	cout << "sigma_r is = " << sigma_r << endl << "sigma_s is = " << sigma_s << endl << "number of iterations are = " << numIter << endl << "window size is = " << windowSize << " " << endl;
	// normalize image so it goes from 0-1 and not 0-255
	img.convertTo(img, CV_64FC3, 1.0 / 255.0);

	Mat output;
	domainTransform(output, img, windowSize, sigma_r, sigma_s, numIter);

    imshow("Input", img);
    imshow("Output", output);
	int k = 0;
	// wait for esc key to close
	while( k != 27)
		k = waitKey(0);
    destroyAllWindows();

	return 0;
}