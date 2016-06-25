#include "opencv2\\opencv.hpp"
#include "opencv2\opencv_modules.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include "gnuplot.h"
//from http://tips.hecomi.com/entry/20101209/1291888423

using namespace std;
using namespace cv;







// http://blog.livedoor.jp/hen_cyberne/archives/51071213.html
// �z���O���t�B�[�s�񂩂��]�E���i�������o�����߂̐��K���֐�
void
normalize(cv::Vec3f&  vec)
{
	float a = vec[0] * vec[0];
	float b = vec[1] * vec[1];
	float c = vec[2] * vec[2];

	float norm = sqrt(a + b + c);

	vec[0] /= norm;
	vec[1] /= norm;
	vec[2] /= norm;
}




// �R�[�h�̃x�[�X�@http://qiita.com/vs4sh/items/4a9ce178f1b2fd26ea30
// orb�̕����@http://independence-sys.net/main/?p=2632

int main(){


	cv::VideoCapture cap(0);//�f�o�C�X�̃I�[�v��
	//cap.open(0);//�������ł��ǂ��D

	if (!cap.isOpened())//�J�����f�o�C�X������ɃI�[�v���������m�F�D
	{
		//�ǂݍ��݂Ɏ��s�����Ƃ��̏���
		return -1;
	}
	cv::Mat frame;
	cv::Mat frame_p;
	//�v���b�g�p
	//gnuplot::CGnuplot gp;


	vector <cv::Mat> vframe;
	double x=0, y=0, theta=0;
	double x0, y0, theta0;
	double fx, fy, cx, cy;
	
	fx = 643.0262;
	fy = 646.7989;
	cx = 291.1886;
	cy = 243.6844;
	
	//�J���������p�����[�^�s��
	//ELECOM UCAM - DLE300TBK
	//	ans =

	//	643.0262         0         0
	//	0  646.7989         0
	//	291.1886  243.6844    1.0000* /
	cv::Mat A = (cv::Mat_<double>(3, 3) << fx,0,cx,0,fy,cy,0,0,1);

	// �����_�����i�[���邽�߂̕ϐ�
	vector<KeyPoint> keypoints;
	vector<KeyPoint> keypoints_p;

	// �摜�̓��������i�[���邽�߂̕ϐ�
	Mat descriptor;
	Mat descriptor_p;

	//
	//�擾�����t���[���摜�ɑ΂��āC�N���[�X�P�[���ϊ���2�l���Ȃǂ̏������������ށD
	//
	// FeatureDetector�I�u�W�F�N�g�̐���
	Ptr<FeatureDetector> detector =new ORB(1000, 1.2f, 8, 31, 0, 2, 0, 31);

	// DescriptionExtractor�I�u�W�F�N�g�̐���
	Ptr<DescriptorExtractor> extractor = new ORB(1000, 1.2f, 8, 31, 0, 2, 0, 31);

	// DescriptorMatcher�I�u�W�F�N�g�̐���
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	//����p�̑O����̎擾
	// �����_���o�̎��s
	cap >> frame_p; // get a new frame from camera

	detector->detect(frame_p, keypoints_p);
	// �����L�q�̌v�Z�����s
	extractor->compute(frame_p, keypoints_p, descriptor_p);


	//�J���}���t�B���^�����@from http://pukulab.blog.fc2.com/blog-entry-39.html
	// ������
	CvKalman *kalman = cvCreateKalman(4, 2);
	cvSetIdentity(kalman->measurement_matrix, cvRealScalar(1.0));
	cvSetIdentity(kalman->process_noise_cov, cvRealScalar(1e-5));
	cvSetIdentity(kalman->measurement_noise_cov, cvRealScalar(0.1));
	cvSetIdentity(kalman->error_cov_post, cvRealScalar(1.0));

	// ���������^�����f��
	kalman->DynamMatr[0] = 1.0; kalman->DynamMatr[1] = 0.0; kalman->DynamMatr[2] = 1.0; kalman->DynamMatr[3] = 0.0;
	kalman->DynamMatr[4] = 0.0; kalman->DynamMatr[5] = 1.0; kalman->DynamMatr[6] = 0.0; kalman->DynamMatr[7] = 1.0;
	kalman->DynamMatr[8] = 0.0; kalman->DynamMatr[9] = 0.0; kalman->DynamMatr[10] = 1.0; kalman->DynamMatr[11] = 0.0;
	kalman->DynamMatr[12] = 0.0; kalman->DynamMatr[13] = 0.0; kalman->DynamMatr[14] = 0.0; kalman->DynamMatr[15] = 1.0;

	
	//�摜�ۑ��pID
	int saveid = 0;
	while (true)//�������[�v
	{
		

		cap >> frame;
		
		//����
		/*frame_p = vframe[0];
		vframe.erase(vframe.begin());
		vframe.push_back(frame);
*/
		// �����_���o�̎��s
		detector->detect(frame, keypoints);

//		detector->detect(frame_p, keypoints_p);


		// �����L�q�̌v�Z�����s
		extractor->compute(frame, keypoints, descriptor);

//		extractor->compute(frame_p, keypoints_p, descriptor_p);

		// �����_�̃}�b�`���O�����i�[����ϐ�
		vector<DMatch> dmatch;

		// �����_�}�b�`���O�̎��s
		matcher->match(descriptor, descriptor_p, dmatch);


		vector<DMatch> dmatchOK;

		map <double, int> distmap;
		// �ŏ������̎擾
		double min_dist = DBL_MAX;
		for (int i = 0; i < (int) dmatch.size(); i++){
			double dist = dmatch[i].distance;
			distmap[dist]++;
			if (dist < min_dist){
				min_dist = dist;
			}
		}
		if (min_dist < 1.0)
		{
			min_dist = 1.0;
		}
			
		

		double ave2 = 0;
		const double threshold = 2.0 * min_dist;//���ꂪ����

		for (int i = 0; i < (int) dmatch.size(); i++){
			if (dmatch[i].distance < threshold){

				dmatchOK.push_back(dmatch[i]);
				ave2 += dmatch[i].distance;
			}
		}

	//	�e�����_�̋���
		//
		//vector <double> distance;
		//vector <double> disnum;
		//for (auto it = distmap.begin(); it != distmap.end(); ++it){
		//	distance.push_back(it->first);
		//	disnum.push_back(it->second);
		//}
		//gp.Plot(distance, disnum);


		////cout << 1 << keypoints[dmatch[0].queryIdx].pt << endl;
		////cout << 2 << keypoints_p[dmatch[0].queryIdx].pt << endl;	
		////cout << 3 << keypoints[dmatch[0].trainIdx].pt << endl;
		////cout << 4 << keypoints_p[dmatch[0].trainIdx].pt << endl;


		/*
		DMatch.distance - �����ʋL�q�q�Ԃ̋����D�Ⴂ�قǗǂ��D
		DMatch.trainIdx - �w�K�L�q�q(�Q�ƃf�[�^)���̋L�q�q�̃C���f�b�N�X�D
		DMatch.queryIdx - �N�G���L�q�q(�����f�[�^)���̋L�q�q�̃C���f�b�N�X�D
		DMatch.imgIdx - �w�K�摜�̃C���f�b�N�X�D*/

		//std::vector< cv::Vec2f > points1(dmatch.size());
		//std::vector< cv::Vec2f > points2(dmatch.size());
		//for (int i = 0; i < dmatch.size(); i++)
		//{
		//	// �Ή��_�y�A�̍쐬
		//	cv::Point2d pt1, pt2;
		//	pt1.x = keypoints_p[dmatch[i].trainIdx].pt.x;
		//	pt1.y = keypoints_p[dmatch[i].trainIdx].pt.y;
		//	pt2.x = keypoints[dmatch[i].queryIdx].pt.x;
		//	pt2.y = keypoints[dmatch[i].queryIdx].pt.y;
		//	points1[i][0] = pt1.x;
		//	points1[i][1] = pt1.y;
		//	points2[i][0] = pt2.x;
		//	points2[i][1] = pt2.y;
		//}

		//cout << "####average = " << ave2 / dmatchOK.size() << endl;
		// �Ή��_�̕\��
		cv::Mat img_matches;
		cv::drawMatches(frame, keypoints, frame_p, keypoints_p, dmatchOK, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//�z������ӂ��s��
		cv::Mat H;
		// �\���ȑΉ��_������
		//cout << dmatchOK.size() << " " << dmatch.size() <<  endl;
		if (dmatchOK.size() > 10) {
			std::vector<cv::Point2f> obj, scene;
			for (int i = 0; i < (int) dmatchOK.size(); i++) {
				obj.push_back(keypoints[dmatchOK[i].queryIdx].pt);
				scene.push_back(keypoints_p[dmatchOK[i].trainIdx].pt);
			}

			// �z���O���t�B�[�s����v�Z
			H = cv::findHomography(obj, scene, cv::RANSAC);
			//std::cout << H << std::endl;

			// �s�񂪋�ł͂Ȃ�
			if (!H.empty()) {
				std::vector<cv::Point2d> obj_corners(4), scene_corners(4);
				obj_corners[0] = scene_corners[0] = cv::Point2d(0, 0);
				obj_corners[1] = scene_corners[1] = cv::Point2d(frame.cols, 0);
				obj_corners[2] = scene_corners[2] = cv::Point2d(frame.cols, frame.rows);
				obj_corners[3] = scene_corners[3] = cv::Point2d(0, frame.rows);

				// �z���O���t�B�s��̐���
				cv::perspectiveTransform(obj_corners, scene_corners, H);

				////// �΂̐��ň͂� (�J�n�_�����摜�����ɂ���̂ŉE�ɃI�t�Z�b�g)
				////cv::line(img_matches, scene_corners[0] + cv::Point2d(frame.cols, 0), scene_corners[1] + cv::Point2d(frame.cols, 0), cv::Scalar(0, 255, 0), 4);
				////cv::line(img_matches, scene_corners[1] + cv::Point2d(frame.cols, 0), scene_corners[2] + cv::Point2d(frame.cols, 0), cv::Scalar(0, 255, 0), 4);
				////cv::line(img_matches, scene_corners[2] + cv::Point2d(frame.cols, 0), scene_corners[3] + cv::Point2d(frame.cols, 0), cv::Scalar(0, 255, 0), 4);
				////cv::line(img_matches, scene_corners[3] + cv::Point2d(frame.cols, 0), scene_corners[0] + cv::Point2d(frame.cols, 0), cv::Scalar(0, 255, 0), 4);
			}
			//cout << H << endl;
			// http://repo.lib.hosei.ac.jp/bitstream/10114/10556/1/13R4130.pdf
			// Zhang �̎�@�@�z������ӂ��s�����b�s���p���ĕ��i�s��Ɖ�]�s��ɕ�����
			cv::Mat invA = A.inv();

			cv::Mat RT = A.inv() * H;

			cv::Vec3f rVec1 = Vec3f(RT.at< double >(0, 0),
				RT.at< double >(1, 0),
				RT.at< double >(2, 0));
			cv::Vec3f rVec2 = Vec3f(RT.at< double >(0, 1),
				RT.at< double >(1, 1),
				RT.at< double >(2, 1));
			cv::Vec3f tVec = Vec3f(RT.at< double >(0, 2),
				RT.at< double >(1, 2),
				RT.at< double >(2, 2));

			cv::Vec3f rVec3 = rVec1.cross(rVec2);

			normalize(rVec1);
			normalize(rVec2);
			normalize(rVec3);
			//cout << rVec1 << endl;
			//cout << rVec2 << endl;
			//cout << rVec3 << endl;
			//cout <<"theta" <<  acos(rVec1[0]) << " " << asin(rVec1[1]) << endl;
			//cout << rVec3(0) << " " << rVec3(1) << endl;
			x += rVec3(0);
			y += rVec3(1);
			theta += acos(rVec1[0]);
			cout << x << " " << y << " " << theta << endl;
		}

		//�J���}���t�B���^
		// �ϑ��l
		float m [] = { x, y };
		CvMat measurement = cvMat(2, 1, CV_32FC1, m);

		// �C���t�F�[�Y
		const CvMat *correction = cvKalmanCorrect(kalman, &measurement);
		// �X�V�t�F�[�Y
		const CvMat *prediction = cvKalmanPredict(kalman);
		//cout << prediction->data.fl[0] << " " << prediction->data.fl[1] << endl;

		
		// �\��
		cv::imshow("camera", img_matches);





		// �������l���̌��ʂŁA�}�b�`���O���ʉ摜�̍쐬
		Mat result_threth;
		drawMatches(frame, keypoints, frame_p, keypoints_p, dmatchOK, result_threth);
//		imshow("matching prm thre", result_threth);

		//printf("match = %d\n", count);
		//printf("matchOK = %d\n", matchCount);



		imwrite("result.jpg", result_threth);
//
		//cv::imshow("window", frame);//�摜��\���D

		int key = cv::waitKey(1);
		if (key == 'q')//q�{�^���������ꂽ�Ƃ�
		{
			break;//while���[�v���甲����D
		}
		else if (key == 's')//s�������ꂽ�Ƃ�
		{
			//�t���[���摜��ۑ�����D
			string  filename = "img"+to_string(saveid)+".png";
			cv::imwrite("img.png", frame);
			saveid++;
		}

		//�ߋ����̕ۑ�
		frame_p = frame.clone();
		keypoints_p = keypoints;
		descriptor_p = descriptor.clone();
		

	}
	cv::destroyAllWindows();
	return 0;
}

