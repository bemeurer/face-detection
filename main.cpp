#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "cascades/frontal_face.h"
#include "cascades/smile.h"

using namespace cv;

void loadFromHeader(CascadeClassifier *foo, char *cascadeArray){
  FileStorage fs(cascadeArray, FileStorage::READ|FileStorage::MEMORY);
  FileNode root = fs.getFirstTopLevelNode();
  (*foo).read(root);
}


int main() {
    VideoCapture vid(0);
    if (!vid.isOpened()) {
        return -1;
    }
    namedWindow("stream", WINDOW_AUTOSIZE);
    CascadeClassifier faceCascade;
    CascadeClassifier smileCascade;
    loadFromHeader(&faceCascade, face_array);
    loadFromHeader(&smileCascade, smile_array);
    Mat frame;
    while (true) {
        vid >> frame;
        Mat frame_gray;
        cvtColor(frame, frame_gray, CV_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        std::vector<Rect> faces;
        faceCascade.detectMultiScale(frame_gray, faces, 1.1, 20, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        for (size_t i = 0; i < faces.size(); i++) {

            rectangle(frame, faces[i], CV_RGB(255, 0, 0), 3);

            int half = faces[i].y + int(faces[i].height * 0.6);
            Rect lower_face(faces[i].x, half, faces[i].width, (faces[i].y + faces[i].height) - half);

            //XXX: These are here for debugging
            //rectangle(frame, lower_face, CV_RGB(0,0,0), 7);
            line(frame, cvPoint(0, half), cvPoint(frame_gray.cols, half), CV_RGB(0, 0, 0), 2);

            std::vector<Rect> smile;
            smileCascade.detectMultiScale(Mat(frame_gray, lower_face), smile, 1.2, 40, 0 | CV_HAAR_DO_CANNY_PRUNING,
                                          Size(30, 30));
            for (size_t x = 0; x < smile.size(); x++) {
                Rect smile_r(faces[i].x + smile[x].x, half + smile[x].y, smile[x].width, smile[x].height);
                rectangle(frame, smile_r, CV_RGB(0, 255, 0), 3);
            }
        }
        imshow("stream", frame);
        if (waitKey(30) >= 0) break;
    }
    return 0;
}
