#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

std::vector<Rect> detectFace(CascadeClassifier faceCascade, Mat frame_gray);

std::vector<Rect> detectNose(CascadeClassifier noseCascade, Mat face_gray);

std::vector<Rect> detectSmile(CascadeClassifier smileCascade, Mat lower_face);

int main() {
    VideoCapture vid(0);
    if (!vid.isOpened()) {
        return -1;
    }
    namedWindow("stream", WINDOW_AUTOSIZE);
    CascadeClassifier faceCascade;
    CascadeClassifier noseCascade;
    CascadeClassifier smileCascade;
    faceCascade.load("cascades/frontal_face.xml");
    noseCascade.load("cascades/nose.xml");
    smileCascade.load("cascades/smile.xml");
    Mat frame;
    while (true) {
        vid >> frame;
        Mat frame_gray;
        cvtColor(frame, frame_gray, CV_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        std::vector<Rect> faces = detectFace(faceCascade, frame_gray);
        for (size_t i = 0; i < faces.size(); i++) {
            Mat lower;
            rectangle(frame, faces[i], CV_RGB(255, 0, 0), 3);
            Mat faceROI = frame_gray(faces[i]);
            std::vector<Rect> noses = detectNose(noseCascade, faceROI);
            int half = 0;
            if (noses.size() < 1) {
                half = faces[i].y + faces[i].height / 2;
                Rect lower_face(faces[i].x, half, faces[i].width, (faces[i].y + faces[i].height) - half);
                lower = Mat(frame_gray, lower_face);
                //rectangle(frame, lower_face, CV_RGB(0,0,0), 7); //XXX: Debug drawings
                //line(frame, cvPoint(0, half), cvPoint(frame_gray.cols, half), CV_RGB(0,0,0), 2);
            }
            for (size_t n = 0; n < noses.size(); n++) {
                Rect nose_r(faces[i].x + noses[n].x, faces[i].y + noses[n].y, noses[n].width, noses[n].height);
                half = nose_r.y + nose_r.height / 2;
                Rect lower_face(faces[i].x, half, faces[i].width, (faces[i].y + faces[i].height) - half);
                lower = Mat(frame_gray, lower_face);
                //rectangle(frame, lower_face, CV_RGB(0,0,0), 7); //XXX: Debug drawings
                //line(frame, cvPoint(0, half), cvPoint(frame_gray.cols, half), CV_RGB(0,0,0), 2);
                //rectangle(frame, nose_r, CV_RGB(0,0,255), 3);
            }
            std::vector<Rect> smile = detectSmile(smileCascade, lower);
            for (size_t x = 0; x < smile.size(); x++) {
                Rect smile_r(faces[i].x + smile[x].x, half + smile[x].y, smile[x].width, smile[x].height);
                rectangle(frame, smile_r, CV_RGB(255, 0, 0));
            }
        }
        imshow("stream", frame);
        if (waitKey(30) >= 0) break;
    }
    return 0;
}

std::vector<Rect> detectFace(CascadeClassifier faceCascade, Mat frame_gray) {
    std::vector<Rect> faces;
    faceCascade.detectMultiScale(frame_gray, faces, 1.1, 20, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    return faces;
}

std::vector<Rect> detectNose(CascadeClassifier noseCascade, Mat face_gray) {
    std::vector<Rect> noses;
    noseCascade.detectMultiScale(face_gray, noses, 1.3, 20, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    return noses;
}

std::vector<Rect> detectSmile(CascadeClassifier smileCascade, Mat lower_face) {
    std::vector<Rect> smile;
    smileCascade.detectMultiScale(lower_face, smile, 1.2, 50, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));
    return smile;
}