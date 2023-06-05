#include <opencv2/opencv.hpp>
using namespace cv;

int main()
{
    // Load the pre-trained face cascade
    CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml"))
    {
        std::cout << "Failed to load Haar cascade file!" << std::endl;
        return -1;
    }

    // Open the default video camera
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "Failed to open the camera!" << std::endl;
        return -1;
    }

    // Create a window to display the output
    namedWindow("Face Detection", WINDOW_NORMAL);

    while (true)
    {
        Mat frame;
        cap.read(frame);

        if (frame.empty())
        {
            std::cout << "Failed to capture frame!" << std::endl;
            break;
        }

        // Convert the frame to grayscale for face detection
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // Detect faces in the grayscale frame
        std::vector<Rect> faces;
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // Draw bounding boxes around the detected faces
        for (const Rect& face : faces)
        {
            rectangle(frame, face, Scalar(0, 255, 0), 2);
        }

        // Display the frame
        imshow("Face Detection", frame);

        // Check for key press to exit
        if (waitKey(1) == 27) // ESC key
            break;
    }

    // Release the camera and destroy the window
    cap.release();
    destroyAllWindows();

    return 0;
}
