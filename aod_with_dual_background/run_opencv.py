import cv2
import torch

from yolo import YoloDetector
from aod_detector import AodDetectorOpencv

# Device agnostic code
device = torch.device("cuda:0" if torch.cuda.is_available() else "")

if __name__ == '__main__':
    cap = cv2.VideoCapture("sample/video/video2.avi")
    yolo = YoloDetector("yolo/model/yolov8n.pt")

    # List all available opencv algorithm for background subtraction
    opencv_algorithm = {
        "mog2": cv2.createBackgroundSubtractorMOG2(history=7000, detectShadows=False),
        "knn": cv2.createBackgroundSubtractorKNN(history=7000, detectShadows=False),
        "cnt": cv2.bgsegm.createBackgroundSubtractorCNT(useHistory=True, isParallel=True),
        "gsoc": cv2.bgsegm.createBackgroundSubtractorGSOC(),
        "lsbp": cv2.bgsegm.createBackgroundSubtractorLSBP(),
        "mog": cv2.bgsegm.createBackgroundSubtractorMOG(history=7000)
    }

    # Setting for opencv algorithm (we choose BackgroundSubtractorKNN)
    # Because: less noise and more stable masking

    print(f"[Default] NSamples : {opencv_algorithm['knn'].getNSamples()}")
    print(f"[Default] kNNSamples : {opencv_algorithm['knn'].getkNNSamples()}")
    print(f"[Default] Dist2Threshold : {opencv_algorithm['knn'].getDist2Threshold()}")

    """
        SETTING INFORMATIONS:
        NSamples        : Returns the number of data samples in the background model.
        
        kNNSamples      : Returns the number of neighbours, the k in the kNN.
                            - K is the number of samples that need to be within dist2Threshold in order
                              to decide that that pixel is matching the kNN background model.
                              
        Dist2Threshold  : Returns the threshold on the squared distance between the pixel and the sample.
                            - The threshold on the squared distance between the pixel and the sample to decide
                              whether a pixel is close to a data sample.
    """
    opencv_algorithm["knn"].setNSamples(7)
    opencv_algorithm["knn"].setkNNSamples(2)
    opencv_algorithm["knn"].setDist2Threshold(450)

    print(f"[Custom] NSamples : {opencv_algorithm['knn'].getNSamples()}")
    print(f"[Custom] kNNSamples : {opencv_algorithm['knn'].getkNNSamples()}")
    print(f"[Custom] Dist2Threshold : {opencv_algorithm['knn'].getDist2Threshold()}")

    aod = AodDetectorOpencv(
        opencv_algorithm=opencv_algorithm["knn"],
        learning_rate=0.0001,
        abandoned_threshold_frame=400
    )

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        return_value = []

        # Convert to grayscale + blue (kernel must be odd)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (7, 7), 0)

        abandoned_objects = aod.detect(frame)

        # Plotting all variable inside the AOD Calculation (Optional, just for visualisation)
        mask, background = aod.get_variables()
        cv2.imshow("Mask", mask)
        cv2.imshow("Background", background)

        # XYXY Coordinate of abandoned object
        for _, x, y, w, h, _ in abandoned_objects:
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=1)

            cropped = frame[y1:y2, x1:x2]

            _class = "unknown"

            result_dict = {
                "region": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                "class": _class,
                "image": cropped
            }
            return_value.append(result_dict)

        cv2.imshow("Original", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
