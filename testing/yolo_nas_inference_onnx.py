import cv2
import imutils
import onnxruntime
import numpy as np

if __name__ == '__main__':
    session = onnxruntime.InferenceSession("yolo_nas_l.onnx", providers=['CUDAExecutionProvider'])
    inputs = [o.name for o in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]

    cap = cv2.VideoCapture(0)
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        orig_frame = frame
        w, h = frame.shape[1], frame.shape[0]

        frame = cv2.resize(frame, (640, 640))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.reshape(1, 3, 640, 640)

        predictions = session.run(outputs, {inputs[0]: frame})
        num_detections, pred_boxes, pred_scores, pred_classes = predictions
        for image_index in range(num_detections.shape[0]):
            for i in range(num_detections[image_index, 0]):
                class_id = pred_classes[image_index, i]
                confidence = pred_scores[image_index, i]
                x_min, y_min, x_max, y_max = pred_boxes[image_index, i]
                print(
                    f"Detected object with class_id={class_id}, confidence={confidence}, x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

        cv2.imshow('webcam', orig_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()


