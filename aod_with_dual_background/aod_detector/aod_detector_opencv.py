import cv2
import numpy as np
import torch
import torch.nn as nn

from .object_tracker import ObjectTracker

# Device agnostic code
device = torch.device("cuda:0" if torch.cuda.is_available() else "")

class AodDetectorOpencv:
    def __init__(self, opencv_algorithm, learning_rate, abandoned_threshold_frame=500):
        self.tracker = ObjectTracker(abandoned_threshold_frame)
        self.algorithm = opencv_algorithm

        self.lr = -1
        self.desired_lr = learning_rate  # Learning rate that user wanted

        self.mask = None
        self.countours = None
        self.n_frames = 0

        # Class settings
        self.light_change_pixel_threshold = 100  # median mask value threshold (value between 0 - 255)
        self.light_change_occurrence_count = 0  # Error evader on bias (flash on mask frame) of KNN Subtract Class
        self.light_change_occur = False
        self.light_change_cooldown_period = None

    @staticmethod
    def calculate_countours(threshold):
        return cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    @staticmethod
    def calculate_average_pooling(mask, number_of_pool=10):
        # Convert mask to PyTorch tensor
        mask = torch.from_numpy(mask).float().to(device)

        # Add channel dimension if necessary (assuming grayscale input)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension

        pool_size = int(mask.size(2) // number_of_pool)

        # Define average pooling layer
        pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size).to(device)

        # Convert output back to NumPy array and squeeze channel dimension if needed
        output = pool(mask).squeeze().detach().cpu().numpy()

        return output

    def detect(self, frame):
        self.n_frames += 1

        ###########################
        # Below is the code for adaptively adjusting the background if light suddenly comes out
        ###########################

        # Skip 500 frame at first (let the camera initialize)
        # Then update the learning rate value to user desire
        # Also stop updating when light change occur
        if self.n_frames > 500 and not self.light_change_occur:
            self.lr = self.desired_lr

        self.mask = self.algorithm.apply(frame, learningRate=self.lr)

        # Skip 500 frame at first (let the camera initialize)
        if self.n_frames > 500:
            average_pooling = self.calculate_average_pooling(self.mask)
            mean = np.mean(average_pooling)
            if mean > self.light_change_pixel_threshold:
                self.light_change_occurrence_count += 1

                # Executed at the exact moment the light changes (only once)
                if not self.light_change_occur and self.light_change_occurrence_count > 50:     # Wait for 50 frames then update
                    self.light_change_cooldown_period = self.n_frames + 20  # Setting up cooldown period (20 frames)
                    self.light_change_occurrence_count = 0

            if self.light_change_cooldown_period is not None:
                if self.n_frames < self.light_change_cooldown_period:
                    self.lr = 0.99  # Updating background immediately
                    self.light_change_occur = True
                else:
                    # Reset to normal setting
                    self.lr = self.desired_lr
                    self.light_change_occur = False

        print(f"learning rate : {self.lr}")
        self.contours, hierarchy = self.calculate_countours(self.mask)
        boxes_xyxy = []
        for contour in self.contours:
            contour_area = cv2.contourArea(contour)
            if 100 < contour_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                boxes_xyxy.append((x, y, w, h))

        _, abandoned_objects = self.tracker.update(boxes_xyxy)
        return abandoned_objects

    def get_variables(self):
        return self.mask, self.algorithm.getBackgroundImage()

    def set_threshold_difference(self, new_threshold):
        self.threshold_difference = new_threshold
