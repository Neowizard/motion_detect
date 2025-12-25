import cv2


class MotionDetector:
    def __init__(self,
                 min_area=400,  # minimum area to consider as motion
                 history=300,  # how many frames to remember
                 var_threshold=30):  # sensitivity to motion

        self.min_area = min_area
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )

    def process_frame(self, frame):
        """
        Parameters:
            frame: numpy array (BGR) - single frame from pyav (format="bgr24")
        Returns:
            tuple (x, y, w, h)  - bounding box motion regions
        """
        fg_mask = self.bg_subtractor.apply(frame)

        _, fg_mask = cv2.threshold(fg_mask, 230, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rois = [cv2.boundingRect(c) for c in contours]
        return rois
