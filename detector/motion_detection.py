import cv2


class MotionDetector:
    def __init__(self, min_area=400, history=300, var_threshold=25):
        self.min_area = min_area
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )

    def process_frame(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)

        _, fg_mask = cv2.threshold(fg_mask, 230, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [c for c in contours if cv2.contourArea(c) >= self.min_area]

        rois = [cv2.boundingRect(c) for c in filtered]
        return rois
