import time
import cv2

class FPS:
    def __init__(self):
        self.prev_frame_time = time.time()  # Initialize for the first frame 
        self.new_frame_time = 0

    def draw_fps(self, frame):
        self.new_frame_time = time.time()

        if self.new_frame_time - self.prev_frame_time > 0:
            fps = 1 / (self.new_frame_time - self.prev_frame_time)
        else:
            fps = 0  # Default for cases of extremely fast processing

        self.prev_frame_time = self.new_frame_time

        cv2.putText(frame, f'FPS:{int(fps)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        return fps  # Return the calculated FPS
