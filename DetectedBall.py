from collections import deque
from DetectedObject import DetectedObject

class DetectedBall:
    def __init__(self, ball: DetectedObject = None):
        self.detections = deque([]) if not ball else deque([ball])

    def __eq__(self, other):
        if not other:
            return False
        if len(self.detections) != len(other.detections):
            return False
        for i in range(len(self.detections)):
            if self.detections[i] != other.detections[i]:
                return False
        return True
        
    
    def add_detection(self, ball_position: DetectedObject):
        self.detections.append(ball_position)
    
    def get_last_detection(self):
        return self.detections[-1]