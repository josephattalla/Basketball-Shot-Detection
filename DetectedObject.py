class DetectedObject:
    def __init__(self, x: int, y: int, w: int, h: int, frame: int, conf: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame = frame
        self.conf = conf
    
    def __eq__(self, other):
        if not other:
            return False
        return self.x == other.x and self.y == other.y and self.w == other.w and self.h == other.h and self.frame == other.frame and self.conf == other.conf