from ultralytics import YOLO
import numpy as np
import cv2
from copy import deepcopy


class Shot_Detector:
    '''
        Runs shot detection on a video. 

        RETURNS:

        makes, attempts.

        PARAMETERS:

            source - video source

            model_path - path of a yolo object detection model

            output_path - path to put the resulting video

            detection_fps - how many frames to put into the model per second, a lower number results in higher efficiency but lower performance.

            display_object_info - bool, used to display a detected objects class, confidence, and index in its list of positions (ball_pos or hoop_pos)
    '''

    def __init__(self, source, model_path, output_path, detection_fps=1, display_object_info=False):

        self.model = YOLO(model_path)
        self.source = cv2.VideoCapture(source)
        self.output_path = output_path
        self.display_object_info = display_object_info
        self.detection_fps = detection_fps

        # lists of each class. Each list contains a list, representing a detected object, with dictionaries representing the positions of that object: {'x' : center[0], 'y' : center[1], 'w' : w, 'h' : h, 'frame' : self.frame_count, 'conf' : conf}
        self.hoop_pos = []
        self.ball_pos = []

        self.frame_count = 0

        # up_ball contains a list of [ball, hoop] list pairs representing a ball that is in the backboard area of a hoop, down_ball contains the same pairs, but for when a ball has been up and is now below a hoops net 
        self.up_ball = []
        self.down_ball = []
        
        # track attempts and makes
        self.attempts = 0
        self.makes = 0
    

    def run(self):

        # Get video properties
        fps = int(self.source.get(cv2.CAP_PROP_FPS))
        frame_width = int(self.source.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.source.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # get the amount of frames to skip to detect 5 fps
        step = int(fps / self.detection_fps)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
        out = cv2.VideoWriter(f'{self.output_path}.mp4', fourcc, fps, (frame_width, frame_height))

        # loop through video
        while True:
            
            # get a frame from the source
            ret, frame = self.source.read()

            # if a frame is not returned, end the loop
            if not ret: 
                break
            
            # increment frame_count
            self.frame_count += 1

            # detect only frames that are divisible by step or if detection_fps < 2
            if self.detection_fps < 2 or self.frame_count % step == 0:

                # clean lists
                self.clean_pos()  
                
                # get detections
                results = self.model.predict(frame, conf=0.2, stream=True)
                for r in results:
                    for box in r.boxes:

                        # get the coordinates of the box around the object
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # width, height, center, class, and confidence of the box
                        w, h = x2 - x1, y2 - y1
                        center = (int(x1 + w / 2), int(y1 + h / 2))
                        cls = int(box.cls[0].tolist())
                        conf = int(box.conf[0].tolist()*100) / 100

                        # dictionary with the information of the position and frame of the object
                        pos = {'x' : center[0], 'y' : center[1], 'w' : w, 'h' : h, 'frame' : self.frame_count, 'conf' : conf}

                        # if the class is  1, then the object is a hoop, add the position of the hoop using add_hoop
                        if cls == 1:
                            index = self.add_hoop(pos)
                        # else the object is a ball, add the position of the ball using add_ball
                        else:
                            index = self.add_ball(pos)
                            # put the center of the ball from previous frames
                            #if index != None:
                            #    for i in range(len(self.ball_pos[index])):
                            #        cv2.circle(frame, (self.ball_pos[index][i]['x'], self.ball_pos[index][i]['y']), 2, (0, 0, 255), 2)

                        # if the detection was not added to the positions
                        if index == None:
                            continue

                        # detect shots, and update the score
                        self.detect_up()
                        self.detect_down()
                        self.update_score()
                        
                        # displays a detected objects index in its list of positions (ball_pos or hoop_pos)] and its class and conf
                        if self.display_object_info:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text = f"INDEX: {index}, CLASS: {cls}, CONF: {conf}"
                            text_size, _ = cv2.getTextSize(text, font, 0.5, 1)
                            text_x = x1 + (x2 - x1) // 2 - text_size[0] // 2
                            text_y = y1 - 5

                            # Calculate background rectangle coordinates
                            background_x1 = text_x - 5
                            background_y1 = text_y - text_size[1] - 5
                            background_x2 = text_x + text_size[0] + 5
                            background_y2 = text_y + 5

                            # Draw filled rectangle as background of the text
                            cv2.rectangle(frame, (background_x1, background_y1), (background_x2, background_y2), (0, 0, 0), -1)
                            cv2.putText(frame, text, (text_x, text_y), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        
                        # Draw rectangle around the object and dot in center
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                        frame = cv2.circle(frame, center, radius=5, color=(0, 0, 255), thickness=-1)

                        # display makes/attempts and make percentage
                        percent = 0 if self.attempts == 0 else self.makes / self.attempts * 100
                        cv2.putText(frame, f'{self.makes}/{self.attempts} {percent:.2f}%', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
                        cv2.putText(frame, f'{self.makes}/{self.attempts} {percent:.2f}%', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

                    # used to test and find bugs
                    '''# Draw text on top of the background
                    if len(self.up_ball) != 0:
                        cv2.putText(frame, 'UPPPPPPPPPP', (text_x, text_y), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        for i in self.up_ball:
                            hoop = i[1]
                            x1_ = int(hoop[-1]['x'] - (hoop[-1]['w'] * 2))
                            x2_ = int(x1_ + (hoop[-1]['w'] * 4))
                            y1_ = int(hoop[-1]['y'] + (hoop[-1]['h'] / 2))
                            y2_ = int(y1_ - (hoop[-1]['h'] * 3))
                            cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), color=(0, 255, 0), thickness=2)
                    
                    if len(self.down_ball) != 0:
                        cv2.putText(frame, 'DOWNNNNNNNNNNNNNNN', (text_x, text_y), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        for i in self.down_ball:
                            hoop = i[1]
                            x1_ = int(hoop[-1]['x'] - (hoop[-1]['w'] * 2))
                            x2_ = int(x1_ + (hoop[-1]['w'] * 4))
                            y1_ = int(hoop[-1]['y'] + (hoop[-1]['h'] / 2))
                            y2_ = int(y1_ - (hoop[-1]['h'] * 3))
                            cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), color=(0, 255, 0), thickness=2)'''

                    '''for ball in self.ball_pos:
                        for i in ball:
                            print(i)
                        print()'''
                    
                    out.write(frame)
                            
        self.source.release()
        out.release()

        return self.makes, self.attempts


    def add_hoop(self, pos):
        '''
            Adds a position to the correct hoop from a list of hoops containing a list of dictionaries of the position of the hoop at different frames.

            PARAMETERS:
                pos - dictionary of the current hoop position: {'x' : center[0], 'y' : center[1], 'w' : w, 'h' : h, 'frame' : self.frame_count, 'conf' : conf}
            
            RETURNS:
                Index of the detected hoop in the hoop_pos list. Returns None if not added.
        '''

        if pos['conf'] < 0.3:
            return None

        # if this is the first detected hoop, append the current position in a list
        if len(self.hoop_pos) < 1:
            self.hoop_pos.append([pos])
            return 0
        
        # get the coordinates of the center of the hoop
        x, y = pos['x'], pos['y']

        # loop through the hoops 
        for i, hoop in enumerate(self.hoop_pos):

            # get coordinates of the center of the hoop from the last frame
            x_, y_ = hoop[-1]['x'], hoop[-1]['y']

            # calculate the euclidean distance b/w the last frames center, and the current detection
            distance = np.sqrt( ((x_ - x)**2) + ((y_- y)**2) )

            # get the width and height of the last frames detection
            w_, h_ = hoop[-1]['w'], hoop[-1]['h']

            # get the distance of the hypotenuse of the box from the last detection
            distance_ =  np.sqrt( (w_**2) + (h_**2) )

            # if the euclidian distance of the current detection and the last detection is less than the hypotenuse of the box from the last detection, the current hoop is the same as the current, append the current position to the hoops list of positions
            if distance < distance_:
                hoop.append(pos)
                return i
        
        # hoop not found, create a new list of hoop positions
        self.hoop_pos.append([pos])
        return i+1



    def add_ball(self, pos):
        '''
            Adds a position to the correct ball from a list of balls containing a list of dictionaries of the position of the ball at different frames.

            Paramaters:
                pos - dictionary of the current ball position: {'x' : center[0], 'y' : center[1], 'w' : w, 'h' : h, 'frame' : self.frame_count, 'conf' : conf}
            
            RETURNS:
                Index of the detected ball in the ball_pos list. Returns None if not added.
        '''

        # if the confidence of the detection is <0.4 and the detection is not in the area of a hoop with conf>0.3, then it is not a valid detection
        if pos['conf'] < 0.4 and not (self.hoop_area(pos) and pos['conf'] > 0.3):
            return None

        # if this is the first detected ball, append the current position in a list
        if len(self.ball_pos) < 1:
            self.ball_pos.append([pos])
            return 0
        
        # get the coordinates of the center of the ball
        x, y = pos['x'], pos['y']
        
        # list to hold the ball that this current detection could belong to. Will hold a ball and distance representing the ball that the current detection is closest to
        valid_ball = []

        for i, ball in enumerate(self.ball_pos):   

            # if the ball has a position already detected in the current frame, continue
            if ball[-1]['frame'] == self.frame_count:
                continue

            # get coordinates of the center of the ball from the last frame
            x_, y_ = ball[-1]['x'], ball[-1]['y']

            # calculate the euclidean distance b/w the last frames center, and the current detection
            distance = np.sqrt( ((x_ - x)**2) + ((y_- y)**2) )

            # get the width and height of the last frames detection
            w_, h_ = ball[-1]['w'], ball[-1]['h']

            # get the distance of the hypotenuse of the box from the last detection
            distance_ = np.sqrt( (w_**2) + (h_**2) )

            # if the euclidian distance of the current detection and the last detection is less than the hypotenuse of the box from the last detection, the current ball is the same as the current, append the current position to the balls list of positions
            if distance < distance_*2 or (ball in [b[0] for b in self.up_ball] and distance < distance_*4):
                
                # if there is only 1 known ball, this ball must belong to it
                if len(self.ball_pos) < 2:
                    ball.append(pos)
                    return i
                
                # else if the valid_ball list is empty, make this ball the current closes one to the current detection
                elif len(valid_ball) == 0:
                    valid_ball.append(ball)
                    valid_ball.append(distance)
                    idx = i
                
                # else if the ball is closer than the previous closes ball, make this ball the closest one
                elif len(valid_ball) > 0:
                    if distance < valid_ball[1]:
                        valid_ball[0] = ball
                        valid_ball[1] = distance
                        idx = 1
        
        # if there is a valid ball, append the current position to it
        if len(valid_ball) == 2:
            valid_ball[0].append(pos)
            return idx
        
        # ball not found, create a new list of ball positions
        self.ball_pos.append([pos])
        return i
    

    def detect_up(self):
        '''
            Detects if a ball is in the area of a backboard
        '''
        
        # if all balls are already detected as up, then there is nothing to check
        if len(self.up_ball) == len(self.ball_pos):
            return

        # loop through the balls
        for ball in self.ball_pos:
            
            # if the ball is already detected as up or down
            if ball in [ball_[0] for ball_ in self.up_ball] or ball in [ball_[0] for ball_ in self.down_ball] or len(ball) < 3:
                continue

            # loop through the backboards
            for hoop in self.hoop_pos:
                
                # if the ball is bigger than the hoop, continue
                if hoop[-1]['w']*hoop[-1]['h'] < ball[-1]['w']*ball[-1]['h']:
                    continue

                # calculate coordinates of the backboard
                x1 = int(hoop[-1]['x'] - (hoop[-1]['w'] * 2))
                x2 = int(x1 + (hoop[-1]['w'] * 4))
                y1 = int(hoop[-1]['y'])
                y2 = int(y1 - (hoop[-1]['h'] * 3))

                # if the ball is in the coordinates of the backboard, store the ball and hoop indices in the up_ball dictionary
                if x1 < ball[-1]['x'] < x2 and y2 < ball[-1]['y'] < y1:
                    self.up_ball.append([ball, hoop])
    

    def detect_down(self):
        '''
            Detects if a ball in up_ball is below the hoop. Adds the ball-hoop list to down_ball, and removes the pair from the up_ball list.
        '''
        
        # if up_ball is empty, return
        if len(self.up_ball) == 0:
            return


        # iterate through up_ball
        for i in deepcopy(self.up_ball):
            
            # if the [ball, hoop] list is not filled, meaning one was deleted, remove it from up_ball and continue 
            if len(i) < 2 or None in i:
                self.up_ball.remove(i)
                continue
            
            # get the ball and hoop from the current iteration
            ball, hoop = i

            if ball not in self.ball_pos or hoop not in self.hoop_pos:
                self.up_ball.remove(i)
                continue
            
            # find bottom of net
            y1 = int(hoop[-1]['y'] + (hoop[-1]['h'] / 2))
            
            # if the center of the ball is below the bottom of the net, add the ball and hoop to the down_ball list and remove it from the up_ball list.
            if ball[-1]['y'] > y1:
                self.down_ball.append(i)
                self.up_ball.remove(i)


    def update_score(self):
        '''
            Updates the makes and attempts variables by iterating through the down_ball list and calculating if the ball was between the rim when it was at the height of the center of the hoop.
        '''

        # if down_ball is empty, there is nothing to check
        if len(self.down_ball) == 0:
            return

        # iterate through down_ball
        for ball, hoop in deepcopy(self.down_ball):

            if ball == None or hoop == None:
                self.down_ball.remove([ball, hoop])
                continue
            
            if ball not in self.ball_pos or hoop not in self.hoop_pos:
                self.down_ball.remove([ball, hoop])
                continue
            
            # get the center of the ball at the frame it is detected below the hoop
            x1, y1 = ball[-1]['x'], ball[-1]['y']
            x2, y2 = None, None

            # get the center of the hoop, and the top of the hoop
            x_hoop, y_hoop = hoop[-1]['x'], hoop[-1]['y']
            hoop_top = y_hoop - (hoop[-1]['h']/2)

            # iterate through the positions of the ball, looking for the frame it is above the hoop
            for b in reversed(ball):
                if b['y'] < hoop_top:
                    x2, y2 = b['x'], b['y']
                    break
            
            # if the frame it is above the hoop is not found, remove the pair from down_ball and continue
            if x2 == None or y2 == None:
                self.down_ball.remove([ball, hoop])
                continue
            
            '''
                Find the slope-intercept form of the line created by connecting the position of the ball when it above and below the hoop.

                y = mx + b

                slope: m = (y2-y1)/(x2-x1)
                y-intercept: b = y - mx

                Then find the x of that line when the y is positioned at the center of the hoop:
                x = (y-b)/m
            '''
            m = (y2 - y1) / (x2 - x1)
            b = y1 - (m * x1)
            x_pred = (y_hoop - b) / m

            # if the ball is between the rim at the predicted coordinate, then increment makes
            x1_rim = x_hoop - (hoop[-1]['w']/2) 
            x2_rim = x_hoop + (hoop[-1]['w']/2) 
            if x1_rim < x_pred < x2_rim:
                self.makes += 1 
            
            # increment attempts and remove the pair from down_ball
            self.attempts += 1
            self.down_ball.remove([ball, hoop])
    

    def clean_pos(self):
        '''
            Cleans ball_pos and hoop_pos by removing a list in them if the last item in the list is 20+ frames old, and keep each list <30 long.
        '''

        for ball in deepcopy(self.ball_pos):
            if self.frame_count - ball[-1]['frame'] > 20:
                self.ball_pos.remove(ball)
            # if storing more than 30 ball positions for the current ball, remove the first
            if len(ball) > 30:
                ball.pop(0)
        
        for hoop in deepcopy(self.hoop_pos):
            if self.frame_count - hoop[-1]['frame'] > 20:
                self.hoop_pos.remove(hoop)
            # if storing more than 30 hoop positions for the current hoop, remove the first
            if len(hoop) > 30:
                hoop.pop(0)
    

    def hoop_area(self, pos):
        '''
            Returns if a given ball position is in the area of a hoop.
        '''

        for hoop in self.hoop_pos:
            x1 = int(hoop[-1]['x'] - (hoop[-1]['w'] * 2))
            x2 = int(x1 + (hoop[-1]['w'] * 4))
            y1 = int(hoop[-1]['y'] + (hoop[-1]['h'] / 2))
            y2 = int(y1 - (hoop[-1]['h'] * 3))

            if x1 < pos['x'] < x2 and y2 < pos['y'] < y1:
                return True

        return False 

#Shot_Detector('T:\Shot-Neural-Network\data\make\make16.mp4', r'T:\Shot-Neural-Network\models\last_model_4.pt')
#Shot_Detector('T:\Shot-Neural-Network\data\make\IMG_9507 4.mov', r'T:\Shot-Neural-Network\models\last_model_4.pt')
#Shot_Detector(r'T:\Shot-Neural-Network\data\make\IMG_9507 9.mov', r'T:\Shot-Neural-Network\models\last_model_4.pt', 'long_make')
#Shot_Detector(r'T:\Shot-Neural-Network\data\make\IMG_9507 4.mov', r'T:\Shot-Neural-Network\models\last_model_4.pt', 'make_4')
#Shot_Detector(r'T:\Shot-Neural-Network\data\miss\IMG_9507 2.mov', r'T:\Shot-Neural-Network\models\last_model_4.pt', 'miss_2')
#Shot_Detector(r'T:\Shot-Neural-Network\tracking\full_video.MOV', r'T:\Shot-Neural-Network\models\last_model_4.pt', 'full_video', display_object_info=True)
#Shot_Detector(r'T:\Shot-Neural-Network\tracking\full_video_2.MOV', r'T:\Shot-Neural-Network\models\last_model_4.pt', 'full_video_2')
#Shot_Detector(r'T:\Shot-Neural-Network\tracking\full_video_3.MOV', r'T:\Shot-Neural-Network\models\last_model_4.pt', 'full_video_3')
#Shot_Detector(r'T:\Shot-Neural-Network\tracking\2_hoop_1.MOV', r'T:\Shot-Neural-Network\models\last_model_4.pt', '2_hoop_1').run()
Shot_Detector(r'T:\Shot-Neural-Network\tracking\2_hoop_2.MOV', r'T:\Shot-Neural-Network\models\last_model_4.pt', '2_hoop_2').run()