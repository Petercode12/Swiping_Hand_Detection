from threading import Thread
import time
import mediapipe as mp
import cv2

mpHands = mp.solutions.hands
draw_utils = mp.solutions.drawing_utils  # For drawing landmarks

def draw_landmarks(image, hand_landmarks):
    draw_utils.draw_landmarks(
        image, hand_landmarks, mpHands.HAND_CONNECTIONS,
        draw_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
        draw_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )

class HandDetect:
    def __init__(self):
        self.hands = mpHands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.8, 
            min_tracking_confidence=0.8
        )
        self.fingers_statuses = {}
        self.count = 0
        self.previous_x = None
        self.swipe_gesture = "NONE"

    def findHandLandMarks(self, hand_landmark, label):
        if label == "Left":
            label = "Right"
        elif label == "Right":
            label = "Left"

        landMarkList = []
        for landmarks in hand_landmark.landmark:
            landMarkList.append([landmarks.x, landmarks.y, label])
        return landMarkList

    def detect_swipe(self, landmarks):
        if not landmarks:
            return

        hand_center_x = sum([point[0] for point in landmarks]) / len(landmarks)

        if self.previous_x is not None:
            movement_threshold = 0.1 # Modify this to change the sensitivity of the detection
            
            if hand_center_x - self.previous_x > movement_threshold:
                print("SWIPE RIGHT")
                self.swipe_gesture = "SWIPE RIGHT"
            elif self.previous_x - hand_center_x > movement_threshold:
                print("SWIPE LEFT")
                self.swipe_gesture = "SWIPE LEFT"
            else:
                self.swipe_gesture = "NONE"

        self.previous_x = hand_center_x

    def check_thumbs_up(self, HandLabel, landmarks):
        if HandLabel == "Left" and landmarks[4][1] < landmarks[8][1]:
            self.fingers_statuses[landmarks[4][2].upper()+'_THUMB_UP'] = True
        if HandLabel == "Right" and landmarks[4][1] < landmarks[8][1]:
            self.fingers_statuses[landmarks[4][2].upper()+'_THUMB_UP'] = True
        if HandLabel == "Left" and landmarks[4][1] > landmarks[0][1]:
            self.fingers_statuses[landmarks[4][2].upper()+'_THUMB_DOWN'] = True
        if HandLabel == "Right" and landmarks[4][1] > landmarks[0][1]:
            self.fingers_statuses[landmarks[4][2].upper()+'_THUMB_DOWN'] = True

    def count_fingers(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        
        fingerCount = 0
        hand_gesture = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN"}

        self.fingers_statuses = {
            'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
            'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
            'LEFT_RING': False, 'LEFT_PINKY': False, 'RIGHT_THUMB_UP': False, 'LEFT_THUMB_UP': False,
            'RIGHT_THUMB_DOWN': False, 'LEFT_THUMB_DOWN': False
        }

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label
                handLandmarks = self.findHandLandMarks(hand_landmarks, handLabel)
                draw_landmarks(frame, hand_landmarks)  # Draw landmarks
                
                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                    fingerCount += 1
                    self.fingers_statuses[handLandmarks[4][2].upper()+'_THUMB'] = True
                elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                    fingerCount += 1
                    self.fingers_statuses[handLandmarks[4][2].upper()+'_THUMB'] = True

                self.check_thumbs_up(handLabel, handLandmarks)
                
                if handLandmarks[8][1] < handLandmarks[6][1]:
                    fingerCount += 1
                    self.fingers_statuses[handLandmarks[8][2].upper()+'_INDEX'] = True
                if handLandmarks[12][1] < handLandmarks[10][1]:
                    fingerCount += 1
                    self.fingers_statuses[handLandmarks[12][2].upper()+'_MIDDLE'] = True
                if handLandmarks[16][1] < handLandmarks[14][1]:
                    fingerCount += 1
                    self.fingers_statuses[handLandmarks[16][2].upper()+'_RING'] = True
                if handLandmarks[20][1] < handLandmarks[18][1]:
                    fingerCount += 1
                    self.fingers_statuses[handLandmarks[20][2].upper()+'_PINKY'] = True

                self.detect_swipe(handLandmarks)

            self.count = fingerCount
            hand_gesture = self.recognizeGesture(hand_gesture)
        
        return fingerCount, hand_gesture
    
    def recognizeGesture(self, hands_gestures):
        hands_labels = {'RIGHT', 'LEFT'}
        
        for hand_label in hands_labels:
            if self.count == 2 and self.fingers_statuses[hand_label+'_MIDDLE'] and self.fingers_statuses[hand_label+'_INDEX']:
                hands_gestures[hand_label] = "V SIGN"
            elif self.count == 3 and self.fingers_statuses[hand_label+'_THUMB'] and self.fingers_statuses[hand_label+'_INDEX'] and self.fingers_statuses[hand_label+'_PINKY']:
                hands_gestures[hand_label] = "SPIDERMAN SIGN"
            elif self.fingers_statuses[hand_label+'_THUMB_UP']:
                hands_gestures[hand_label] = "THUMBS UP"
            elif self.fingers_statuses[hand_label+'_THUMB_DOWN']:
                hands_gestures[hand_label] = "THUMBS DOWN"

        return hands_gestures
