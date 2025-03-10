import cv2
from hand_detect import HandDetect  # Importing your HandDetect class

def main():
    cap = cv2.VideoCapture(0)  # Open the default webcam
    hand_detector = HandDetect()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Couldn't read frame from webcam.")
            break
        
        # Flip frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Get finger count and hand gesture
        finger_count, hand_gesture = hand_detector.count_fingers(frame)
        swipe_gesture = hand_detector.swipe_gesture  # Get swipe gesture
        
        # Display the results
        cv2.putText(frame, f"Fingers: {finger_count}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Right: {hand_gesture['RIGHT']}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Left: {hand_gesture['LEFT']}", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.putText(frame, f"Swipe: {swipe_gesture}", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Hand Tracking", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
