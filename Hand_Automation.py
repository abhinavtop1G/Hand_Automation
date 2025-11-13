import cv2
import time
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Controller
import cvzone

# --------------------------
# MODE FLAGS
# --------------------------
MODE_MENU = 0
MODE_FINGER_COUNTER = 1
MODE_KEYBOARD = 2
current_mode = MODE_MENU

selection_start_time = 0
SELECTION_HOLD_TIME = 3  # seconds needed to confirm a menu option

# --------------------------
# MediaPipe Finger Counter
# --------------------------
class FingerCounter:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def run(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        lmList = []

        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        if lmList:
            fingers = []
            # Thumb
            if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # Other 4 fingers
            for i in range(1, 5):
                if lmList[self.tipIds[i]][2] < lmList[self.tipIds[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total = fingers.count(1)

            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total), (45, 375),
                        cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        return img

# --------------------------
# Virtual Keyboard
# --------------------------
class VirtualKeyboard:
    def __init__(self):
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)
        self.keyboard = Controller()
        self.finalText = ""

        self.keys = [
            ["Q","W","E","R","T","Y","U","I","O","P"],
            ["A","S","D","F","G","H","J","K","L",";"],
            ["Z","X","C","V","B","N","M",",",".","/"]
        ]

        self.buttonList = []
        for i in range(len(self.keys)):
            for j, key in enumerate(self.keys[i]):
                self.buttonList.append(Button([100*j+50, 100*i+50], key))

    def drawAll(self, img):
        for button in self.buttonList:
            x, y = button.pos
            w, h = button.size
            cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), cv2.FILLED)
            cv2.putText(img, button.text, (x+20, y+65),
                        cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
        return img

    def run(self, img):
        hands, img = self.detector.findHands(img)
        img = self.drawAll(img)

        if hands:
            lmList = hands[0]["lmList"]
            index_x, index_y = lmList[8][0], lmList[8][1]

            for button in self.buttonList:
                x,y = button.pos
                w,h = button.size

                # Hover
                if x < index_x < x+w and y < index_y < y+h:
                    cv2.rectangle(img, (x-5,y-5),(x+w+5,y+h+5),(175,0,175),cv2.FILLED)

                    cv2.putText(img, button.text, (x+20,y+65),
                                cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)

                    # Distance for click
                    x1, y1 = lmList[8][0], lmList[8][1]
                    x2, y2 = lmList[12][0], lmList[12][1]
                    length, _, _ = self.detector.findDistance((x1,y1),(x2,y2))

                    if length < 40:
                        self.keyboard.press(button.text)

                        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), cv2.FILLED)
                        cv2.putText(img, button.text, (x+20,y+65),
                                    cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)

                        self.finalText += button.text
                        time.sleep(0.25)

        # Text box
        cv2.rectangle(img,(50,350),(1100,450),(175,0,175),cv2.FILLED)
        cv2.putText(img, self.finalText,(60,425),
                    cv2.FONT_HERSHEY_PLAIN,5,(255,255,255),5)

        return img

class Button:
    def __init__(self,pos,text,size=[85,85]):
        self.pos = pos
        self.size = size
        self.text = text


# --------------------------
# MAIN PROGRAM
# --------------------------

cap = cv2.VideoCapture(0)
fingerCounter = FingerCounter()
keyboardApp = VirtualKeyboard()

prev_fingers = -1

while True:
    success, img = cap.read()

    # -----------------------
    # MODE: MENU
    # -----------------------
    if current_mode == MODE_MENU:
        img = cv2.flip(img, 1)
        detector = mp.solutions.hands.Hands(max_num_hands=1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = detector.process(imgRGB)

        lmList = []
        if res.multi_hand_landmarks:
            handLms = res.multi_hand_landmarks[0]
            h,w,c = img.shape
            for id,lm in enumerate(handLms.landmark):
                lmList.append([id,int(lm.x*w), int(lm.y*h)])

        fingers = 0
        if lmList:
            # Count fingers
            tips = [4,8,12,16,20]
            fingers = 0
            if lmList[tips[0]][1] > lmList[tips[0]-1][1]: fingers+=1
            for i in range(1,5):
                if lmList[tips[i]][2] < lmList[tips[i]-2][2]:
                    fingers+=1

        # MENU UI TEXT
        cv2.putText(img, "MENU", (250,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),3)
        cv2.putText(img, "Show 1 Finger = Finger Counter", (80,180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255),3)
        cv2.putText(img, "Show 2 Fingers = Virtual Keyboard", (80,260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255),3)
        cv2.putText(img, f"Detected: {fingers}", (80,350), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),3)

        # Start hold timer for selection
        if fingers != prev_fingers:
            selection_start_time = time.time()
            prev_fingers = fingers

        hold_time = time.time() - selection_start_time

        if fingers == 1 and hold_time > SELECTION_HOLD_TIME:
            current_mode = MODE_FINGER_COUNTER

        if fingers == 2 and hold_time > SELECTION_HOLD_TIME:
            current_mode = MODE_KEYBOARD

    # -----------------------
    # MODE: FINGER COUNTER
    # -----------------------
    elif current_mode == MODE_FINGER_COUNTER:
        img = fingerCounter.run(img)
        cv2.putText(img, "Press Q to return to menu", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    # -----------------------
    # MODE: KEYBOARD
    # -----------------------
    elif current_mode == MODE_KEYBOARD:
        img = keyboardApp.run(img)
        cv2.putText(img, "Press Q to return to menu", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    # -----------------------
    # Return to menu
    # -----------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        current_mode = MODE_MENU

    cv2.imshow("Image", img)
    cv2.waitKey(1)
