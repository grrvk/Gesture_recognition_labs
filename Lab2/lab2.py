import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


wrist_idx = [0]

fingers_tips = [4, 8, 12, 16, 20]
fingers_dips = [3, 7, 11, 15, 19]
fingers_pips = [2, 6, 10, 14, 18]
fingers_mcps = [1, 5, 9, 13, 17]

def get_lm_coords(landmarks, indices):
    tips_of_interest = [landmarks.landmark[idx] for idx in indices]
    toi_coords = [(lm.x, lm.y, lm.z) for lm in tips_of_interest]
    return toi_coords

def check_values(arr, rate):
    arr.sort()
    for i in range(len(arr) - 1):
        if abs(arr[i] - arr[i + 1]) > rate:
            return False
    return True

def check_palm_facing(hands_landmarks, idx):
    # check palm rotation for 90 degrees depending on distance between 4 fingers
    wrist_coords = get_lm_coords(hands_landmarks[idx], wrist_idx)
    fingers_coords = get_lm_coords(hands_landmarks[idx], fingers_mcps)
    coords = [entry[2] for entry in (fingers_coords[1:])]
    mcp_angle_check = check_values(coords, rate=0.05)

    if not mcp_angle_check:
        print('Palm rotated 90')
        return False

    # check palm facing back depending on wrist and thumb Z
    if wrist_coords[0][2] < fingers_coords[0][2]:
        print('Palm facing back')
        return False

    #check palm rotation around X-axis based on middle finger tip and wrist coords
    tips_coords = get_lm_coords(hands_landmarks[idx], fingers_tips)
    coords = [entry[2] for entry in (wrist_coords + [tips_coords[2]])]
    tips_angle_check = check_values(coords, rate=0.1)

    if not tips_angle_check:
        print('fingers facing back/forward')
        return False

    return True

# check that fingers
def check_palm_fingers(hands_landmarks, idx):
    # check that fingers are straight
    for tip, dip, pip, mcp in zip(fingers_tips, fingers_dips, fingers_pips, fingers_mcps):
        y_mcp = hands_landmarks[idx].landmark[mcp].y
        y_pip = hands_landmarks[idx].landmark[pip].y
        y_dip = hands_landmarks[idx].landmark[dip].y
        y_tip = hands_landmarks[idx].landmark[tip].y

        if not (y_mcp >= y_pip >= y_dip >= y_tip):
            print('At least one finger at tip is down')
            return False  # Finger is not straight

    # Finger are stuck together
    tip_xs = [hands_landmarks[idx].landmark[tip].x for tip in fingers_tips[1:]]
    stick_together = check_values(tip_xs, rate=0.05)
    if not stick_together:
        print('Fingers are spread')
        return False

    # check is tip of middle finger the highest
    y_tips = [hands_landmarks[idx].landmark[tip].y for tip in fingers_tips]
    y_middle_tip = y_tips[2]

    middle_high = all(y_middle_tip < y for y in y_tips if y != y_middle_tip)
    if not middle_high:
        print('Middle finger is not the highest')
        return False

    #check is thumb stretched out
    x_pip = hands_landmarks[idx].landmark[2].x
    x_dip = hands_landmarks[idx].landmark[3].x
    x_tip = hands_landmarks[idx].landmark[4].x
    if x_pip > x_dip or x_pip > x_tip:
        print('Thumb is inward')
        return False

    return True

def hand_is_symbol(hands_landmarks, idx=0):
    if check_palm_facing(hands_landmarks, idx) and check_palm_fingers(hands_landmarks, idx):
        return True
    return False


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    landmarks = results.multi_hand_landmarks
    if landmarks:
        state = hand_is_symbol(landmarks)
        c = (0, 255, 0) if state else (0, 0, 255)
        custom_connection_style = mp_drawing.DrawingSpec(color=c, thickness=2)
        for hand_landmarks in landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                custom_connection_style)

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()