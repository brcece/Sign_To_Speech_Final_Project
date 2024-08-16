import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3

# Load the trained model
clf = joblib.load('hand_gesture_model.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def predict_with_confidence(model, X, threshold=0.7):
    try:
        probs = model.predict_proba(X)
        max_prob = np.max(probs)
        if max_prob >= threshold:
            prediction = model.predict(X)[0]
        else:
            prediction = 'Uncertain'
        return prediction, max_prob
    except:
        print('algılanamadı')

recognized_words = []
previous_prediction = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    
    if result.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        
        # Ensure the length of landmarks to be consistent (126 values)
        if len(result.multi_hand_landmarks) == 1:
            landmarks.extend([0] * 63)
        
        landmarks = np.array(landmarks).flatten().reshape(1, -1)
        prediction, confidence = predict_with_confidence(clf, landmarks)
        if prediction != 'Uncertain' and prediction != previous_prediction:
            recognized_words.append(prediction)
            previous_prediction = prediction
            text = f'{prediction} ({confidence * 100:.2f}%)'
        else:
            text = prediction
    
        cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    
    sentence = ' '.join(recognized_words)
    cv2.putText(frame, sentence, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Hand Gesture', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print the final sentence after closing the camera
print(f"Final sentence: {sentence}")
text_speech = pyttsx3.init()

text_speech.setProperty('rate', 110)  
text_speech.setProperty('volume', 1.0) 

answer = sentence

text_speech.say(answer)
text_speech.runAndWait()
