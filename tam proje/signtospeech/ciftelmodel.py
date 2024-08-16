from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import joblib

# Load the data
def load_data(word):
    data = []
    labels = []
    for file in os.listdir(f'data/{word}'):
        if file.endswith('.txt'):
            landmarks = np.loadtxt(f'data/{word}/{file}')
            # Ensure the length of landmarks to be consistent (126 values)
            if len(landmarks) == 63:
                landmarks = np.concatenate([landmarks, np.zeros(63)])
            data.append(landmarks)
            labels.append(word)
    return data, labels

words = ['hello','i am','you','okey','can','i','have','coffe','espresso','thanks','large','milk','card','ice']
data = []
labels = []

for word in words:
    word_data, word_labels = load_data(word)
    data.extend(word_data)
    labels.extend(word_labels)

data = np.array(data)
labels = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test the model
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(clf, 'hand_gesture_model.pkl')
