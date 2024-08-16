These files has own datasets but if you want to create your datasets and retrain the model :

for sign to speech :

tam proje\signtospeech\ciftelveri.

for every word you should change these lines .
os.makedirs('deneme/data/hello', exist_ok=True)  #the file path which is you will save in it.


cap = cv2.VideoCapture(0)

word = "hello" #the word which is you will save a new data about it.

You can't add a data files in file with content.

AFTER THAT 
You should retrain the model by using ciftelmodel . It will save a new model with the same name . 
(If you want to change the model name joblib.dump(clf, 'hand_gesture_model.pkl') change this line as you want)

words = ['hello','i am','you','okey','can','i','have','coffe','espresso','thanks','large','milk','card','ice']
Whatever words you want to train your model with, you should write the same names you used when recording data here.



for speech to sign : 
I used a model which is already created from https://sign.mt/.
tam proje\speechtıtexttovideo
the assets file has videos about words. 
For new videos just change the file content.

The details about the project in the word file. 


[SigntoSpeech_Backend_Raport.docx](https://github.com/user-attachments/files/16640386/SigntoSpeech_Backend_Raport.docx)



