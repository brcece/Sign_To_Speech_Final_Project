import speech_recognition as sr
import os
import re
import cv2
import numpy as np

def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Konuşmaya başlayın...")
        recognizer.adjust_for_ambient_noise(source)
        
        audio = recognizer.listen(source)

        try:
                # Kullanıcıdan metin al
            text = recognizer.recognize_google(audio, language="en-US")
            print(text)

            processed_text = text.lower()
            replacement_dict = {
                'thank you': 'thanks',
                'hi': 'hello',
                'cold':'iced',
                'short':'small',
                'tall' : 'medium',
                'grande':'large',
                'nice':'good',
                }
            for old_word, new_word in replacement_dict.items():
                    processed_text = processed_text.replace(old_word, new_word)
            print(processed_text)
            kelimeler = re.findall(r'\b\w+\b', processed_text)

            dosya_konumu = "app\\assets"
            webm_dosyalari = [dosya for dosya in os.listdir(dosya_konumu) if dosya.endswith(".webm")]

            for kelime_sira, kelime in enumerate(kelimeler, start=1):
                for dosya in webm_dosyalari:
                    if kelime.lower() == dosya.lower().split(".webm")[0]:  # Tam eşleşme kontrolü
                        dosya_yolu = os.path.join(dosya_konumu, dosya)
                        cap = cv2.VideoCapture(dosya_yolu)
                        if not cap.isOpened():
                            print(f"{dosya} açılamadı.")
                            continue

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            cv2.imshow('Video', frame)
                            if cv2.waitKey(25) & 0xFF == ord('q'):
                                break

                        cap.release()
                        cv2.destroyAllWindows()
            

        except sr.UnknownValueError:
                print("Anlaşılamadı")
        except sr.RequestError as e:
                print(f"Hata: {e}")

        
recognize_speech()
