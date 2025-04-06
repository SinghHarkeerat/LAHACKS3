import os
import sys
import cv2
import dlib
import math
import json
import statistics
import subprocess
import threading
from PIL import Image
import imageio.v2 as imageio
import numpy as np
from collections import deque
import time
import openai

# Install dependencies if necessary
def install_pyaudio():
    try:
        import pyaudio
    except ImportError:
        print("Installing PyAudio...")
        if sys.platform == "win32":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pipwin"])
            subprocess.check_call([sys.executable, "-m", "pipwin", "install", "pyaudio"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaudio"])

def install_speechrecognition():
    try:
        import speech_recognition as sr
    except ImportError:
        print("Installing SpeechRecognition...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "SpeechRecognition"])

install_pyaudio()
install_speechrecognition()

# Import Speech Recognition
import speech_recognition as sr

# Speech recognition setup
recognizer = sr.Recognizer()

def live_captioning():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎤 Live captioning started...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=3)
                text = recognizer.recognize_google(audio)
                print(f"[Speech]: {text}")
            except sr.UnknownValueError:
                print("[Speech]: (Could not understand)")
            except sr.WaitTimeoutError:
                pass
            except sr.RequestError:
                print("[Speech]: (Speech service unavailable)")
                break

# Start speech recognition in a separate thread (optional)
enable_voice = "yes"
if enable_voice == "yes":
    threading.Thread(target=live_captioning, daemon=True).start()

# Load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../model/face_weights.dat")

cap = cv2.VideoCapture(0)

# Define constants for buffer size and lip measurements
PAST_BUFFER_SIZE = 10  # Size of the buffer for storing previous lip frames
LIP_WIDTH = 60  # Width of the lip frame to extract (adjust this based on your use case)
LIP_HEIGHT = 40  # Height of the lip frame to extract (adjust this based on your use case)
LIP_DISTANCE_THRESHOLD = 40  # Distance threshold for lip movement detection (can be calibrated)
VALID_WORD_THRESHOLD = 10  # Minimum frames to validate a word
NOT_TALKING_THRESHOLD = 5  # Number of frames before considering the person is not talking

TOTAL_FRAMES = PAST_BUFFER_SIZE + VALID_WORD_THRESHOLD  # Total frames for one word (buffer + word frames)

all_words = []
curr_word_frames = []
not_talking_counter = 0
data_count = 1
past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)

words = [""]
options = ", ".join(words)
label = "no"
labels = []

custom_distance = 0
clean_output_dir = input("Clean output directory for this word? (yes/no): ")

if clean_output_dir == "yes":
    outputs_dir = os.path.abspath(os.path.join(os.getcwd(), "../outputs"))
    for folder_name in os.listdir(outputs_dir):
        folder_path = os.path.join(outputs_dir, folder_name)
        if os.path.isdir(folder_path) and label in folder_path:
            print(f"Removing folder {folder_name}...")
            os.system(f"rm -rf \"{folder_path}\"")

determining_lip_distance = 50
lip_distances = []

if custom_distance != "-1" and custom_distance.isdigit() and int(custom_distance) > 0:
    LIP_DISTANCE_THRESHOLD = int(custom_distance)
    determining_lip_distance = 0
    print("✅ Using custom lip distance threshold.")

def process_frames(all_words, labels):
    median_length = int(statistics.median([len(sublist) for sublist in all_words]))
    indices_to_keep = [i for i, sublist in enumerate(all_words) if median_length <= len(sublist) <= median_length + 2]
    all_words = [all_words[i] for i in indices_to_keep]
    labels = [labels[i] for i in indices_to_keep]
    all_words = [sublist[:median_length] for sublist in all_words]
    return all_words, labels

def saveAllWords(all_words, labels):
    output_dir = "../collected_data"
    next_dir_number = 1
    for i, word_frames in enumerate(all_words):
        label = labels[i]
        word_folder = os.path.join(output_dir, f"{label}_{next_dir_number}")
        while os.path.exists(word_folder):
            next_dir_number += 1
            word_folder = os.path.join(output_dir, f"{label}_{next_dir_number}")
        os.makedirs(word_folder)

        with open(os.path.join(word_folder, "data.txt"), "w") as f:
            f.write(json.dumps(word_frames))

        images = []
        for j, img_data in enumerate(word_frames):
            img = Image.new('RGB', (len(img_data[0]), len(img_data)))
            pixels = img.load()
            for y in range(len(img_data)):
                for x in range(len(img_data[y])):
                    pixels[x, y] = tuple(img_data[y][x])
            img_path = os.path.join(word_folder, f"{j}.png")
            img.save(img_path)
            images.append(imageio.imread(img_path))

        video_path = os.path.join(word_folder, "video.mp4")
        imageio.mimsave(video_path, images, fps=int(cap.get(cv2.CAP_PROP_FPS)))
        next_dir_number += 1

def rephrase_sentence(text):
    try:
        response = openai.Completion.create(
            model="gpt-4",  # Corrected to use 'model' instead of 'engine'
            prompt=f"Rephrase the following sentence: '{text}'",
            max_tokens=100,
            temperature=0.7
        )
        rephrased_text = response['choices'][0]['text'].strip()
        return rephrased_text
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return text  # Return the original sentence if an error occurs

# Main loop
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
        mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
        lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])
        lip_left = landmarks.part(48).x
        lip_right = landmarks.part(54).x
        lip_top = landmarks.part(50).y
        lip_bottom = landmarks.part(58).y

        if determining_lip_distance != 0 and LIP_DISTANCE_THRESHOLD is None:
            determining_lip_distance -= 1
            distance = landmarks.part(58).y - landmarks.part(50).y
            lip_distances.append(distance)
            cv2.putText(frame, "Calibrating... Keep mouth closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            if determining_lip_distance == 0:
                LIP_DISTANCE_THRESHOLD = sum(lip_distances) / len(lip_distances) + 2
                print("✅ Calibrated lip distance threshold:", LIP_DISTANCE_THRESHOLD)
            continue

        pad_w = LIP_WIDTH - (lip_right - lip_left)
        pad_h = LIP_HEIGHT - (lip_bottom - lip_top)
        pad_left = min(pad_w // 2, lip_left)
        pad_right = min(pad_w - pad_left, frame.shape[1] - lip_right)
        pad_top = min(pad_h // 2, lip_top)
        pad_bottom = min(pad_h - pad_top, frame.shape[0] - lip_bottom)

        lip_frame = frame[lip_top - pad_top:lip_bottom + pad_bottom, lip_left - pad_left:lip_right + pad_right]
        lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))
        lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lip_frame_lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
        l_eq = clahe.apply(l)
        lip_frame_eq = cv2.merge((l_eq, a, b))
        lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
        lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
        lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
        lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)

        ORANGE, BLUE, RED = (0, 180, 255), (255, 0, 0), (0, 0, 255)

        if lip_distance > LIP_DISTANCE_THRESHOLD:
            cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Recording...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 2)
            curr_word_frames += [lip_frame_eq.tolist()]
            not_talking_counter = 0
        else:
            not_talking_counter += 1
            if not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE == TOTAL_FRAMES:
                data_count += 1
                curr_word_frames = list(past_word_frames) + curr_word_frames
                print(f"[+] Saving word '{label.upper()}' | count: {data_count}")
                all_words.append(curr_word_frames)
                labels.append(label)
                curr_word_frames = []
                not_talking_counter = 0
            elif len(curr_word_frames) > VALID_WORD_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE < TOTAL_FRAMES:
                curr_word_frames += [lip_frame_eq.tolist()]
                not_talking_counter = 0
            elif len(curr_word_frames) < VALID_WORD_THRESHOLD or (not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE > TOTAL_FRAMES):
                curr_word_frames = []

            past_word_frames += [lip_frame_eq.tolist()]
            if len(past_word_frames) > PAST_BUFFER_SIZE:
                past_word_frames.popleft()

        for n in range(48, 61):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.putText(frame, "Press ESC to exit", (900, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Mouth", frame)

    if cv2.waitKey(1) == 27:
        break

print("📝 Processing and saving frames...")
saveAllWords(all_words, labels)

# Rephrase the sentence (example for testing)
detected_sentence = "This is a test sentence."  # Example, replace with actual detected sentence
rephrased_sentence = rephrase_sentence(detected_sentence)
print(f"Original Sentence: {detected_sentence}")
print(f"Rephrased Sentence: {rephrased_sentence}")

cap.release()
cv2.destroyAllWindows()
