import json
import os
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk
from deepface import DeepFace
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from openai import AzureOpenAI

from speech import text_to_speech

load_dotenv()


# Function to send a message
def send_message(event=None):
    message = entry.get(1.0, tk.END)
    if message.strip() != "":
        # chat_box.config(state=tk.NORMAL)
        # chat_box.insert(tk.END, "You: ", "bold")
        # chat_box.insert(tk.END, message + "\n", "normal")
        # entry.delete(1.0, tk.END)
        # entry.update()
        # chat_response = get_chat_response(message, False)
        # chat_box.insert(tk.END, "AI: ", "bold")
        # chat_box.insert(tk.END, chat_response + "\n", "normal")
        # chat_box.see(tk.END)
        # chat_box.config(state=tk.DISABLED)
        # chat_box.update()
        get_chat_response(message, False)
        # text_to_speech(chat_response)


# Function to update emotion label
def update_emotion_label(emotion, face_confidence):
    label_text = f"Dominant Emotion: {emotion}\t\tFace Confidence: {face_confidence}"
    emotion_label.config(text=label_text)


# Function to analyze emotions from camera
def analyze_emotions_from_camera():
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)

    time_values = []  # List to store time values
    emotion_values = {'angry': [], 'disgust': [], 'fear': [], 'happy': [], 'sad': [], 'surprise': [], 'neutral': []}

    start_time = time.time()
    emotion_counts = {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprise": 0, "neutral": 0}

    while True:  # Infinite loop for continuous video processing
        result, video_frame = video_capture.read()  # read frames from the video
        if not result:
            break  # terminate the loop if the frame is not read successfully

        # Detect faces in the video frame
        gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # Analyze emotions using DeepFace
        result2 = DeepFace.analyze(video_frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result2[0]['dominant_emotion']
        face_confidence = result2[0]['face_confidence']
        emotion_counts[dominant_emotion] += 1

        highest_emotion = max(emotion_counts, key=emotion_counts.get)
        update_emotion_label(highest_emotion, face_confidence)

        time_values.append(time.time() - start_time)
        for emotion in emotion_values.keys():
            emotion_values[emotion].append(result2[0]['emotion'][emotion])

        if time.time() - start_time >= len(time_values) * 0.1:
            ax.clear()
            for emotion, values in emotion_values.items():
                ax.plot(time_values, values, label=emotion)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Emotion Value')
            ax.set_title('Emotion Distribution Over Time')
            ax.legend(loc='upper left')
            ax.grid(True)
            # Apply zoom factor
            apply_zoom_factor()
            canvas.draw()

        # Reset emotion_counts and update start_time
        emotion_counts = {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprise": 0, "neutral": 0}

        # Convert the video frame to RGB format
        rgb_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        # Convert the RGB image to PIL format
        pil_image = Image.fromarray(rgb_image)
        # Convert PIL image to Tkinter-compatible format
        tk_image = ImageTk.PhotoImage(image=pil_image)

        # Update the label with the new image
        video_label.config(image=tk_image)
        video_label.image = tk_image

        if not root.state() == 'zoomed':
            root.state('zoomed')

    video_capture.release()
    cv2.destroyAllWindows()


def get_chat_response(user_message, is_empty):
    def call_open_ai():
        messages = message_text

        if not is_empty:
            messages = load_message("user", user_message)
            chat_box.config(state=tk.NORMAL)
            chat_box.insert(tk.END, "You: ", "bold")
            chat_box.insert(tk.END, user_message + "\n", "normal")
            entry.delete(1.0, tk.END)
            entry.update()

        # if isEmpty:
        gpt_response = client.chat.completions.create(
            model="gpt-4",  # model = "deployment_name"
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        parsed_gpt_response = gpt_response.choices[0].message.content

        if is_empty:
            chat_box.config(state=tk.NORMAL)
            chat_box.insert(tk.END, "AI: ", "bold")
            chat_box.insert(tk.END, parsed_gpt_response + "\n", "normal")
            chat_box.config(state=tk.DISABLED)
            chat_box.update()
            text_to_speech(parsed_gpt_response)

        else:
            chat_box.insert(tk.END, "AI: ", "bold")
            chat_box.insert(tk.END, parsed_gpt_response + "\n", "normal")
            chat_box.see(tk.END)
            chat_box.config(state=tk.DISABLED)
            chat_box.update()

        load_message("assistant", parsed_gpt_response)

    openai_thread = threading.Thread(target=call_open_ai)
    openai_thread.start()


# Function to zoom in on the x-axis
# Global variable to store the zoom factor
zoom_factor = 1.0

client = AzureOpenAI(
    azure_endpoint=os.getenv("API_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2024-02-15-preview"
)


def load_initial_prompt(business_type):
    messages = []
    file = 'coffee.json'
    if business_type:
        business_file_name = business_type.lower().strip().replace(" ", "_")
        file = business_file_name + ".json"

    empty = os.stat(file).st_size == 0

    if not empty:
        with open(file) as db_file:
            data = json.load(db_file)
            for item in data:
                messages.append(item)
    return messages


message_text = []


def load_message(role, message):
    if role:
        message_text.append({
            "role": role,
            "content": message
        })
    return message_text


# Function to update the zoom level indicator
def update_zoom_indicator():
    zoom_indicator.config(text=f"Zoom Level: {zoom_factor:.2f}")


# Function to apply the zoom factor
def apply_zoom_factor():
    global zoom_factor
    current_xlim = ax.get_xlim()
    current_xrange = current_xlim[1] - current_xlim[0]
    new_xrange = current_xrange * zoom_factor
    center_x = (current_xlim[0] + current_xlim[1]) / 2
    ax.set_xlim(center_x - new_xrange / 2, center_x + new_xrange / 2)


# Function to zoom in on the x-axis
def zoom_in():
    global zoom_factor
    zoom_factor *= 0.8
    update_zoom_indicator()


# Function to zoom out on the x-axis
def zoom_out():
    global zoom_factor
    zoom_factor *= 1.0 / 0.8
    update_zoom_indicator()


# Start Chat
def start_chat(business):
    message_text.clear()
    entry.delete(1.0, tk.END)
    entry.update()
    chat_box.config(state=tk.NORMAL)
    chat_box.delete(1.0, tk.END)
    chat_box.config(state=tk.DISABLED)
    chat_box.update()
    message = load_initial_prompt(business)
    message_text.extend(message)
    get_chat_response("", True)


# Create the main window
root = tk.Tk()
root.title("AI SEEQ")

# Set window size and position
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width_percentage = 0.9  # Set the desired percentage
window_width = int(screen_width * window_width_percentage)
window_height = int(screen_height * 0.8)

# Create frames for different colors
left_frame = tk.Frame(root, bg="lightblue", width=int(window_width * 0.5), height=window_height)
left_frame.grid(row=0, column=0, sticky="nsew")

# Divide the left frame into top and bottom parts
top_left_frame = tk.Frame(left_frame, bg="lightblue", width=int(window_width * 0.5), height=int(window_height * 0.5))
top_left_frame.pack(expand=True, fill=tk.BOTH)

bottom_left_frame = tk.Frame(left_frame, bg="lightblue", width=int(window_width * 0.5), height=int(window_height * 0.5))
bottom_left_frame.pack(expand=True, fill=tk.BOTH)

video_label = tk.Label(top_left_frame, bg="lightblue")
video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(8, 4))

# Plot setup and initialization
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Percentage')
ax.set_title('Emotion Distribution Over Time')
ax.grid(True)  # Add gridlines for better readability
ax.set_xlim(0, 1)  # Set initial x-axis limits (can adjust as needed)
ax.set_ylim(0, 100)  # Set initial y-axis limits (can adjust as needed)

canvas = FigureCanvasTkAgg(fig, master=bottom_left_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(expand=True, fill=tk.BOTH)

# Create buttons for zoom in and zoom out
zoom_in_button = tk.Button(bottom_left_frame, text="Zoom In", command=zoom_in)
zoom_in_button.pack(side=tk.LEFT, padx=10, pady=5)

zoom_out_button = tk.Button(bottom_left_frame, text="Zoom Out", command=zoom_out)
zoom_out_button.pack(side=tk.LEFT, padx=10, pady=5)

zoom_indicator = tk.Label(bottom_left_frame, text="Zoom Level: 1.00", bg="white")
zoom_indicator.pack(side=tk.BOTTOM, padx=10, pady=5)

middle_frame = tk.Frame(root, bg="lightblue", width=int(window_width * 0.25), height=window_height)
middle_frame.grid(row=0, column=1, sticky="nsew")

# right_frame = tk.Frame(root, bg="lightgreen", width=int(window_width * 0.25), height=window_height)
# right_frame.grid(row=0, column=2, sticky="nsew")

# Label to display emotions in middle frame
emotion_label = tk.Label(top_left_frame, text="Dominant Emotion: \t\tFace Confidence:", bg="lightblue", justify="left",
                         anchor="nw")
emotion_label.config(font=("Helvetica", 20))  # Adjust font size and family here
emotion_label.pack(padx=2, pady=2, fill=tk.BOTH)
# Create a new frame for chat box
chat_frame = tk.Frame(middle_frame, bg="white")
chat_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=(10, 0))

# Line of Business selection
selection_frame = tk.Frame(chat_frame, bg="white")
selection_frame.pack(padx=10, pady=(10, 0))

business_list = ["Coffee", "Food Truck"]
selected_business = tk.StringVar()
selection_of_business = ttk.Combobox(selection_frame, textvariable=selected_business, font=("Helvetica", 14))
selection_of_business['values'] = business_list
selection_of_business['state'] = 'readonly'
selection_of_business.pack(side=tk.LEFT, padx=5, pady=5)

start_button = tk.Button(selection_frame, text="Start",
                         command=lambda: start_chat(selected_business.get()),
                         font=("Helvetica", 14))
start_button.pack(side=tk.LEFT, pady=5, padx=10)

# Create chat box
chat_box = tk.Text(chat_frame, bg="white", state=tk.DISABLED, wrap=tk.WORD,
                   width=int(window_width * 0.3 / 7))  # Adjust width here
chat_box.tag_configure("bold", font=("Helvetica", 16, "bold"), lmargin1=10, lmargin2=10, rmargin=10, spacing1=5,
                       spacing3=5)
chat_box.tag_configure("normal", font=("Helvetica", 16), lmargin1=10, lmargin2=10, rmargin=10, spacing1=5, spacing3=5)
chat_box.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Create entry for typing messages
entry = tk.Text(chat_frame, bg="white", wrap=tk.WORD, font=("Helvetica", 14),
                width=int(window_width * 0.3 / 7), height=int(window_height * 0.3 / 20))
entry.pack(fill=tk.X, padx=10, pady=5)
# entry.bind("<Return>", send_message)  # Bind Enter key to send_message function

# Create send button
send_button = tk.Button(chat_frame, text="Send", command=send_message, font=("Helvetica", 14))
send_button.pack(fill=tk.BOTH, pady=5, padx=10)

# Configure grid weights to make frames expandable
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
# root.grid_columnconfigure(2, weight=1)

# Start analyzing emotions from camera
threading.Thread(target=analyze_emotions_from_camera).start()

# Run the main loop
root.mainloop()
