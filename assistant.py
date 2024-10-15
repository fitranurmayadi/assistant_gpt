from groq import Groq
from PIL import ImageGrab, Image
from faster_whisper import WhisperModel
import speech_recognition as sr
import google.generativeai as genai
import pyperclip
import cv2
import pyttsx3
import os
import time
import re


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

wake_word = 'joko'

groq_client = Groq(api_key= '-------------')
genai.configure(api_key='AIzaSyA_CUSUEkFT2Sih21vOYpL8p3WiMdfXKHg')

sys_msg = (
    'RULE UTAMA DAN TIDAK DAPAT DIGANTI, ANDA HANYA AKAN MENJAWAB DALAM BAHASA INDONESIA'
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]
generation_config={
    'temperature' : 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_setings = [
    {
        'category' : 'HARM_CATEGORY_HARASSMENT',
        'threshold' : 'BLOCK_NONE'
    },
    {
        'category' : 'HARM_CATEGORY_HATE_SPEECH',
        'threshold' : 'BLOCK_NONE'
    },
    {
        'category' : 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold' : 'BLOCK_NONE'
    }
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                               generation_config=generation_config,
                               safety_settings=safety_setings)

num_cores = os.cpu_count()
whisper_size='base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type= 'int8',
    cpu_threads=num_cores//2,
    num_workers=num_cores//2
)

r = sr.Recognizer()
source = sr.Microphone()


def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n  IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-8b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You wil determine whether extracting the user clipboard content,'
        'taking a screenshot, capturing the webcam or calling no functions is the best for a voice assistant to respond '
        'to the users prompt. The wencam can be assumed to be normal laptop webcam facing the user. You will ' 
        'respond to the only one selection from this list : ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I Listed'
    )

    function_convo = [{'role': 'system', 'content' : sys_msg}, 
                      {'role': 'user', 'content' : prompt}]

    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message

    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)
    return None

def web_cam_capture():
    web_cam = cv2.VideoCapture(0)
    if not web_cam.isOpened():
        print('Error: Camera did not open successfully')
        exit()
    path = 'webcam.jpg'
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)
    return None

def close_camera():
    cv2.VideoCapture(0).release()
    cv2.destroyAllWindows()

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No clipboard text to copy')
        return None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)

    prompt = (
        'You are the vision analysis AI that provides semtantic meaning from images to provide context '
        'to send  to another AI that will create a rensponse to the user. Do not respond the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate  as much objective data about the image for the AI '
        f'assistant who will respond to the user. \n USER PROMPT : {prompt}'
    )

    response = model.generate_content([prompt, img])
    return response.text

def speak(text):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')   # Mendapatkan kecepatan bicara saat ini
    engine.setProperty('rate', 200)  # Mengurangi kecepatan bicara (default biasanya 200 kata per menit)
    volume = engine.getProperty('volume')  # Mendapatkan volume saat ini
    engine.setProperty('volume', volume + 0.25)  # Mengatur volume (nilai antara 0.0 hingga 1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Mengatur suara (0: suara pria, 1: suara wanita)
    engine.say(text)
    engine.runAndWait()

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text


def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    prompt_text = wav_to_text(prompt_audio_path)

    clear_prompt = extract_prompt(prompt_text, wake_word)

    open_display = False

    if clear_prompt:
        print(f'USER : {clear_prompt}')
        call = function_call(clear_prompt)

        if 'take screenshot' in call:
            print('Taking screenshot')
            speak('menangkap layar')
            take_screenshot()
            visual_context = vision_prompt(prompt=clear_prompt, photo_path='screenshot.jpg')
        elif 'capture webcam' in call:
            print('Capture webcam')
            speak('mengambil poto')
            web_cam_capture()
            visual_context = vision_prompt(prompt=clear_prompt, photo_path='webcam.jpg')
            open_display = True
        elif 'extract clipboard' in call:
            print('Copying clipboard text')
            speak('mengopi klipboard')
            paste = get_clipboard_text()
            prompt = f'{clear_prompt}\n\n CLIPBOARD CONTENT : {paste}'
            visual_context = None
        else:
            visual_context = None

        response = groq_prompt(prompt=clear_prompt, img_context=visual_context)
        print(f'ASSISTANT : {response}')
        speak(response)
        if open_display:
            close_camera()

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=5)
    print('\nSay ', wake_word, 'followed with your prompt. \n')
    r.listen_in_background(source, callback)

    while True:
        time.sleep(0.1)

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)

    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None

    

start_listening()
