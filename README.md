Before running the app, download the diddybludden dependencies:
pip install -r requirements.txt 

If previous step failed, create a new virtual environment and then do it:
python -m venv venv
venv\Scripts\activate
THEN run
pip install -r requirements.txt

ALSO you need to download ollama, then open a cmd terminal and run:
ollama pull llama3.2:1b
ollama serve

To run the chatbot:
python travel_assistant.py
Then copy the local or netwrok access url found in terminal

to run the mood detection on camera, run computer-vision\enhanced_face_detection.py 
to run on a video, run computer-vision\enhanced_face_detection.py --video path-to-video-here
i hate my chud life
