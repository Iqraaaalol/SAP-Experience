If previous step failed, create a new virtual environment and then do it:
python -m venv venv
venv\Scripts\activate
THEN run
pip install -r requirements.txt

ALSO you need to download ollama, then open a cmd terminal and run:
ollama pull llama3.2:1b

### For Local Use Only:
ollama serve

### For Remote Access (Connect from Other Devices):
1. Set environment variable (run in PowerShell as Administrator):
   $env:OLLAMA_HOST="0.0.0.0:11434"
   ollama serve

2. Allow firewall access:
   New-NetFirewallRule -DisplayName "OLLAMA" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow

3. Find your IP address:
   ipconfig
   (Look for IPv4 address, e.g., 192.168.1.100)

4. Update your .env file:
   OLLAMA_URL=http://YOUR_IP_HERE:11434

5. Test from another device:
   curl http://YOUR_IP_HERE:11434/api/version

To run the chatbot:
python travel_assistant.py
Then copy the local or netwrok access url found in terminal

to run the mood detection on camera, run computer-vision\enhanced_face_detection.py 
to run on a video, run computer-vision\enhanced_face_detection.py --video path-to-video-here
i hate my chud life
