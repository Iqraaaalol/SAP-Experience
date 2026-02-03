### Setup:
1. Create new virtual enviorntment 

2. pip install -r requirements.txt (Change torch version based on which cuda version youre using)

3. Run "python download_dataset.py"

4. Run "python chroma_ingest.py" (Takes a minute)

5. Download Ollama and run 'ollama pull llama 3.2:3b' (link: https://ollama.com/download/OllamaSetup.exe)


### For Local Use Only:
ollama serve or have app open

### For Remote Access (Connect from Other Devices):
1. Set environment variable (run in PowerShell as Administrator):
   $env:OLLAMA_HOST="0.0.0.0:11434"
   ollama serve or have app open

2. Allow firewall access:
   New-NetFirewallRule -DisplayName "OLLAMA" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow

3. Find your IP address:
   ipconfig
   (Look for IPv4 address, e.g., 192.168.1.100)

4. Update your .env file:
   OLLAMA_URL=http://YOUR_IP_HERE:11434

5. Test from another device:
   curl http://YOUR_IP_HERE:11434/api/version


### How to run app:
1. Run "python run.py"

2. Access the local passenger interface: "localhost:8000"
   Access the crew dashboard with: "localhost:8000/crew-dashboard"

3. Access through network local passenger interface: "{Your_IP_here}:8000"
   Access through network crew dashboard with: "{Your_IP_here}:8000/crew-dashboard"