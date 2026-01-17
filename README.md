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

To run the program:
uvicorn travel_assistant:app --host 127.0.0.1 --port 8000 --reload
then open the travel assistant react.html file (Keep the ollama cmd terminal open)

i hate my chud life
