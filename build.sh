set -o errexit

pip install --upgrade Flask Werkzeug
pip install pyaudio --global-option=build_ext --global-option="C:\Users\anjal\Downloads\portaudio\include"

pip install --upgrade pip
pip install -r requirements.txt
