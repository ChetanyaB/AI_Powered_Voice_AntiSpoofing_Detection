# AI Powered Voice Antispoofing Detection

AI Powered Voice Antispoofing Detection is a deep learning based system that classifies voice audio as genuine human speech or spoofed audio such as deepfake, replay attack, or synthetic voice. The project focuses on improving security in voice-based authentication systems and preventing misuse of AI-generated voices.



## Overview

The system analyzes speech signals and learns acoustic patterns that differentiate real human voices from artificially generated or replayed audio. A deep learning model is trained using extracted voice features and is used to predict whether an input audio sample is real or fake.



## How to Run

1. Clone the repository and move into the project directory:

git clone https://github.com/ChetanyaB/AI_Powered_Voice_AntiSpoofing_Detection.git  

cd AI_Powered_Voice_Antispoofing_Detection  


2. Create a virtual environment:

python -m venv venv  


3. Activate the virtual environment:

For Windows:  
venv\Scripts\activate  

For Linux or macOS:  
source venv/bin/activate  


4. Move to Directory:

cd API


5. Install dependencies:

pip install -r requirements.txt  


6. Run file

streamlit run streamlit_app.py



## Testing with Custom Audio

Add your .wav audio file in the browse option or record your voice on the streamlit app opened. 
