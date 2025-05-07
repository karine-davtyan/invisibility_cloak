## Real-Time Invisibility Cloak GUI

This project applies a “cloak” effect by semantically segmenting out people in your webcam feed (DeepLabV3+ResNet101) and compositing a static background. It runs in a PyQt5 GUI and can be containerized with Docker.

## Features

- Live webcam preview with real-time person masking  
- Adjustable **Blur Radius** and **Threshold** sliders  
- **Capture Background** (from camera) or **Load Background** (from image file)  
- Packaged via Docker for easy deployment

## Requirements

- Python 3.9+  
- GPU optional (CPU works, but slower)  
- Windows/macOS/Linux

## Setup

1. Clone the repo  
   ```bash
   git clone https://github.com/<your-username>/invisibility_cloak_gui.git
   cd invisibility_cloak_gui
Create & activate a virtual environment

```bash
 Copy
Edit
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # macOS/Linux
Install Python dependencies

```bash
Copy
Edit
pip install --upgrade pip
pip install -r requirements.txt
Run the GUI

bash
Copy
Edit
python invisibility_cloak_gui.py
