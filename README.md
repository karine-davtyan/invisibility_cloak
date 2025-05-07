# Real-Time Invisibility Cloak GUI

This project implements a “magic cloak” effect by semantically segmenting out people in your webcam feed (using DeepLabV3+ResNet101) and replacing them with a static background.  
It runs in a PyQt5 GUI with live sliders, and can be packaged via Docker.

---

## 📦 Repository Contents

    invisibility_cloak/
    ├── invisibility_cloak_gui.py    # PyQt5 application
    ├── requirements.txt              # Python dependencies
    ├── Dockerfile                    # Container setup
    ├── README.md                     # This file
    └── .gitignore                    # Ignore patterns

---

## 🚀 Features

- **Live webcam preview** with person-mask compositing  
- **Adjustable** Blur Radius & Threshold sliders  
- **Capture Background** from your camera or **Load Background** from an image file  
- **Docker** container for zero-install deployment  

---

## ⚙️ Requirements

- Python 3.9+  
- Windows/macOS/Linux (GUI + camera support)  
- (Optional) NVIDIA GPU for faster inference

---

## 🛠️ Setup

1. Clone the repo  
   
       git clone https://github.com/karine-davtyan/invisibility_cloak.git  
       cd invisibility_cloak  

2. Create and activate a virtual environment  
   - **Windows (PowerShell)**  
         
           python -m venv .venv  
           .\.venv\Scripts\Activate.ps1  

   - **macOS/Linux**  
         
           python3 -m venv .venv  
           source .venv/bin/activate  

3. Install dependencies  
       
       pip install --upgrade pip  
       pip install -r requirements.txt  

4. Run the GUI  
       
       python invisibility_cloak_gui.py  

### 🐳 Docker

- **Build the image**  
      
       docker build -t inviscloak-gui .  

- **Run the container**  
  - **Linux/macOS (with X11 display)**  
        
         xhost +local:docker  
         docker run --rm \
           --device /dev/video0 \
           -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           inviscloak-gui  

  - **Windows**  
    1. Install an X-server (e.g. VcXsrv) and set `DISPLAY` accordingly.  
    2. Run:  
           
           docker run --rm `
             --device video="Integrated Camera":rwm `
             -e DISPLAY=host.docker.internal:0 `
             inviscloak-gui  

---

## 🎮 Usage

- **Capture Background:** grab the current camera frame as your static backdrop  
- **Load Background:** choose any image (JPG/PNG) from disk  
- **Blur Radius:** smooths the mask edges (odd values only)  
- **Threshold:** adjust mask transparency cutoff  
