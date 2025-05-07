# Real‑Time Invisibility Cloak with GUI & Docker

## Overview
This application applies a real‑time "invisibility cloak" effect using semantic segmentation. It features:

- A PyQt5 GUI with sliders for blur radius and threshold
- Button to load a custom background image
- Live video preview in the app window
- Dockerfile for easy containerization and deployment

## Setup

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/invisibility_cloak_gui.git
   cd invisibility_cloak_gui

## With Docker or locally
docker build -t inviscloak-gui .
docker run --rm --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix inviscloak-gui

pip install -r requirements.txt
python invisibility_cloak_gui.py