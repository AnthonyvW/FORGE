# FORGE - Fabricated Optical Resource for Gigapixel Exploration

![Pre-release in progress](https://img.shields.io/badge/Pre--release-in%20progress-yellow)
[![Windows](https://custom-icon-badges.demolab.com/badge/Windows-0078D6?logo=windows11&logoColor=white)](#)
[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](#)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)

> **📅 Scheduled Release:**  
> FORGE is currently in active development and is scheduled for release on **June 23, 2025**.
>


FORGE is an open-source, gigapixel-scale imaging system designed to scan tree core samples with high precision. Built upon a modified off-the-shelf 3D printer, it automates the imaging of multiple samples, producing ultra-high-resolution images suitable for dendrochronology and related research.

[<img src="https://github.com/user-attachments/assets/acc4b7ab-5bc3-4d7d-95b1-6783f011dd43" width="300">](https://github.com/user-attachments/assets/acc4b7ab-5bc3-4d7d-95b1-6783f011dd43)

> *Above: A stitched image covering a 1.5mm x 3mm area of a tree core sample. Click to view full resolution.*

---

## Features

* **Automated Scanning**: Utilizes 3D printer mechanics for precise, repeatable sample movement.
* **High-Resolution Imaging**: Captures gigapixel images of tree core samples.
* **Image Processing**: Includes tools for stitching captured images.
* **Modular Design**: Easily adaptable to different sample types and imaging requirements.

---

## Project Status

FORGE is currently under active development. Recent updates include enhancements to 3D printer control, auto-homing features, and improvements to image stitching algorithms. The project aims to provide a reliable and efficient solution for high-resolution scanning of tree cores.

---

## Getting Started

### Prerequisites

* **Hardware**: A compatible 3D printer modified for imaging purposes and an amscope camera.
* **Software**: Python 3.x and the dependencies listed in `requirements.txt`.
* **Operating System**: Linux, Windows 10, and Windows 11

### Installation

1\. Clone the repository:

   ```bash
   git clone https://github.com/AnthonyvW/FORGE.git
   cd FORGE
   ```


2\. Install the required Python packages:

  ```bash
  pip install -r requirements.txt
  ```


3\. Configure the camera settings using `amscope_camera_configuration.yaml`.

4\. Run the main application:
  
  ```bash
  python main.py
  ```


---

## Repository Structure

* **Camera/**: Camera control scripts and configurations.
* **UI/**: User interface components.
* **image\_processing/**: Scripts for stitching and analyzing images.
* **printer/**: 3D printer control and configuration files.
* **main.py**: Entry point for the application.
* **requirements.txt**: List of Python dependencies.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements. For major changes, open an issue first to discuss your proposed modifications.

---
