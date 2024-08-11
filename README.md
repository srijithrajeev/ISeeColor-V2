# ISeeColor Visualization App

ISeeColor is a Python-based application that visualizes objects and their corresponding fixation points in video frames using YOLOv8 segmentation models. This tool is designed to help analyze visual attention data by displaying objects of interest in video footage.

This implementation is based on and extends the methods described in the paper:

**Panetta, Karen, Qianwen Wan, Srijith Rajeev, Aleksandra Kaszowska, Aaron L. Gardony, Kevin Naranjo, Holly A. Taylor, and Sos Agaian. "ISeeColor: Method for Advanced Visual Analytics of Eye Tracking Data." IEEE Access 8 (2020): 52278-52287.** [Link to the Paper](https://ieeexplore.ieee.org/document/9036879)

## Minimum Requirements

- **Operating System**: Windows 10+ / Ubuntu 20.04+
- **Python Version**: Python 3.8+
- **Memory**: 8 GB RAM (16 GB recommended)
- **Disk Space**: 1 GB free disk space
- **GPU**: Nvidia GPU with CUDA support (for faster inference)

## Installation and Setup

### Windows

1. **Install Python 3.8+**:
   - Download and install the latest version of Python from the [official Python website](https://www.python.org/downloads/).
   - During installation, ensure that the "Add Python to PATH" option is checked.

2. **Install Poetry**:
   - Open Command Prompt / PS and run:
     ```bash
     curl -sSL https://install.python-poetry.org | python -
     ```
   - Add Poetry to your PATH by following the instructions provided after installation.

3. **Clone the Repository**:
   - Open Command Prompt (`cmd`) and navigate to the directory where you want to clone the repository:
     ```bash
     git clone https://github.com/srijithrajeev/ISeeColor-V2.git
     cd ISeeColor-V2
     ```

4. **Install Dependencies**:
   - Install the necessary dependencies using Poetry:
     ```bash
     poetry install
     ```

5. **Run the Application**:
   - Start the application with:
     ```bash
     poetry run python iseecolor_app.py
     ```

### Ubuntu

1. **Install Python 3.8+**:
   - Ensure Python 3.8+ is installed by running:
     ```bash
     sudo apt update
     sudo apt install python3 python3-pip python3-venv
     ```

2. **Install Poetry**:
   - Run the following commands in the terminal:
     ```bash
     curl -sSL https://install.python-poetry.org | python3 -
     ```
   - Add Poetry to your PATH by modifying your shell configuration file (`~/.bashrc`, `~/.zshrc`, etc.) and adding:
     ```bash
     export PATH="$HOME/.poetry/bin:$PATH"
     ```
   - Then, apply the changes:
     ```bash
     source ~/.bashrc  # or source ~/.zshrc
     ```

3. **Clone the Repository**:
   - Open a terminal and navigate to the directory where you want to clone the repository:
     ```bash
     git clone https://github.com/srijithrajeev/ISeeColor-V2.git
     cd ISeeColor-V2
     ```

4. **Install Dependencies**:
   - Install the necessary dependencies using Poetry:
     ```bash
     poetry install
     ```

5. **Run the Application**:
   - Start the application with:
     ```bash
     poetry run python iseecolor_app.py
     ```

## Usage

Once the application is running:

1. **Load Video**: Click on the "Load Video" button to open a video file (supported formats: `.mp4`, `.avi`, `.mov`).
2. **Load Fixation Data**: Click on the "Load Fixation Data" button to load a CSV file containing eye-tracking data.
   - A sample video file and eye-tracking data file in the correct format are provided in the `data/` folder of this repository.
3. **Play Video**: Use the play, pause, stop, and frame navigation buttons to control video playback.
4. **Analyze**: The application will display objects detected by the YOLO model and highlight fixation points.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues, please open an issue on GitHub or contact the project maintainer.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8) for the object segmentation model.
- [PyQt5](https://riverbankcomputing.com/software/pyqt/intro) for the GUI framework.
- **Panetta, Karen, et al. "ISeeColor: Method for Advanced Visual Analytics of Eye Tracking Data." IEEE Access 8 (2020): 52278-52287.** [Link to the Paper](https://ieeexplore.ieee.org/document/9036879)
