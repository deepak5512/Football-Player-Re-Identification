# ⚽ Football Player Re-identification in a single feed

This project is designed to re-identify football players from input videos using object detection. It allows you to track players frame-by-frame and generate annotated videos as output.

---

## 🚀 How to Run the Project

Follow the steps below to set up and run the project on your local machine:

---

### 1. Clone the Repository

Use the following command to clone the repository:

```bash
git clone https://github.com/deepak5512/Football-Player-Re-Identification.git
cd Football-Player-Re-Identification
```

---

### 2. (Optional) Set Up a Virtual Environment

Setting up a virtual environment is recommended to manage project dependencies cleanly.

**Create the virtual environment:**

```bash
python -m venv .venv
```

**Activate the virtual environment:**

- **On Windows:**
    
    ```bash
    .venv/Scripts/activate
    ```
    
- **On macOS/Linux:**
    
    ```bash
    source .venv/bin/activate
    ```
    

**To deactivate the environment:**

```bash
deactivate
```

---

### 3. Add the Trained Model

Place your trained model file (e.g., `best.pt`) inside the `models/` directory.

---

### 4. Add Input Video

Place the input video you want to process inside the `inputVideos/` folder.

---

### 5. Configure Paths

Ensure the paths to the model and input video are correctly specified in the `main.py` file.

---

### 6. Install the dependencies

Execute the following command in the terminal:

```bash
pip install -r requirements.txt
```

---

### 7. Run the Program

Execute the following command in the terminal:

```bash
python main.py
```

---

### 7. View the Output

Once the process completes, check the `outputVideos/` folder for the annotated output video.

---

## 📁 Folder Structure

```
Football-Player-Re-Identification/
│
├── inputVideos/        # Folder to place raw input videos
├── models/             # Folder to store trained model files
├── outputVideos/       # Folder to store processed output videos
├── trackers/           # Folder to store the code for tracking of players in video
├── utils/              # Folder to store miscellaneous utility functions
├── .gitignore          # Files to be ignored when pushed to GitHub
├── main.py             # Main script to run the pipeline
├── README.md           # Markdown file for information
└── requirements.txt    # All the required dependencies & libraries
```

---

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- Numpy