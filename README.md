# AI-Powered Sign Language Translator using MediaPipe and Optimized GRU

## 📌 Overview

AI-Powered Sign Language Translator is a real-time sign language recognition system that leverages MediaPipe for hand landmark detection and an optimized GRU (Gated Recurrent Unit) neural network for gesture classification. The system captures hand gestures through a webcam, processes sequential landmark data, and accurately predicts sign language gestures in real time.

This project aims to bridge communication gaps by enabling efficient interaction between hearing-impaired individuals and others through automated sign language recognition.

---

## 🚀 Features

- Real-time sign language recognition
- Hand landmark extraction using MediaPipe
- Optimized GRU-based deep learning model
- Webcam-based gesture detection
- Fast and accurate predictions
- User-friendly and lightweight implementation

---

## 🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Scikit-Learn
- Matplotlib

---

## ⚙️ How It Works

1. Capture hand gestures using a webcam.
2. Detect hand landmarks with MediaPipe.
3. Convert landmark positions into sequential data.
4. Train an optimized GRU model on the collected dataset.
5. Perform real-time gesture classification.
6. Display the predicted sign language output.

---

## 📂 Project Structure

```text
Sign-Language-Translator/
│
├── data/                  # Dataset and collected gesture sequences
├── models/                # Trained GRU model files
├── app.py                 # Main application for real-time prediction
├── train.py               # Model training script
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── assets/                # Images and additional resources
```

---

## 📦 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/sign-language-translator.git
cd sign-language-translator
```

### Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the Model

```bash
python train.py
```

### Run Real-Time Sign Language Recognition

```bash
python app.py
```

The webcam will start automatically, and the system will recognize trained sign language gestures in real time.

---

## 🧠 Model Architecture

The recognition pipeline consists of:

- **MediaPipe Hands** for extracting hand landmarks
- **Optimized GRU Network** for learning temporal gesture patterns
- **Dense Classification Layer** for final gesture prediction

This architecture provides efficient sequence learning while maintaining low computational overhead.

---

## 📈 Results

- Accurate real-time gesture recognition
- Fast inference speed
- Lightweight and efficient deployment
- Robust hand landmark tracking using MediaPipe

---

## 🎯 Applications

- Assistive communication tools
- Sign language learning platforms
- Accessibility solutions
- Human-computer interaction systems
- Educational and research purposes

---

## 🔮 Future Improvements

- Support for larger sign language vocabularies
- Sentence and phrase generation
- Text-to-speech conversion
- Mobile application deployment
- Multi-hand gesture recognition
- Support for multiple sign languages

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository, create a feature branch, and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👥 Project Team

This project was developed as a collaborative effort by:

- **Garv Gupta**
- **Ashlesha Agrawal**
- **Aashi Soni**

---

If you found this project useful, consider giving it a ⭐ on GitHub.