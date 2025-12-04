# 🎨 AI Image Colorization

### *Deep Learning-based Colorization of Grayscale Images*

This project implements an **AI-based Image Colorization System** using a **U-Net deep learning model** that predicts color channels (a, b) from a grayscale image (L). A **Flask web application** is included for easy user interaction, allowing seamless upload and colorization of images.

---

## 📂 Project Structure

```
AI-Image-Colorization/
│
├── app/                                # Web Application (Flask)
│   ├── app.py                          # Flask main server
│   ├── utils.py                        # Preprocessing & model loading helpers
│   ├── templates/                      # UI HTML templates
│   │   ├── index.html
│   │   └── result.html
│   ├── static/                         # CSS, JS, and images
│   │   ├── css/
│   │   │   └── styles.css
│   │   ├── js/
│   │   └── images/
│   ├── uploads/                        # Uploaded grayscale images
│   └── outputs/                        # Colorized images
│
├── model/                              # Deep Learning Model + Scripts
│   ├── unet_colorization.py            # U-Net architecture
│   ├── dataset_loader.py               # Dataset preprocessing (LAB conversion)
│   ├── train_model.py                  # Model training script
│   └── inference.py                    # Run colorization model
│
├── saved_models/                       # Trained model weights
│   └── colorization_model_best.pth
│
├── data/                               # (Optional) Dataset
│   ├── train/
│   ├── test/
│   └── sample/
│
├── docker/                             # Containerization setup
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── requirements.txt                    # Python libraries
├── README.md                           # Documentation
└── .gitignore                          # Ignore temp/cache files
```

---

## 🚀 Features

* 🌈 **Automatic AI colorization** from grayscale to realistic color.
* 🧠 **U-Net deep learning model** trained on LAB color space.
* 🌐 **Flask-based web app** with drag-and-drop upload.
* 📥 Upload grayscale → 🔄 AI processing → 🎨 Download colorized output.
* 🐳 **Docker support** for easy deployment.
* ⚡ GPU acceleration support for training and inference.

---

## 🛠️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/AI-Image-Colorization.git
cd AI-Image-Colorization
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download or Train Model

Place pretrained model here:

```
saved_models/colorization_model_best.pth
```

Or train your own model:

```bash
python model/train_model.py
```

---

## 🧪 Training the Model

Run:

```bash
python model/train_model.py
```

This script:

* Loads dataset
* Converts RGB → LAB
* Extracts **L-channel** as input
* Extracts **a,b channels** as ground truth
* Trains U-Net using MSELoss
* Saves best model weights

---

## 🎯 Running Inference

Colorize a single image:

```bash
python model/inference.py --image sample.jpg
```

Output saved to:

```
app/outputs/colorized.png
```

---

## 🌍 Running the Web Application

```bash
cd app
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

Upload a grayscale image → get instant colorized output.

---

## 🐳 Docker Deployment

### Build image

```bash
docker build -t ai-colorization .
```

### Run container

```bash
docker run -p 5000:5000 ai-colorization
```

---

## 📊 Model Architecture – U-Net

* Encoder: Extracts high-level grayscale features
* Decoder: Reconstructs colorized output
* Skip Connections: Preserve spatial details
* Output: 2-channel (a, b) prediction

---

## 📈 Dataset

You can use datasets like:

* ImageNet
* COCO
* Places365
* CelebA

Place training images in:

```
data/train/
```

---

## 📤 Outputs

After running inference:

```
app/uploads/       → grayscale inputs
app/outputs/       → colorized results
saved_models/      → trained weights
```

---

## 🎓 Learning Outcomes

* Understanding of CNN & U-Net architecture
* Working with LAB color space
* ML model training & inference
* Flask deployment & web integration
* Docker containerization
* End-to-end AI project workflow

---

## 🤝 Contributing

Pull requests are welcome.
Please follow clean coding standards and document any major changes.

---


