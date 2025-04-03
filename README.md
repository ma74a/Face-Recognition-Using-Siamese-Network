# Face Recognition Using Siamese Network

A deep learning project that implements face recognition using Siamese Neural Networks with contrastive loss. This project can identify and match faces from images by learning robust face embeddings.

## Overview

This project implements a face recognition system using Siamese Neural Networks, which are particularly effective for one-shot learning tasks. The system learns to generate embeddings for face images and can identify whether two face images belong to the same person or different people.

### Architecture

The project uses a Siamese Network architecture with the following components:

1. **Siamese Network Structure**:
   - Two identical neural networks that share weights
   - Each network processes one input image
   - Outputs face embeddings for comparison

2. **Contrastive Loss**:
   - Measures the similarity between pairs of face embeddings
   - Minimizes distance for different-person pairs
   - Maximizes distance for same-person pairs

### Key Features

- Face detection and cropping using MTCNN
- Siamese Network for face embedding generation
- Contrastive loss for training
- Face database management
- Real-time face matching

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Face-Recognition-Using-Siamese-Network.git
cd Face-Recognition-Using-Siamese-Network
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Project Structure

```
Face-Recognition-Using-Siamese-Network/
├── src/                    # Source code
│   ├── embeddings.py       # Embedding generation and matching
│   ├── model.py           # Siamese network architecture
│   ├── training.py        # Training utilities
│   └── custom_dataset.py  # Dataset handling
├── scripts/               # Utility scripts
│   └── get_faces.py       # Face detection and cropping
├── utils/                 # Helper functions
│   ├── config.py         # Configuration settings
│   └── visualize.py      # Visualization utilities
├── models/               # Saved model checkpoints
├── embeddings/          # Face embedding database
├── train_model.py       # Training script
├── main.py             # Main application script
└── requirements.txt    # Project dependencies
```

## Usage

### Training the Model

To train the model on your own dataset:

```bash
python3 train_model.py
```

### Face Recognition

To perform face recognition on an image:

```bash
python3 main.py path/to/image.jpg
```

## Model Architecture

### Siamese Network
![Siamese Network Architecture](https://raw.githubusercontent.com/yourusername/Face-Recognition-Using-Siamese-Network/main/docs/siamese_network.png)

### Contrastive Loss
The contrastive loss function is defined as:

![Contrastive Loss Formula](https://raw.githubusercontent.com/yourusername/Face-Recognition-Using-Siamese-Network/main/docs/contrastive_loss.png)

Where:
- Y = 0 for same person, 1 for different people
- D is the Euclidean distance between embeddings
- m is the margin parameter

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MTCNN](https://github.com/ipazc/mtcnn) for face detection
- PyTorch for deep learning framework
- [FaceNet](https://arxiv.org/abs/1503.03832) paper for inspiration
