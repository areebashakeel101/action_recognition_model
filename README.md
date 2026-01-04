
# action_recognition_model
Action recognition using ResNet50 + LSTM on UCF101 with FastAPI + Gradio frontend
=======
# Action Recognition — ResNet50 + LSTM (UCF101 subset)

One-line: Action recognition using ResNet50 + LSTM on UCF101 with FastAPI + Gradio frontend.

## Overview
- **Project:** Video action recognition pipeline implemented in a single notebook. It extracts frames from videos, encodes frames with a pretrained ResNet50, aggregates temporal features with an LSTM, and serves predictions via a FastAPI endpoint with a Gradio frontend.
- **Notebook:** See [221349_AreebaShakeel.ipynb](221349_AreebaShakeel.ipynb) for full code: dataset loading, model, training loop, evaluation, and demo UI.

## Features
- **Dataset:** Subset of UCF101 (8 selected classes).
- **Preprocessing:** Frame extraction, resizing, and normalization.
- **Model:** `ActionModel` — ResNet50 feature extractor + LSTM + classifier.
- **Training:** Cross-entropy loss, Adam optimizer, early stopping, train/validation loops, and loss plots.
- **Serving:** FastAPI `/predict` endpoint that accepts uploaded video/image files.
- **Frontend:** Gradio UI with sample-video buttons and upload support.

## Quickstart
1. Open and run the notebook: [221349_AreebaShakeel.ipynb](221349_AreebaShakeel.ipynb).
2. Install dependencies (example):

```bash
pip install torch torchvision opencv-python pillow matplotlib fastapi uvicorn gradio requests
```

3. From the notebook you can:
- Start training and evaluation by running the training cells.
- Start the API server (notebook runs `uvicorn.run(app, host="127.0.0.1", port=8000)`).
- Launch the Gradio frontend (`demo.launch(share=True)` in the notebook).

## API
- POST a file to `http://127.0.0.1:8000/predict` with form field `file` to get a JSON response:

```json
{ "filename": "...", "predicted_action": "Basketball" }
```

## Files
- Notebook: [221349_AreebaShakeel.ipynb](221349_AreebaShakeel.ipynb)
- Uploads directory used by API: `uploads/` (created by the notebook)

## Notes & Next Steps
- Consider using 3D CNN backbones (I3D, R3D) or temporal attention for better temporal modeling.
- Improve data augmentation, batch size, and sequence length for performance.
- Productionize: add Dockerfile, model versioning, async file handling, and authentication.

---
Created from the notebook in this repository. Run the notebook to reproduce training and launch the demo.

