# Blueprint Analyzer

AI-powered construction blueprint analyzer for material calculations and structural analysis.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kishan0703/Structural-Stability-Assessment-Tool-.git
cd blueprint-analyzer
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Poppler (for PDF processing):
- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/
- Linux: `sudo apt-get install poppler-utils`
- Mac: `brew install poppler`

## Usage

1. Start the application:
```bash
python src/app.py
```

2. Open http://localhost:5000 in your browser
3. Upload blueprints and annotate them
4. Get material calculations

## Project Structure

```
blueprint-analyzer/
├── data/
│   ├── blueprints/        # PDF blueprints
│   ├── converted_images/   # Converted images
│   ├── annotations/        # YOLO annotations
│   └── models/            # Trained models
├── src/
│   ├── app.py              # Flask application
│   ├── blueprint_processor.py
│   └── yolo_trainer.py
└── requirements.txt
```
