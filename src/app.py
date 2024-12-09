from flask import Flask, render_template, request, jsonify
from blueprint_processor import BlueprintProcessor
from pathlib import Path
import os

app = Flask(__name__)

# Initialize processor with config
blueprint_dir = os.getenv('BLUEPRINT_DIR', '../data/blueprints')
processor = BlueprintProcessor(blueprint_dir)

@app.route('/')
def index():
    converted_files = processor.convert_pdfs()
    return render_template('annotator.html', files=converted_files)

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    filename = data['filename']
    annotations = data['annotations']
    
    output_file = processor.save_annotations(filename, annotations)
    return jsonify({'status': 'success', 'file': str(output_file)})

@app.route('/analyze', methods=['POST'])
def analyze_blueprint():
    file = request.files['blueprint']
    results = processor.analyze_blueprint(file)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)