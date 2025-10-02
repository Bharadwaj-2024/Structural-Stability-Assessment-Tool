from flask import Flask, render_template, request, jsonify, send_from_directory
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

@app.route('/static/converted/<filename>')
def serve_converted_file(filename):
    """Serve converted blueprint images"""
    try:
        converted_dir = processor.converted_dir
        return send_from_directory(converted_dir, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

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

@app.route('/save_damage', methods=['POST'])
def save_damage_assessment():
    data = request.json
    filename = data['filename']
    damage_marks = data['damage_marks']
    
    results = processor.save_damage_assessment(filename, damage_marks)
    return jsonify(results)

@app.route('/find_safe_path', methods=['POST'])
def find_safe_path():
    try:
        data = request.json
        print(f"Received pathfinding request: {data}")
        
        filename = data['filename']
        start_point = tuple(data['start_point'])
        end_point = tuple(data['end_point'])
        damage_marks = data.get('damage_marks', [])
        
        print(f"Start: {start_point}, End: {end_point}, Damage marks: {len(damage_marks)}")
        
        path_results = processor.find_safe_path(filename, start_point, end_point, damage_marks)
        
        print(f"Pathfinding result: {path_results.get('status')}, Path length: {len(path_results.get('path', []))}")
        
        return jsonify(path_results)
    except Exception as e:
        print(f"Error in find_safe_path route: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)