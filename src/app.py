from flask import Flask, render_template, request, jsonify, send_from_directory, session
from flask_socketio import SocketIO, emit, join_room, leave_room
from blueprint_processor import BlueprintProcessor
from pathlib import Path
import os
import json
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'rescue-blueprint-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize processor with config
blueprint_dir = os.getenv('BLUEPRINT_DIR', '../data/blueprints')
processor = BlueprintProcessor(blueprint_dir)

# Store active users and their rooms (blueprints)
active_users = {}
blueprint_rooms = {}

@app.route('/')
def index():
    converted_files = processor.convert_pdfs()
    return render_template('annotator.html', files=converted_files)
    
@app.route('/dashboard')
def dashboard():
    """Command center dashboard route"""
    converted_files = processor.convert_pdfs()
    # Get all assessment data for dashboard
    assessments = processor.get_all_assessments()
    return render_template('dashboard.html', files=converted_files, assessments=assessments)

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

@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    """Generate risk heatmap"""
    try:
        data = request.json
        filename = data['filename']
        damage_marks = data['damage_marks']
        victim_markers = data.get('victim_markers', [])
        secondary_hazards = data.get('secondary_hazards', [])
        intensity = data.get('intensity', 1.0)
        
        # Call the processor to generate the heatmap
        heatmap_result = processor.generate_heatmap(
            filename, 
            damage_marks, 
            victim_markers, 
            secondary_hazards,
            intensity
        )
        
        if heatmap_result.get('status') == 'success':
            # If the user is in a room, broadcast the heatmap to all users in the room
            sid = request.sid
            if sid in active_users:
                user = active_users[sid]
                room = user.get('room')
                
                if room:
                    # Broadcast to all users in the room except the sender
                    emit('heatmap_generated', {
                        'filename': filename,
                        'heatmap_image': heatmap_result['heatmap_image'],
                        'stats': heatmap_result['stats'],
                        'username': user.get('username', 'Unknown user')
                    }, room=room, skip_sid=sid)
            
            return jsonify(heatmap_result)
        else:
            return jsonify({'status': 'error', 'message': heatmap_result.get('message', 'Unknown error')})
            
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/find_safe_path', methods=['POST'])
def find_safe_path():
    try:
        data = request.json
        print(f"Received pathfinding request: {data}")
        
        filename = data['filename']
        start_point = tuple(data['start_point'])
        end_point = tuple(data['end_point'])
        damage_marks = data.get('damage_marks', [])
        victim_markers = data.get('victim_markers', [])
        
        print(f"Start: {start_point}, End: {end_point}, Damage marks: {len(damage_marks)}, Victims: {len(victim_markers)}")
        
        # Optionally prioritize path to victims
        prioritize_victims = data.get('prioritize_victims', False)
        
        path_results = processor.find_safe_path(filename, start_point, end_point, damage_marks, 
                                               victim_markers=victim_markers,
                                               prioritize_victims=prioritize_victims)
        
        print(f"Pathfinding result: {path_results.get('status')}, Path length: {len(path_results.get('path', []))}")
        
        return jsonify(path_results)
    except Exception as e:
        print(f"Error in find_safe_path route: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)})
        
@app.route('/save_victims', methods=['POST'])
def save_victim_locations():
    """Save victim location markers with optional photos"""
    try:
        data = request.json
        filename = data['filename']
        victim_markers = data['victim_markers']
        
        results = processor.save_victim_locations(filename, victim_markers)
        return jsonify(results)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/upload_victim_photo', methods=['POST'])
def upload_victim_photo():
    """Upload a photo associated with a victim marker"""
    try:
        if 'photo' not in request.files:
            return jsonify({'status': 'error', 'error': 'No photo file provided'})
            
        photo_file = request.files['photo']
        filename = request.form.get('filename', '')
        victim_data = json.loads(request.form.get('victim_data', '{}'))
        
        if not photo_file.filename or not filename or not victim_data:
            return jsonify({'status': 'error', 'error': 'Missing required data'})
            
        results = processor.save_victim_photo(filename, victim_data, photo_file)
        return jsonify(results)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})
        
@app.route('/save_hazards', methods=['POST'])
def save_secondary_hazards():
    """Save secondary hazards markers"""
    try:
        data = request.json
        filename = data['filename']
        hazards = data['hazards']
        
        results = processor.save_secondary_hazards(filename, hazards)
        return jsonify(results)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/get_assessments', methods=['GET'])
def get_all_assessments():
    """Get all damage assessments for the command center dashboard"""
    try:
        assessments = processor.get_all_assessments()
        return jsonify({'status': 'success', 'assessments': assessments})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle new user connections"""
    # Generate unique user ID if not already set
    user_id = str(uuid.uuid4())
    active_users[request.sid] = {
        'id': user_id,
        'username': f"Rescuer-{user_id[:5]}",
        'room': None,
        'color': f"hsl({hash(user_id) % 360}, 70%, 50%)"  # Random color based on user ID
    }
    emit('user_connected', {'user': active_users[request.sid]}, to=request.sid)

@socketio.on('join_blueprint')
def handle_join_blueprint(data):
    """Handle user joining a blueprint room for collaboration"""
    room = data['blueprint']
    username = data.get('username', active_users[request.sid]['username'])
    
    # Update user info
    active_users[request.sid]['room'] = room
    active_users[request.sid]['username'] = username
    
    # Join the room
    join_room(room)
    
    # Initialize room if not exists
    if room not in blueprint_rooms:
        blueprint_rooms[room] = {
            'users': {},
            'damage_marks': []
        }
    
    # Add user to room
    blueprint_rooms[room]['users'][request.sid] = active_users[request.sid]
    
    # Notify room about new user
    room_users = [user for user in blueprint_rooms[room]['users'].values()]
    emit('room_update', {
        'users': room_users,
        'damage_marks': blueprint_rooms[room].get('damage_marks', [])
    }, to=room)
    
    print(f"User {username} joined blueprint room: {room}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle user disconnection"""
    if request.sid in active_users:
        user = active_users[request.sid]
        room = user.get('room')
        
        if room and room in blueprint_rooms and request.sid in blueprint_rooms[room]['users']:
            # Remove user from room
            del blueprint_rooms[room]['users'][request.sid]
            
            # Notify room about user leaving
            room_users = [u for u in blueprint_rooms[room]['users'].values()]
            emit('room_update', {'users': room_users}, to=room)
            
            # Clean up empty rooms
            if not blueprint_rooms[room]['users']:
                del blueprint_rooms[room]
        
        # Remove user from active users
        del active_users[request.sid]

@socketio.on('toggle_heatmap')
def handle_toggle_heatmap(data):
    """Handle heatmap visibility toggle for collaboration"""
    if request.sid in active_users:
        room = active_users[request.sid]['room']
        if room:
            # Add user info to the visibility data
            data['username'] = active_users[request.sid]['username']
            
            # Broadcast to all users in the room except the sender
            emit('heatmap_visibility', data, room=room, skip_sid=request.sid)

@socketio.on('cursor_move')
def handle_cursor_move(data):
    """Handle user cursor movement for collaboration"""
    if request.sid in active_users:
        room = active_users[request.sid]['room']
        if room:
            # Add user info to the cursor data
            data['user'] = {
                'id': active_users[request.sid]['id'],
                'username': active_users[request.sid]['username'],
                'color': active_users[request.sid]['color']
            }
            # Broadcast to room except sender
            emit('cursor_update', data, to=room, skip_sid=request.sid)

@socketio.on('damage_mark_added')
def handle_damage_mark_added(data):
    """Handle new damage mark added by user"""
    if request.sid in active_users:
        room = active_users[request.sid]['room']
        if room and room in blueprint_rooms:
            # Add user info to the damage mark
            damage_mark = data['mark']
            damage_mark['added_by'] = active_users[request.sid]['username']
            
            # Add to room damage marks
            blueprint_rooms[room]['damage_marks'].append(damage_mark)
            
            # Broadcast to room including sender
            emit('damage_mark_update', {
                'action': 'add',
                'mark': damage_mark
            }, to=room)

@socketio.on('voice_command')
def handle_voice_command(data):
    """Handle voice command from user"""
    if request.sid in active_users:
        room = active_users[request.sid]['room']
        command = data.get('command', '').lower()
        position = data.get('position', {'x': 0, 'y': 0})
        
        # Process the command and generate response
        result = processor.process_voice_command(command, position)
        
        if result.get('action') == 'add_damage_mark' and room in blueprint_rooms:
            # Add the damage mark from voice command
            damage_mark = result.get('mark', {})
            damage_mark['added_by'] = active_users[request.sid]['username']
            damage_mark['via_voice'] = True
            
            # Add to room damage marks
            blueprint_rooms[room]['damage_marks'].append(damage_mark)
            
            # Broadcast to room
            emit('damage_mark_update', {
                'action': 'add',
                'mark': damage_mark
            }, to=room)
        
        # Send response back to the user
        emit('voice_command_response', result, to=request.sid)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)