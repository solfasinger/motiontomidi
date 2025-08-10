from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
import rtmidi
import time
import threading

app = Flask(__name__)

# Initialize MIDI output with better debugging and IAC Driver support
midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()

print(f"Available MIDI ports: {available_ports}")

# Look for IAC Driver specifically (most reliable on Mac)
iac_port = None
for i, port in enumerate(available_ports):
    if "IAC" in port:
        iac_port = i
        break

if iac_port is not None:
    midiout.open_port(iac_port)
    print(f"Connected to IAC Driver: {available_ports[iac_port]}")
else:
    # Create virtual port if no IAC Driver found
    midiout.open_virtual_port("MotionToMIDI")
    print("Created virtual MIDI port: MotionToMIDI")
    print("Note: If you don't see this port in Audio MIDI Setup, enable the IAC Driver instead")

# Store the previous frame and multiple ROIs for motion detection
prev_frame = None
roi_list = {}  # Dictionary to store multiple ROIs {roi_id: roi_coords}
sound_files = {}  # Dictionary to store sound file paths {roi_id: sound_file_path}
roi_midi_notes = {}  # Dictionary to store MIDI notes {roi_id: midi_note}
roi_play_modes = {}  # Dictionary to store play modes {roi_id: 'restart'/'finish'}
roi_last_trigger = {}  # Dictionary to track last trigger times {roi_id: timestamp}
roi_playing_status = {}  # Dictionary to track if sound is currently playing {roi_id: boolean}
roi_note_active = {}  # Dictionary to track if MIDI note is currently active {roi_id: boolean}
simultaneous_play = True  # Global setting for simultaneous vs single play mode

@app.route('/')
def index():
    print("Trying to render index.html")
    try:
        return render_template('index.html')
    except Exception as e:
        print("Error rendering template:", e)
        return f"Template error: {e}"

@app.route('/detect', methods=['POST'])
def detect():
    global prev_frame, roi_list
    
    data = request.json
    image_data = data['image']
    
    # Remove the header of the base64 string
    header, encoded = image_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    frame = np.array(img)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)

    motion_results = {}
    
    if prev_frame is not None:
        frame_delta = cv2.absdiff(prev_frame, frame_gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Check motion for each ROI
        for roi_id, roi_coords in roi_list.items():
            if roi_coords:
                x1, y1, x2, y2 = roi_coords['x1'], roi_coords['y1'], roi_coords['x2'], roi_coords['y2']
                # Convert percentages to pixel coordinates
                h, w = thresh.shape
                x1, x2 = int(x1 * w / 100), int(x2 * w / 100)
                y1, y2 = int(y1 * h / 100), int(y2 * h / 100)
                
                roi_thresh = thresh[y1:y2, x1:x2]
                motion_area = cv2.countNonZero(roi_thresh)
                motion_threshold = 500
                
                motion_detected = motion_area > motion_threshold
                
                # Check play mode and current playing status
                play_mode = roi_play_modes.get(roi_id, 'restart')  # Default to restart
                currently_playing = roi_playing_status.get(roi_id, False)
                
                should_trigger = False
                if motion_detected:
                    # Check if enough time has passed since last trigger (2 second cooldown)
                    current_time = time.time()
                    last_trigger_time = roi_last_trigger.get(roi_id, 0)
                    time_since_last_trigger = current_time - last_trigger_time
                    
                    if time_since_last_trigger >= 2.0:  # 2 second cooldown
                        if play_mode == 'restart':
                            should_trigger = True
                        elif play_mode == 'finish' and not currently_playing:
                            should_trigger = True
                
                # Send MIDI note if motion detected and should trigger
                if should_trigger:
                    # In single play mode, stop all other playing sounds
                    if not simultaneous_play:
                        for other_roi_id in roi_playing_status:
                            if other_roi_id != roi_id and roi_playing_status.get(other_roi_id, False):
                                roi_playing_status[other_roi_id] = False
                                print(f"Stopped sound for ROI {other_roi_id} due to single play mode")
                    
                    midi_note = roi_midi_notes.get(roi_id)
                    if midi_note is not None:
                        try:
                            midiout.send_message([0x90, midi_note, 100])  # Note on, velocity 100
                            print(f"✓ Motion detected in ROI {roi_id}, sent MIDI note {midi_note} (mode: {play_mode})")
                            
                            # Schedule MIDI note off after 2 seconds
                            def send_note_off():
                                time.sleep(2.0)
                                try:
                                    midiout.send_message([0x80, midi_note, 0])  # Note off
                                except:
                                    pass
                            threading.Thread(target=send_note_off, daemon=True).start()
                            
                        except Exception as e:
                            print(f"✗ Error sending MIDI note {midi_note} for ROI {roi_id}: {e}")
                    else:
                        print(f"! Motion detected in ROI {roi_id}, but no MIDI note assigned")
                    
                    # Update trigger time and playing status
                    roi_last_trigger[roi_id] = time.time()
                    if play_mode == 'finish':
                        roi_playing_status[roi_id] = True
                
                motion_results[roi_id] = {
                    'motion': motion_detected,
                    'motion_area': int(motion_area),
                    'sound_file': sound_files.get(roi_id, None),
                    'midi_note': roi_midi_notes.get(roi_id, None),
                    'play_mode': play_mode,
                    'should_trigger': should_trigger,
                    'currently_playing': currently_playing,
                    'stop_others': should_trigger and not simultaneous_play
                }
        
        # If no ROIs are set, check global motion
        if not roi_list:
            motion_area = cv2.countNonZero(thresh)
            motion_threshold = 2000
            global_motion = motion_area > motion_threshold
            if global_motion:
                print(f"Global motion detected (area: {motion_area})")
            motion_results['global'] = {
                'motion': global_motion,
                'motion_area': int(motion_area),
                'sound_file': None,
                'midi_note': None,
                'play_mode': 'restart',
                'should_trigger': global_motion,
                'currently_playing': False,
                'stop_others': False
            }

    prev_frame = frame_gray
    return jsonify({
        'motion_results': motion_results,
        'roi_list': roi_list
    })

@app.route('/update_roi', methods=['POST'])
def update_roi():
    global roi_list
    data = request.json
    roi_id = data['roi_id']
    roi_coords = data['roi_coords']
    
    roi_list[roi_id] = roi_coords
    print(f"Updated ROI {roi_id}: {roi_coords}")
    return jsonify({'success': True})

@app.route('/get_roi_list', methods=['GET'])
def get_roi_list():
    """New endpoint to get current ROI list for editing"""
    return jsonify({'roi_list': roi_list})

@app.route('/clear_roi', methods=['POST'])
def clear_roi():
    global roi_list
    data = request.json
    roi_id = data.get('roi_id', None)
    
    if roi_id and roi_id in roi_list:
        del roi_list[roi_id]
        # Also clear related data
        roi_playing_status.pop(roi_id, None)
        roi_last_trigger.pop(roi_id, None)
        print(f"Cleared ROI {roi_id}")
    else:
        roi_list.clear()
        roi_playing_status.clear()
        roi_last_trigger.clear()
        print("Cleared all ROIs")
    
    return jsonify({'success': True})

@app.route('/set_midi_note', methods=['POST'])
def set_midi_note():
    data = request.json
    roi_id = data['roi_id']
    midi_note = int(data['midi_note'])
    roi_midi_notes[roi_id] = midi_note
    print(f"Set MIDI note {midi_note} for ROI {roi_id}")
    return jsonify({'success': True})

@app.route('/set_play_mode', methods=['POST'])
def set_play_mode():
    data = request.json
    roi_id = data['roi_id']
    play_mode = data['play_mode']  # 'restart' or 'finish'
    roi_play_modes[roi_id] = play_mode
    print(f"Set play mode '{play_mode}' for ROI {roi_id}")
    return jsonify({'success': True})

@app.route('/set_simultaneous_play', methods=['POST'])
def set_simultaneous_play():
    global simultaneous_play
    data = request.json
    simultaneous_play = data['simultaneous_play']
    print(f"Set simultaneous play mode: {simultaneous_play}")
    return jsonify({'success': True})

@app.route('/get_simultaneous_play', methods=['GET'])
def get_simultaneous_play():
    return jsonify({'simultaneous_play': simultaneous_play})

@app.route('/sound_finished', methods=['POST'])
def sound_finished():
    data = request.json
    roi_id = data['roi_id']
    roi_playing_status[roi_id] = False
    print(f"Sound finished playing for ROI {roi_id}")
    return jsonify({'success': True})

@app.route('/upload_sound', methods=['POST'])
def upload_sound():
    global sound_files
    
    if 'sound_file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['sound_file']
    roi_id = request.form.get('roi_id')
    
    if file.filename == '' or not roi_id:
        return jsonify({'error': 'No file selected or ROI ID missing'}), 400
    
    # Save the file
    filename = f"roi_{roi_id}_{file.filename}"
    filepath = os.path.join('static', filename)
    file.save(filepath)
    
    # Store the sound file path for this ROI
    sound_files[roi_id] = filename
    print(f"Uploaded sound file {filename} for ROI {roi_id}")
    
    return jsonify({'success': True, 'filename': filename})

@app.route('/remove_sound', methods=['POST'])
def remove_sound():
    global sound_files
    
    data = request.json
    roi_id = data.get('roi_id')
    
    if not roi_id:
        return jsonify({'error': 'ROI ID missing'}), 400
    
    if roi_id in sound_files:
        # Get the filename to delete
        filename = sound_files[roi_id]
        filepath = os.path.join('static', filename)
        
        # Delete the file from disk if it exists
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Deleted sound file: {filepath}")
            else:
                print(f"Sound file not found on disk: {filepath}")
        except Exception as e:
            print(f"Error deleting sound file {filepath}: {e}")
        
        # Remove from sound_files dictionary
        del sound_files[roi_id]
        print(f"Removed sound file for ROI {roi_id}")
        
        return jsonify({'success': True, 'message': f'Sound file removed for ROI {roi_id}'})
    else:
        return jsonify({'error': f'No sound file found for ROI {roi_id}'}), 404

@app.route('/get_sound_files', methods=['GET'])
def get_sound_files():
    return jsonify(sound_files)

@app.route('/get_midi_notes', methods=['GET'])
def get_midi_notes():
    return jsonify(roi_midi_notes)

@app.route('/get_play_modes', methods=['GET'])
def get_play_modes():
    return jsonify(roi_play_modes)

# Serve static files (for sound files)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Create static directory for sound files
    if not os.path.exists('static'):
        os.makedirs('static')
    
    print("\n" + "="*50)
    print("MotionToMIDI Flask App Starting")
    print("="*50)
    print("MIDI Setup Complete!")
    print("Now:")
    print("1. Open your browser to http://localhost:5000")
    print("2. Set up ROIs and assign MIDI notes")
    print("3. Check your DAW/MIDI software for incoming MIDI")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)