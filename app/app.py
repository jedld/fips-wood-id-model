from flask import Flask, request, redirect, url_for, render_template_string
from werkzeug.utils import secure_filename
import os
import socket
import threading
import time
import json
from collections import defaultdict
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'tmp/uploads'  # Change this to the directory where you want to save images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
BROADCAST_PORT = 5001  # Port for UDP broadcasting
BROADCAST_INTERVAL = 5  # Broadcast every 5 seconds

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    raise Exception(f'The specified upload folder does not exist: {UPLOAD_FOLDER}')

# Global variable to control broadcasting
broadcast_running = False
broadcast_thread = None

# Server state
selected_class = None  # Currently selected class for uploads

# Upload statistics
upload_stats = {
    'total_uploads': 0,
    'uploads_by_class': defaultdict(int),
    'uploads_by_date': defaultdict(int),
    'last_upload': None
}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to get server IP address
def get_server_ip():
    try:
        # Get the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# Helper function to get available classes
def get_available_classes():
    try:
        with open('class_labels.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes
    except FileNotFoundError:
        # Fallback classes if file not found
        return ['acacia_auriculiformis', 'acacia_mangium', 'eucalyptus_camaldulensis']

# Helper function to update upload statistics
def update_upload_stats(class_name):
    """Update upload statistics when a new image is uploaded"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    upload_stats['total_uploads'] += 1
    upload_stats['uploads_by_class'][class_name] += 1
    upload_stats['uploads_by_date'][today] += 1
    upload_stats['last_upload'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Helper function to get upload statistics
def get_upload_statistics():
    """Get current upload statistics"""
    stats = {
        'total_uploads': upload_stats['total_uploads'],
        'uploads_by_class': dict(upload_stats['uploads_by_class']),
        'uploads_by_date': dict(upload_stats['uploads_by_date']),
        'last_upload': upload_stats['last_upload'],
        'total_classes': len(upload_stats['uploads_by_class']),
        'today_uploads': upload_stats['uploads_by_date'].get(datetime.now().strftime('%Y-%m-%d'), 0),
        'selected_class': selected_class
    }
    
    # Get top classes by upload count
    sorted_classes = sorted(
        upload_stats['uploads_by_class'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    stats['top_classes'] = sorted_classes[:5]  # Top 5 classes
    
    return stats

# Helper function to count existing files
def count_existing_uploads():
    """Count existing uploads from the file system"""
    total_count = 0
    class_counts = defaultdict(int)
    
    if os.path.exists(UPLOAD_FOLDER):
        for class_name in os.listdir(UPLOAD_FOLDER):
            class_path = os.path.join(UPLOAD_FOLDER, class_name)
            if os.path.isdir(class_path):
                # Count image files in this class directory
                image_count = 0
                for filename in os.listdir(class_path):
                    if allowed_file(filename):
                        image_count += 1
                
                class_counts[class_name] = image_count
                total_count += image_count
    
    return total_count, dict(class_counts)

# Initialize statistics from existing files
def initialize_statistics():
    """Initialize upload statistics from existing files"""
    global upload_stats
    
    total_count, class_counts = count_existing_uploads()
    
    upload_stats['total_uploads'] = total_count
    upload_stats['uploads_by_class'] = defaultdict(int, class_counts)
    
    # Set last upload to current time if there are existing files
    if total_count > 0:
        upload_stats['last_upload'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Initialized statistics: {total_count} total uploads across {len(class_counts)} classes")

# UDP Broadcasting function
def broadcast_server_info():
    """Broadcast server information via UDP"""
    global broadcast_running
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Set socket timeout
    sock.settimeout(1)
    
    server_ip = get_server_ip()
    server_info = {
        "type": "wood_id_server",
        "ip": server_ip,
        "port": 5000,
        "name": "Wood ID Upload Server",
        "version": "1.0"
    }
    
    message = json.dumps(server_info).encode('utf-8')
    
    print(f"Starting UDP broadcast on port {BROADCAST_PORT}")
    print(f"Server IP: {server_ip}")
    
    while broadcast_running:
        try:
            # Broadcast to all interfaces
            sock.sendto(message, ('<broadcast>', BROADCAST_PORT))
            print(f"Broadcasted server info: {server_info}")
            time.sleep(BROADCAST_INTERVAL)
        except Exception as e:
            print(f"Broadcast error: {e}")
            time.sleep(BROADCAST_INTERVAL)
    
    sock.close()
    print("UDP broadcast stopped")

# Start broadcasting
def start_broadcasting():
    """Start the UDP broadcasting in a separate thread"""
    global broadcast_running, broadcast_thread
    
    if not broadcast_running:
        broadcast_running = True
        broadcast_thread = threading.Thread(target=broadcast_server_info, daemon=True)
        broadcast_thread.start()
        print("UDP broadcasting started")

# Stop broadcasting
def stop_broadcasting():
    """Stop the UDP broadcasting"""
    global broadcast_running
    
    if broadcast_running:
        broadcast_running = False
        if broadcast_thread:
            broadcast_thread.join(timeout=2)
        print("UDP broadcasting stopped")

# HTML template for the homepage
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wood ID Image Upload</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .info-box {
            background-color: #e8f4fd;
            border: 1px solid #bee5eb;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .broadcast-status {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .selected-class-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            text-align: center;
        }
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 10px;
        }
        .stat-label {
            color: #6c757d;
            font-size: 1.1em;
        }
        .top-classes {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .class-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .class-item:last-child {
            border-bottom: none;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        select, input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        button:hover {
            background-color: #0056b3;
        }
        .status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 5px;
            display: none;
        }
        .success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .refresh-btn {
            background-color: #28a745;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            background-color: #218838;
        }
        .set-class-btn {
            background-color: #17a2b8;
            margin-top: 10px;
        }
        .set-class-btn:hover {
            background-color: #138496;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌳 Wood ID Image Upload</h1>
        
        <div class="info-box">
            <strong>Server Information:</strong><br>
            IP Address: {{ server_ip }}<br>
            Port: 5000<br>
            UDP Broadcast Port: 5001
        </div>
        
        <div class="broadcast-status">
            <strong>🔔 UDP Broadcasting Active</strong><br>
            This server is broadcasting its location for automatic discovery by mobile apps.
        </div>
        
        {% if stats.selected_class %}
        <div class="selected-class-box">
            <strong>🎯 Currently Selected Class:</strong> {{ stats.selected_class.replace('_', ' ').title() }}
        </div>
        {% else %}
        <div class="selected-class-box" style="background-color: #f8d7da; border-color: #f5c6cb;">
            <strong>⚠️ No Class Selected</strong><br>
            Please select a wood species before uploading images.
        </div>
        {% endif %}
        
        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Statistics</button>
        
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-number">{{ stats.total_uploads }}</div>
                <div class="stat-label">Total Uploads</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.total_classes }}</div>
                <div class="stat-label">Classes with Uploads</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.today_uploads }}</div>
                <div class="stat-label">Today's Uploads</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.uploads_by_class|length }}</div>
                <div class="stat-label">Available Classes</div>
            </div>
        </div>
        
        {% if stats.top_classes %}
        <div class="top-classes">
            <h3>📊 Top Classes by Upload Count</h3>
            {% for class_name, count in stats.top_classes %}
            <div class="class-item">
                <span>{{ class_name.replace('_', ' ').title() }}</span>
                <span><strong>{{ count }}</strong> uploads</span>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if stats.last_upload %}
        <div class="info-box">
            <strong>📅 Last Upload:</strong> {{ stats.last_upload }}
        </div>
        {% endif %}
        
        <form id="classSelectForm">
            <div class="form-group">
                <label for="class_select">Select Wood Species:</label>
                <select id="class_select" name="class_name" required>
                    <option value="">-- Select a wood species --</option>
                    {% for class_name in classes %}
                    <option value="{{ class_name }}" {% if stats.selected_class == class_name %}selected{% endif %}>{{ class_name.replace('_', ' ').title() }}</option>
                    {% endfor %}
                </select>
            </div>
            <button type="submit" class="set-class-btn">Set Selected Class</button>
        </form>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="image">Select Image File:</label>
                <input type="file" id="image" name="image" accept=".png,.jpg,.jpeg,.gif" required>
            </div>
            
            <button type="submit">Upload Image</button>
        </form>
        
        <div id="status" class="status"></div>
    </div>

    <script>
        // Handle class selection
        document.getElementById('classSelectForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const classSelect = document.getElementById('class_select');
            const selectedClass = classSelect.value;
            
            if (!selectedClass) {
                showStatus('Please select a wood species', 'error');
                return;
            }
            
            fetch('/set_class', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({class_name: selectedClass})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(`Class set to: ${selectedClass.replace('_', ' ').title()}`, 'success');
                    setTimeout(() => location.reload(), 1500);
                } else {
                    showStatus('Failed to set class: ' + data.error, 'error');
                }
            })
            .catch(error => {
                showStatus('Error setting class: ' + error.message, 'error');
            });
        });
        
        // Handle image upload
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('image');
            const statusDiv = document.getElementById('status');
            
            if (!fileInput.files[0]) {
                showStatus('Please select a file', 'error');
                return;
            }
            
            formData.append('image', fileInput.files[0]);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.text())
            .then(data => {
                if (data.includes('successfully')) {
                    showStatus('File uploaded successfully!', 'success');
                    document.getElementById('uploadForm').reset();
                    // Refresh the page to update statistics
                    setTimeout(() => location.reload(), 2000);
                } else {
                    showStatus('Upload failed: ' + data, 'error');
                }
            })
            .catch(error => {
                showStatus('Upload failed: ' + error.message, 'error');
            });
        });
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = 'status ' + type;
            statusDiv.style.display = 'block';
            
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }
    </script>
</body>
</html>
'''

# Route for the homepage
@app.route('/')
def index():
    classes = get_available_classes()
    server_ip = get_server_ip()
    stats = get_upload_statistics()
    return render_template_string(HTML_TEMPLATE, classes=classes, server_ip=server_ip, stats=stats)

# Route to set the selected class
@app.route('/set_class', methods=['POST'])
def set_class():
    """Set the currently selected class for uploads"""
    global selected_class
    
    try:
        data = request.get_json()
        class_name = data.get('class_name')
        
        if not class_name:
            return json.dumps({'success': False, 'error': 'No class name provided'})
        
        # Validate that the class exists
        available_classes = get_available_classes()
        if class_name not in available_classes:
            return json.dumps({'success': False, 'error': f'Invalid class: {class_name}'})
        
        selected_class = class_name
        print(f"Selected class set to: {class_name}")
        
        return json.dumps({'success': True, 'class_name': class_name})
    
    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)})

# Route to get the currently selected class
@app.route('/get_class')
def get_class():
    """Get the currently selected class"""
    return json.dumps({'selected_class': selected_class})

# Route to handle the image upload
@app.route('/upload', methods=['POST'])
def upload_file():
    global selected_class
    
    # Check if a class is selected
    if not selected_class:
        print("No class selected. Please select a wood species first.")
        return 'No class selected. Please select a wood species first.', 400
    
    # Check if the POST request has the 'image' part
    if 'image' not in request.files:
        print("No image part in the request")
        return 'No image part in the request', 400

    file = request.files['image']

    # If no file is selected
    if file.filename == '':
        print("No selected file")
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Create class-specific folder
        class_folder = os.path.join(app.config['UPLOAD_FOLDER'], selected_class)
        os.makedirs(class_folder, exist_ok=True)
        
        # Save the file to the class-specific folder
        file_path = os.path.join(class_folder, filename)
        file.save(file_path)
        
        # Update upload statistics
        update_upload_stats(selected_class)
        
        return f'File successfully uploaded to {selected_class} folder', 200
    else:
        return 'Invalid file type', 400

# Route to get statistics as JSON
@app.route('/stats')
def get_stats():
    """API endpoint to get upload statistics as JSON"""
    stats = get_upload_statistics()
    return json.dumps(stats, indent=2)

if __name__ == '__main__':
    # Initialize statistics from existing files
    initialize_statistics()
    
    # Start UDP broadcasting before running the Flask app
    start_broadcasting()
    
    try:
        print("Starting Flask server...")
        app.run(host='0.0.0.0')
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_broadcasting()
        print("Server stopped.")
