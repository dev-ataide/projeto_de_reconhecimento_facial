from flask import Flask, request, jsonify, render_template
import json
app = Flask(__name__)

log_file_path = "recognized_faces_log.txt"

@app.route('/recognized_faces', methods=['GET', 'POST'])
def save_recognized_faces():
    if request.method == 'POST':
        data = request.json
        log_data = json.dumps(data)
        with open(log_file_path, 'a') as log_file:
            log_file.write(log_data + '\n')
        return jsonify({"message": "Data saved successfully"})
    elif request.method == 'GET':
        with open(log_file_path, 'r') as log_file:
            logs = log_file.readlines()
        return render_template('logs.html', logs=logs)

if __name__ == '__main__':
    app.run(debug=True)
