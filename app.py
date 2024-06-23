from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_cors import cross_origin
from llm import GSpaceBot_3

app = Flask(__name__)
cors = CORS(app, resources={r"/process": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

start = False
model = None

@app.route('/api', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def predict():
    global start, model
    data = request.get_json()
    chat = data['messages']
    print(chat)
    

    if not start:
        start = True
        model = GSpaceBot_3()        
        response = model.get_answer(chat)
        return jsonify({'response': response})
    response = model.get_answer(chat)
    return jsonify({'response': response})
    
    
    return jsonify({'error': 'File upload failed'}), 500


@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')
if __name__ == '__main__':
    app.run(debug=True)


    