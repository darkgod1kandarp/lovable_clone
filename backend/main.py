# Creating server that can show rh 
from flask import Flask, render_template, request  
from flask_cors import CORS
from agent.graph import run_agent
import requests

# Create a Flask app  
app = Flask(__name__)  
# Define a route for the home page  

# ADD CORS support to allow cross-origin requests  
CORS(app)    

# Create Testing API  
@app.route('/api/test')  
def test_api():  
    return {'message': 'Hello, this is a test API!'}

@app.route('/api/prompt', methods=['POST'])
def prompt_api():
    data = request.get_json()
    prompt = data.get('prompt', '')
    use_ollama = False
    # Process the prompt here (e.g., call a model or perform some action)
    result = run_agent(prompt, use_ollama)
    return {'result': result}

# Create a server    
if __name__ == '__main__':  
    app.run(debug=True, host='0.0.0.0', port=5000)
    
 
