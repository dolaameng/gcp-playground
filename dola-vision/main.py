from flask import Flask, request, make_response, redirect, render_template
import base64
import io
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import json

credentials = GoogleCredentials.get_application_default()
service = discovery.build('vision', 'v1', credentials=credentials)

app = Flask(__name__)
app.config['debug'] = True

@app.route('/')
def index():
  return render_template('index.html')
  
@app.route('/detect_label', methods=['POST'])
def detect_label():
  image64 = request.form['image64']
  #image = base64.b64decode(image64)
  #im = Image.open(io.BytesIO(image))
  service_request = service.images().annotate(body={
            'requests': [{
                'image': {
                    'content': image64.decode('UTF-8')
                },
                'features': [{
                    'type': 'LABEL_DETECTION',
                    'maxResults': 1
                }]
            }]
        })
  response = service_request.execute()
  try:
    label = response['responses'][0]['labelAnnotations'][0]['description']
    result = {'label': label}
  except:
    result = {'label': 'unknown'}
  result_response = make_response(json.dumps(result))
  result_response.headers['Content-Type'] = "application/json"
  return result_response
  
  
@app.route('/detect_landmark', methods=['POST'])
def detect_landmark():
  image64 = request.form['image64']
  service_request = service.images().annotate(body={
            'requests': [{
                'image': {
                    'content': image64.decode('UTF-8')
                },
                'features': [{
                    'type': 'LANDMARK_DETECTION',
                    'maxResults': 1
                }]
            }]
        })
  response = service_request.execute()
  try:
    label = response['responses'][0]['landmarkAnnotations'][0]['description']
    result = {'landmark': label}
  except:
    result = {'landmark': 'unknown'}
  result_response = make_response(json.dumps(result))
  result_response.headers['Content-Type'] = "application/json"
  return result_response
  
@app.route('/detect_ocr', methods=['POST'])
def detect_ocr():
  image64 = request.form['image64']
  service_request = service.images().annotate(body={
            'requests': [{
                'image': {
                    'content': image64.decode('UTF-8')
                },
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': 1
                }]
            }]
        })
  response = service_request.execute()
  try:
    label = response['responses'][0]['textAnnotations'][0]['description']
    result = {'ocr': label}
  except:
    result = {'ocr': 'unknown'}
  result_response = make_response(json.dumps(result))
  result_response.headers['Content-Type'] = "application/json"
  return result_response
