from flask import Flask, request, make_response, redirect, render_template
import base64
import io
import json
from PIL import Image
from time import time
from detector import *


app = Flask(__name__)
app.config['debug'] = True

@app.route('/')
def index():
  return render_template('index.html')
  
@app.route('/detect_label', methods=['POST'])
def detect_label():
  image64 = request.form['image64']
  image = base64.b64decode(image64)
  im = Image.open(io.BytesIO(image))
  img_path = "test_images/"+str(time())+".png" 
  im.save( img_path )
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  try:
    response = sign_classifier.predict(img).sort_values("probs", ascending=False).head(5).to_html()
    label = response
    result = {'label': label}
  except:
    result = {'label': 'unknown'}
  result_response = make_response(json.dumps(result))
  result_response.headers['Content-Type'] = "application/json"
  return result_response

# from OpenSSL import SSL
# context = SSL.Context(SSL.SSLv23_METHOD)
# context.use_privatekey_file('yourserver.key')
# context.use_certificate_file('yourserver.crt')

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8080)
