'''
Examples adopted from https://googlecloudplatform.github.io/google-cloud-python/stable/vision-usage.html
'''


from google.cloud import vision
from google.cloud.vision.image import Feature, FeatureTypes
import io

# should just work if application has been authorized
# otherwise pass credential explicitly
client = vision.Client()

# as bytes
image_content = io.open('images/cat.jpg', 'rb').read()
image_cat = client.image(content=image_content)
image_face = client.image(source_uri='gs://dolva-vision/face.jpg')

# label detection
labels = image_cat.detect_labels(limit=3)
for label in labels:
  print label.description
#
### FACE DETECTION DOES NOT SEEM TO WORK WITH THE IMAGE I CHOSE  
## face detection
#try:
#  faces = image_face.detect_faces(limit=5)
#  print faces[0]
#except:
#  pass
#  
### LANDMARK DETECTION DOES NOT SEEM TO WORK WITH THE IMAGE I CHOSE 
## landmark detection
##image_marinabay = client.image(source_uri='https://indesemen.files.wordpress.com/2013/09/marina-bay-sands.jpg')
#try:
#  image_marinabay = client.image(content=io.open("images/marina-bay-sands.jpg", "rb").read())
#  landmarks = image_marinabay.detect_landmarks()
#  print landmarks[0]
#except:
#  pass
#  
## OCR
#image_ocr = client.image(source_uri="http://www.cambridgeassessment.org.uk/Images/ocr.jpg")
#ocr_texts = image_ocr.detect_text()
#print ocr_texts[0].description