from flask import Flask, request, make_response, render_template, redirect

from google.appengine.api import channel, users

from datetime import datetime
import json

CHANNEL_TIMEOUT = 60

app = Flask(__name__)
app.config['debug'] = True

## TODO: update channel_clients on open and close events
app.channel_clients = set()

def send_to_all(message):
  for client in app.channel_clients:
    channel.send_message(client, json.dumps(message))

@app.route('/')
def index():
  user = users.get_current_user()
  if not user:
    login_url = users.create_login_url(request.url)
    return redirect(login_url)
  user_name = user.nickname()
  logout_url = users.create_logout_url(request.url)
  
  channel_id = user.user_id()
  token = channel.create_channel(channel_id, CHANNEL_TIMEOUT)
  app.channel_clients.add(channel_id)
  send_to_all({
    'type': 'connect',
    'who': user.nickname()
  })
#  print app.channel_clients
  
  return render_template('chat.html', user_name=user_name, logout_url=logout_url, token=token)
  
@app.route('/send_message', methods=["POST"])
def send_message():
  user = users.get_current_user()
  if not user:
    return "user not logged in", 404
  user_id = user.user_id()
  message = request.form['message']
  channel_message = {
      'type': 'message',
      'who': user.nickname(),
      'what': message
  }
  send_to_all(channel_message)
  return "message distributed"

@app.route('/_ah/channel/connected/', methods=['POST'])
def client_connected():
#  client = request.form['from']
#  print "debug:   ", request.form
#  print "client %s connected" % client
#  app.channel_clients.add(client)
#  send_to_all({
#    'type': 'connect',
#    'who': users.User(_user_id=client).nickname()
#  })
  return "noted"

@app.route('/_ah/channel/disconnected/', methods=['POST'])
def client_disconnected():
#  client = request.form['from']
#  print "client %s disconnected" % client
#  if client in app.channel_clients:
#    app.channel_clients.remove(client)
#  send_to_all({
#    'type': 'disconnect',
#    'who': users.User(_user_id=client).nickname()
#  })
  return "noted"