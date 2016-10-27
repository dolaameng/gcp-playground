var startChat = function(token) {
  var channel = new goog.appengine.Channel(token);
  var socket = channel.open();
  socket.onopen = function() {
    console.log("channel open");
  };
  socket.onclose = function() {
    console.log("channel closed");
  };
  socket.onerror = function(err) {
    console.log(err);
  };
  socket.onmessage = function(msg) {
    var data = JSON.parse(msg.data);
    if (data.type === "message"){
      $("<p>" + data.who + ":" + data.what + "</p>").appendTo("#chatlog");
    } else if (data.type === "connect") {
      $("<p>" + data.who + " is now online" + "</p>").appendTo("#chatlog");
    } else if (data.type === "disconnect") {
      $("<p>" + data.who + " is now offline" + "</p>").appendTo("#chatlog");
    }

  };
};

var init = function(token) {
  console.log("init");
  $("#send").on("click", function(evnt) {
    var msg = $("#message").val();
    $.ajax({
      type: "post",
      url: "/send_message",
      data: {'message': msg},
      success: function(data) {
        console.log(data);
      }
    });
  });
  
  startChat(token);
};