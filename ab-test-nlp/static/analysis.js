var analyzeElem = $("#analyze");
var questionElem = $("#question");
var answerText = $("#answer-text");
var answerPlot = $("#answer-plot");

analyzeElem.click(function(){
    var question = questionElem.val().trim();
    if (question) {
        console.log("Question:", question);
        $.post("/ask", 
        {
            "question": question
        }, 
        function(answer){
            console.log("Answer:", answer);
            answerText.html(answer["text"]);
            answerPlot.html(answer["plot"]);
            $("body").append(answer["script"]);
        });
    }
});

$('body').keypress(function(event){
  if(event.keyCode == 13){
    $('#analyze').click();
  }
});