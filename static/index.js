const http = new XMLHttpRequest(); 

var canvas = document.querySelector("#input");
var ctx = canvas.getContext("2d");
var mousedown = false; 

const submitButton = document.querySelector(".submit")

// Set the drawing color to black
ctx.strokestyle = "black"; 
ctx.lineWidth = 6; 
ctx.lineJoin= "round"; 


function draw_stroke(e){
  if(!mousedown){
    return 
  }

  var x = e.clientX - canvas.offsetLeft; 
  var y = e.clientY - canvas.offsetTop; 

  ctx.lineTo(x, y); 
  ctx.stroke(); 
  ctx.beginPath(); 
  ctx.moveTo(x,y); 

}

function submit(){

  const image = canvas.toDataURL();
  // use this dataurl to send the file to the server, 
  
  let url = "/predict"

  http.open("POST", url) 

  http.send(image) 

  http.onload = function(){
    let response_obj = http.response; 

    let predictionElement = document.querySelector("#class-pred"); 
    predictionElement.textContent = `predicted class from the model: ${response_obj}`; 
  }

}

canvas.addEventListener("mousedown", ()=>{mousedown = true}); 
canvas.addEventListener("mousemove", draw_stroke); 
canvas.addEventListener("mouseup", ()=>{ mousedown = false; ctx.beginPath()}); 

submitButton.addEventListener("click", submit); 