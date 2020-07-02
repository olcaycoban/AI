let puffin;

function imageReady(){
    image(puffin,0,0,width,height);
}

function setup(){
    createCanvas(640,480);
    puffin=createImg('images/puffin.jpg',imageReady);
    puffin.hide();
    background(0);
}