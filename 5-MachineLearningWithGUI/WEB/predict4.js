let mobilenet;
let video;

function modelReady() {
    console.log('Model Ready');
    mobilenet.predict(gotResults);
}

function gotResults(error, results ){
    if(error){
        console.log(error);
    }
    else{
        console.log(results);
        let label=results[0].className;
        let prob=results[0].probability;
        fill(0);
        textSize(640);
        text(label,10,height-100);
    }

}

function setup(){
    createCanvas(640,480);
    video = createCapture(VIDEO);
    video.hide();
    background(0);
    classifier = ml5.imageClassifier('MobileNet', video, modelReady);
}

function draw(){
    image(video,0,0);
}

