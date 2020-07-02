let classifier;

let img;

function preload() {
  classifier = ml5.imageClassifier('MobileNet');
  img = loadImage('https://img3.goodfon.com/original/2560x1600/1/2f/ozero-gory-lesa-derevya.jpg');
}
//https://img3.goodfon.com/original/2560x1600/1/2f/ozero-gory-lesa-derevya.jpg
function setup() {
  createCanvas(400, 400);
  classifier.classify(img, gotResult);
  image(img, 0, 0);
}

function gotResult(error, results) {

  if (error) {
    console.error(error);
  }

  console.log(results);
  createDiv("Label:" + results[0].label);
  createDiv("Confidence: " + nf(results[0].confidence, 0, 2));
}