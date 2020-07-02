let classifier; let video; let resultsP;
function setup() {
  noCanvas();
  video = createCapture(VIDEO);
  classifier = ml5.imageClassifier('MobileNet', video, modelReady);
  resultsP = createP('Loading model and video...');
}

function modelReady() {
  console.log('Model Ready');
  classifyVideo();
}

function classifyVideo() {
  classifier.classify(gotResult);
}

function gotResult(err, results) {
  resultsP.html(results[0].label + ' ' + nf(results[0].confidence, 0, 2));
  classifyVideo();
}