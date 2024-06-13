let cropper = "";
let model;
let cropperData;
const previewImage = document.getElementById("preview-image");
const test = document.getElementById("test");
const webcam = new Webcam(document.getElementById('wc'));

async function init(){
  await webcam.setup("device-list");
  model = await tf.loadLayersModel('tf_model/model.json');
  console.log(model.summary());

}

async function captureNow(){
  console.log("pressed");
  const dataURL = await webcam.captureWithoutTensorFlow();
  previewImage.src = dataURL;
  if(cropper){
    cropper.destroy();
  }
  cropper = new Cropper(previewImage);

}
// - id (unique)
// - left
// - top
// - width
// - height
// - column
// - row

async function cropCapture(){
  if(cropper){
    let data = cropper.getCropBoxData();//JSON
    console.log(data);
    const column = document.getElementById("column_input");
    const row = document.getElementById("row_input");

    //kirim variable data, row, column ke database

  }
}

function convertToTf(imgSrc){
  return tf.tidy(() => {
    const webcamImage = tf.browser.fromPixels(imgSrc);
    const reversedImage = webcamImage.reverse(1);
    const croppedImage = webcam.cropImage(reversedImage);
    const batchedImage = croppedImage.expandDims(0);
    const expandedImg = batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    return tf.image.resizeBilinear(expandedImg,[150, 150]);
  });
}

function sendData(){
  console.log("Pressed Send Data")
}

function cropReset(){
  if(cropper){
    cropper.reset();
  }
}

async function predict(img){
  let processedImg = convertToTf(img);
  let predictions = await model.predict(processedImg);
  // const myJSON = JSON.stringify(predictions.argMax());
  const classId = (await predictions.data())[0];
  if(classId >= 10e-20){
    return 0;
  }else {
    return 1;
  }
}

function predictAll(){
  // cropperData = //Ambil semua data dari database;
  for (let i = 0; i < cropperData.length; i++) {
    cropper.setCropBoxData(cropperData[i]);
    let img = cropper.getCroppedCanvas({});
    let classId = predict(img);
    //update data di database

  }
}

init();
