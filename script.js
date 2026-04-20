const X_MIN = 1978.0;
const X_MAX = 2021.0;
const Y_MIN = 193.44;
const Y_MAX = 1798.61;

async function predict() {
    const year = parseFloat(document.getElementById('yearInput').value);
    const resultElement = document.getElementById('result');
    resultElement.innerText = "Loading model... please wait.";

    try {
        // Log to console so we can track progress
        console.log("Attempting to load model...");
        
        const session = await ort.InferenceSession.create('./gold_rate_model.onnx');
        
        console.log("Model loaded successfully!");

        const scaledInput = (year - X_MIN) / (X_MAX - X_MIN);
        const inputTensor = new ort.Tensor('float32', new Float32Array([scaledInput]), [1, 1]);
        
        const results = await session.run({ input: inputTensor });
        const scaledOutput = results.output.data[0];
        const prediction = scaledOutput * (Y_MAX - Y_MIN) + Y_MIN;

        resultElement.innerText = `Predicted USD Gold Rate: $${prediction.toFixed(2)}`;
    } catch (e) {
        console.error("Inference Error:", e);
        resultElement.innerText = `Error: ${e.message}`;
    }
}
