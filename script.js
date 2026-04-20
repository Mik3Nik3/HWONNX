const X_MIN = 1978.0;
const X_MAX = 2021.0;
const Y_MIN = 193.44;
const Y_MAX = 1798.61;

async function predict() {
    const year = parseFloat(document.getElementById('yearInput').value);
    
    // 1. Scale Input: (x - min) / (max - min)
    const scaledInput = (year - X_MIN) / (X_MAX - X_MIN);
    
    try {
        // 2. Load and Run Model
        const session = await ort.InferenceSession.create('./gold_rate_model.onnx');
        const inputTensor = new ort.Tensor('float32', [scaledInput], [1, 1]);
        const results = await session.run({ input: inputTensor });
        const scaledOutput = results.output.data[0];

        // 3. Unscale Output: y * (max - min) + min
        const prediction = scaledOutput * (Y_MAX - Y_MIN) + Y_MIN;

        document.getElementById('result').innerText = 
            `Predicted USD Gold Rate: $${prediction.toFixed(2)}`;
    } catch (e) {
        console.error(e);
        document.getElementById('result').innerText = "Error loading model.";
    }
}
