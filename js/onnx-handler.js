import { nonMaxSuppression } from './utils.js';

export function preprocess(img, modelInputShape, preProcessInfo) {
    const [modelWidth, modelHeight] = modelInputShape.slice(2);

    // Scaling factor
    const ratio = Math.min(modelWidth / img.width, modelHeight / img.height);
    const newWidth = Math.round(img.width * ratio);
    const newHeight = Math.round(img.height * ratio);

    // Padding
    const padX = (modelWidth - newWidth) / 2;
    const padY = (modelHeight - newHeight) / 2;

    preProcessInfo.ratio = ratio;
    preProcessInfo.padX = padX;
    preProcessInfo.padY = padY;

    const canvas = document.createElement('canvas');
    canvas.width = modelWidth;
    canvas.height = modelHeight;
    const context = canvas.getContext('2d');
    context.fillStyle = '#000000'; // Fill with black padding
    context.fillRect(0, 0, modelWidth, modelHeight);
    context.drawImage(img, padX, padY, newWidth, newHeight);
    const imageData = context.getImageData(0, 0, modelWidth, modelHeight);
    const { data } = imageData;

    const red = [], green = [], blue = [];
    for (let i = 0; i < data.length; i += 4) {
        red.push(data[i] / 255.0);
        green.push(data[i + 1] / 255.0);
        blue.push(data[i + 2] / 255.0);
    }
    const float32Data = new Float32Array([...red, ...green, ...blue]);

    return new ort.Tensor('float32', float32Data, modelInputShape);
}

export function postprocess(tensor, preProcessInfo, outputShape, iouThreshold) {
    const data = tensor.data;
    const { ratio, padX, padY } = preProcessInfo;

    // Handle different output shapes
    if (outputShape[1] === 5 && outputShape[2] === 8400) { // lin_0725 and cheng
        const numClasses = outputShape[1] - 4;
        const numProposals = outputShape[2];

        const boxes = [];
        for (let i = 0; i < numProposals; i++) {
            const classScores = [];
            for (let j = 0; j < numClasses; j++) {
                classScores.push(data[(4 + j) * numProposals + i]);
            }
            const maxScore = Math.max(...classScores);
            const classId = classScores.indexOf(maxScore);

            if (maxScore > 0.5) { // Confidence threshold
                const x_center = data[0 * numProposals + i];
                const y_center = data[1 * numProposals + i];
                const w = data[2 * numProposals + i];
                const h = data[3 * numProposals + i];

                const x1 = (x_center - w / 2 - padX) / ratio;
                const y1 = (y_center - h / 2 - padY) / ratio;
                const x2 = (x_center + w / 2 - padX) / ratio;
                const y2 = (y_center + h / 2 - padY) / ratio;

                boxes.push({
                    classId: classId,
                    score: maxScore,
                    box: [x1, y1, x2 - x1, y2 - y1]
                });
            }
        }
        return nonMaxSuppression(boxes, 0.5, iouThreshold);
    } else if (outputShape[1] === 300 && outputShape[2] === 5) { // lai model (rt-detr)
        const numProposals = outputShape[1];

        const boxes = [];
        for (let i = 0; i < numProposals; i++) {
            const score = data[i * 5 + 4];

            if (score > 0.5) {
                const x1_raw = data[i * 5 + 0];
                const y1_raw = data[i * 5 + 1];
                const x2_raw = data[i * 5 + 2];
                const y2_raw = data[i * 5 + 3];

                const x1 = (x1_raw - padX) / ratio;
                const y1 = (y1_raw - padY) / ratio;
                const x2 = (x2_raw - padX) / ratio;
                const y2 = (y2_raw - padY) / ratio;

                boxes.push({
                    classId: 0, // Assuming single class for this model
                    score: score,
                    box: [x1, y1, x2 - x1, y2 - y1]
                });
            }
        }
        return nonMaxSuppression(boxes, 0.5, iouThreshold);
    } else {
        console.error("Unsupported model output shape:", outputShape);
        return [];
    }
}