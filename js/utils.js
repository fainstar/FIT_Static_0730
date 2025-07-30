export function nonMaxSuppression(boxes, scoreThreshold, iouThreshold) {
    boxes.sort((a, b) => b.score - a.score);
    const result = [];
    while (boxes.length > 0) {
        result.push(boxes[0]);
        boxes = boxes.filter(box => {
            if (boxes.length === 1) return false; // Keep the last box
            const iou = intersectionOverUnion(boxes[0], box);
            return iou < iouThreshold;
        });
    }
    return result;
}

export function intersectionOverUnion(boxA, boxB) {
    const [xA, yA, widthA, heightA] = boxA.box;
    const [xB, yB, widthB, heightB] = boxB.box;

    const x1 = Math.max(xA, xB);
    const y1 = Math.max(yA, yB);
    const x2 = Math.min(xA + widthA, xB + widthB);
    const y2 = Math.min(yA + heightA, yB + heightB);

    const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const boxAArea = widthA * heightA;
    const boxBArea = widthB * heightB;

    return intersectionArea / (boxAArea + boxBArea - intersectionArea);
}

export function getColorForScore(score) {
    if (score > 0.85) {
        return '#00FF00'; // Green
    } else if (score > 0.6) {
        return '#FFFF00'; // Yellow
    } else {
        return '#FF0000'; // Red
    }
}