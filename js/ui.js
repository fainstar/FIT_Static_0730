import { getColorForScore } from './utils.js';

export function drawBoundingBoxes(ctx, boxes, imageCanvas, scale) {
    const baseLineWidth = 2;
    const baseFontSize = 16;
    const uiScale = Math.min(imageCanvas.width, imageCanvas.height) / 640; // Scale for UI elements

    boxes.forEach(({ classId, score, box }) => {
        // Scale box coordinates from original image size to canvas size
        const [x, y, width, height] = box.map(coord => coord * scale);
        const color = getColorForScore(score);

        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(1, baseLineWidth * uiScale);
        ctx.strokeRect(x, y, width, height);

        ctx.fillStyle = color;
        const fontSize = Math.max(10, baseFontSize * uiScale);
        ctx.font = `${fontSize}px sans-serif`;
        const label = `${score.toFixed(2)}`;
        ctx.fillText(label, x, y > fontSize ? y - 5 : fontSize);
    });
}