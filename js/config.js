export const models = {
    'lin_0725': {
        path: './model/lin_0725.onnx',
        inputShape: [1, 3, 640, 640],
        outputShape: [1, 5, 8400] // [batch, 4_coords + 1_class, 8400_proposals]
    },
    'cheng': {
        path: './model/cheng.onnx',
        inputShape: [1, 3, 640, 640],
        outputShape: [1, 5, 8400] // [batch, 4_coords + 1_class, 8400_proposals]
     },
     // 'lai': {
     //    path: './model/lai.onnx',
     //    inputShape: [1, 3, 640, 640],
     //    outputShape: [1, 300, 5] // [batch, 300_proposals, 4_coords + 1_class]
     // }
 };