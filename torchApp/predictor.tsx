import {useCallback, useState} from 'react';
import {
  Image,
  media,
  MobileModel,
  Module,
  Tensor,
  torch,
  torchvision,
} from 'react-native-pytorch-core';

const T = torchvision.transforms;
let model: any = null;

const packFn = async (image: Image): Promise<Tensor> => {
  const width = image.getWidth();
  const height = image.getHeight();

  const blob = media.toBlob(image);

  let tensor = torch.fromBlob(blob, [height, width, 3]);
  tensor = tensor.permute([2, 0, 1]);
  tensor = tensor.div(255);
  const centerCrop = T.centerCrop(Math.min(width, height));
  tensor = centerCrop(tensor);

  const resize = T.resize(300);
  tensor = resize(tensor);

  const normalize = T.normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
  tensor = normalize(tensor);

  return tensor.unsqueeze(0);
};

const inferenceFn = async (model: Module, tensor: Tensor): Promise<Tensor> => {
  return await model.forward(tensor);
};

const unpackFn = async (scores: Tensor): Promise<number> => {
  // Get the index of the value with the highest probability
  scores = scores[0];

  let probs = [];
  for (let i = 0; i < scores.shape[0]; i++) {
    probs.push(parseFloat(scores[i][1]));
  }
  let maxProb = Math.max(...probs);

  return maxProb;
};

export default function useImageModelInference(imageClasses: {
  [key: string]: string;
}) {
  const [noseProb, setNoseProb] = useState();

  const processImage = useCallback(async (image: Image) => {
    if (model === null) {
      console.log('loading...');
      const filePath = await MobileModel.download(
        require('./assets/mb2-ssd-lite-Epoch-99-Loss-2.27.ptl'),
      );
      model = await torch.jit._loadForMobile(filePath);
      console.log('finished');
    }
    const inputs = await packFn(image);
    const [scores] = await inferenceFn(model, inputs);
    const maxProb = await unpackFn(scores);
    // Resolve the most likely class label and return it

    setNoseProb(maxProb);
  }, []);

  return {
    noseProb,
    processImage,
  };
}
