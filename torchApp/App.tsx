import React from 'react';
import {useCallback, useRef, useState} from 'react';
import {View, Text, Image} from 'react-native';
import {Camera, ImageUtil} from 'react-native-pytorch-core';
import useImageClassification from './predictor';

const imageClasses = require('./assets/ImageNetClasses.json');

export default function CameraExample() {
  const cameraRef = useRef(null);
  const [imgClass, setImgClass] = useState('테스트');
  const {noseProb, processImage} = useImageClassification(imageClasses);
  const [imageURL, setImageURL] = useState();

  async function handleCapture(image) {
    const filePath = await ImageUtil.toFile(image);
    console.log(filePath);
    await setImageURL(filePath);
    image.release();
  }

  const handleFrame = useCallback(
    async (image: Image) => {
      await processImage(image);
      console.log('강아지 코일 확률:', noseProb);
      setImgClass(noseProb);
      image.release();
    },
    [processImage, noseProb],
  );

  return (
    <View style={{flex: 1, justifyContent: 'center'}}>
      <Camera
        ref={cameraRef}
        hideFlipButton={true}
        onFrame={handleFrame}
        style={{height: '50%'}}
        onCapture={handleCapture}
      />
      <Text style={{position: 'absolute', top: 100, left: 100}}>
        강아지 코 확률: {imgClass}
      </Text>
      {imageURL && (
        <Image
          style={{
            position: 'absolute',
            top: 200,
            right: 200,
            width: 100,
            height: 100,
          }}
          source={{uri: imageURL}}
        />
      )}
    </View>
  );
}
