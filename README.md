# Training an AI Model with PyTorch and Deploying It to a Mobile App

This project is based on the [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd.git). You can find more details on [my blog](https://froggydisk.github.io/ai-on-app/).

We are applying a dog nose recoginition model to a React Native App by using Object Detection.

<div align="center">
  <img src="/assets/demo.png" width="50%">
</div>

## Process

1. Data Labeling
2. Data Preprocessing
3. Model Training
4. Model Test
5. Model Transformation
6. Model Application

### Data Labeling

The raw dataset is [Stanfod Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). It contains 20,580 images of 120 different breeds of dogs.

I randomly selected a subset of images from this dataset using stratified sampling and created a new dataset, called `dognose_dataset`. You can find the datasets in `data` folder.

I used a labeling tool called [labelImg](https://github.com/HumanSignal/labelImg), which is a simple tool for annotating images.

This tool is quite easy to use, but there can be some challenges when installing it.

Here is a tip:

```bash
brew install qt qt@5
brew install libxml2
brew install pyqt@5
pip3 install pyqt5 lxml
git clone https://github.com/HumanSignal/labelImg.git
cd labelImg
make qt5py3
python3 labelImg.py
```

Don't forget to install `qt@5`. The official page does not mension this package, but it is essential.

You can see how to use it [here](https://github.com/HumanSignal/labelImg).

![labelImg](/assets/labelImg.png)

### Data Preprocessing

We need to structure the data so it can be ingested by the ML/DL model.

I added some code to organize the data.

Put all the files `image.jpg + label.xml` in one folder and run the command below:

```bash
python3 xml_to_csv.py
```

It will generate files as shown below:

```bash
🗂️ label
sub-test-annotations-bbox.csv
sub-train-annotations-bbox.csv
sub-validation-annotations-bbox.csv
🗂️ test
🗂️ train
🗂️ validation
```

### Model Training

Before training, we need to prepare two things:

- A pre-trained model
- A development environment

`torch-ssd` provides a pre-trained model trained on the `PASCAL VOC` dataset.

Building your own environment for ML/DL is possible but time-consuming. It is better to use a Docker container designed for data science.

If you have a GPU, you can save a lot of time. Decide whether to use a GPU or CPU first.

- GPU

```bash
# Assuming nvidia-docker2 is installed
docker pull ufoym/deepo # image for GPU
docker run -it --gpus all --shm-size 8G -v "$(pwd)":/mount ufoym/deepo bash
```

- CPU

```bash
docker pull ufoym/deepo:cpu # image for CPU
docker run -it --shm-size 8G -v "$(pwd)":/mount ufoym/deepo:cpu bash
```

You need to install additional packages in the container:

```bash
pip install opencv-python
apt-get update
apt-get install libgl1-mesa-glx
```

Finally, let's train the model:

```bash
python train_ssd.py --dataset_type open_images --datasets /mount/data/dognose_dataset --net mb2-ssd-lite --pretrained_ssd models/mb2-ssd-lite-mp-0_686.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 10 --num_epochs 100 --base_net_lr 0.001 --batch_size 8 --debug_steps 10
```

Repeat this process, adjusting hyperparameters to find the best model.

### Model Test

Ensure you are in the right development environment.

- Test on a single image:

```bash
python3 run_ssd_example.py mb2-ssd-lite models/mb2-ssd-lite-Epoch-99-Loss-2.2699455499649046.pth models/open-images-model-labels.txt dog.jpg
```

The result file `open-images-model-labels.txt` will be created. Check if the model recognizes the object well.

<div align="center">
  <img src="/assets/dog-nose.jpg" width="50%">
</div>

- Test on the test dataset:

```bash
python eval_ssd.py --dataset_type open_images --net mb2-ssd-lite --dataset /mount/data/dognose_dataset --trained_model models/mb2-ssd-lite-Epoch-99-Loss-2.2699455499649046.pth --label_file models/open-images-model-labels.txt
```

The result looks like this:

```bash
Average Precision Per-class:
DogNose: 0.8976893924291722
Average Precision Across All Classes:0.8976893924291722
```

- Test in real-time:

```bash
pip3 install torch torchvision opencv-python
python3 run_ssd_live_demo.py mb2-ssd-lite models/mb2-ssd-lite-Epoch-99-Loss-2.2699455499649046.pth models/open-images-model-labels.txt
```

Try showing it a photo of a real dog and see the result.

### Model Transformation

We need to transform the model from `.pth` to `.ptl` for deploying it to a mobile app.

```bash
python3 export_model.py
```

### Model Application

You can check a sample app in the torchApp folder. Just run it.

If you want to deploy the model directly to your app, here is what you need:

```bash
App.js
predictor.js
metro.config.js
mb2-ssd-lite-Epoch-99-Loss-2.2699455499649046.ptl
ImageNetClasses.json
```

Use the components in `App.js` and `predictor.js`, and modify the settings in `metro.config.js` to read the `.ptl` file.

Don't forget to restart npm with the following command:

```bash
npm start --reset-cache
```

You must add `Privacy - Camera Usage Description` to `info.plist` when testing on iOS.

## Conclusion

Everything is set. It seems working well.

<div align="center">
  <img src="/assets/perf-test.png" width="50%">
</div>

There might be performance issues due to the limited hardware capabilities of mobile devices.

You may also experience thermal issues when you use camera with deep learning models.
