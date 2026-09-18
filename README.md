# NightSight
Low light street enhancement.

This primary purpose of the project is to perform illumination by transforming the images instead of image generation (RetinexNet). Some aspects of CycleGAN were explored, but are abandoned. It uses the open-source code by the authors of the [RetinexNet model](https://github.com/weichen582/RetinexNet). **However, their code lacks the implementation for the latest tensorflow and keras versions (supported for tensorflow 1.x). This repository allows the execution in latest tensorflow and keras version (tf 2.16.0 and keras 3.3.3 tested).**

Refer to the notebooks for the model trainings. The RetinexNet training notebook is located [locally](Model/RetinexNet/RetinexNet-LOLDataset.ipynb) and at [kaggle](https://www.kaggle.com/code/samyak03/retinexnet).

The RetinexNet models are fully trained and model data included.\
The CycleGANs are abandoned midway in training due to higher training time and limited GPU time from Kaggle. 

The codes are fully functional and last tested on tensorflow 2.16.0 and keras 3.3.3.

## Model Training
The snippets of the training epoch-wise on a sample image and how the model improves. Half of the training (epoch 01-50) trains `Illumination_Low` parameter and the rest (epoch 51-100) trains `Illumination_Delta` parameter of the model.

### Epoch 1
![Epoch 1 training](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___16_2.png)

### Epoch 50
![Epoch 50 training](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___16_100.png)

### Epoch 51
![Epoch 51 training](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___16_102.png)

### Epoch 75
![Epoch 75 training](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___16_150.png)

### Epoch 100
![Epoch 100 training](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___16_200.png)

## Results
- Input - The input image that is fed to the model.
- Ground Truth - The model output should match closely to it.
- Sample - Model parameter output that encodes color information to the result.
- Illumination - Model parameter output that encodes resultant illumination to the input.
- Result - The combination of Sample and Illumination with the Input. This is the final output image.

![Inference 1](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_4.png)

![Inference 2](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_6.png)

![Inference 3](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_8.png)

![Inference 4](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_10.png)

![Inference 5](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_12.png)

![Inference 6](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_14.png)

![Inference 7](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_16.png)

![Inference 8](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_18.png)

![Inference 9](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_20.png)

![Inference 10](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_24.png)

![Inference 11](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_26.png)

![Inference 13](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_28.png)

![Inference 14](https://raw.githubusercontent.com/SAM-DEV007/NightSight/refs/heads/main/Model/RetinexNet/results/__results___files/__results___19_30.png)

