 # Land Cover Classification — Machine Learning & Deep Learning

 An end-to-end **machine-learning and deep-learning pipeline for land-cover classification** using Sentinel-2 satellite imagery and the EuroSAT 10-class dataset.

 This repository contains my individual work from the **ROSPIN Summer School Land Cover Classification project**, with a focus on understanding machine-learning models, training and evaluating classification models, and developing a high-performance model using multispectral Sentinel-2 imagery.

 The original collaborative project is available here:

 **Original project:**\
 https://github.com/DariusSasarman/Land-cover-classification-ROSPIN-Summer-School

---

 # 1\. My Contribution

 During the project, I focused on the **machine-learning and model-development aspects** of the land-cover classification pipeline.

 My work included understanding how the existing models were structured, how satellite imagery was processed, how the training pipeline worked, and how the models were evaluated.

 ### Main contributions

 - Studied and understood the existing machine-learning and deep-learning models.
- Investigated the training and evaluation pipeline.
- Worked with Sentinel-2 satellite imagery and the EuroSAT dataset.
- Learned how RGB and multispectral satellite data can be used for classification.
- Worked with both traditional machine-learning models and deep-learning models.
- Trained and evaluated classification models.
- Worked with a **13-band multispectral ResNet-18 model** using TorchGeo.
- Trained the high-accuracy model that achieved approximately **99% accuracy and 0.99 macro F1** on the benchmark evaluation dataset.
- Analyzed classification reports and confusion matrices.
- Investigated the differences between RGB and multispectral approaches.
- Studied the limitations of benchmark-trained models when applied to real-world Sentinel-2 areas.
- Worked on the model documentation and README.
- Analyzed model predictions on a real Romanian area of interest.

 The main result of this work was a **Spectral ResNet-18 model using 13 Sentinel-2 bands**, which achieved approximately **99% accuracy** on the EuroSAT evaluation set.

---

 # 2\. Project Overview

 The goal of the project is to automatically classify satellite imagery into different land-cover categories.

 The classification task contains 10 classes:

 | Class | Description |
| --- | --- |
| AnnualCrop | Agricultural fields with annual crops |
| Forest | Forested areas |
| HerbaceousVegetation | Herbaceous vegetation |
| Highway | Roads and highways |
| Industrial | Industrial areas |
| Pasture | Pasture and grassland |
| PermanentCrop | Permanent agricultural crops |
| Residential | Residential areas |
| River | Rivers and waterways |
| SeaLake | Seas and lakes |

The project explores several approaches, ranging from traditional machine-learning methods such as **Random Forest** to convolutional neural networks such as **ResNet-18 and ResNet-50**.

---

 # 3\. Models

 Several models were trained and evaluated during the project.

 | Model | Input | Accuracy | Macro F1 | Purpose |
| --- | --- | --- | --- | --- |
| ResNet-18 RGB | 3 bands | 97% | 0.97 | Production web model |
| **Spectral ResNet-18** | **13 bands** | **99%** | **0.99** | **High-accuracy benchmark** |
| ResNet-50 RGB | 3 bands | 98% | 0.98 | Deep-learning benchmark |
| Spectral Random Forest | Spectral features | 87% | 0.86 | Traditional ML baseline |
| Random Forest | Basic features | 80% | 0.78 | Initial ML baseline |

The models demonstrate the progression from traditional machine learning to deep-learning approaches capable of learning more complex spatial and spectral patterns.

---

 # 4\. My Main Model — Spectral ResNet-18

 The main model associated with my work is a **Spectral ResNet-18 model using all 13 Sentinel-2 spectral bands**.

 Unlike the production RGB model, which only uses three channels, the spectral model uses substantially more information from the Sentinel-2 imagery.

 ### Performance

 - **Accuracy:** 99%
- **Macro F1:** 0.99
- **Evaluation samples:** 4,050
- **Input:** 13 Sentinel-2 bands
- **Architecture:** ResNet-18
- **Framework:** PyTorch / TorchGeo

 The model achieved approximately **99% accuracy** on the EuroSAT evaluation dataset.

 ### Model checkpoint

 The trained checkpoint is available through the project's model repository:

 https://huggingface.co/Airam18/land-cover-clasification-model-all-bands/tree/main

---

 # 5\. Spectral ResNet-18 Results

 The benchmark evaluation produced the following results:

```
              precision    recall  f1-score   support

AnnualCrop       0.99      0.99      0.99       450
Forest           1.00      1.00      1.00       450
HerbaceousVegetation  0.99    0.99      0.99       450
Highway          0.98      0.99      0.99       375
Industrial       0.99      0.99      0.99       375
Pasture          0.97      0.98      0.98       300
PermanentCrop    0.99      0.99      0.99       375
Residential      1.00      1.00      1.00       450
River            0.99      0.99      0.99       375
SeaLake          1.00      1.00      1.00       450

            accuracy                           0.99      4050
           macro avg       0.99      0.99      0.99      4050
        weighted avg       0.99      0.99      0.99      4050
```

 The results show consistently high precision, recall, and F1 scores across the majority of the 10 classes.

---

 # 6\. Confusion Matrix

 The confusion matrix for the Spectral ResNet-18 model was:

```
[[445   0   0   0   0   2   3   0   0   0]
 [  0 449   1   0   0   0   0   0   0   0]
 [  1   0 445   0   0   3   1   0   0   0]
 [  0   0   0 373   0   1   1   0   0   0]
 [  0   0   0   3 371   0   0   1   0   0]
 [  2   0   2   0   0 295   0   0   1   0]
 [  2   0   2   1   0   0 370   0   0   0]
 [  0   0   0   0   2   0   0 448   0   0]
 [  0   0   0   2   0   2   0   0 371   0]
 [  0   0   0   0   0   0   0   0   1 449]]
```

 The relatively small number of off-diagonal predictions demonstrates the strong benchmark performance of the model.

---

 # 7\. Understanding the Training Process

 One of the main objectives of my work was to understand not only the final accuracy, but also **how the model learns**.

 The general training process can be represented as:

```
Sentinel-2 / EuroSAT Dataset
            │
            ▼
      Data Preprocessing
            │
            ▼
    Train / Validation Data
            │
            ▼
      Spectral ResNet-18
            │
            ▼
        Prediction
            │
            ▼
      Loss Calculation
            │
            ▼
    Backpropagation
            │
            ▼
     Parameter Update
            │
            ▼
       Next Training Step
```

 During training, the model learns parameters that allow it to recognize patterns associated with different land-cover classes.

 For example, different classes can have characteristic:

 - spectral responses;
- textures;
- spatial structures;
- shapes;
- vegetation patterns;
- water signatures;
- urban patterns.

 The use of multiple Sentinel-2 bands allows the spectral model to access substantially more information than an RGB-only model.

---

 # 8\. RGB vs Multispectral Classification

 An important part of the project was understanding the difference between RGB and multispectral classification.

 ### RGB ResNet-18

 The production model uses:

 - Band 4 — Red
- Band 3 — Green
- Band 2 — Blue

 This produces a standard three-channel RGB image.

 Advantages include:

 - simpler preprocessing;
- smaller input;
- faster inference;
- easier web deployment;
- lower computational requirements.

 The RGB ResNet-18 achieved approximately **97% accuracy**.

 ### Spectral ResNet-18

 The spectral model uses all **13 Sentinel-2 bands**.

 This provides additional information outside the visible RGB spectrum.

 The spectral model achieved approximately:

 **99% accuracy**

 This demonstrates the potential benefit of using multispectral information for land-cover classification.

---

 # 9\. Other Models Evaluated

 ## ResNet-18 RGB

 The RGB ResNet-18 model achieved:

 - Accuracy: **97%**
- Macro F1: **0.97**

 It was selected for the production web application because of its balance between performance and computational requirements.

 Model:

 https://huggingface.co/dariussasarman/ROSPIN-Land-Classification/tree/main

---

 ## ResNet-50 RGB

 The larger ResNet-50 model achieved:

 - Accuracy: **98%**
- Macro F1: **0.98**

 It provided strong classification performance but required more computational resources than ResNet-18.

---

 ## Spectral Random Forest

 A traditional Random Forest model was also trained using spectral features.

 Results:

 - Accuracy: **87%**
- Macro F1: **0.86**

 Important spectral features included:

```
B12_mean
B12_std
R_B8_B12_std
```

 This provided a useful traditional machine-learning baseline for comparison with the deep-learning models.

---

 ## Standard Random Forest

 The initial Random Forest baseline achieved:

 - Accuracy: **80%**
- Macro F1: **0.78**

 This provided a baseline for understanding how much improvement could be obtained by using more advanced feature representations and deep-learning architectures.

---

 # 10\. Model Comparison

 The overall results can be summarized as:

```
Model                         Accuracy       Macro F1

Random Forest                  80%             0.78
Spectral Random Forest         87%             0.86
ResNet-18 RGB                  97%             0.97
ResNet-50 RGB                  98%             0.98
Spectral ResNet-18             99%             0.99
```

 The progression illustrates the differences between traditional machine learning, RGB deep learning, and multispectral deep learning approaches.

---

 # 11\. Why the 99% Model Was Not Used for the Web Application

 Although the Spectral ResNet-18 achieved the highest benchmark accuracy, the production web application uses the RGB ResNet-18 model.

 The main reason is that a real-time web application has additional requirements beyond benchmark accuracy.

 The RGB model provides:

 - faster inference;
- lower memory requirements;
- simpler preprocessing;
- easier integration;
- easier handling of standard image inputs;
- reduced data requirements.

 The 13-band model requires more complex multispectral processing and is therefore more difficult to integrate into a live web pipeline.

 This creates an important distinction between:

 **Benchmark performance**

 and

 **Production suitability**

 The 99% model demonstrated excellent benchmark performance, while the 97% RGB model provided a more practical solution for the live web application.

---

 # 12\. Sentinel-2 Data Pipeline

 The live application uses Sentinel-2 satellite data.

 The production pipeline can be summarized as:

```
Sentinel-2 Satellite
        │
        ▼
   Copernicus API
        │
        ▼
 Satellite Imagery
        │
        ▼
 Preprocessing
        │
        ▼
 RGB Band Selection
 B4 / B3 / B2
        │
        ▼
     ResNet-18
        │
        ▼
 Feature Extraction
        │
        ▼
 Classification
        │
        ▼
 Land-Cover Prediction
        │
        ▼
 Web Application
```

---

 # 13\. System Architecture

 Mermaid flowchart: Sentinel-2 Satellite, Copernicus API, Satellite Imagery, Preprocessing, RGB Bands B4 B3 B2, 13-Band Multispectral Data, ResNet-18 RGB, Spectral ResNet-18, Feature Extraction, Multispectral Feature Extraction, Classification, Land-Cover Prediction, Web Application

---

 # 14\. Real-World Testing

 In addition to evaluating the models on the benchmark dataset, the project also investigated their behavior on real-world Sentinel-2 areas of interest.

 This was important because high benchmark accuracy does not necessarily guarantee identical performance on real-world satellite scenes.

 For example, when testing areas around **Cluj-Napoca, Romania**, some residential and industrial regions containing significant vegetation were incorrectly classified as forest.

 This demonstrates one of the challenges of transferring a model trained on benchmark datasets to real-world satellite imagery.

---

 # 15\. Generalization Challenges

 The Spectral ResNet-18 model showed strong performance on the EuroSAT benchmark but experienced domain-shift issues when applied to real Sentinel-2 AOIs.

 Potential factors include:

 ### Spectral differences

 The spectral distribution and normalization of benchmark EuroSAT patches may differ from the real Sentinel-2 L2A imagery used during live inference.

 ### Mixed pixels

 Real satellite images frequently contain pixels representing combinations of land-cover types.

 For example, an urban region can contain:

 - buildings;
- trees;
- roads;
- grass;
- parking areas.

 A benchmark image may have a single dominant label, while real-world scenes can contain several land-cover types simultaneously.

 ### Dataset differences

 EuroSAT consists of relatively standardized image patches, whereas real-world AOIs can contain much larger and more complex spatial structures.

 This creates a **domain gap** between benchmark evaluation and real-world deployment.

---

 # 16\. Example — Romanian Area of Interest

 The model was also tested on a real area of interest in Romania.

 One observed failure case occurred in areas around **Cluj-Napoca**, where highly vegetated residential and industrial areas could sometimes be classified as **Forest**.

 This is an example of why real-world validation is important even when a model achieves very high benchmark accuracy.

---

 # 17\. Evaluation Reports

 Detailed evaluation reports are available in the repository.

 | Model | Report |
| --- | --- |
| ResNet-18 RGB | `reports/resnet18_m3_report.txt` |
| Spectral ResNet-18 | `reports/spectral_resnet_torchgeo_report.txt` |
| ResNet-50 | `reports/resnet50_report.txt` |
| Spectral Random Forest | `reports/spectral_rf_report.txt` |
| Random Forest | `reports/baseline_rf_report.txt` |

---

 # 18\. Repository Structure

```
.
├── models/
│   └── ...
│
├── checkpoints/
│   └── spectral_resnet_torchgeo_best.pth
│
├── reports/
│   ├── baseline_rf_report.txt
│   ├── spectral_rf_report.txt
│   ├── spectral_resnet_torchgeo_report.txt
│   ├── resnet18_m3_report.txt
│   └── resnet50_report.txt
│
├── model/
│   └── resnet18_m3_best.pth
│
├── training/
│   └── ...
│
├── evaluation/
│   └── ...
│
└── README.md
```

---

 # 19\. Technologies

 The project uses technologies from machine learning, deep learning, computer vision, and remote sensing.

 ### Programming

 - Python

 ### Machine Learning

 - Supervised Machine Learning
- Random Forest
- Deep Learning
- Convolutional Neural Networks
- Image Classification
- Multispectral Classification

 ### Deep Learning

 - ResNet-18
- ResNet-50
- PyTorch
- TorchGeo

 ### Data Processing

 - NumPy
- OpenCV
- Scikit-learn
- Matplotlib

 ### Remote Sensing

 - Sentinel-2
- EuroSAT
- Multispectral imagery
- Satellite image processing

---

 # 20\. Key Takeaways

 This project provided practical experience with the complete machine-learning workflow:

```
Dataset
   ↓
Data Processing
   ↓
Feature Representation
   ↓
Model Selection
   ↓
Training
   ↓
Evaluation
   ↓
Error Analysis
   ↓
Real-World Testing
```

 The most important results from my work were:

 - Understanding existing machine-learning and deep-learning models.
- Working with Sentinel-2 multispectral imagery.
- Understanding the training and evaluation pipeline.
- Training a **13-band Spectral ResNet-18** model.
- Achieving approximately **99% benchmark accuracy** and **0.99 macro F1**.
- Comparing traditional machine-learning models with deep-learning architectures.
- Understanding the differences between RGB and multispectral classification.
- Investigating real-world model failures and domain-shift problems.
- Understanding the trade-off between benchmark accuracy and production deployment requirements.

---

 # 21\. Original Collaborative Project

 This repository is based on work carried out collaboratively during the **ROSPIN Summer School**.

 The original project can be found here:

 **Land Cover Classification — ROSPIN Summer School**

 https://github.com/DariusSasarman/Land-cover-classification-ROSPIN-Summer-School

 The original repository contains the broader collaborative project.

 This repository focuses specifically on documenting **my work with the machine-learning models, model training, evaluation, the Spectral ResNet-18 model, and the analysis of its performance on both benchmark and real-world satellite imagery.**

---
