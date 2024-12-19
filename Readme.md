<center>

# **Application of Convolutional Vision Transformers for the Classification of Infectious Diseases in Chest Radiological images.**


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

<center>

# **Introduction**

</center>

**Background**:
- **Global Threats**: COVID-19, TB, pneumonia.
- **Importance of X-rays**: Essential non-invasive tools; however, manual analysis is inconsistent and often inaccurate.

**Deep Learning Evolution**:
- Last decade: Rise of CNNs for chest X-rays. Key studies: Rajpurkar et al., Wang et al.
- Emergence of Transformers in imaging, comparable to CNNs (Dosovitskiy et al.)

**Personal Drive**:
- Initial goal: Thesis on deep learning in computer-aided diagnosis.
- Discovery of the chest X-ray dataset during research.
- Influenced by "Do Preprocessing and Class Imbalance Matter to the Deep Image Classifiers for COVID-19 Detection?".
- Decision to tackle this significant issue for my master's thesis solidified.

**Project Objective**:
- Venture beyond CNNs; explore transformers for chest X-rays.
- Use pre-trained models like vgg19, resnet50 along side a Vision Transformer for disease classification: COVID-19, TB, lung opacity, etc.
- Goal: High accuracy and comprehensive understanding of the model's diagnostic skills.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

<center>

# **Aims and Objectives**

</center>

## Dataset Acquisition & Refinement
1. **Collection**: Sourced a comprehensive chest X-ray dataset from open-source platforms.
2. **Image Preprocessing**:
   - **Zooming**: Emphasis on regions of clinical interest.
   - **CLAHE**: Adaptive contrast enhancement preserving essential details.
   - **Sharpening**: Enhancing edges and fine details.
   - **Scaling**: Uniformity in image size and resolution.
   - **Zero-Centering**: Neutralize mean pixel value for faster model convergence.
3. **Shuffling & Batching**: Stratified sampling for balanced class exposure during training.
4. **Gblobal Class Based Weights**: Counter dataset imbalance and prevent model bias.

## Model Development & Evaluation
5. **Architectures**:
   - Established architectures: VGG19, ResNet50.
   - Introduce Custom CNN and Vision Transformer.
6. **Evaluation Matrix**: Analyze using accuracy, precision, recall, F-score, and ROC-AUC.
7. **Interpretability**:
   - Visualization methods: Convolutional visualization & attention map.
   - Goal: Understand model decision-making and identify regions of importance.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

<center>

# **About Dataset**

</center>

## ***Origin & Collection Process***
- **Dataset link**: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WNQ3GI](#)
- A collection of chest radiological images, primarily X-rays publicly released on `25th September, 2021`.
- **Creators**: Researchers from Qatar University, Doha, and Dhaka University, with contributions from Pakistan and Malaysia.
- **Expertise**: Continuous consultations with medical experts to ensure precision.
- **Sources**: Mainly from the COVID-19 Radiography Database on Kaggle, augmented with additional Pneumonia and COVID-19 images from other platforms.

## ***Composition & Diversity***
- **Categories**: Originally four - COVID-19, Lung Opacity, Normal, and Viral Pneumonia. Tuberculosis was added to enhance scope and relevance.
- **Image Type**: Mainly frontal view X-rays. Lateral views excluded for consistency.

## ***Distribution & Labelling***
| Disease Name    | Samples | Label |
|-----------------|-------:|-----:|
| COVID-19        | 4,189  | 0   |
| Lung Opacity    | 6,012  | 1   |
| Normal          | 10,192 | 2   |
| Viral Pneumonia | 7,397  | 3   |
| Tuberculosis    | 4,897  | 4   |
| **Total**       |32,687  |     |

- The dataset serves as a cornerstone for evaluating the potential of deep learning models such as (pre-trained conv-nets, vision transformers or custom convolutional neural network architectures in chest X-ray classification.

- It is a fairly recent, comprehensive dataset for disease diagnosis or classification from chest radiology images (chest X-rays).

- The authors of this dataset, split the dataset into `Training`, `Testing` and `Validation` set.

<center>

## ***Samples from the Datasets splits is shown below.***

### Training

![Samples from the Train Datasets](https://drive.google.com/uc?id=1-ITbHStnr3aDhOwMI-wPNoQ9hSD0st-X)

### Testing

![Samples from the Test Datasets](https://drive.google.com/uc?id=10HOmEXr3_dmUTWrRxow-An5JsghfkCfI)

### Validation

![Samples from the Validation Datasets](https://drive.google.com/uc?id=1091q8USZzQIlfJifAC9MT7tCRP0M4QUT)

<br><br>

---

<br><br>

## ***Distribution of classes in training, testing, and validation sets.***

### Training

![Value Counts Train Datasets](https://drive.google.com/uc?id=1cogvTqzBzLCTTIZ703hkl-INoKfD0G-W)


### Testing

![Value Counts Test Datasets](https://drive.google.com/uc?id=1-jN1UXQgu_YJ4fE1yohuf2GC7t0KMe05)

### Validation

![Value Counts Validation Datasets](https://drive.google.com/uc?id=1-hia-kkpa-Aie-nWqvfU1ef7NRA8Xbph)

</center>

<br><br>

---

<br><br>




<center>

# **Literature Review Limitations**

</center>

- **Black-box Nature of Deep Learning Models:** Deep models like VGG19 and ResNet50 often lack transparency in decision-making, impacting medical practitioners' trust.

- **Overfitting of Models:** Indications from loss-accuracy curves suggest prevalent overfitting in some studies, hindering real-world reliability.

- **Inadequate Performance Metrics:** Some studies might have used unsuitable metrics given the class imbalances in chest X-ray datasets.

- **Binary Classification Emphasis:** Many studies, including the dataset's authors, narrowly focused on binary classification, possibly overlooking key diagnostic nuances.

- **Local Feature Concentration:** Traditional CNNs might not capture the broader context in medical images effectively.

- **Limited Disease Diversity:** Some focused primarily on COVID-19, neglecting other respiratory diseases with similar X-ray features.

- **Class Imbalance:** The datasets used in most studies were imbalanced. However, no steps were taken to address class imabalance during training. The skewed representation in datasets might induce biased model predictions.

- **Lack of Interpretability Tools:** There's a marked absence of tools to illustrate model decisions, essential for clinician trust.

## **Necessity of the Proposed Research:**

- **Holistic Model Comparison:** An exhaustive evaluation of models ranging from VGG19 and ResNet50 to Vision Transformers and custom CNNs will be undertaken.

- **Mitigating Overfitting:** This work will use techniques like regularization and data augmentation to combat overfitting.

- **Comprehensive Performance Metrics:** Given the dataset imbalances, appropriate metrics will be employed for a genuine model assessment. Metrics such as precision, recall, f1 score and Area Under Curve of Receiving Operating Characteristics.

- **Embracing Multi-class Classification:** Shifting focus from binary to multi-class classification to capture the intricacies of respiratory diseases.

- **Incorporating Global Context with Transformers:** Vision Transformers could address the limitation of traditional CNNs by recognizing broader image contexts.

- **Enhanced Interpretability with Custom CNN:** The custom CNN will emphasize diagnostic regions, balancing performance with transparency due to the use of spatial convolutional attention.

- **Efficiency and Performance Equilibrium:** The custom CNN, with features like depth-wise separable convolution, aims for a balance between efficiency and accuracy.

- **Addressing Black-box Models via model interpretability:** This research will focus on illustrating the decision-making of deep classifiers by using gradient or activation visualization tools such as GRAD-CAM or Guided GRAD-CAM as well as Attention map visualization.

- **Effective Handling of Class Imbalance:** Class Imbalance will be handled during training by using global class based weighting.

<center>

<br><br>

---

<br><br>


# **Research Questions**

</center>


*1. How does the performance of Convolutional Vision Transformers compare with traditional Convolutional Neural Networks and other deep learning models such as VGG19, ResNet50, in the classification of chest radiological images of infectious respiratory diseases?*

*2. What insights can be derived from activation visualization or attention map visualization about the decision-making process of the pre-trained vgg19, resnet50 models or custom CNN and vision transformer models, respectively, in predicting respiratory diseases from chest radiological images?*


<br><br>

---

<br><br>


<center>

# **Methodology**

![Methodology Diagram](https://drive.google.com/uc?id=1QCg8MyzPd-cRDzUA6rgzz4Sd3moe7Dvt)

</center>

<br><br>

---

<br><br>

## **Training Protocol**

* The vision transformer model and the custom CNN model was trained from scratch.

* **In case of Pre-trained Models, a 2-step training protocol is adopted:**

  * **Transfer Learning:**
    * This involves harnessing the learned features from a previously trained model to accelerate and enhance the training for a new, related task.
    * Initially, only the top layers (often the fully connected layers) are retrained to adapt to the new task while the base layers are kept frozen.
    * At this stage, the pre-trained model effectively acts as a feature extractor, with its learned patterns benefiting the new task.

  * **Fine-Tuning:**
    * This phase dives deeper, adjusting not just the top layers but also some or all of the deeper layers of the pre-trained model.
    * By doing so, the model is further specialized, tailoring its learned features more closely to the nuances of the specific task at hand.


<br><br>

---

<br><br>



## **Class Imbalance Hanlding with Global class based weighting.**

<center>

**Class Weights:**

|   |   |
|---|---|
| 0 | 1.5613057324840764|
| 1 | 1.0873301912947047|
| 2 | 0.6413736713000817|
| 3 | 0.8837314105452907|
| 4 | 1.334921715452689|

---

<br>
<br>
</center>

<center>

# **EDA**

## *1. Average pixel intensity distribution.*

![Train Datasets](https://drive.google.com/uc?id=1-2NW9reC0KzWdHdi2V6iY6geDzuxqJO6)


![Validation Datasets](https://drive.google.com/uc?id=1-kCMKCCbTBWmzgje80gyAlfacQE-DUcf)


![Test Datasets](https://drive.google.com/uc?id=1-rlj1z3_w3Itxhsulq5hyf9ngNK6QA5b)

<br><br>

---

<br><br>


## *2. Size distribution of the images.*

![Train Datasets](https://drive.google.com/uc?id=1-Aznb5ZXwxhZtcQex8Q0e9W1N5KQPwLt)


![Validation Datasets](https://drive.google.com/uc?id=1-v1mHSp1EWINXCJ-M3cPjG7Sjo353lok)


![Test Datasets](https://drive.google.com/uc?id=104q_84T6cKSZp9Ro2ueMtentO7idv2rF)


<!-- ## 5. Check for missing values (None). -->

<br><br>

---

<br><br>

</center>


## *3. Distribution of image intensities for each class.*

### **Training**

![Train Datasets](https://drive.google.com/uc?id=1EBKjLf5_vxGwVh97APDmdzjuBoK1vuRW)

### **Validation**

![Validation Datasets](https://drive.google.com/uc?id=1kPqtNK7KXKg3Sk-uD1mabPbfj2zqNBs5)

### **Testing**
![Test Datasets](https://drive.google.com/uc?id=1-O4Tu0E-31-CE01hDv9a_VBiAjbUN8a8)

## **Insights**

- **Variation Across Classes:**
  - Different diseases exhibit unique pixel intensity distributions, reflecting their distinct radiological manifestations.
  - These variations are indicative of the specific pathophysiological changes associated with each condition.

- **Significance of Outliers:**
  - Outliers in intensity distributions can stem from imaging artifacts, data inconsistencies, or rare case presentations.
  - Their presence demands careful preprocessing, considering their potential to either introduce noise or represent crucial data points.

- **Potential for Model Discrimination:**
  - The disparities in intensity histograms among most classes bolster the claim that deep learning models can effectively differentiate between these classes.

- **Challenges with 'Lung Opacity' and 'Normal' Classes:**
  - These two classes display nearly identical intensity distributions, highlighting possible challenges in model differentiation.
  - Such overlap underscores the importance of advanced feature extraction or specific modeling techniques.

- **Class-specific Observations:**
  - **COVID-19:** Consistent imaging or disease presentation is hinted at by its near-normal distribution.
  - **Lung Opacity:** Multiple radiological patterns might be present, as suggested by the secondary peak post-mean.
  - **Normal:** Despite its resemblance to 'Lung Opacity', a notable secondary peak implies potential overlap with pathological conditions.
  - **Viral Pneumonia:** Its histogram suggests a mix of typical and atypical disease presentations in the dataset.
  - **Tuberculosis:** Two primary radiological patterns seem to dominate, as indicated by the twin peaks centered around the mean.

<br><br>

---

<br><br>

<center>
<br>
<br>

# **Image Pre-Processing**

</center>

* Zoom
* CLAHE
* Image Sharpening
* Rescaling
* 0 centering

<br>
<br>

---

<br><br>


<center>

# 1. CLAHE


* ## Grid Size Test on Original Images

![Grid Size](https://drive.google.com/uc?id=1-imhzmhfult2frNjRTy8AP6F9N9wnSQk)

* ## Clip Size Test on Original Images

![Clip Size](https://drive.google.com/uc?id=1-kyU0ZwMnKQeljfuEPLwYKUW3DURA6Q4)

<br><br>

---

<br><br>


# 2. Image Sharpening on Original Images

<br>
<br>

* ## Kernels Tested

**Basic Sharpening Kernel:**

|   |   |   |
|---|---|---|
| 0 | -1| 0 |
| -1| 5 | -1|
| 0 | -1| 0 |

<!-- --- --> <br>

**Laplacian Sharpening Kernel:**

|   |   |   |
|---|---|---|
| 0 | 1 | 0 |
| 1 |-4 | 1 |
| 0 | 1 | 0 |

<!-- --- --> <br>

**Diagonal Edge Sharpening:**

|   |   |   |
|---|---|---|
|-1 | 0 |-1 |
| 0 | 8 | 0 |
|-1 | 0 |-1 |

<!-- --- --> <br>

**Exaggerated Sharpening:**

|   |   |   |
|---|---|---|
|-1 |-1 |-1 |
|-1 | 9 |-1 |
|-1 |-1 |-1 |

<!-- --- --> <br>

**Another variant of Laplacian:**

|   |   |   |
|---|---|---|
| 1 | 1 | 1 |
| 1 |-7 | 1 |
| 1 | 1 | 1 |

<!-- --- -->

<br>
<br>

* ## Testing Different Image Sharpening Kernels

![Image Sharpening](https://drive.google.com/uc?id=1-yIHBALoHG-Gvg4_6-JWMjblMdkyMdjT)

<br><br>

---

<br><br>


# 3. Image Sharpening on CLAHE pre-processed Images

![C+S with different kernels](https://drive.google.com/uc?id=1-ynrn9psOLTzxhFw3yfqGjeRWdumhXjK)

<br><br>

---

<br><br>


# 4. Zoom

![Zoom](https://drive.google.com/uc?id=1HTKmwIooGuFWAQFmoE3vheFRn1NQYLyF)

<br><br>

---

<br><br>


# 5. Zoom + CLAHE + Image Sharpening
## Clip Size : 2, Grid Size: 15, Basic Sharpening Kernel, Zoom Factor : 0.83

![Zoom + CLAHE + Sharpening](https://drive.google.com/uc?id=1--t1dKzQ-w0F3-p4P0p9WdNWt0iALsZx)

<br><br>

---

<br><br>

# **6. Rescale and 0 Centering**

</center>

## Why Rescale and 0 center ?

* some
* some
* some

<center>

* # Distribution of Average Pixel Intensity of Original Images : Training Set

![Distribution of Pixel Intensity of Original Images](https://drive.google.com/uc?id=18xeWT3Of1QCwvStIN_Y8BLHAqVmhbhLC)

* # Distribution of Average Pixel Intensity of Pre-Processed Images : Training Set

![Distribution of Pixel Intensity of Original Images](https://drive.google.com/uc?id=1uZNQOpnLKOOgwjt0HtqNiX2TTRRfgiJ6)


* # Distribution of Average Pixel Intensity of Pre-Processed Rescaled Images : Training Set

![Distribution of Pixel Intensity of Original Images](https://drive.google.com/uc?id=1XD0M5F1Tc73VvxytqPaIGc8Yf282Ii6M)


* # Distribution of Average Pixel Intensity of Pre-Processed Rescaled Zero Centered Images : Training Set

![Distribution of Pixel Intensity of Original Images](https://drive.google.com/uc?id=17luHoNO5bdqpFqO1gVoEzFgbEeYk6oGN)


</center>

<br><br>

---

<br><br>


## **Image Pre-processing Pipeline for Chest X-ray: Rationale and Observations**

**Combining multiple techniques offers a synergistic effect for X-ray image optimization. A stepwise methodology was employed to highlight the significance of each stage:**

- **Zoom (Factor of 0.83):**
  - **Purpose:** Crop out potential artifacts and noise along the image borders.
  - **Benefits:** Standardizes the primary region of interest due to diverse radiological image presentations from various sources.

- **CLAHE (Clip Limit of 2 and Grid Size of 15):**
  - **Purpose:** Enhance image contrast.
  - **Benefits:** Especially effective for X-rays with dark backgrounds, it provides optimized contrast while preserving genuine image features.

- **Basic Sharpening Kernel:**
  - **Purpose:** Accentuate the edges and details within the image.
  - **Benefits:** Sharpens the image without adding unnecessary noise or artifacts, making the structures within the chest clearly visible.

- **Rescaling:**
  - **Purpose:** Normalize pixel values to a specific range, commonly between 0 and 1.
  - **Benefits:** Ensures consistent input scale for deep learning models, which can aid in faster and more stable convergence during training.

- **Zero-Centering:**
  - **Purpose:** Shift pixel values so that the mean is zero.
  - **Benefits:** Reduces the dominance of particular pixel intensity ranges, allowing models to learn features more effectively.

**Key Observations from Visualization:**
  - **Zooming:** Effectively eliminates extraneous details, offering a clearer view of the primary chest region.
  - **CLAHE:** Achieves uniform and enhanced contrast, which makes minute details more discernible.
  - **Sharpening:** Further emphasizes the edges and contours, clearly outlining structures within the chest.

**Conclusion:** The comprehensive approach of Zoom, CLAHE, Image Sharpening, Rescaling, and Zero-Centering offers a robust and efficient pre-processing pipeline for chest X-rays. Each step complements the others, ensuring the transformed image is medically relevant, standardized, and optimized for deep learning applications.



<br><br>

---

<br><br>

<center>

# **Dataset Processing & Model Training Essentials**

</center>

## ***Dataset Preparation:***
- Shuffled, Batched, and One-Hot Encoded.
- Type: `tf.data.Dataset` (MapDataset, PrefetchDataset)

## ***Why use `tf.data.Dataset`?***
- Efficiently handles large datasets
- Seamless data feeding for GPU/TPU
- On-the-fly transformations with `map()`
- Overlaps preprocessing & execution with `prefetch()`
- Efficient batching and shuffling
- Memory-efficient & integrated with TensorFlow ecosystem

## **Training Hyperparameters:**
- **Seed:** 123
- **Batch Size:** 128
- **Image Size:** 224
- **Classes:** 5
- Set random seeds for both numpy and TensorFlow

<br><br>

---

<br><br>


<br>
<br>

<center>

# **Class Distribution Across Batched Datasets.**

</center>

- Histograms for Training, Testing, and Validation sets.
- Maintained class ratio from original to final pre-processed batches.

<center>

* # **Training Set**

![Training Set](https://drive.google.com/uc?id=1-PVNe2L26HlZpAFuXIItgjrfz1EjHaHk)

* # **Validation Set**

![Validation Set](https://drive.google.com/uc?id=1-z766R_utECCy9RzMufuGyT-zeTimh1X)

* # **Testing Set**

![Testing Set](https://drive.google.com/uc?id=1MUdTHqggt3CAnVAa5kCesSP0GB1ZqmEO)

<br>
<br>

</center>

---

<br><br>


<br>
<br>

# **Models Trained, Evaluated and Compared.**

1. Preliminary Simple Convnet Model to ensure the final datasets (Train,Validation and Testing) were transformed and compiled correctly.
2. VGG19.
3. Resnet50.
4. Vision Transformer.
5. Custom Convnet Architecture based on skip connection, depthwise separable conv layers, spatial convolutional attention, dubbed, CustomCNN.

## **Why Use These Models ?**

- **Comprehensive Performance Analysis:** Using a mix of traditional architectures like VGG19 and Resnet50, alongside modern models such as Vision Transformers and a tailored architecture, provides a holistic evaluation of their efficacy in chest X-ray classification. This diversity allows for a rigorous cross-architecture comparison.

- **Building Upon Previous Work:** Traditional CNNs like VGG19 and Resnet50 have established successes in image classification. Assessing their fit for chest X-rays helps understand their specific advantages and challenges in this domain.

- **Exploring New Paradigms:** Vision Transformers, diverging from the traditional CNN approach, raise a question: can segmenting images into patches and viewing them as sequences enhance medically-relevant feature extraction?

- **Customized Approach:** CustomCNN, emphasizing skip connections, depthwise separable conv layers, and spatial convolutional attention, is crafted to better discern the nuances in chest X-rays. It tackles specific challenges like overlapping structures and subtle anomalies, which may be missed by general architectures.

- **Addressing Interpretability:** Deep learning models can often be "black-boxes", especially concerning in medical contexts. The custom architecture aims to bolster model transparency, spotlighting diagnostically significant regions.

- **Flexibility and Adaptability:** Evaluating a variety of models offers flexibility. Should one model underperform or face unexpected challenges, alternatives are at the ready to ensure project goals are met.

In summary, the chosen models represent a balance of reliable methods and innovative paradigms, all geared towards optimal chest X-ray classification.


<br>
<br>


<br>
<br>

---

<br><br>


## **1\. Preliminary Model Training for Testing.**

  - Trained a basic ConvNet on pre-processed chest x-ray dataset.
  - ***Objective***: Verify correct image pre-processing & achieve good performance metrics.
  - This step preceded transfer learning and fine-tuning with VGG19 and ResNet50.


<br>
<br>

<center>

## Model Architecture Summary

| Layer Type          | Output Shape          | Param #    |
|---------------------|-----------------------|------------|
| InputLayer          | (None, 224, 224, 3)   | 0          |
| Conv2D              | (None, 224, 224, 64)  | 1,792      |
| Activation          | (None, 224, 224, 64)  | 0          |
| Conv2D              | (None, 224, 224, 64)  | 36,928     |
| Activation          | (None, 224, 224, 64)  | 0          |
| MaxPooling2D        | (None, 112, 112, 64)  | 0          |
| Dropout             | (None, 112, 112, 64)  | 0          |
| ...                 | ...                   | ...        |
| Activation          | (None, 256)           | 0          |
| Dropout             | (None, 256)           | 0          |
| Dense               | (None, 5)             | 1,285      |

Total params: 104,038,981  
Trainable params: 104,038,981  
Non-trainable params: 0

<br>
<br>

---

<br><br>


## Classification Report

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.88      | 0.95   | 0.91     | 838     |
| 1     | 0.88      | 0.80   | 0.84     | 1203    |
| 2     | 0.89      | 0.92   | 0.90     | 2039    |
| 3     | 0.96      | 0.96   | 0.96     | 1480    |
| 4     | 0.98      | 0.96   | 0.97     | 980     |

- **Accuracy**: 0.92
- **Macro Avg**: Precision: 0.92, Recall: 0.92, F1-Score: 0.92, Support: 6540
- **Weighted Avg**: Precision: 0.92, Recall: 0.92, F1-Score: 0.92, Support: 6540

<br>
<br>

---

<br><br>


## Confusion Matrix on Test Set of the Prediction by Base Line Model

![Confusion Matrix](https://drive.google.com/uc?id=1us1knxvJKZ4TDtGRTBr5BKu578a5THur)


<br>
<br>

---

<br><br>


</center>


<center>
<br>
<br>

## **2\. VGG19 Model**

<br>
<br>

## Model Architecture Summary

| Layer Type                        | Output Shape          | Param #    |
|-----------------------------------|-----------------------|------------|
| InputLayer                        | (None, 224, 224, 3)   | 0          |
| block1_conv1 (Conv2D)             | (None, 224, 224, 64)  | 1,792      |
| block1_conv2 (Conv2D)             | (None, 224, 224, 64)  | 36,928     |
| block1_pool (MaxPooling2D)        | (None, 112, 112, 64)  | 0          |
| block2_conv1 (Conv2D)             | (None, 112, 112, 128) | 73,856     |
| block2_conv2 (Conv2D)             | (None, 112, 112, 128) | 147,584    |
| block2_pool (MaxPooling2D)        | (None, 56, 56, 128)   | 0          |
| block3_conv1 (Conv2D)             | (None, 56, 56, 256)   | 295,168    |
| block3_conv2 (Conv2D)             | (None, 56, 56, 256)   | 590,080    |
| block3_conv3 (Conv2D)             | (None, 56, 56, 256)   | 590,080    |
| block3_conv4 (Conv2D)             | (None, 56, 56, 256)   | 590,080    |
| block3_pool (MaxPooling2D)        | (None, 28, 28, 256)   | 0          |
| block4_conv1 (Conv2D)             | (None, 28, 28, 512)   | 1,180,160  |
| block4_conv2 (Conv2D)             | (None, 28, 28, 512)   | 2,359,808  |
| block4_conv3 (Conv2D)             | (None, 28, 28, 512)   | 2,359,808  |
| block4_conv4 (Conv2D)             | (None, 28, 28, 512)   | 2,359,808  |
| block4_pool (MaxPooling2D)        | (None, 14, 14, 512)   | 0          |
| block5_conv1 (Conv2D)             | (None, 14, 14, 512)   | 2,359,808  |
| block5_conv2 (Conv2D)             | (None, 14, 14, 512)   | 2,359,808  |
| block5_conv3 (Conv2D)             | (None, 14, 14, 512)   | 2,359,808  |
| block5_conv4 (Conv2D)             | (None, 14, 14, 512)   | 2,359,808  |
| block5_pool (MaxPooling2D)        | (None, 7, 7, 512)     | 0          |
| global_average_pooling2d (GlobalAveragePooling2D) | (None, 512) | 0   |
| dense (Dense)                     | (None, 4096)          | 2,101,248  |
| dropout (Dropout)                 | (None, 4096)          | 0          |
| dense_1 (Dense)                   | (None, 4096)          | 16,781,312 |
| dense_2 (Dense)                   | (None, 5)             | 20,485     |

- **Total params**: 38,927,429  
- **Trainable params**: 38,927,429   
- **Non-trainable params**: 0


<br>
<br>

---

<br><br>


## Classification Report

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0-Covid 19         | 0.94      | 0.95   | 0.95     | 838     |
| 1-Lung Opacity         | 0.89      | 0.92   | 0.90     | 1203    |
| 2-Normal         | 0.94      | 0.93   | 0.94     | 2039    |
| 3-Viral Pneumonia         | 0.98      | 0.96   | 0.97     | 1480    |
| 4-Tuberculosis         | 0.97      | 0.98   | 0.98     | 980     |
| **accuracy**  |           |        | 0.95     | 6540    |
| **macro avg**  | 0.95      | 0.95   | 0.95     | 6540    |
| **weighted avg** | 0.95      | 0.95   | 0.95     | 6540    |

---

<br><br>

## Model Architecture
![VGG19 Architecture](https://drive.google.com/uc?id=1B8SAjeO4hNMtNOKDNuxRgGvliPwAMvBr)


<br>
<br>

---

<br><br>


## Confusion Matrix on Test Set of the Prediction by VGG19 Model

![Confusion Matrix](https://drive.google.com/uc?id=1-CMkPNDpR1n2KqD-RVsHGQBA-3UH1oPo)

<br>
<br>

---

<br><br>


## Performance Metric Curves of VGG19 During Transfer Learning and Fine-Tuning

## Transfer Learning
![Transfer Learning](https://drive.google.com/uc?id=14YfbonP5gsfBOs8X2Liaq-WhPJckZ6EU)


## Fine Tuning
![Fine-Tuning](https://drive.google.com/uc?id=1-90NS0wL6hNHpOwXpSqufU-815d2heuc)

<br>
<br>

---

<br><br>


</center>


<center>
<br>
<br>

## **3\. Resnet50 Model**

<br>
<br>

---

<br><br>


## Model Architecture Summary

| Layer (type) | Output Shape | Param # | Connected to |
|--------------|--------------|---------|--------------|
| input_1 (InputLayer) | [(None, 224, 224, 3)] | 0 | |
| conv1_pad (ZeroPadding2D) | (None, 230, 230, 3) | 0 | input_1[0][0] |
| conv1_conv (Conv2D) | (None, 112, 112, 64) | 9472 | conv1_pad[0][0] |
| conv1_bn (BatchNormalization) | (None, 112, 112, 64) | 256 | conv1_conv[0][0] |
| conv1_relu (Activation) | (None, 112, 112, 64) | 0 | conv1_bn[0][0] |
| pool1_pad (ZeroPadding2D) | (None, 114, 114, 64) | 0 | conv1_relu[0][0] |
| pool1_pool (MaxPooling2D) | (None, 56, 56, 64) | 0 | pool1_pad[0][0] |
| conv2_block1_1_conv (Conv2D) | (None, 56, 56, 64) | 4160 | pool1_pool[0][0] |
| ... | ... | ... | ... |
| ... | ... | ... | ... |
| ... | ... | ... | ... |
| conv5_block3_1_conv (Conv2D) | (None, 7, 7, 512) | 1049088 | conv5_block2_out[0][0] |
| conv5_block3_1_bn (BatchNormalization) | (None, 7, 7, 512) | 2048 | conv5_block3_1_conv[0][0] |
| conv5_block3_1_relu (Activation) | (None, 7, 7, 512) | 0 | conv5_block3_1_bn[0][0] |
| conv5_block3_2_conv (Conv2D) | (None, 7, 7, 512) | 2359808 | conv5_block3_1_relu[0][0] |
| conv5_block3_2_bn (BatchNormalization) | (None, 7, 7, 512) | 2048 | conv5_block3_2_conv[0][0] |
| conv5_block3_2_relu (Activation) | (None, 7, 7, 512) | 0 | conv5_block3_2_bn[0][0] |
| conv5_block3_3_conv (Conv2D) | (None, 7, 7, 2048) | 1050624 | conv5_block3_2_relu[0][0] |
| conv5_block3_3_bn (BatchNormalization) | (None, 7, 7, 2048) | 8192 | conv5_block3_3_conv[0][0] |
| conv5_block3_add (Add) | (None, 7, 7, 2048) | 0 | conv5_block2_out[0][0], conv5_block3_3_bn[0][0] |
| conv5_block3_out (Activation) | (None, 7, 7, 2048) | 0 | conv5_block3_add[0][0] |
| global_average_pooling2d (GlobalAveragePooling2D) | (None, 2048) | 0 | conv5_block3_out[0][0] |
| dense (Dense) | (None, 5) | 10245 | global_average_pooling2d[0][0] |

**Total params:** 23,597,957  
**Trainable params:** 23,544,837  
**Non-trainable params:** 53,120


<br>
<br>

---

<br><br>


## Classification Report

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.97      | 0.91   | 0.94     | 838     |
| 1         | 0.94      | 0.87   | 0.90     | 1203    |
| 2         | 0.91      | 0.97   | 0.94     | 2039    |
| 3         | 0.93      | 0.99   | 0.95     | 1480    |
| 4         | 0.99      | 0.89   | 0.94     | 980     |
| **accuracy**  |           |        | 0.94     | 6540    |
| **macro avg** | 0.95      | 0.93   | 0.94     | 6540    |
| **weighted avg** | 0.94      | 0.94   | 0.94     | 6540    |

<br>
<br>

---

<br><br>


## Model Architecture
![Resnet50 Architecture](https://drive.google.com/uc?id=1--9AtPmo6cTeN0nxxK9N8Qlx9vEByrsJ)

<br>
<br>

---

<br><br>


## Confusion Matrix on Test Set of the Prediction by RESNET50 Model

![Confusion Matrix](https://drive.google.com/uc?id=1-9EK6C9rC1PCr6Y44Pe7gcHwQ_lqUneK)

<br>
<br>

---

<br><br>


## Performance Metric Curves of RESNET50 During Transfer Learning and Fine-Tuning

## Transfer Learning
![Transfer Learning](https://drive.google.com/uc?id=1N9WBWJenl6y9VjYWNynrb48ml4yQyIhC)


## Fine Tuning
![Fine-Tuning](https://drive.google.com/uc?id=1-3mspBvgC63XwpHEnusyYGDcAoVRYtvT)

<br>
<br>

---

<br><br>


</center>


<center>
<br>
<br>

## **4\. Custom CNN Model using Spatial Attention, Depthwise Convolution and Skip Connections.**

<br>
<br>

---

<br><br>

## Model Architecture Summary

| Layer (type)                                         | Output Shape          | Param #  | Connected to                                      |
|------------------------------------------------------|-----------------------|----------|----------------------------------------------------|
| input_layer (InputLayer)                             | (None, 224, 224, 3)   | 0        | []                                                |
| spatialAttention_conv_preVGG_Spatial_Attention (Conv2D)| (None, 224, 224, 1)   | 28       | ['input_layer[0][0]']                             |
| spatialAttention_multiply_preVGG_Spatial_Attention (Multiply)| (None, 224, 224, 3) | 0    | ['input_layer[0][0]', 'spatialAttention_conv_preVGG_Spatial_Attention[0][0]'] |
| vggBlock_1_conv_1 (Conv2D)                           | (None, 224, 224, 32)  | 896      | ['spatialAttention_multiply_preVGG_Spatial_Attention[0][0]'] |
| vggBlock_1_bn_1 (BatchNormalization)                 | (None, 224, 224, 32)  | 128      | ['vggBlock_1_conv_1[0][0]']                       |
| vggBlock_1_act_1 (Activation)                        | (None, 224, 224, 32)  | 0        | ['vggBlock_1_bn_1[0][0]']                         |
| vggBlock_1_conv_2 (Conv2D)                           | (None, 224, 224, 32)  | 9248     | ['vggBlock_1_act_1[0][0]']                        |
| vggBlock_1_bn_2 (BatchNormalization)                 | (None, 224, 224, 32)  | 128      | ['vggBlock_1_conv_2[0][0]']                       |
| vggBlock_1_act_2 (Activation)                        | (None, 224, 224, 32)  | 0        | ['vggBlock_1_bn_2[0][0]']                         |
| spatialAttention_conv_prePoolVGG_1 (Conv2D)          | (None, 224, 224, 1)   | 289      | ['vggBlock_1_act_2[0][0]']                        |
| resBlock_1_adjust_conv (Conv2D)                      | (None, 224, 224, 32)  | 128      | ['spatialAttention_multiply_preVGG_Spatial_Attention[0][0]'] |
| spatialAttention_multiply_prePoolVGG_1 (Multiply)    | (None, 224, 224, 32)  | 0        | ['vggBlock_1_act_2[0][0]', 'spatialAttention_conv_prePoolVGG_1[0][0]'] |
| resBlock_1_adjust_bn (BatchNormalization)            | (None, 224, 224, 32)  | 128      | ['resBlock_1_adjust_conv[0][0]']                  |
| vggBlock_1_pool (MaxPooling2D)                       | (None, 112, 112, 32)  | 0        | ['spatialAttention_multiply_prePoolVGG_1[0][0]']  |
| resBlock_1_adjust_pool (MaxPooling2D)                | (None, 112, 112, 32)  | 0        | ['resBlock_1_adjust_bn[0][0]']                    |
| resBlock_1_add (Add)                                 | (None, 112, 112, 32)  | 0        | ['vggBlock_1_pool[0][0]', 'resBlock_1_adjust_pool[0][0]'] |
| ... | ... | ... | ... |
| ... | ... | ... | ... |
| ... | ... | ... | ... |
| vggBlock_4_conv_1 (Conv2D)                           | (None, 28, 28, 256)   | 295168   | ['resBlock_3_add[0][0]']                          |
| vggBlock_4_bn_1 (BatchNormalization)                 | (None, 28, 28, 256)   | 1024     | ['vggBlock_4_conv_1[0][0]']                       |
| vggBlock_4_act_1 (Activation)                        | (None, 28, 28, 256)   | 0        | ['vggBlock_4_bn_1[0][0]']                         |
| vggBlock_4_conv_2 (Conv2D)                           | (None, 28, 28, 256)   | 590080   | ['vggBlock_4_act_1[0][0]']                        |
| vggBlock_4_bn_2 (BatchNormalization)                 | (None, 28, 28, 256)   | 1024     | ['vggBlock_4_conv_2[0][0]']                       |
| vggBlock_4_act_2 (Activation)                        | (None, 28, 28, 256)   | 0        | ['vggBlock_4_bn_2[0][0]']                         |
| spatialAttention_conv_prePoolVGG_4 (Conv2D)          | (None, 28, 28, 1)     | 2305     | ['vggBlock_4_act_2[0][0]']                        |
| resBlock_4_adjust_conv (Conv2D)                      | (None, 28, 28, 256)   | 33024    | ['resBlock_3_add[0][0]']                          |
| spatialAttention_multiply_prePoolVGG_4 (Multiply)    | (None, 28, 28, 256)   | 0        | ['vggBlock_4_act_2[0][0]', 'spatialAttention_conv_prePoolVGG_4[0][0]'] |
| resBlock_4_adjust_bn (BatchNormalization)            | (None, 28, 28, 256)   | 1024     | ['resBlock_4_adjust_conv[0][0]']                  |
| vggBlock_4_pool (MaxPooling2D)                       | (None, 14, 14, 256)   | 0        | ['spatialAttention_multiply_prePoolVGG_4[0][0]']  |
| resBlock_4_adjust_pool (MaxPooling2D)                | (None, 14, 14, 256)   | 0        | ['resBlock_4_adjust_bn[0][0]']                    |
| resBlock_4_add (Add)                                 | (None, 14, 14, 256)   | 0        | ['vggBlock_4_pool[0][0]', 'resBlock_4_adjust_pool[0][0]'] |
| global_avg_pool (GlobalAveragePooling2D)             | (None, 256)           | 0        | ['resBlock_4_add[0][0]']                          |
| dense_1 (Dense)                                      | (None, 1024)          | 263168   | ['global_avg_pool[0][0]']                         |
| dropout_1 (Dropout)                                  | (None, 1024)          | 0        | ['dense_1[0][0]']                                 |
| dense_2 (Dense)                                      | (None, 1024)          | 1049600  | ['dropout_1[0][0]']                               |
| output_layer (Dense)                                 | (None, 5)             | 5125     | ['dense_2[0][0]']                                 |

Total params: 2,543,845  
Trainable params: 2,540,965  
Non-trainable params: 2

<br>
<br>

---

<br><br>


## Classification Report

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.91      | 0.89   | 0.90     | 838     |
| 1         | 0.87      | 0.86   | 0.86     | 1203    |
| 2         | 0.92      | 0.92   | 0.92     | 2039    |
| 3         | 0.96      | 0.97   | 0.97     | 1480    |
| 4         | 0.97      | 0.97   | 0.97     | 980     |
|-----------|-----------|--------|----------|---------|
| Accuracy  |           |        | 0.92     | 6540    |
| Macro Avg | 0.92      | 0.92   | 0.92     | 6540    |
| Weighted Avg | 0.92   | 0.92   | 0.92     | 6540    |


<br>
<br>

---

<br><br>


## Model Architecture
![Custom CNN Architecture](https://drive.google.com/uc?id=14IkriobdiqvJtywEELB1dRK1Pi-_yfDR)

<br>
<br>

---

<br><br>


## Confusion Matrix on Test Set of the Prediction by Custom CNN Model

![Confusion Matrix](https://drive.google.com/uc?id=1bfmpTf2LC5GMPybrf3owd7soUaJRU5DB)

<br>
<br>

---

<br><br>


## Performance Metric Curves of Custom CNN Model During Training.

![Transfer From Scratch](https://drive.google.com/uc?id=1-3WQ8cP8EWTse3d18S5HEa-GwMopBY4K)


<br>
<br>

---

<br><br>


</center>


<center>
<br>
<br>

## **5\. Vision Transformer with Multi Head Self Attention Layer**

<br>
<br>

---

<br><br>


## Model Architecture Summary

| Layer (type)                               | Output Shape         | Param #    | Connected to                               |
|--------------------------------------------|----------------------|------------|--------------------------------------------|
| input_1 (InputLayer)                       | (None, 224, 224, 3)  | 0          | -                                          |
| tf.image.extract_patches (TFOpLambda)      | (None, 11, 11, 1200) | 0          | input_1[0][0]                              |
| reshape (Reshape)                          | (None, 121, 1200)    | 0          | tf.image.extract_patches[0][0]             |
| dense (Dense)                              | (None, 121, 768)     | 922,368    | reshape[0][0]                              |
| token_and_position_embedding               | (None, 122, 768)     | 94,464     | dense[0][0]                                |
| layer_normalization (LayerNormalization)   | (None, 122, 768)     | 1,536      | token_and_position_embedding[0][0]         |
| multi_head_attention (MultiHeadAttention)  | (None, 122, 768)     | 11,808,768 | layer_normalization[0][0], layer_normalization[0][0] |
| add (Add)                                  | (None, 122, 768)     | 0          | multi_head_attention[0][0], token_and_position_embedding[0][0] |
| layer_normalization_1 (LayerNormalization) | (None, 122, 768)     | 1,536      | add[0][0]                                  |
| dense_1 (Dense)                            | (None, 122, 3072)    | 2,362,368  | layer_normalization_1[0][0]                |
| dropout (Dropout)                          | (None, 122, 3072)    | 0          | dense_1[0][0]                              |
| dense_2 (Dense)                            | (None, 122, 768)     | 2,360,064  | dropout[0][0]                              |
| add_1 (Add)                                | (None, 122, 768)     | 0          | dense_2[0][0], layer_normalization_1[0][0] |
| layer_normalization_2 (LayerNormalization) | (None, 122, 768)     | 1,536      | add_1[0][0]                                |
| multi_head_attention_1 (MultiHeadAttention)| (None, 122, 768)     | 11,808,768 | layer_normalization_2[0][0], layer_normalization_2[0][0] |
| add_2 (Add)                                | (None, 122, 768)     | 0          | multi_head_attention_1[0][0], add_1[0][0]  |
| layer_normalization_3 (LayerNormalization) | (None, 122, 768)     | 1,536      | add_2[0][0]                                |
| ... | ... | ... | ... |
| ... | ... | ... | ... |
| ... | ... | ... | ... |
| dense_9 (Dense)                            | (None, 122, 3072)    | 2,362,368  | layer_normalization_9[0][0]                |
| dropout_4 (Dropout)                        | (None, 122, 3072)    | 0          | dense_9[0][0]                              |
| dense_10 (Dense)                           | (None, 122, 768)     | 2,360,064  | dropout_4[0][0]                            |
| add_9 (Add)                                | (None, 122, 768)     | 0          | dense_10[0][0], layer_normalization_9[0][0]|
| dense_11 (Dense)                           | (None, 122, 1)       | 769        | add_9[0][0]                                |
| tf.math.multiply (TFOpLambda)              | (None, 122, 768)     | 0          | add_9[0][0], dense_11[0][0]                |
| tf.math.reduce_sum (TFOpLambda)            | (None, 768)          | 0          | tf.math.multiply[0][0]                     |
| dense_12 (Dense)                           | (None, 2096)         | 1,611,824  | tf.math.reduce_sum[0][0]                   |
| dropout_5 (Dropout)                        | (None, 2096)         | 0          | dense_12[0][0]                             |
| dense_13 (Dense)                           | (None, 2096)         | 4,395,312  | dropout_5[0][0]                            |
| global_average_pooling1d (GlobalAveragePooling1D) | (None, 768)  | 0          | add_9[0][0]                                |
| concatenate (Concatenate)                  | (None, 2864)         | 0          | dense_13[0][0], global_average_pooling1d[0][0] |
| dense_14 (Dense)                           | (None, 5)            | 14,325     | concatenate[0][0]                          |

**Total params:** 89,710,422  
**Trainable params:** 89,710,422  
**Non-trainable params:** 0  


<br>
<br>

---

<br><br>


## Classification Report

|            | Precision | Recall | F1-Score | Support |
|------------|:---------:|:------:|:-------:|:-------:|
| **0**      | 0.59      | 0.67   | 0.63    | 838     |
| **1**      | 0.74      | 0.73   | 0.73    | 1203    |
| **2**      | 0.82      | 0.80   | 0.81    | 2039    |
| **3**      | 0.90      | 0.86   | 0.88    | 1480    |
| **4**      | 0.81      | 0.81   | 0.81    | 980     |
|------------| --------- | ------ | ------- | ------- |
| **accuracy** |          |        | 0.79    | 6540    |
| **macro avg** | 0.77    | 0.77   | 0.77    | 6540    |
| **weighted avg** | 0.79 | 0.79   | 0.79    | 6540    |


<br>
<br>

---

<br><br>


## Model Architecture
![VIT Architecture](https://drive.google.com/uc?id=1-2Rr6CBMcAAfmbuCzbwtofD7sRVy4th2)

<br>
<br>

---

<br>
<br>


## Confusion Matrix on Test Set of the Prediction by VIT Model

![Confusion Matrix](https://drive.google.com/uc?id=1-853W6MINvZj4yMgiMhJrF_xB00fpI7m)

<br>
<br>

---

<br>
<br>


## Performance Metric Curves of VIT Model During Training.

![Transfer From Scratch](https://drive.google.com/uc?id=1XDcEgQT54f43vI73sCm3ykB-_olWzqHs)


<br>
<br>

---

<br>
<br>

</center>



# **Research Question 1 and Performance Comparison.**

## **Research Question 1** : *How does the performance of Convolutional Vision Transformers compare with traditional Convolutional Neural Networks and other deep learning models such as VGG19, ResNet50, in the classification of chest radiological images of infectious respiratory diseases?*

### **1. Overview:**
Comparing the performance of Convolutional Vision Transformers (ViT) with traditional Convolutional Neural Networks (CNNs) like VGG19, ResNet50, and the custom CNN architecture for chest radiological images provides several insights.

<center>

# **Performance Metric Scores of the trained models.**

| Model Name | Loss     | Test Accuracy | Precision | Recall  | AUC     | F1 Score | Top-2-Accuracy |
|------------|----------|---------------|-----------|---------|---------|----------|----------------|
| VGG19      | 0.159846 | 0.945260      | 0.946319  | 0.943425| 0.994903| 0.944870 | Not Available  |
| CustomCNN  | 0.208691 | 0.970000      | 0.928472  | 0.920948| 0.992963| 0.924695 | Not Available  |
| ResNet50   | 0.222941 | 0.936697      | 0.939757  | 0.935015| 0.990741| 0.937380 | Not Available  |
| VIT        | 0.662379 | 0.786544      | 0.811604  | 0.761468| 0.951327| 0.785737 | 0.936544       |

</center>

### **2. Dataset Size and Training Complexity:**
ViTs, especially those adapted from "ViT b16", shine when trained on vast datasets. The study's dataset might not be expansive enough to fully leverage ViTs, and their computational demands could hinder optimal training, as indicated by the elevated loss and accuracy metrics during training.

### **3. Historical Strength of CNNs:**
CNNs like VGG19 and ResNet50, optimized for image tasks over the years, display superior performance. The custom CNN, with its specialized layers, further underlines the effectiveness of task-tailored CNNs.

### **4. Conclusion:**
While Vision Transformers have made significant advancements in computer vision, when classifying chest radiological images—with current dataset and computational resources—they may be outperformed by established CNNs like VGG19, ResNet50, and the custom CNN model.


<br><br>

---

<br><br>

<center>

# **Gradient activation and attention map visualisation for Model Interpretability.**


<center>
<br><br>

---

<br><br>

## **VGG19 Activation Visualisation Using Grad Cam and Guided Grad Cam.**

![Activation Map Visualisation](https://drive.google.com/uc?id=1bA6kyiiLwnUjArdawizsmxfn-ZJ9zmfk)

![Activation Map Visualisation](https://drive.google.com/uc?id=1xg95m-mnLQuFaMpBHKN6EqaDNMZnEZW2)

![Activation Map Visualisation](https://drive.google.com/uc?id=1tZjq7PS3b5oaZlNXet_0oVzCAF-aNtgS)

![Activation Map Visualisation](https://drive.google.com/uc?id=1dxv6bkSTU794prtq4kjBnffOzGjxcX1M)

![Activation Map Visualisation](https://drive.google.com/uc?id=1mRg1anRWNKYfjhu8uV_XjrPuOaxwmmlL)
</center>

<br><br>

<center>
<br><br>

---

<br>
<br>

## **Resnet50 Activation Visualisation Using Grad Cam and Guided Grad Cam.**
<br>


* ## *Testing Different Alpha Values for super imposition of gradcam heatmap with original image.*

![Activation Map Visualisation](https://drive.google.com/uc?id=15P9RGpu5YLZMJVgYqfQu-8macQO8uzlH)

![Activation Map Visualisation](https://drive.google.com/uc?id=1A5F91RF6LXgvaZe3Fpfi49vGpA_48PAq)

![Activation Map Visualisation](https://drive.google.com/uc?id=11ASbpCfytdalzDD_smht4Y2Wsn1owo7b)

![Activation Map Visualisation](https://drive.google.com/uc?id=1BOzB_KW8M-PvKVvuUx6PEGq2Yz97vvQA)

![Activation Map Visualisation](https://drive.google.com/uc?id=1ddd1hPmuVrpjfCJ4bNfL4DYybZMn0lSv)

<br><br>

---

<br><br>

* ## *Comparing Original Image with Grad Cam Heatmap Superimposed Image.*

![Activation Map Visualisation](https://drive.google.com/uc?id=1qF8-Iwnh9DAcZmz3UxA3UnNpUtLtY-C2)

![Activation Map Visualisation](https://drive.google.com/uc?id=1ihzzuUqmIIGibYa6Hr7DQ__t94N0dtzz)

![Activation Map Visualisation](https://drive.google.com/uc?id=1TO3F6zRTzOH7mcNNqQgwV3D8e19tp47y)

![Activation Map Visualisation](https://drive.google.com/uc?id=1Z8KXcnvGgoN_Envad6-znyKStorhiR57)

![Activation Map Visualisation](https://drive.google.com/uc?id=1WW8HFD_ZVSVR_ZKzOEEDSwZxcr-gGLZY)

<br><br>

---

<br><br>


* ## *Testing Different Alpha Values for Super Imposition of Guided GradCam HeatMap with Original Image.*

![Activation Map Visualisation](https://drive.google.com/uc?id=1seRI2rpCkQX4lb4h5VhvyVOodiIXGf0B)

![Activation Map Visualisation](https://drive.google.com/uc?id=17THvq6IQ88E4wCpA6wXiiQBPBF4Soyld)

![Activation Map Visualisation](https://drive.google.com/uc?id=1SbhA19qtIANtQ12Uc8uEB4AC_x3XumaM)

![Activation Map Visualisation](https://drive.google.com/uc?id=1j0LtIHO3wdg6DhylLRYElMfJbMqxI8eP)

![Activation Map Visualisation](https://drive.google.com/uc?id=1wa7EB2WiKPzJWcsgl1dYElvMo56njdMA)

<br><br>

---

<br><br>


* ## *Comparing Original Image with Grad Cam and Guided Grad Cam Heatmap Superimposed Image respectively.*

![Activation Map Visualisation](https://drive.google.com/uc?id=18IihMjcSXkjRA7dQtUd583or_zoaJJ4c)

![Activation Map Visualisation](https://drive.google.com/uc?id=18KarbgFVK6eRPZqpFxpWqKMi0-Hlc0Fd)

![Activation Map Visualisation](https://drive.google.com/uc?id=1v93ZkXHDjRNY-7G8S3swuHI1QXOLtnqU)

![Activation Map Visualisation](https://drive.google.com/uc?id=16gpZJF0gZ9mPhD6HzTlbhsRPA4ejeWoU)

![Activation Map Visualisation](https://drive.google.com/uc?id=1K5PM7IvtJUTtxxy33nTCvyVj9cqLZH1O)

<br><br>

---

<br><br>


* ## *Comparing Original Image with Grad Cam and Guided Grad Cam Heatmap Superimposed Image respectively along with Grad Cam and Guided Grad Cam Heatmaps.*

![Activation Map Visualisation](https://drive.google.com/uc?id=1c4MdN7BEadfYAOMusBidNyv1U2UWDnMq)

![Activation Map Visualisation](https://drive.google.com/uc?id=1C_y9KwmvjVaPZPFU7gUpP9xDZZaljwsW)

![Activation Map Visualisation](https://drive.google.com/uc?id=1O3-jGg6GtOTD5SemxNwt3aZ1H9Urk36o)

![Activation Map Visualisation](https://drive.google.com/uc?id=15pfRSD-jtCBJjsXIVdxbdu2gWD5IF8ZV)

![Activation Map Visualisation](https://drive.google.com/uc?id=1r3xx-1csfnuAgMttfF79lVCi4y7GUPLh)

</center>

<br><br>

---

<br><br>


<center>

---

## **Custom CNN Spatial Attention Map Visualisation.**

<br><br>

![Activation Map Visualisation](https://drive.google.com/uc?id=1UaeZK6hH83NG3yBiAmpKGjtd9OQtxcry)

![Activation Map Visualisation](https://drive.google.com/uc?id=1uRLsJaSu_vRYAzj7K8kpVlI6mQ1OkaVM)

![Activation Map Visualisation](https://drive.google.com/uc?id=1ran2uZlCu1EUvNrau2harhFpOTaJ9c_7)

![Activation Map Visualisation](https://drive.google.com/uc?id=1cijKbdd-6aF_m7YqwQRxB4eAbSno6aPT)

![Activation Map Visualisation](https://drive.google.com/uc?id=1fThCTjLqGAt9VoP4xnfrRmTzWHmjEEVQ)

</center>

<br><br>

---

<!-- <br><br> -->


<!-- <center>

## **Vision Transformer Attention Map Visualisation.**

<br><br>

# ***Block 1***

![Activation Map Visualisation](https://drive.google.com/uc?id=1o0AKz8sbwlmcoynvPm30mWKJN7M6owal)

<br>

---

<br>

# ***Block 2***

![Activation Map Visualisation](https://drive.google.com/uc?id=1TMKV09IMOODSxBuhXIUVvfnOyyq5KOXn)

<br>

---

<br>

# ***Block 3***

![Activation Map Visualisation](https://drive.google.com/uc?id=1143yDu8B7bhnMQ9zbXkiFQTqVzIzAsis)

<br>

---

<br>

# ***Block 4***

![Activation Map Visualisation](https://drive.google.com/uc?id=1p1FSx3p9HM5p3sUpsKKogf-lNxRcTXuq)

<br>

---

<br>

# ***Block 5***

![Activation Map Visualisation](https://drive.google.com/uc?id=1_x3BrYuj2L4kEROZvhozfTp3OgsmCsKh)

</center>

<br><br> -->


# **Research Question 2 and Model Interpretability.**

* ### **Research Question 2 :** *What insights can be derived from activation visualization or attention map visualization about the decision-making process of the pre-trained vgg19, resnet50 models or custom CNN and vision transformer models, respectively, in predicting respiratory diseases from chest radiological images?*

* ### **Insights from Activation and Attention Map Visualizations in Chest Radiological Images Classification**

  - Activation and attention map visualizations offer transparency into neural networks, highlighting areas deemed significant during predictions. Such insights are crucial for medical imaging applications, where the understanding of a model's focus can aid in validation and trust.

  - ### **1. ResNet50:**
Using gradient-based visualization, ResNet50 pinpoints distinct chest X-ray features indicative of respiratory diseases, like opacities. However, its attention sometimes extends to noise and image edges. This could be influenced by the zoom preprocessing, emphasizing both critical regions and noise.

  - ### **2. VGG19:**
Gradient visualizations for VGG19 revealed its attention predominantly on image noise and artifacts, potentially due to the absence of a zoom preprocessing step. This highlights the importance of diligent preprocessing in medical imaging.

  - ### **3. Custom CNN:**
Spatial attention maps from the custom CNN delineate its focus from broad chest X-ray regions to specific areas like the lungs and then potential pathological zones. This layered attention mirrors a radiologist's diagnostic process. The integration of spatial attention mechanisms boosts the model's ability to focus on vital image areas, essential for medical imaging tasks.

  - ### **4. Vision Transformers (ViT):**
ViTs, with their self-attention mechanisms, process images as sequences of non-overlapping patches. Though the attention maps from our ViT experiments exhibited inconsistencies, ideally, ViTs would emphasize diagnostic-relevant patches, providing a comprehensive image interpretation. However, their efficiency is deeply tied to training data quality and size.

  - ### **Conclusion:**
While visualizations grant insights into model decision-making, they should be interpreted with care. Sole reliance on network focus isn't conclusive proof of correct predictions. Clinical validation remains vital. Nevertheless, these visualization tools undoubtedly foster understanding and collaboration between neural networks and clinicians.


# **Conclusion**

## Conclusion and Future Work

- **Deep Learning in Medical Imaging**
  - Application: Both opportunities and challenges observed.
  - Focus on:
    - **CNNs**: Traditional architectures like VGG19 and ResNet50.
    - **ViTs**: Emerging model in the landscape.
    - Task: Classify chest radiological images of infectious respiratory diseases.
  - Additional Investigation:
    - Custom convolution network architecture with features:
      - Spatial attention
      - Depth-wise convolutions
      - VGG and ResNet inspired blocks.

- **Outcomes**
  - **Vision Transformers**
    - Advancements recognized in general computer vision domain.
    - Limitations: Medical imaging, especially with dataset size and computational constraints.
  - **Traditional CNNs**
    - Demonstrated superior performance.
    - Benefit: Spatial hierarchies tailored for image data.
    - Custom CNN: Highlighted advantages of task-specific models.
  - **Model Interpretability**
    - Emphasis using:
      - Gradient-based visualization
      - Attention map methodologies.
    - Purpose: Transparent insight into model decisions.
    - Opportunity: Enhanced collaboration between ML models and medical professionals.

### Future Work

1. **Dataset Expansion** : ViTs typically perform better on larger datasets.
2. **Incorporation of Transfer Learning** : Incorporate other pre-trained models like inception, xception etc for a more robust and comprehensive comparison.
3. **Model Fusion and Ensembling** : Combine the strengths of CNNs and ViTs.
4. **Clinical Integration and Validation** : Collaborative efforts with radiologists.
5. **Enhanced Model Interpretability** : Beyond current methods, explore advanced interpretability techniques.
6. **Real-time Application** : Integrate models into real-time diagnostic platforms for radiologists.

##### **Overall**: Study highlighted both potential and challenges of CNNs and ViTs in medical imaging. Future direction involves continuous iteration, refinement, and enhanced collaboration.



```python

```
