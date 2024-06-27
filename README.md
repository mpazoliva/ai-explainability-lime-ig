The goal of this project is to assess the performance of two different explainability approaches, perturbation-based versus gradient-based, on a ResNet18 CNN classifier trained on the CUB-200 dataset. Particularly, Local Interpretable Model-agnostic Explanations and Integrated Gradients methods were used. The analysis showed that both methodologies created equally accurate explanations, although in terms of interpretability and efficiency there are important differences: being LIME easier to implement and interpret but more computationally expensive, while IG offered a more computationally efficient alternative, though it was less straightforward to interpret.

1. Introduction

In the field of Machine Learning (ML), different algorithmic and statistical models are developed to enable computers to learn from and make predictions or decisions based on data. In this sense, instead of being explicitly programmed to perform a task, the model is provided with data, and it automatically identifies patterns and make decisions based on them. This is good because it allows the automatic processing of big amounts of information with minimum human intervention and usually has very good results. But in this path, sometimes, the interpretability of why and how a model is making a decision gets lost.
Some models are interpretable and can provide insights into their decision-making processes, while others are less transparent, typically call black-box models. That is why, the goal of Explainable Artificial Intelligence (XAI) is to provide insights into how ML models make decisions, especially in complex or high-stakes applications, where understanding the reasoning behind a decision is crucial. The goal is to make the inner workings and decisions of machine learning models more understandable and transparent.
But, what is an explanation? An explanation in XAI is a description or visualization that makes the model's behavior more understandable to humans. There are different ways to categorize explanations: factual (explains why a particular prediction was made) or counterfactual (explores what changes to the input data would have led to a different model prediction); global (provides an overview of how a model works across the entire dataset) or regional (focuses on a subset of the data or a specific region of the input space); at attribute level (break down a prediction by highlighting the importance of individual features or attributes in the input data) or at example level (provide insights into why a particular prediction was made for a specific data point). 
Also, in the field of XAI there are many approaches. Particularly in this project, two factual, regional and attribute level explanations will be explored and compared:
Perturbation-based techniques: create many variations of the input data and see how the model's predictions change to explain what features are important. They work with any model but are slower because they need to test many inputs.
Gradient-based techniques:   use the model's gradients, which show how changes in input affect the output, to explain important features directly. They are faster because they don't need extra data testing, but they only work with models where gradients can be calculated.
In a nutshell, the purpose of this report is to implement and compare the behavior of these two explanation techniques on a CNN classifier trained on the CUB-200 dataset. 

2. Data.
   
Bird species classification is notably challenging due to high intraclass variance caused by variations in lighting, background, and pose, as well as the subtle visual differences between some species. These factors can result in difficulties for both human and machine classifiers, sometimes leading to misclassifications or performance based on irrelevant features (as seen in the “Clever Hans” phenomenon). For all this, the CUB-200-2011 dataset is ideal for testing XAI approaches, as it provides a rich set of annotations that enable detailed analysis. The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset is a comprehensive collection of 200 bird species classification. It contains a total of 11,788 images with annotations (15 part locations, 312 binary attributes, 1 bounding box). Each species is associated with a Wikipedia article and is organized by scientific classification. The images were harvested from Flickr and annotated using Mechanical Turk, providing bounding boxes, part locations, and binary attribute labels that describe visual characteristics like color, pattern, and shape. The dataset’s extensive annotations include the pixel locations and visibility of specific bird parts.
For the purpose of this project, the data was downloaded from the CUB-200-2011 website. The folder structure had various metadata files and the images were organized in subdirectories. In the code, the CUB200Dataset class was load as a custom PyTorch Dataset to allow efficiently feed data into the CNN. This class would: load the images and their labels, apply transformations to the images (resize, normalize, etc.), and provide an efficient way to access each image and its label using indices. Then, the loader function reads and processes metadata files from the CUB-200 dataset to split image file paths and labels into training and testing sets.
The transformations applied were: resizing (resizes image to 224x224 pixels, as pre-trained CNNs expect input images to have consistent size); converting the images to PyTorch tensors (as PyTorch models expect inputs to be in this format); normalization (normalizes the tensors with the given mean and standard deviation values for each channel RGB). All this standardize the inputs, making training more stable and faster. 

3. Methods
   
After loading data, this project needed of three steps to complete the goal: 1. train a CNN, 2. implement XAI approaches, 3. use the approaches on the CNN and compare their results. 

3.1. The model: modified ResNet 18 CNN.
	To simplify the work, as it was not the main goal of this project, the CNN classifier was build using a pre-trained Residual Network with 18 layers model. ResNet18 is a popular CNN architecture used for image classification tasks. It is known for its efficiency and effectiveness in learning complex patterns in images. The key feature of ResNet18 is its use of “residual connections” which help the network learn better and faster. However, some modifications were made. The original ResNet18 model is pre-trained on the ImageNet dataset, which has 1000 classes; I modified the final layer of ResNet18 to adapt it to the CUB-200 dataset, which has 200 bird species. 
	Once the model was defined on the CUBResNet class, I implemented a training and evaluation function. Then set up the device, model, loss function, and optimize. Model was trained and evaluated. As it can be observed in the code, after 9 epochs the model loss reduced to 0.1096. On the evaluation, test loss was 2.6222 and accuracy 0.4424. Even though these are not great results for a model I am going to consider them as enough for the goal of this project, as the main purpose was not training a CNN.
3.2. The XAI Approaches: LIME and Integrated Gradients
To develop the XAI methods and explore their capabilities from two different perspectives, I compared perturbation-based and gradient-based algorithms, both from primary attribution (evaluates the contribution of each input feature to the output of a model).

3.2.1 Perturbation-based: Local Interpretable Model-agnostic Explanations (LIME)
LIME is an interpretability method that trains an interpretable surrogate model by sampling data points around a specified input example and using model evaluations at these points to train a simpler interpretable 'surrogate' model, such as a linear model. LIME explains the predictions of any classifier in an interpretable and faithful manner by learning an interpretable model locally around the prediction.

Implementation Steps:
Sampling: generates a set of perturbed samples around the input image.
Prediction: for each perturbed sample, it obtains predictions from the original model.
Weighting:weights these samples based on their similarity to the original image.
Training: trains an interpretable model on these weighted samples.
Explanation: the coefficients of the interpretable model are used to highlight the important features (superpixels) in the image.

Code: to generate LIME explanations, I used the lime library, an open-source library designed to explain the predictions of machine learning models.

Example of LIME explanation:
Here, we can see a bird image with label “0” from the CUB-200 dataset. The overlay with yellow boundaries and colored regions is the LIME explanation, which responds to:

Yellow Boundaries: these lines are segmentations or superpixels created by the LIME algorithm. LIME segments the image into interpretable regions and perturbs these segments to understand their impact on the model's predictions.
Colored Regions: the color intensity within these superpixels indicates the importance of each region in the prediction. Regions that contribute positively are highlighted in warm colors (like yellow), while those that detract from the prediction are shown in cooler colors (like blue).

In conclusion, the colored regions and their intensities show which parts of the image were most significant for the model's prediction of the label "0". In this case, the regions around the bird's head and beak seem to be highlighted, indicating that these areas were significant in predicting the label.

3.2.2 Gradient-based: Integrated Gradients
Integrated Gradients is a gradient-based method that attributes the prediction of the model to its input features by considering the integral of the gradients. Integrated gradients represent the integral of gradients with respect to inputs along the path from a given baseline to input. The integral can be approximated using a Riemann Sum or Gauss Legendre quadrature rule. Formally, it can be described as follows:

Implementation Steps:
Initialization: initialize the Integrated Gradients object with the model.
Attribution Calculation: calculate the attributions using the attribute method.
Visualization: visualize the attributions as a heatmap.

Code: to generate Integrated Gradients explanations, I used Captum library, an open-source extensible library for model interpretability built on PyTorch. The attribution algorithms in Captum are methods used to determine the contribution of different input features, neurons, or layers to the final output of a model. 

Example of Integrated Gradients explanation:
Here, we can see the same bird image with label “0” from the CUB-200 dataset, but in this case with IG explanation. This responds to a color map where darker colors (black, purple) represent lower attribution values, and brighter colors (yellow, white) represent higher attribution values.

The visualization shows very sparse and faint highlights, indicating that only a few pixels have high attribution values, and most of the image has low attribution. The brighter regions are likely the areas where the model focused on to make the prediction. These should correspond to important features of the bird in the image, such as the head and beak.

3.3 Comparison: using the two approaches on the CNN.
The methodology for comparing the two explainability approaches will be based on their feedback when actually assessing the models output: how well the explanations they generate align with the known, ground-truth features of the images. With each method, I will analyze the overlap between the model's output of most significant features and the annotations on the data. For this, the comparison will be quantified using the Intersection over Union (IoU),  a standard metric used to measure the accuracy of an object detector on a particular dataset, as it measures the overlap between two bounding boxes: the ground truth bounding box (the actual region of interest in the image according to annotation) and the prediction bounding box (the region identified by the model).  It's calculated by dividing the area of overlap between the predicted bounding box and the ground truth bounding box by the area of their union.

Also, for a deeper analysis, these two aspects will be considered: 
Interpretability: I will generate visual explanations for a random subset of images from the test dataset with both methods and evaluate the interpretability based on the clarity and intuitiveness of the explanations provided by each method.
Computational Efficiency: I will measure the average time taken to generate explanations by each method. 

4. Results.

Is the model Clever Hans? 
The IoU scores for a subset of 10 images showed the same results for both LIME and IG explanations. This means that the explanations detected by both approaches agree equally with the annotations on the data,  both systems are identifying similar regions as important. Higher IoU scores suggest better alignment and, hence, that the model is paying attention to the correct features, and lower scores suggest the opposite. In this case, the IoU scores vary across the images, indicating that the model's attention can sometimes closely align with the ground truth (e.g., Image 9) and other times not as much (e.g., Image 3).

Interpretability of Explanations
A subset of ten images were randomly chosen to generated visual explanations from each method (see appendix 1). This provided insights into the important features contributing to the model's predictions. On one hand, LIME explanations produced superpixel-based visualizations, where specific regions of the image were highlighted. The interpretability was high as the superpixels provided clear and distinct regions that contributed to the predictions. On the other hand, Integrated Gradients Explanations produced heatmap-based visualizations with continuous pixel-wise attributions. The interpretability was moderate to high, with heatmaps showing the gradient of attributions across the image.

Computational Efficiency
In general, the average time taken to generate LIME explanations (90.16 seconds) was higher due to the need for generating perturbed samples and training a surrogate model for each explanation. While average time for Integrated Gradients was lower ( 53.37 seconds), as it only involved calculating gradients and integrating them over a baseline. 

5. Discussion
To conclude, this analysis shows that both methods generated equally accurate explanations. However, a further analysis showed some differences that can count as advantages or disadvantages: in terms of interpretability and efficiency, LIME provided more interpretable visualizations but it was computationally more expensive, whereas, Integrated Gradients were faster but produced more diffuse attributions. In conclusion, both LIME and Integrated Gradients have their strengths and weaknesses and the choice between these methods depends on the specific requirements of interpretability and computational resources available for the task.
