# Similar Movie Poster Retrieval and Year Prediction

In this project, given one movie poster, we will predict 3 similar looking movie posters from the dataset. We use ResNet50 model to generate embeddings and KNN Regressor to obtain similar movie posters. 

### ResNet50

ResNet50, short for "Residual Network with 50 layers," is a groundbreaking deep learning architecture that has revolutionized computer vision. Key points include:

ResNet50's innovation lies in skip connections, enabling training of extremely deep neural networks.
Widely used for image classification and object recognition tasks.
Residual blocks help mitigate the vanishing gradient problem, allowing for the training of deep networks.
Pre-trained versions are available for transfer learning on various datasets.
Exceptional accuracy in identifying objects and patterns in images.
A cornerstone in modern computer vision research and applications.
Utilize ResNet50 to achieve state-of-the-art results in image classification and object recognition. 🌟🖼️🤖 

### K-Nearest Neighbours

KNN Regressor is an adaptable machine learning algorithm for regression tasks. It predicts values based on the similarity of data points, making it suitable for various applications. Key features include:

No assumptions about data distribution.
Use of a distance metric (e.g., Euclidean) to measure similarity.
Tunable hyperparameter 'k' for perforxmance optimization.
Local approximation for non-linear relationships.
Evaluated using metrics like MAE, MSE, and R2 score.


### Model Implementation

Initially, we are using a pretrained version of ResNet50 model which is trained on more than a million images from the ImageNet database. We will use the embeddings calculated from this model to predict similar movie posters using K-Nearest Neighbours. Later, we will use a basic Neural Network to build a predictive model to obtain the year of the movie based on poster. After that, we will finetune the model and compare the final results.

### Final Results

The pretrained model performed well with a small train and test loss. Although, finetuned model showed signs of overfitting and the loss was comparatively high than the base model. Also, while finetunning we used multiple layers while earlier we only used one layer. This can be the reason for making the model complex and moving towards, overfitting.

