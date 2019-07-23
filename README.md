# Dog Breed Classification
A project intended to classify the breed of a dog based on an image input. The model, once trained and compiled, returns the predicted class of a dog image from an option of 120 different breeds.


### Required Dependencies
* Numpy
* Pandas
* Matplotlib
* Ski Kit Learn
* Keras


### Execution Instructions
1. Download the Stanford Dogs Dataset from Kaggle by following the link provided: https://www.kaggle.com/jessicali9530/stanford-dogs-dataset

2. Extract data from the .zip and .tar files.
~~~~
python setup.py
~~~~~~~~ 

3. Run the configuration file to initilaise the constants.
~~~~
python config.py
~~~~~~~~ 

4. Sort the raw images files into train, validation and test directories.
~~~~
python sort.py
~~~~~~~~ 

5. Create and train the neural network model.
~~~~
python model.py
~~~~~~~~ 

6. Generate predictions by passing in a file path to replace 'imageFileName' in the following command.
~~~~
python -i <imageFileName> predict.py
~~~~~~~~ 
