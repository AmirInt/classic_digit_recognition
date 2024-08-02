# Classic Digit Recognition

The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centred in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this dataset and in this project, you will get a chance to experiment with the task of classifying these images into the correct digit using some of the methods you have learned so far.

We will try linear regression (just for the sake of seeing the failure), SVM, Multinomial softmax regression, dimensionality reduction using PCA and various kernels such as cubic and RBF on non-linear SVM and display the test errors.

## Run

Make sure you've installed the dependencies of the project:
```
pip3 install -r requirements.txt
```

Run `main.py` with option `help` to see a list of options to execute:
```
python3 main.py help
```