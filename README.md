# Sentiment Analysis on IMDb

*Sentiment* *analysis* on IMDb with the following steps, 

pre-process the data, select features, train several machine learning models and tune the parameters to improve the accuracy.

## Dataset Information

The dataset is located in *./IMDb* contains `training set`, `test set` and `development set` in txt format within which the reviews are divided into positive and negative polar.

## Requirements

There are several python libraries requirement for the project as follows.
* `numpy`
* `nltk`
* `scikit-learn`
* `matplotlib`
* `re`

To install the libraries, use the command line to deploy installation from [[PyPI]](https://pypi.python.org/pypi) using pip.

```python
    > pip install package_name
```

Or simply use command below if you have installed [[Anaconda]](https://www.anaconda.com/distribution/).
```python
    > conda install package_name
```


## Usage

To run the program, navigate to root directory of the project folder with terminal and use the command line with the following to run the program.
```python
    > python Sentiment_IMDb.py
```

## Process design

The process design of the whole program divided into several parts as the chart shown below. 


![image](README.assets/process%20design.jpg)

## Functions and Parameters

### Functions
* `preprocess`: read the dataset from path into dataset_file_full.
* `random_shuffle`: random shuffle the dataset_file_full and read the data into X and Y.
* `remove_html`: the pre-processer function of the vectorizer of scikit-learn to remove the html symbols.
* `train_classifier`: train the data set with the clf classifier model using chi-squared test.
* `get_res_test`: get the classification report of the test set.
* `get_res_dev`: get the accuracy of the development set.
### Parameters

* `num_features`: the max_features parameter of the vectorizer of scikit-learn.
* `num_features_chi2`: the number of features selected by chi-squared test.

## Running Error
Several running errors may encountered while running. Here is the solutions.
*  *[Ubuntu] [Python] MemoryError: Unable to allocate array with shape (x, x) and data type float64*
Open the terminal to run the following.
```
    $ sudo passwd root
    $ echo 1 > /proc/sys/vm/overcommit_memory
```

* *Process finished with exit code 137 (interrupted by signal 9: SIGKILL)* 
The program process was killed by the system due to exhaustion of CPU or RAM resources. 
Host with more resources required.
