## HW3: Text Clustering

### Installation

1. Unzip HW3_tekwani.zip
2. cd/HW3_tekwani
3. If you want to create a virtual environment, run virtualenv <the name of the environment, say 'tekwani'>
4. To begin using the virtual environment, you must activate it.
   $ source tekwani/bin/activate
5. Now install the packages specified in requirements.txt. You can do this using
   pip freeze > requirements.txt (freeze the current state of the environment)
   pip install -r requirements.txt


### Running the solution

The folder HW3_tekwani/data must contain a input.mat and iris.csv file AS IT IS since the code references the `data` directory.


1. Run kmeans_text.py without any command line arguments. It will generate a predictions.txt file in the `predictions` folder.
2. Run kmeans_iris.py without any command line arguments. It will generate a iris_pred.txt file in the `predictions` folder.



### Other files

1. `predict.py`: A quick script I wrote to evaluate my Iris clustering better. I downloaded the Iris dataset off the UCI website and stored the original labels (the ground truth values)
separately. After running my kmeans_iris.py script, I compare my results with the true labels and get the homogeneity, completeness and silhouette scores to analyse how my k-Means model is performing.

2. `pca_10.pkl`: The PCA model pickled to save execution time. PCA with 10 components takes about 85-90 minutes on my system.
`pca_10.pkl_01.npy` through `pca_10.pkl_04.npy` are part files generated by `joblib`. These are needed to load `pca_10.pkl`.

3. `feature_selec.py` will just load the model so you don't have to. The actual model fitting in `feature_selec.py` is commented out.

4. `feature_selection.py`: It's the same as feature_selec.py but will fit the model again - PCA takes about 85-90 minutes to run.

