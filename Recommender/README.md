## HW4 : Movie Recommender System

### Installation

1. Unzip HW4_tekwani.zip
2. cd/HW4_tekwani
3. If you want to create a virtual environment, run virtualenv <the name of the environment, say 'tekwani'>
4. To begin using the virtual environment, you must activate it.
   $ source tekwani/bin/activate
5. Now install the packages specified in requirements.txt. You can do this using
   pip freeze > requirements.txt (freeze the current state of the environment)
   pip install -r requirements.txt
6. Install GraphLab. 
   I have an Education license and you can use my license key if you have to run my solution.
   Install GraphLab in the `tekwani` virtual environment using:
   
   pip install --upgrade --no-cache-dir https://get.graphlab.com/GraphLab-Create/2.1/btekwani@gmu.edu/4EAC-A0B8-1CC4-B8FB-97F6-26F7-78CE-A6AE/GraphLab-Create-License.tar.gz


### Running the solution

The folder HW4_tekwani/data must contain all the *.dat files as they are because the code references the `data` directory.

The folder HW4_tekwani/explore contains SQL scripts used for basic exploratory analysis referenced in my report. These scripts have been used to generate
the top_actors.txt and top_directors.txt in the `data` directory.

The file `recommender.py` is the model I used to submit the solution on the leaderboard.
You only need to run this file to get the final.txt file.


### Other files

1. `explore.py` creates a reduced training dataset - based on the thresholds set for the occurrence of the user and movie
2. `recenter.py` takes a prediction file and converts any rating marginally greater than 5.0 to 5.0 and ratings less than 1.0 to 1.0
3. `data_prep.py` creates the DataFrames used in the final model - all feature engineering and selection is done here. The result is `movies_sf.csv` which contains the feature
    matrix used for every model. 
4. `model1.py` through `model7.py` uses grid search and cross validation to get the best parameters for the FactorizationRecommender model. These models also use different
 combinations of side data - user and movies and evaluate model performance based on the presence of these features. Running each of these files can take anywhere from 1 hour to
  20 hours each - so this is not advisable.
5. I have included the output for some of my cross validation and model choices (model1 to model7) in files ending with the *.out extension. GraphLab writes the output to
   log files so I've only copied a few of these log files into *.out files.

 
