# Encoders
Testing different Encoding Schemes for Categorical variables

Categorical variables cannot be used in the Machine Learning Model, hence they need to be encoded. Categorical variables need to be encoded properly to get the most out of the data. There are many encoding techniques which can be used. Hence, this PR is an attempt to investigate various encoders for different types of data. So, when one needs to encode something, instead of exploring all the techniques, one can just go through the notebook and use top 3-5 encoders.
This PR adds the following notebooks:

Encoders - gives a general overview and implementation of the encoders investigated
Extracting data - used to create datasets with the metrics for each case
Analysis -  Analyze the results produced in the previous step

All the encoders are tested for Linear model (Linear Regression) and Tree based model(Random Forest) and are analysed using RMSE and MAE metrics.

Also, we took 3 datasets which have different dimensionality and have categorical variables of different cardinality.

#Note The purpose here is just to investigate encoders and hence no efforts has been put to increase the efficiency of the models. Instead, we are more interested in the percentage change of the metric that we want to optimize rather than optimizing the model.

Instructions - Install the reuirements using - pip install -r requirements.txt
