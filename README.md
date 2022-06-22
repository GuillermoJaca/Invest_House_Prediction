# Predict House Prices to invest in Germany


4 Notebooks with their utils:

1. 'Analysis_Data.ipynb' where I analyse the data, clean the data and create new features. For the final pipeline I create the script 'clean_pipeline.py'. From which the train-val data and the test data where I will test my final solution.

2. 'Task.ipynb' where I make decisions of features based on CV scores, select the best features, select model and its hyperparameters. I leave ensembling as an extension.

More details of ehat I do in 'Task.ipynb':

Data Modelling in order to create the best model with the best features according to a Cross-Validation strategy. I carry out this process in which I visualize the error and count in all features to 'see' in which intervals of each feature the model is permorming poorly. I additionally look at how many examples there are in each class, because I suspected that the biggest error might come from categories in which we dont have a lot of data. E.g To predict the price of a haus in a small village, where we don't have more data, will be way more difficult than from the center of Berlin.
Additionally, in order to add extra value to real state inversors, I predict the expected error, so that investors have more information and know the confidence of the predictions. They do not know how the model work internally. What we are really optimizing is the money that they earn (not just the price of the haus) so, by investing more money (ot buying more hauses) of the ones in which the confidence is higher will yield much better results for our customers than buying without that information. We can additionally , with Shap values explain, why the prediction might have low confidence. i.e by looking at the Shap values of the features of each prediction. Then the investors can incorporate also unique information that they might have. Let's me explain this: If a real estate investor have very good information about the prices in properties in a small village, and our model yield a bad confidence because of the feature count of properties in the village, then the real estate investor can make decisions incorporating this extra information that he/she has.

3. 'Test_prediction.ipynb' where I evaluate the results in test set.

4. 'nlp.ipynb' where I use NLP to predict price of hauses.

I included also an illustrative example of pytest, and a Docker file.


