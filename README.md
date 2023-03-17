# deep-learning-challenge

Source Data: https://github.com/xnotynot/deep-learning-challenge/blob/main/Resources/charity_data.csv

Primary Project Notebook: https://github.com/xnotynot/deep-learning-challenge/blob/main/adv_ml_fund_analysis.ipynb

Optimization Notebook: https://github.com/xnotynot/deep-learning-challenge/blob/main/adv_ml_fund_analysis_optimization.ipynb

Saved Models: https://github.com/xnotynot/deep-learning-challenge/tree/main/Results

## Overview

This project was designed to help create a predictive model to determine whether applicants for aid from the non-profit group Alphabet Soup will successfully use the funding they request.

With a sample size of approximately 34K records of historical data, it uses a neural net deep learning model to make a binary prediction of funding success

Tensorflow Keras were used to build and compile a neural net model based

The initial model's accuracy fell slightly below 75%, further attempts were made to optimize the model to increase its accuracy.

## Variables

>EIN and NAME—Identification columns<br>APPLICATION_TYPE—Alphabet Soup application type<br>AFFILIATION—Affiliated sector of industry<br>CLASSIFICATION—Government organization classification<br>USE_CASE—Use case for funding<br>ORGANIZATION—Organization type<br>STATUS—Active status<br>INCOME_AMT—Income classification<br>SPECIAL_CONSIDERATIONS—Special consideration for application<br>ASK_AMT—Funding amount requested<br>IS_SUCCESSFUL—Was the money used effectively

## Results

- **Preprocessing**
 1. The only target variable in the dataset is `IS_SUCCESSFUL`.
 2. The features which contribute to the analysis include: 
 `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, and `ASK_AMT`.
 3. `EIN` and `NAME` are both identifications for the specific businesses that received funding in the past. As such, they do not contribute directly to the success of the funding, they are neither target nor features.

- **Compiling, Training, and Evaluating the Model**
 1. Since we have high number of inputs, choosing 3 hidden layers seemed optimal
 
 2. Later it was changed to 4 layers having little impact on the model's accuracy.

 3. For the number of neurons, I went with the rule-of-thumb stating that the number should be less than twice the size of the input layer, with that leading to the number of 80 neurons for the first layer. 
 4. For the second hidden layer, used 30 neurons, which is fewer than the number of inputs.
 5. I used ReLU as the method for both hidden layers and sigmoid for the output layer as learned from the sessions (need to do more research to understand the significance of this)
 6. None of the models could reach the target accuracy of 75%. The peak value was close to 73%  
 7. The 3 different methods to increase the performance of my model. 
    * By dropping the `STATUS` and `SPECIAL_CONSIDERATIONS` variables to see if they were reducing the effectivity of the analysis.
    * By adding another hidden layer between my two original hidden layers with 60 neurons.
    * By doubling the number of neurons in each hidden layer
<br>
None of these methods yielded positive results.

## Summary
Overall, the models never reached the target accuracy of 75%. The best iteration yielded an accuracy close at 73%.

It is possible that we could have eliminated other features to improve model accuracy.  Determining which features are important could be done via the feature analysis with confusion matrix.
