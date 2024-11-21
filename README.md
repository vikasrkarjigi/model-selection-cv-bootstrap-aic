## Team Members
1. Ayyappa Boovadira Kushalappa **(A20555146)**
2. Binaal Bopanna Machanda Beliappa **(A20599896)**
3. Vikas Ravikumar Karjigi **(A20581687)**
4. Yashas Basavaraju **(A20586428)**

# Model Selection

## Overview
This project provides an implementation of two fundamental model selection techniques: So, the methods are the following two: **k-fold cross-validation** and **bootstrapping**. These methods are intended to assess and compare abilities of machine learning models on the data that they have never seen before, which may provide information on how well a particular model performs when confronted with new inputs, and how accurately a model can predict outcomes. The implementation is generic and not tied to any specific model and allows users to tune it to one’s needs and budgets by adjusting its parameters.

---

## Features

### K-Fold Cross-Validation
- **Flexible Configuration:** Determine the number of folds, i.e. the number of sets that will be used to estimate the model in order to tune the degree of bias/variance.`
- **Shuffling Option:** It is recommended that before splitting the dataset, it should be first shuffled, so as to minimize selection bias.
- **Reproducibility:** Optional Parameter: random_state as this is used to initialized the random number generator for reproduction of the partitions of the data for testing and evaluation.

### Bootstrapping
- **Custom Sampling:*Decrease the frequency and the size of the bootstrap samples in relation to the dataset you are testing.
- **With Replacement:** Any individual datum may also be repeated within one sample, allowing for more reliable resampling.
- **Reproducibility:** It is recommended you set a means of `random_state` for deterministic and reproducible purposes.

---

## Frequently Asked Questions

### 1. Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?
In simple cases such as linear regression, cross validation and boot strap can give out the same results as other simpler selection techniques such as AIC (Akaike Information Criterion).
- **Why They Align** :
AIC tries to compare the complexity of the model with the fitness of the model by using a penalty term that is the number of parameters in the model. AIC in simple models like the linear regression model frequently identifies models that eliminate overfitness in a model at the same time. In the same way, cross-validation estimates how the model performs on new data through an empirical evaluation of predictive accuracy of the model on a validation set which is generally more predisposed to choosing simple models when data is scanty or stochastic. Another technique that is used, often in combination with cross-validation, which involves re-sampling the dataset, and estimating the performance measure on different test sets, is also bootstrapping, which naturally prefers models of intermediate complexity and good fit.

- **Why They May Differ** :
The main weakness of AIC is that it depends on approximation of the log likelihood function of the model commonly used, which sometimes is assumed without necessarily having basis. However, cross validation and bootstrapping are more flexible methods and does not put such assumption like the above methods. Therefore in situations where these assumptions are violated (for instance when the error terms are non Gaussian) cross validation or boot strap could give results that are different from AIC.


### 2. In what cases might the methods you've written fail or give incorrect or undesirable results?
While cross-validation and bootstrapping are robust methods, they have limitations and can produce undesirable results under certain circumstances:
- **Imbalanced Datasets:**
Cross validation is potentially misleading in the situations where the number of instance in some folds is relatively small in case of the datasets with imbalanced classes. Because F1 measure averages the precision and recall rates this can make the model look better than everyone else in most of the majority class but worst in the minority class.

- **Computational Complexity:**
Despite this, both methods, in particularly in their generic form, require significant computational resources for large dataset or complex models and may be thus not suitable for real-time application or where computational power is limited.


### 3. What could you implement given more time to mitigate these cases or help users of your methods?
Several enhancements can be implemented to address the limitations and improve the robustness and usability of the methods:
- **Stratified K-Fold Cross-Validation:** It is valuable for balanced datasets but especially helpful when the imbalanced nature of the data should be preserved in each fold.
- **Early Stopping and Regularization:** Early stopping or penalization during training within cross-validation, some of the mechanisms can minimize overfitting risks especially within complex forms of the estimate model.


### 4. What parameters have you exposed to your users in order to use your model selectors?
The implementation provides flexibility by exposing the following parameters:
- **For K-Fold Cross-Validation:**
  - k: It shows the number of divisions that should be made to splits the dataset into folds. The default value for them is 5 or 10 that is depending on the number of records in the involved data set.
  - shuffle: Boolean indicating whether the block of data should be shuffled before dividing into folds. Default is True to eliminate selection bias which is otherwise known as risky bias.
  - random_state: Optional for the sake of reproducibility of the splits: an integer that is used to randomize the splits.

- **For Bootstrapping:**
  - n_samples: This is the number of Bootstrap samples to be produced as indicated in the balanced Bootstrap algorithm above. Default is 1000 so that the sample size is big enough for accurate estimation.
  - sample_size: The size of each bootstrap sample as a proportion of the full sample sizes. It is 1.0 by default, so every record in the dataset will be sampled with replacement.
  - random_state: A non-negative integer such that setting this argument repeatedly to the same value will make the resampling statistically reproducible.

The ability to configure these parameters, makes the methods versatile enough to handle a wide range of datasets and users needs for any other machine learning processes.

## Usage
This implementation is planned to be incorporated as a regular piece in any machine learning process. Parameters are available where the user may adjust to fit the need of the evaluation process.


### Installation

To get started with this project, first you need **Python 3.x**. Then follow these installation steps:

#### 1. Clone the Repository to your local machine:

```bash
git clone https://github.com/vikasrkarjigi/Project1.git
```
#### 2. Navigate to the project directory
```bash
cd ElasticNetModel
```
#### 3. You can install the required libraries using pip:
```bash
pip install numpy matplotlib
```
#### 4. Run the Test Script
```bash
python test_ElasticNetModel.py
```
This will run the test cases and print out the evaluation metrics and generate the plots.

### Basic Usage
To use the ElasticNet model, follow the example below:
```bash
from elasticnet.models.ElasticNet import ElasticNetModel
import numpy as numpy

# Create some synthetic data
X, y = numpy.random.rand(100, 3), numpy.random.rand(100)

# Initialize and fit the model
model = ElasticNetModel(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4)
model.fit(X, y)

# Predict using the trained model
predictions = model.predict(X)

# Evaluate the model
model.evaluate(X, y, "ElasticNet Model Synthetic Data")

```
### Test Case Outputs

Below are the screenshots showing the results of each test case included in this project:

- **test_predict()**: 
  Tests the model on a small dataset and verifies if the predictions are reasonable.
<div style="text-align: center;">
  <img src="elasticnet/images/small_test_dataset_output.png" alt="test_predict Output" width="700"/>
</div>

- **test_zero_variance_features()**: 
  Tests the model on a dataset where one feature has zero variance.
<div style="text-align: center;">
  <img src="elasticnet/images/zero_variance_dataset_output.png" alt="test_zero_variance_features Output" width="700"/>
</div>

- **test_highly_correlated_features()**: 
  Tests the model on highly correlated features to ensure it handles multicollinearity appropriately.
 <div style="text-align: center;">
  <img src="elasticnet/images/high_correlated_dataset_output.png" alt="test_highly_correlated_features Output" width="700"/>
</div>

- **test_sparse_data()**: 
  Tests the model on sparse data with many zero entries to check if regularization works correctly.
<div style="text-align: center;">
  <img src="elasticnet/images/sparse_dataset_output.png" alt="test_sparse_data Output" width="700"/>
</div>

- **test_with_outliers()**: 
  Tests the model on a dataset with outliers to ensure predictions remain reasonable.
<div style="text-align: center;">
  <img src="elasticnet/images/outliers_dataset_output.png" alt="test_with_outliers Output" width="700"/>
</div>

- **test_large_dataset()**: 
  Evaluates the model on a large dataset to ensure scalability and stability.
<div style="text-align: center;">
  <img src="elasticnet/images/large_dataset_output.png" alt="test_large_dataset Output" width="700"/>
</div>

- **test_different_alpha_l1_ratios()**: 
  Tests the model with various values for `alpha` and `l1_ratio` to evaluate the impact of these parameters on performance.
<div style="text-align: center;">
  <img src="elasticnet/images/different_alpha_l1ratios_dataset_output_1.png" alt="test_different_alpha_l1_ratios Output 1" width="700"/>
  <img src="elasticnet/images/different_alpha_l1ratios_dataset_output_2.png" alt="test_different_alpha_l1_ratios Output 2" width="700"/>
  <img src="elasticnet/images/different_alpha_l1ratios_dataset_output_3.png" alt="test_different_alpha_l1_ratios Output 3" width="700"/>
</div>

- **test_non_normalized_data()**: 
  Tests the model on non-normalized data to verify its ability to handle such inputs.
<div style="text-align: center;">
  <img src="elasticnet/images/non_normalized_dataset_output.png" alt="test_non_normalized_data Output" width="700"/>
</div>

Each screenshot corresponds to the results from the respective test case and provides a `visual representation` of the model's performance under various conditions.

---

### Additional Features

In addition to the basic requirements, the following enhancements have been implemented to go above and beyond what was asked:

1. **Comprehensive Model Evaluation**:
   - Every test checks model performance depending on **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)** and **R-Squared (R2)**.
   - Besides, the model makes **residual** and **scatter plots** for the purpose of visual evaluation of predictions and residuals to attained further understanding of the model.

2. **Extensive Test Coverage**:
   - Explained many use cases: zero variance features, perfectly correlated features, sparse data, outliers, large datasets and different values of `alpha` and `l1_ratio`.
   - All these test cases are integrated into the `test_ElasticNetModel.py` in order to ensure it follows the right behavior in several conditions.

3. **Tolerance for Early Stopping**:
   - As a stopping criterion, there is another parameter known as `tol` parameter has been implemented. If this change in coefficients in two successive iterations is less than this value, it means the convergence between iterations is so rapid and therefore the coordinate descent algorithm will stop early to enhance efficiency of computations.

4. **Flexibility in Regularization**:
   - Made `alpha` and `l1_ratio` parameters available to the users, enabling mass control of the **L1/L2** regularizing option and tuning of the model according to the particular application.
   - To demonstrate how alpha and l1_ratio influence the outcome and strength of regularization on the prediction capability of the model, various sets of alpha and l1_ratio have been used.
   
These additional features and enhancements make the model functional as well as efficient, flexible and adaptable for the different approaches that can be applied in data.

### Conclusion
If you have not heard of K-fold cross-validation and bootstrapping, you cannot overemphasize how valuable these techniques are when it comes to evaluating and selecting different machine learning models. These methods assist in giving reliable and impartial evaluation of how well a model or a system does and how preditably it shall generalize the new unseen data. K-fold cross-validation provides the guideline for every data disseminating its experiences to the training and testing phases, making the model to be fully overestimated and also underestimated. On the other hand, bootstrapping use resampling to assess the stability and accuracy of the model, it provides more insights of the model than the cross-validation. Both methods are free of restrictions and can be used together with virtually any datasets and models of analysis without any strict constraints.

However as with any other tool these approaches are not without their weaknesses. They can be slow for large data sets, whereas for small data sets or data sets with unequal proportions the performance can deteriorate. Gladly, most of these issues can easily be solved by employment of techniques such as, use of stratified sampling, adjusting fold or sample sizes, or simply running the simulations for a higher number of iterations to improve the accuracy. K-fold cross validation and Bootstrapping comes very handy when applied in the right manners; For any one working on the Machine learning it is a potent weapon to have from model building perspective.
