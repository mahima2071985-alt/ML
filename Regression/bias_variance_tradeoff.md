# Bias–Variance Tradeoff – Complete Exam Notes
## AIML ZG565 | BITS Pilani

---

## 1. What is Bias–Variance Tradeoff?

Bias–Variance Tradeoff explains the balance between two sources of prediction error in a Machine Learning model:

1. **Bias**
2. **Variance**

The objective is to build a model that performs well on both training data and unseen data.

| Concept | Meaning |
|:---|:---|
| **Bias** | Error caused by overly simple assumptions in a model |
| **Variance** | Error caused by the model being too sensitive to its training data |
| **Tradeoff** | Balancing bias and variance to achieve good generalization |

---

## 2. Bias

Bias is the error caused by a model being too simple to capture the actual relationship in the data.

**Example:**

Actual relationship:

$$
Y = X^2
$$

But we use a Linear Regression model:

$$
\hat{y} = w_0 + w_1X
$$

A straight line cannot properly represent this curved relationship. This leads to **high bias**.

### Characteristics of High Bias
- Model is too simple.
- Cannot learn the actual pattern.
- Training error is high.
- Testing error is usually high.
- Leads to **underfitting**.

---

## 3. Variance

Variance measures how sensitive a model is to changes in its training data. A highly complex model may learn not only the actual pattern but also noise in the training data.

**Example:**

Using a Degree 10 Polynomial Regression model for a simple quadratic relationship:

$$
\hat{y} = w_0 + w_1X + w_2X^2 + \dots + w_{10}X^{10}
$$

The model may fit the training data extremely well but perform poorly on unseen data. This leads to **high variance**.

### Characteristics of High Variance
- Model is too complex.
- Learns noise in training data.
- Training error is very low.
- Testing error can be high.
- Leads to **overfitting**.

---

## 4. Underfitting

Underfitting happens when a model is too simple to learn the actual pattern in the data (e.g., using a straight line to represent a curved relationship).

### Characteristics
- High bias.
- Low model complexity.
- High training error.
- High testing error.
- Poor prediction performance.

### Solutions
- Increase model complexity.
- Add useful features.
- Choose a more suitable algorithm.

---

## 5. Overfitting

Overfitting happens when a model learns the training data too closely, including noise and unnecessary details.

### Characteristics
- High variance.
- High model complexity.
- Very low training error.
- High testing error.
- Poor generalization.

### Solutions
- Reduce model complexity.
- Use regularization (Ridge / Lasso).
- Collect more training data.
- Use cross-validation.
- Remove unnecessary features.

---

## 6. Good Fit

A good fit occurs when a model captures the actual pattern without learning unnecessary noise.

### Characteristics
- Reasonably low bias.
- Reasonably low variance.
- Low training error.
- Low testing error.
- Good generalization.

---

## 7. Comparison: Underfitting vs Good Fit vs Overfitting

| Feature | Underfitting | Good Fit | Overfitting |
|:---|:---:|:---:|:---:|
| **Model complexity** | Too simple | Appropriate | Too complex |
| **Bias** | High | Balanced | Usually low |
| **Variance** | Usually low | Balanced | High |
| **Training error** | High | Low | Very low |
| **Testing error** | High | Low | High |
| **Performance** | Poor | Good | Poor on unseen data |

### Exam Shortcut
- **High Bias** = Underfitting
- **High Variance** = Overfitting
- **Balanced Bias & Variance** = Good Generalization

---

## 8. Connection with Polynomial Regression

Consider the dataset:

| $X$ | $Y$ |
|:---:|:---:|
| 1 | 1 |
| 2 | 4 |
| 3 | 9 |
| 4 | 16 |
| 5 | 25 |

Actual relationship:

$$
Y = X^2
$$

We compare different polynomial degrees:

| Model | Equation | Training SSE |
|:---|:---|:---:|
| **Degree 1** | $w_0 + w_1X$ | 14 |
| **Degree 2** | $w_0 + w_1X + w_2X^2$ | 0 |
| **Degree 3** | $w_0 + w_1X + w_2X^2 + w_3X^3$ | 0 |

### Interpretation
- Degree 1 cannot represent the quadratic relationship exactly.
- Degree 2 represents the actual relationship exactly.
- Degree 3 can also represent the quadratic relationship, but has additional flexibility.

*Important: A higher-degree model does not automatically mean overfitting. We need to evaluate performance on unseen data.*

---

## 9. Numerical Example: Training MSE vs Testing MSE

Suppose we train three models and get these illustrative results:

| Model | Training MSE | Testing MSE | Conclusion |
|:---|:---:|:---:|:---|
| **Degree 1** | 25 | 30 | **Underfitting** (Both errors high) |
| **Degree 2** | 5 | 7 | **Good Fit** (Both errors low) |
| **Degree 10** | 0.5 | 28 | **Overfitting** (Train error low, test error high) |

---

## 10. Generalization Gap

Generalization Gap measures the difference between testing error and training error:

$$
\text{Generalization Gap} = \text{Testing MSE} - \text{Training MSE}
$$

Using the previous example:

| Model | Calculation | Generalization Gap |
|:---|:---:|:---:|
| **Degree 1** | $30 - 25$ | 5 |
| **Degree 2** | $7 - 5$ | 2 |
| **Degree 10** | $28 - 0.5$ | 27.5 |

*A large positive gap indicates overfitting. Generalization gap is a diagnostic measure.*

---

## 11. Bias–Variance Tradeoff Formula

For the traditional squared prediction error decomposition:

$$
\boxed{\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}}
$$

Where:
- **$\text{Bias}^2$:** Error caused by overly simple model assumptions.
- **$\text{Variance}$:** Error caused by sensitivity to the training dataset.
- **$\text{Irreducible Error}$:** Noise that cannot be eliminated by changing the model.

### Important Observation
As model complexity increases:
- **Bias** generally decreases.
- **Variance** generally increases.

The objective is to choose an appropriate model complexity that minimizes total expected prediction error.

---

## 12. How to Identify Overfitting and Underfitting?

Compare training error and testing error:

| Training Error | Testing Error | Interpretation |
|:---:|:---:|:---|
| High | High | **Underfitting** |
| Low | Low | **Good Fit** |
| Very low | High | **Overfitting** |

**Example:**  
Training accuracy = 98%, Testing accuracy = 60% $\longrightarrow$ **Overfitting**

---

## 13. How to Choose the Best Model?

1. Split dataset into training, validation, and testing sets.
2. Train models with different complexities.
3. Compare training errors.
4. Compare validation errors.
5. Select a suitable model based on validation performance.
6. Evaluate the final selected model using the test dataset.

*Important Exam Point: Use validation data or cross-validation for model selection. Do not repeatedly use the test dataset for model selection.*

---

## 14. How to Reduce Underfitting and Overfitting?

| Problem | Possible Solutions |
|:---|:---|
| **Underfitting** | Increase model complexity, Add useful features |
| **Overfitting** | Reduce model complexity, Apply regularization, Collect more training data, Use cross-validation |

*Regularization penalizes unnecessarily large model weights (e.g., Ridge and Lasso).*

---

## 15. Numerical Problem: Calculate MSE

### Given:

| House | Actual Price ($Y$) | Predicted Price ($\hat{Y}$) |
|:---:|:---:|:---:|
| A | 10 | 8 |
| B | 20 | 18 |
| C | 30 | 27 |

Find Mean Squared Error (MSE).

### Step 1: Formula

$$
\text{MSE} = \frac{\sum (Y - \hat{Y})^2}{n}
$$

### Step 2: Calculate Errors

| House | Actual $- \text{Predicted}$ | Squared Error |
|:---:|:---:|:---:|
| A | $10 - 8 = 2$ | $2^2 = 4$ |
| B | $20 - 18 = 2$ | $2^2 = 4$ |
| C | $30 - 27 = 3$ | $3^2 = 9$ |

### Step 3: Calculate SSE

$$
\text{SSE} = 4 + 4 + 9 = 17
$$

### Step 4: Calculate MSE

$$
\text{MSE} = \frac{17}{3} = \boxed{5.67}
$$

---

## 16. Important Exam Questions and Answers

- **Q1. What is Bias?**  
  Bias is the error caused by overly simple assumptions in a model.
- **Q2. What is Variance?**  
  Variance measures how sensitive a model is to changes in training data.
- **Q3. What is Underfitting?**  
  Underfitting occurs when a model is too simple to learn the actual pattern in the data.
- **Q4. What is Overfitting?**  
  Overfitting occurs when a model learns the training data too closely, including noise.
- **Q5. What happens when Bias is high?**  
  The model generally underfits.
- **Q6. What happens when Variance is high?**  
  The model generally overfits.
- **Q7. What is the objective of Bias–Variance Tradeoff?**  
  To balance bias and variance and achieve good generalization on unseen data.
- **Q8. Does lower training error always mean a better model?**  
  No. A model can have very low training error but high testing error because of overfitting.
- **Q9. What is Generalization Gap?**  
  The difference between testing error and training error ($\text{Testing Error} - \text{Training Error}$).
- **Q10. How can overfitting be reduced?**  
  By regularization, reducing model complexity, collecting more training data, and using cross-validation.

---

## 17. Practice Questions & Solutions

### Question 1
A model has Training accuracy = 99% and Testing accuracy = 65%. Identify the problem.  
**Answer:** Overfitting (High Variance).

### Question 2
A model is too simple and cannot capture the actual relationship. Which is high?  
A. Bias  
B. Variance  
C. Irreducible Error  
**Answer:** A. Bias.

### Question 3
A model has Training MSE = 8 and Testing MSE = 10. Calculate the Generalization Gap.  
**Answer:** $\text{Gap} = 10 - 8 = 2$.

### Question 4
What is the expected prediction error decomposition?  
**Answer:** $\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$.

### Question 5
Why should we not use testing data repeatedly for model selection?  
**Answer:** Repeating model selection on test data causes the model choices to overfit to the test set, making it an unreliable estimate of true generalization error.

---

## Final Revision Cheat Sheet

| Term | Remember |
|:---|:---|
| **Bias** | Error due to simple assumptions |
| **Variance** | Sensitivity to training data |
| **High Bias** | Underfitting |
| **High Variance** | Overfitting |
| **Good Fit** | Balanced bias and variance |
| **Training Error** | Error on training data |
| **Testing Error** | Error on unseen test data |
| **Generalization Gap** | Testing error $-$ Training error |
| **Regularization** | Helps control model complexity |
| **Cross-validation** | Helps evaluate model stability and select models |

### Main Formula:

$$
\boxed{\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}}
$$

*Remember for the exam: A good ML model should not simply memorize training data. It should generalize well to unseen data.*