# Machine Learning – Linear Regression Notes
## AIML ZG565 | BITS Pilani

---

## 1. Introduction to Linear Regression

### Definition
Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values by finding the relationship between input variables ($X$) and output variables ($Y$).

**Example:** Predicting exam marks based on study hours.

### Dataset

| Study Hours ($X$) | Actual Marks ($Y$) |
|:---:|:---:|
| 1 | 2 |
| 2 | 4 |
| 3 | 5 |
| 4 | 4 |
| 5 | 5 |

### Objective
Find the best-fit straight line that minimizes the prediction error.

---

## 2. Linear Regression Equation

$$
\hat{y} = mx + c
$$

Where:
- $X$ = Input variable
- $Y$ = Actual output
- $\hat{y}$ = Predicted output
- $m$ = Slope
- $c$ = Intercept

---

## 3. Ordinary Least Squares (OLS)

OLS is a direct mathematical method used to find the best-fit regression line by minimizing the Sum of Squared Errors (SSE).

### Step 1: Calculate Mean

$$
\bar{x} = \frac{\sum x}{n} = \frac{15}{5} = 3
$$

$$
\bar{y} = \frac{\sum y}{n} = \frac{20}{5} = 4
$$

### Step 2: Calculate Slope

**Formula:**

$$
m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
$$

| $X$ | $Y$ | $X - 3$ | $Y - 4$ | Product | $(X - 3)^2$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2 | -2 | -2 | 4 | 4 |
| 2 | 4 | -1 | 0 | 0 | 1 |
| 3 | 5 | 0 | 1 | 0 | 0 |
| 4 | 4 | 1 | 0 | 0 | 1 |
| 5 | 5 | 2 | 1 | 2 | 4 |
| **Total** | | | | **6** | **10** |

$$
m = \frac{6}{10} = 0.6
$$

### Step 3: Calculate Intercept

$$
c = \bar{y} - m\bar{x}
$$

$$
c = 4 - (0.6 \times 3) = 2.2
$$

### Final Regression Equation

$$
\hat{y} = 0.6X + 2.2
$$

---

## 4. Prediction Using Regression Equation

Substitute each $X$ value into the regression equation: $\hat{y} = 0.6X + 2.2$

| $X$ | Calculation | Predicted $\hat{Y}$ |
|:---:|:---:|:---:|
| 1 | $0.6(1) + 2.2$ | 2.8 |
| 2 | $0.6(2) + 2.2$ | 3.4 |
| 3 | $0.6(3) + 2.2$ | 4.0 |
| 4 | $0.6(4) + 2.2$ | 4.6 |
| 5 | $0.6(5) + 2.2$ | 5.2 |

---

## 5. Error Calculation

Error measures the difference between actual and predicted values.

$$
\text{Error} = Y - \hat{Y}
$$

| $X$ | Actual $Y$ | Predicted $\hat{Y}$ | Error |
|:---:|:---:|:---:|:---:|
| 1 | 2 | 2.8 | -0.8 |
| 2 | 4 | 3.4 | 0.6 |
| 3 | 5 | 4.0 | 1.0 |
| 4 | 4 | 4.6 | -0.6 |
| 5 | 5 | 5.2 | -0.2 |

---

## 6. Sum of Squared Errors (SSE)

SSE measures the total squared difference between actual and predicted values.

$$
\text{SSE} = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2
$$

**Why square the errors?**
- To eliminate negative signs.
- To prevent positive and negative errors cancelling each other.
- To penalize larger errors more heavily.

**Calculation:**

$$
\text{SSE} = (-0.8)^2 + (0.6)^2 + (1.0)^2 + (-0.6)^2 + (-0.2)^2
$$

$$
\text{SSE} = 0.64 + 0.36 + 1.00 + 0.36 + 0.04 = 2.4
$$

**Objective:** Minimize SSE to find the best-fit line.

---

## 7. Mean Squared Error (MSE)

MSE is the average squared prediction error.

$$
\text{MSE} = \frac{\text{SSE}}{n}
$$

$$
\text{MSE} = \frac{2.4}{5} = 0.48
$$

---

## 8. Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

$$
\text{RMSE} = \sqrt{0.48} \approx 0.693
$$

*RMSE is expressed in the original output scale.*

---

## 9. Gradient Descent

### Definition
Gradient Descent is an iterative optimization algorithm that updates model parameters to minimize the cost function. Unlike OLS, which calculates the solution directly, Gradient Descent repeatedly updates slope and intercept.

### Cost Function

$$
J(m, c) = \frac{1}{2n} \sum (Y - \hat{Y})^2
$$

*The factor 2 makes derivative calculations simpler.*

### Step 1: Initialize Parameters
Assume $m = 0$ and $c = 0$.

Initial equation: $\hat{y} = 0X + 0$

### Step 2: Calculate Predictions

| $X$ | Actual $Y$ | Predicted $\hat{Y}$ |
|:---:|:---:|:---:|
| 1 | 2 | 0 |
| 2 | 4 | 0 |
| 3 | 5 | 0 |
| 4 | 4 | 0 |
| 5 | 5 | 0 |

### Step 3: Calculate Initial Cost

$$
\text{SSE} = 2^2 + 4^2 + 5^2 + 4^2 + 5^2 = 86
$$

$$
J = \frac{86}{2(5)} = 8.6
$$

*Purpose: This is the initial cost used as a baseline to evaluate future iterations.*

### Step 4: Calculate Gradients

Gradient for slope:

$$
\frac{\partial J}{\partial m} = -\frac{1}{n} \sum X(Y - \hat{Y})
$$

$$
\frac{\partial J}{\partial m} = -\frac{1}{5}(2 + 8 + 15 + 16 + 25) = -13.2
$$

Gradient for intercept:

$$
\frac{\partial J}{\partial c} = -\frac{1}{n} \sum (Y - \hat{Y})
$$

$$
\frac{\partial J}{\partial c} = -\frac{20}{5} = -4.0
$$

### Step 5: Update Parameters

Learning rate: $\alpha = 0.01$

Slope update:

$$
m_{\text{new}} = m_{\text{old}} - \alpha \frac{\partial J}{\partial m} = 0 - 0.01(-13.2) = 0.132
$$

Intercept update:

$$
c_{\text{new}} = c_{\text{old}} - \alpha \frac{\partial J}{\partial c} = 0 - 0.01(-4.0) = 0.04
$$

New equation:

$$
\hat{y} = 0.132X + 0.04
$$

---

## 10. Gradient Descent – Iteration 2

Previous values: $m = 0.132,\ c = 0.04$

### Step 1: Calculate Predictions
Equation: $\hat{y} = 0.132X + 0.04$

| $X$ | Predicted $\hat{Y}$ |
|:---:|:---:|
| 1 | 0.172 |
| 2 | 0.304 |
| 3 | 0.436 |
| 4 | 0.568 |
| 5 | 0.700 |

### Step 2: Calculate Errors

| $X$ | Actual $Y$ | Predicted $\hat{Y}$ | Error ($Y - \hat{Y}$) |
|:---:|:---:|:---:|:---:|
| 1 | 2 | 0.172 | 1.828 |
| 2 | 4 | 0.304 | 3.696 |
| 3 | 5 | 0.436 | 4.564 |
| 4 | 4 | 0.568 | 3.432 |
| 5 | 5 | 0.700 | 4.300 |

### Step 3: Calculate Gradients

Slope:

$$
\frac{\partial J}{\partial m} = -\frac{58.14}{5} = -11.628
$$

Intercept:

$$
\frac{\partial J}{\partial c} = -\frac{17.82}{5} = -3.564
$$

### Step 4: Update Parameters

$$
m = 0.132 - 0.01(-11.628) = 0.24828
$$

$$
c = 0.04 - 0.01(-3.564) = 0.07564
$$

New equation:

$$
\hat{y} = 0.24828X + 0.07564
$$

### Step 5: Calculate New Cost

$$
\text{SSE} = 54.1838
$$

$$
J = \frac{54.1838}{10} = 5.41838
$$

---

## 11. Gradient Descent – Cost Comparison

| Iteration | Cost ($J$) |
|:---|:---:|
| Initial | 8.60000 |
| Iteration 1 | 6.81007 |
| Iteration 2 | 5.41838 |

**Observation:** Cost decreases with each iteration, showing that the model is improving.

---

## 12. When to Stop Gradient Descent?

Gradient Descent continues updating parameters until convergence.

Common stopping conditions:
1. Cost reduction becomes very small.
2. Gradients become close to zero.
3. Maximum number of iterations is reached.

---

## 13. Important Formulas – Quick Revision

| Concept | Formula |
|:---|:---|
| Regression equation | $\hat{y} = mX + c$ |
| Slope (OLS) | $m = \frac{\sum (X - \bar{X})(Y - \bar{Y})}{\sum (X - \bar{X})^2}$ |
| Intercept | $c = \bar{Y} - m\bar{X}$ |
| Error | $E = Y - \hat{Y}$ |
| SSE | $\text{SSE} = \sum (Y - \hat{Y})^2$ |
| MSE | $\text{MSE} = \frac{\text{SSE}}{n}$ |
| RMSE | $\text{RMSE} = \sqrt{\text{MSE}}$ |
| Cost function | $J(m, c) = \frac{\sum (Y - \hat{Y})^2}{2n}$ |
| Gradient of $m$ | $\frac{\partial J}{\partial m} = -\frac{\sum X(Y - \hat{Y})}{n}$ |
| Gradient of $c$ | $\frac{\partial J}{\partial c} = -\frac{\sum (Y - \hat{Y})}{n}$ |
| Update $m$ | $m = m - \alpha \left(\frac{\partial J}{\partial m}\right)$ |
| Update $c$ | $c = c - \alpha \left(\frac{\partial J}{\partial c}\right)$ |

---

## 14. Exam Important Concepts

- **OLS** finds the best-fit line directly.
- **Gradient Descent** finds the best-fit line iteratively.
- **SSE** measures total squared error.
- **MSE** measures average squared error.
- **RMSE** gives error in the original output scale.
- **Learning rate ($\alpha$)** controls the size of parameter updates.
- **Gradient** indicates the direction and rate of cost change.
- **Convergence** means the algorithm has reached a stable minimum or stopping condition.
- **Lower cost** generally indicates a better fit on the same dataset.

---

### Progress Tracking

**Completed Topics:**
1. Introduction to Linear Regression
2. Regression Equation
3. OLS – Direct Solution
4. Prediction
5. Error Calculation
6. SSE, MSE and RMSE
7. Gradient Descent – Initialization
8. Gradient Calculation
9. Parameter Updates
10. Iterations 1 and 2
11. Convergence

**Next Topics:**
1. Batch Gradient Descent
2. Stochastic Gradient Descent
3. Mini-batch Gradient Descent
4. Multiple Linear Regression
5. Polynomial Regression
6. Ridge and Lasso Regression
7. Linear Basis Function Models
8. Bias-Variance Tradeoff
