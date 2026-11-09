# Machine Learning – Gradient Descent Types
## AIML ZG565 | BITS Pilani

---

## 1. Introduction

Gradient Descent is an optimization algorithm used to minimize the cost function by updating model parameters.

There are three main types:
1. **Batch Gradient Descent (BGD)**
2. **Stochastic Gradient Descent (SGD)**
3. **Mini-batch Gradient Descent**

---

## 2. Common Dataset

We use the following dataset for all three methods.

| Student | Study Hours ($X$) | Marks ($Y$) |
|:---:|:---:|:---:|
| A | 1 | 2 |
| B | 2 | 4 |
| C | 3 | 5 |
| D | 4 | 4 |
| E | 5 | 5 |

### Initial Configuration

Regression equation:

$$
\hat{y} = mx + c
$$

Initial parameters:

$$
m = 0, \quad c = 0
$$

Learning rate:

$$
\alpha = 0.01
$$

---

## 3. Batch Gradient Descent (BGD)

### Definition
Batch Gradient Descent uses the **entire training dataset** to calculate gradients and updates model parameters once per iteration.

### Step 1: Calculate Predictions
Initial equation: $\hat{y} = 0X + 0$

| $X$ | Actual $Y$ | Predicted $\hat{Y}$ |
|:---:|:---:|:---:|
| 1 | 2 | 0 |
| 2 | 4 | 0 |
| 3 | 5 | 0 |
| 4 | 4 | 0 |
| 5 | 5 | 0 |

### Step 2: Calculate Errors

$$
\text{Error} = Y - \hat{Y}
$$

Errors: $2,\ 4,\ 5,\ 4,\ 5$

### Step 3: Calculate Gradients

Slope gradient:

$$
\frac{\partial J}{\partial m} = -\frac{1}{n} \sum X(Y - \hat{Y}) = -\frac{1}{5}(2 + 8 + 15 + 16 + 25) = -13.2
$$

Intercept gradient:

$$
\frac{\partial J}{\partial c} = -\frac{1}{n} \sum (Y - \hat{Y}) = -\frac{1}{5}(2 + 4 + 5 + 4 + 5) = -4.0
$$

### Step 4: Update Parameters

Slope:

$$
m_{\text{new}} = m_{\text{old}} - \alpha \frac{\partial J}{\partial m} = 0 - 0.01(-13.2) = 0.132
$$

Intercept:

$$
c_{\text{new}} = c_{\text{old}} - \alpha \frac{\partial J}{\partial c} = 0 - 0.01(-4.0) = 0.04
$$

New regression equation:

$$
\hat{y} = 0.132X + 0.04
$$

*Important: All training samples are processed before performing one parameter update.*

---

## 4. Stochastic Gradient Descent (SGD)

### Definition
Stochastic Gradient Descent updates model parameters after processing **each individual training sample**. Unlike Batch GD, it does not wait for the entire dataset.

### Step 1: Initialize Parameters

$$
m = 0, \quad c = 0, \quad \alpha = 0.01
$$

### Step 2: First Student (A)

Given: $X = 1,\ Y = 2$

Prediction:

$$
\hat{y} = 0(1) + 0 = 0
$$

Error:

$$
e = 2 - 0 = 2
$$

Slope gradient:

$$
-X(Y - \hat{Y}) = -1(2) = -2
$$

Intercept gradient:

$$
-(Y - \hat{Y}) = -2
$$

### Step 3: Update Parameters

$$
m = 0 - 0.01(-2) = 0.02
$$

$$
c = 0 - 0.01(-2) = 0.02
$$

### Step 4: Second Student (B)

Given: $X = 2,\ Y = 4$

Use the updated equation: $\hat{y} = 0.02X + 0.02$

Prediction:

$$
\hat{y} = 0.02(2) + 0.02 = 0.06
$$

Error:

$$
e = 4 - 0.06 = 3.94
$$

Slope gradient:

$$
-2(3.94) = -7.88
$$

Intercept gradient:

$$
-3.94
$$

### Step 5: Update Again

$$
m = 0.02 - 0.01(-7.88) = 0.0988
$$

$$
c = 0.02 - 0.01(-3.94) = 0.0594
$$

*Important: Every sample produces an immediate parameter update. The next sample uses the latest updated parameters.*

---

## 5. Mini-batch Gradient Descent

### Definition
Mini-batch Gradient Descent divides the training dataset into **smaller groups (batches)** and updates parameters after processing each group.

Assume: $\text{Batch Size} = 2$

### Step 1: Divide Dataset

| Batch | Students |
|:---:|:---:|
| Batch 1 | A, B |
| Batch 2 | C, D |
| Batch 3 | E |

### Step 2: Initialize Parameters

$$
m = 0, \quad c = 0
$$

### Step 3: Calculate Gradients for Batch 1

Students A and B:

| $X$ | $Y$ | Error ($Y - \hat{Y}$) |
|:---:|:---:|:---:|
| 1 | 2 | 2 |
| 2 | 4 | 4 |

Slope gradient:

$$
=-\frac{1}{2}\left[1(2) + 2(4)\right] = -\frac{10}{2} = -5
$$

Intercept gradient:

$$
=-\frac{1}{2}(2 + 4) = -3
$$

### Step 4: Update Parameters

$$
m = 0 - 0.01(-5) = 0.05
$$

$$
c = 0 - 0.01(-3) = 0.03
$$

New equation:

$$
\hat{y} = 0.05X + 0.03
$$

*Batch 1 is completed. The next batch uses these updated parameters.*

---

## 6. Comparison of Three Methods

| Feature | Batch GD | SGD | Mini-batch GD |
|:---|:---:|:---:|:---:|
| **Data per update** | Entire dataset | One sample | Small group |
| **Update frequency** | Once per epoch | Every sample | Every batch |
| **Computation per update** | High | Low | Moderate |
| **Memory requirement** | High | Low | Moderate |
| **Gradient behaviour** | Stable | Noisy | Relatively stable |

---

## 7. Important Terminology

- **Epoch:** One complete pass through the entire training dataset.
- **Batch Size:** Number of training samples used for one parameter update.
- **Iteration:** One parameter update.

---

## 8. Numerical Example – Number of Updates

Given:
- Total training samples = 1000
- Batch size = 100

**Batch Gradient Descent:**

$$
\text{Updates per epoch} = 1
$$

**Stochastic Gradient Descent:**

$$
\text{Updates per epoch} = 1000
$$

**Mini-batch Gradient Descent:**

$$
\text{Updates per epoch} = \frac{1000}{100} = 10
$$

---

## 9. Advantages and Disadvantages

### Batch Gradient Descent
- **Advantages:** Stable gradient calculation; uses the entire dataset.
- **Disadvantages:** Requires more memory; parameter updates are less frequent.

### Stochastic Gradient Descent
- **Advantages:** Frequent parameter updates; low memory requirement.
- **Disadvantages:** Noisy updates; cost may fluctuate.

### Mini-batch Gradient Descent
- **Advantages:** Balances computational efficiency and update frequency; suitable for large datasets; standard in deep learning.
- **Disadvantages:** Batch size needs tuning; gradient updates are slightly less stable than full-batch.

---

## 10. Exam Quick Revision

| Question | Answer |
|:---|:---|
| Uses entire dataset per update? | Batch GD |
| Uses one sample per update? | SGD |
| Uses small groups per update? | Mini-batch GD |
| Updates after every sample? | SGD |
| Updates after every batch? | Mini-batch GD |
| What is an epoch? | One complete dataset pass |
| What is batch size? | Samples per update |
| What is an iteration? | One parameter update |

---

## 11. Important Formulas

### Cost Function

$$
J(m, c) = \frac{1}{2n} \sum (Y - \hat{Y})^2
$$

### Batch Gradients

$$
\frac{\partial J}{\partial m} = -\frac{1}{n} \sum X(Y - \hat{Y})
$$

$$
\frac{\partial J}{\partial c} = -\frac{1}{n} \sum (Y - \hat{Y})
$$

### Parameter Updates

$$
m_{\text{new}} = m_{\text{old}} - \alpha \frac{\partial J}{\partial m}
$$

$$
c_{\text{new}} = c_{\text{old}} - \alpha \frac{\partial J}{\partial c}
$$

*For SGD, set $n=1$ for one sample. For Mini-batch GD, use the samples in the selected batch and average their gradients.*

---

### Progress Tracking

**Completed Topics:**
1. Linear Regression Introduction
2. OLS – Direct Solution
3. Prediction and Error Calculation
4. SSE, MSE and RMSE
5. Gradient Descent
6. Gradient Calculation
7. Parameter Updates
8. Batch Gradient Descent
9. Stochastic Gradient Descent
10. Mini-batch Gradient Descent

**Next Topics:**
1. Multiple Linear Regression
2. Polynomial Regression
3. Ridge Regression
4. Lasso Regression
5. Linear Basis Function Models
6. Bias-Variance Tradeoff
