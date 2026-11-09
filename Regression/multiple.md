# Multiple Linear Regression using Gradient Descent
## AIML ZG565 | BITS Pilani

---

## 1. What is Multiple Linear Regression?

Multiple Linear Regression is used to predict a dependent variable using two or more independent variables.

**Example:** Predicting house price using:
- House size
- Number of bedrooms

### Model Equation

$$
\hat{y} = b_0 + b_1x_1 + b_2x_2
$$

Where:

| Symbol | Meaning |
|:---:|:---|
| $\hat{y}$ | Predicted output |
| $b_0$ | Intercept |
| $b_1$ | Coefficient of feature 1 |
| $b_2$ | Coefficient of feature 2 |
| $x_1$ | House size |
| $x_2$ | Number of bedrooms |

The objective is to find the best values of $(b_0, b_1, b_2)$.

---

## 2. What is Gradient Descent?

Gradient Descent is an optimization algorithm used to find coefficient values that minimize the cost function.

It works by:
1. Initializing coefficients.
2. Calculating predictions.
3. Calculating errors.
4. Calculating cost.
5. Calculating gradients.
6. Updating coefficients.
7. Repeating until convergence.

---

## 3. Dataset

| House Size ($X_1$) | Bedrooms ($X_2$) | Actual Price ($Y$) |
|:---:|:---:|:---:|
| 1 | 1 | 10 |
| 2 | 1 | 15 |
| 3 | 2 | 20 |
| 4 | 2 | 25 |
| 5 | 3 | 30 |

Number of observations:

$$
n = 5
$$

---

## 4. Initialize Coefficients

Initial values:

$$
b_0 = 0, \quad b_1 = 0, \quad b_2 = 0
$$

Learning rate:

$$
\alpha = 0.01
$$

Initial model:

$$
\hat{y} = 0 + 0x_1 + 0x_2
$$

*Therefore, all initial predictions are zero.*

---

## 5. Prediction and Error

Prediction formula:

$$
\hat{y} = b_0 + b_1x_1 + b_2x_2
$$

Error formula:

$$
\text{Error} = \hat{y} - Y
$$

### Initial Predictions and Errors

| House | Actual ($Y$) | Prediction ($\hat{y}$) | Error ($\hat{y} - Y$) |
|:---:|:---:|:---:|:---:|
| A | 10 | 0 | -10 |
| B | 15 | 0 | -15 |
| C | 20 | 0 | -20 |
| D | 25 | 0 | -25 |
| E | 30 | 0 | -30 |

---

## 6. Cost Function

Cost measures how far the predictions are from actual values.

$$
J(b) = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

Substitute the errors:

$$
J(b) = \frac{1}{2(5)} \left[(-10)^2 + (-15)^2 + (-20)^2 + (-25)^2 + (-30)^2\right]
$$

$$
= \frac{100 + 225 + 400 + 625 + 900}{10}
$$

$$
= \frac{2250}{10}
$$

$$
\boxed{J(b) = 225}
$$

**Initial cost = 225.**

---

## 7. Gradient Calculation

We have three coefficients, so we calculate three gradients.

### Gradient for intercept ($b_0$)

$$
\frac{\partial J}{\partial b_0} = \frac{1}{n} \sum (\hat{y} - Y) = \frac{-10 - 15 - 20 - 25 - 30}{5} = \frac{-100}{5} = \boxed{-20}
$$

### Gradient for size coefficient ($b_1$)

$$
\frac{\partial J}{\partial b_1} = \frac{1}{n} \sum (\hat{y} - Y)x_1 = \frac{(-10)(1) + (-15)(2) + (-20)(3) + (-25)(4) + (-30)(5)}{5} = \frac{-350}{5} = \boxed{-70}
$$

### Gradient for bedroom coefficient ($b_2$)

$$
\frac{\partial J}{\partial b_2} = \frac{1}{n} \sum (\hat{y} - Y)x_2 = \frac{(-10)(1) + (-15)(1) + (-20)(2) + (-25)(2) + (-30)(3)}{5} = \frac{-205}{5} = \boxed{-41}
$$

### Gradient Summary

| Coefficient | Gradient |
|:---:|:---:|
| $b_0$ | -20 |
| $b_1$ | -70 |
| $b_2$ | -41 |

---

## 8. Update Coefficients

Update formula:

$$
b_j^{\text{new}} = b_j^{\text{old}} - \alpha \times \text{Gradient}
$$

Learning rate: $\alpha = 0.01$

- **Update $b_0$:**
  $$
  b_0 = 0 - (0.01 \times -20) = \boxed{0.20}
  $$

- **Update $b_1$:**
  $$
  b_1 = 0 - (0.01 \times -70) = \boxed{0.70}
  $$

- **Update $b_2$:**
  $$
  b_2 = 0 - (0.01 \times -41) = \boxed{0.41}
  $$

### New Model

$$
\boxed{\hat{y} = 0.20 + 0.70x_1 + 0.41x_2}
$$

*This completes Iteration 1.*

---

## 9. Subsequent Iterations

For each new iteration:
1. Use the latest coefficients.
2. Calculate new predictions.
3. Calculate new errors.
4. Calculate gradients.
5. Update all coefficients.
6. Calculate the new cost.

*Do not reuse the previous iteration's errors or gradients.*

### Verified Iteration Results

| Stage | $b_0$ | $b_1$ | $b_2$ | Cost |
|:---:|:---:|:---:|:---:|:---:|
| Initial | 0 | 0 | 0 | 225.00000 |
| 1 | 0.20000 | 0.70000 | 0.41000 | 160.62879 |
| 2 | 0.36962 | 1.29076 | 0.75602 | 114.75485 |
| 3 | 0.51359 | 1.78930 | 1.04803 | 82.06270 |
| 4 | 0.63591 | 2.21000 | 1.29444 | 58.76437 |
| 5 | 0.73995 | 2.56498 | 1.50237 | 42.16039 |

*The cost decreases with each iteration. This shows that the model is improving.*

---

## 10. When Do We Stop Gradient Descent?

Gradient Descent can stop when:
1. The cost reduction becomes very small.
2. The gradients become very close to zero.
3. The maximum number of iterations is reached.

### Example:
- Previous cost = 0.5000
- Current cost = 0.4999

Cost reduction:

$$
0.5000 - 0.4999 = 0.0001
$$

If the chosen tolerance is 0.001, this reduction is sufficiently small to consider stopping.

*Important: We do not stop just because a fixed number of iterations is completed. We check the convergence condition.*

---

## 11. OLS vs Gradient Descent

Both methods can be used to find Multiple Linear Regression coefficients.

| OLS using Normal Equation | Gradient Descent |
|:---|:---|
| Direct solution method | Iterative optimization method |
| Uses matrix operations | Uses gradients |
| No learning rate | Requires learning rate |
| No repeated updates | Repeated coefficient updates |
| Can use matrix inverse or Gaussian Elimination to solve normal equations | Does not require Gaussian Elimination |

### Equations
- **OLS Normal Equation:**
  $$
  \boxed{\beta = (X^TX)^{-1}X^TY}
  $$
- **Gradient Descent Update:**
  $$
  \boxed{b_j = b_j - \alpha \frac{\partial J}{\partial b_j}}
  $$

*Important distinction:*
- OLS describes the least-squares estimation objective.
- Gaussian Elimination is one method of solving the normal equations.
- Gradient Descent is an iterative method for minimizing the same squared-error objective.

---

## 12. Important Exam Formulas

| Purpose | Formula |
|:---|:---|
| Prediction | $\hat{y} = b_0 + b_1x_1 + b_2x_2$ |
| Error | $\hat{y} - Y$ |
| Cost | $\frac{1}{2n} \sum (\hat{y} - Y)^2$ |
| Gradient ($b_0$) | $\frac{1}{n} \sum \text{Error}$ |
| Gradient ($b_1$) | $\frac{1}{n} \sum \text{Error} \times x_1$ |
| Gradient ($b_2$) | $\frac{1}{n} \sum \text{Error} \times x_2$ |
| Coefficient update | $b_j - \alpha \times \text{Gradient}$ |

---

## 13. Exam-Style Numerical Problem

### Question:
Consider the following dataset:

| Observation | $X_1$ | $X_2$ | $Y$ |
|:---:|:---:|:---:|:---:|
| 1 | 1 | 1 | 10 |
| 2 | 2 | 1 | 15 |
| 3 | 3 | 2 | 20 |
| 4 | 4 | 2 | 25 |
| 5 | 5 | 3 | 30 |

Apply one iteration of Gradient Descent using:
- $b_0 = b_1 = b_2 = 0$
- $\alpha = 0.01$

Find:
1. Initial predictions.
2. Errors.
3. Initial cost.
4. Gradients for all three coefficients.
5. Updated coefficients.
6. New model equation.
7. Cost after the first update.

### Solution

- **Step 1:** Initial predictions are zero because all initial coefficients are zero.
- **Step 2:** Errors = $[-10, -15, -20, -25, -30]$
- **Step 3:** Initial cost $J = \frac{2250}{10} = \boxed{225}$
- **Step 4:** Gradients:
  $$
  \frac{\partial J}{\partial b_0} = -20, \quad \frac{\partial J}{\partial b_1} = -70, \quad \frac{\partial J}{\partial b_2} = -41
  $$
- **Step 5:** Update coefficients:
  - $b_0 = 0 - 0.01(-20) = 0.20$
  - $b_1 = 0 - 0.01(-70) = 0.70$
  - $b_2 = 0 - 0.01(-41) = 0.41$
- **Step 6:** New model:
  $$
  \boxed{\hat{y} = 0.20 + 0.70x_1 + 0.41x_2}
  $$
- **Step 7:** New cost:
  - New predictions: $[1.31,\ 2.01,\ 3.12,\ 3.82,\ 4.93]$
  - New errors: $[-8.69,\ -12.99,\ -16.88,\ -21.18,\ -25.07]$
  - $J = \frac{1}{10} \left[8.69^2 + 12.99^2 + 16.88^2 + 21.18^2 + 25.07^2\right] = \boxed{160.62879}$

*Therefore, cost reduced from 225 to 160.62879 after the first iteration.*

**Conclusion:** Gradient Descent updates all coefficients to reduce prediction error.

---

## 14. Common Exam Mistakes

1. Forgetting to divide the gradient summation by $n$.
2. Forgetting to multiply the error by the corresponding feature when calculating feature gradients.
3. Using old errors after updating coefficients.
4. Updating only one coefficient instead of all coefficients.
5. Forgetting the learning rate ($\alpha$).
6. Confusing cost function with the gradient.
7. Assuming Gradient Descent must reach zero cost.

---

### Final Memory Shortcut

$$\text{Predict} \longrightarrow \text{Error} \longrightarrow \text{Cost} \longrightarrow \text{Gradient} \longrightarrow \text{Update} \longrightarrow \text{Repeat}$$

**Remember:**
- Number of coefficients = Number of gradients.
- Every iteration uses the latest coefficients.
- Cost should generally decrease with a suitable learning rate.
- Stop when the convergence condition is satisfied.
