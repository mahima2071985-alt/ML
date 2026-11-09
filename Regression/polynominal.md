# Polynomial Regression using Gradient Descent
## AIML ZG565 | BITS Pilani

---

## 1. What is Polynomial Regression?

Polynomial Regression is used when the relationship between input ($X$) and output ($Y$) is curved rather than a straight line.

### Degree 1
$$
\hat{y} = w_0 + w_1X
$$

### Degree 2
$$
\hat{y} = w_0 + w_1X + w_2X^2
$$

### Degree 3
$$
\hat{y} = w_0 + w_1X + w_2X^2 + w_3X^3
$$

*The highest power of $X$ determines the polynomial degree.*

---

## 2. Dataset

Consider the following dataset:

| Observation | $X$ | $Y$ |
|:---:|:---:|:---:|
| 1 | 1 | 1 |
| 2 | 2 | 4 |
| 3 | 3 | 9 |
| 4 | 4 | 16 |
| 5 | 5 | 25 |

Here, the relationship is:

$$
Y = X^2
$$

We will use Degree 2 Polynomial Regression.

---

## 3. Model Equation

$$
\boxed{\hat{y} = w_0 + w_1X + w_2X^2}
$$

Where:
- $w_0$ = Intercept
- $w_1$ = Weight for $X$
- $w_2$ = Weight for $X^2$

We need to find the best values of these three weights.

---

## 4. Initialize Parameters

Initial weights:

$$
w_0 = 0, \quad w_1 = 0, \quad w_2 = 0
$$

Learning rate:

$$
\alpha = 0.001
$$

Number of observations:

$$
n = 5
$$

---

## 5. Calculate Initial Predictions

Using the model:

$$
\hat{y} = 0 + 0X + 0X^2
$$

All predictions are zero.

| $X$ | Actual $Y$ | Prediction ($\hat{y}$) | Error ($\hat{y} - Y$) |
|:---:|:---:|:---:|:---:|
| 1 | 1 | 0 | -1 |
| 2 | 4 | 0 | -4 |
| 3 | 9 | 0 | -9 |
| 4 | 16 | 0 | -16 |
| 5 | 25 | 0 | -25 |

Error formula:

$$
\text{Error} = \hat{y} - Y
$$

---

## 6. Calculate Initial Cost

Cost function:

$$
J(w) = \frac{1}{2n} \sum (\hat{y} - Y)^2
$$

Substitute the values:

$$
J(w) = \frac{1}{2(5)} \left[(-1)^2 + (-4)^2 + (-9)^2 + (-16)^2 + (-25)^2\right]
$$

$$
= \frac{1}{10} [1 + 16 + 81 + 256 + 625]
$$

$$
= \frac{979}{10}
$$

$$
\boxed{J(w) = 97.9}
$$

---

## 7. Calculate Gradients

We have three weights, so we calculate three gradients.

### Gradient for $w_0$

$$
\frac{\partial J}{\partial w_0} = \frac{1}{n} \sum \text{Error} = \frac{-1 - 4 - 9 - 16 - 25}{5} = \frac{-55}{5} = \boxed{-11}
$$

### Gradient for $w_1$

$$
\frac{\partial J}{\partial w_1} = \frac{1}{n} \sum (\text{Error} \times X) = \frac{(-1)(1) + (-4)(2) + (-9)(3) + (-16)(4) + (-25)(5)}{5} = \frac{-225}{5} = \boxed{-45}
$$

### Gradient for $w_2$

$$
\frac{\partial J}{\partial w_2} = \frac{1}{n} \sum (\text{Error} \times X^2) = \frac{(-1)(1) + (-4)(4) + (-9)(9) + (-16)(16) + (-25)(25)}{5} = \frac{-979}{5} = \boxed{-195.8}
$$

### Gradient Summary

| Weight | Gradient |
|:---:|:---:|
| $w_0$ | -11 |
| $w_1$ | -45 |
| $w_2$ | -195.8 |

---

## 8. Update Weights – Iteration 1

Gradient Descent update formula:

$$
w_j^{\text{new}} = w_j^{\text{old}} - \alpha \times \text{Gradient}
$$

Learning rate: $\alpha = 0.001$

- **Update $w_0$:**
  $$
  w_0 = 0 - (0.001 \times -11) = \boxed{0.011}
  $$

- **Update $w_1$:**
  $$
  w_1 = 0 - (0.001 \times -45) = \boxed{0.045}
  $$

- **Update $w_2$:**
  $$
  w_2 = 0 - (0.001 \times -195.8) = \boxed{0.1958}
  $$

### Updated Model

$$
\boxed{\hat{y} = 0.011 + 0.045X + 0.1958X^2}
$$

---

## 9. Calculate New Predictions

- **For $X = 1$:**
  $$
  \hat{y} = 0.011 + 0.045(1) + 0.1958(1)^2 = 0.011 + 0.045 + 0.1958 = 0.2518
  $$

- **For $X = 2$:**
  $$
  \hat{y} = 0.011 + 0.045(2) + 0.1958(2)^2 = 0.011 + 0.09 + 0.7832 = 0.8842
  $$

- **For $X = 3$:**
  $$
  \hat{y} = 0.011 + 0.045(3) + 0.1958(3)^2 = 0.011 + 0.135 + 1.7622 = 1.9082
  $$

- **For $X = 4$:**
  $$
  \hat{y} = 0.011 + 0.045(4) + 0.1958(4)^2 = 0.011 + 0.18 + 3.1328 = 3.3238
  $$

- **For $X = 5$:**
  $$
  \hat{y} = 0.011 + 0.045(5) + 0.1958(5)^2 = 0.011 + 0.225 + 4.895 = 5.1310
  $$

### Prediction Table

| $X$ | Actual $Y$ | Prediction ($\hat{y}$) |
|:---:|:---:|:---:|
| 1 | 1 | 0.2518 |
| 2 | 4 | 0.8842 |
| 3 | 9 | 1.9082 |
| 4 | 16 | 3.3238 |
| 5 | 25 | 5.1310 |

---

## 10. Calculate New Cost

First calculate errors:

$$
\text{Error} = \hat{y} - Y
$$

| $X$ | Actual $Y$ | Prediction ($\hat{y}$) | Error |
|:---:|:---:|:---:|:---:|
| 1 | 1 | 0.2518 | -0.7482 |
| 2 | 4 | 0.8842 | -3.1158 |
| 3 | 9 | 1.9082 | -7.0918 |
| 4 | 16 | 3.3238 | -12.6762 |
| 5 | 25 | 5.1310 | -19.8690 |

Cost formula:

$$
J(w) = \frac{1}{2n} \sum \text{Error}^2
$$

$$
= \frac{1}{10} [0.55980324 + 9.70820964 + 50.29361824 + 160.68603444 + 394.776261]
$$

$$
= \frac{616.02392658}{10}
$$

$$
\boxed{J(w) \approx 61.60239}
$$

### Cost Comparison

| Stage | Cost ($J$) |
|:---|:---:|
| Initial | 97.90000 |
| After Iteration 1 | 61.60239 |

*The cost decreased, which means the model improved after the first update.*

---

## 11. Why Does Each Gradient Use a Different Feature?

Our model is:

$$
\hat{y} = w_0 + w_1X + w_2X^2
$$

Each weight is associated with a different feature.

| Weight | Feature used in gradient |
|:---:|:---:|
| $w_0$ | $1$ |
| $w_1$ | $X$ |
| $w_2$ | $X^2$ |

Therefore:

$$
\frac{\partial J}{\partial w_0} = \frac{1}{n} \sum \text{Error}
$$

$$
\frac{\partial J}{\partial w_1} = \frac{1}{n} \sum (\text{Error} \times X)
$$

$$
\frac{\partial J}{\partial w_2} = \frac{1}{n} \sum (\text{Error} \times X^2)
$$

*Important: Multiply the error by the corresponding feature for that weight.*

---

## 12. Degree 3 Polynomial Regression using Gradient Descent

Degree 3 model:

$$
\boxed{\hat{y} = w_0 + w_1X + w_2X^2 + w_3X^3}
$$

Four weights need to be calculated.

Gradients:

$$
\frac{\partial J}{\partial w_0} = \frac{1}{n} \sum \text{Error}
$$

$$
\frac{\partial J}{\partial w_1} = \frac{1}{n} \sum (\text{Error} \times X)
$$

$$
\frac{\partial J}{\partial w_2} = \frac{1}{n} \sum (\text{Error} \times X^2)
$$

$$
\frac{\partial J}{\partial w_3} = \frac{1}{n} \sum (\text{Error} \times X^3)
$$

Update each weight using:

$$
w_j^{\text{new}} = w_j^{\text{old}} - \alpha \frac{\partial J}{\partial w_j}
$$

*The procedure is the same as Degree 2, but Degree 3 has one additional weight and one additional feature, $X^3$.*

---

## 13. When Should Gradient Descent Stop?

Gradient Descent can stop when:
1. Cost becomes almost constant between iterations.
2. The change in cost becomes very small.
3. A predefined maximum number of iterations is reached.

*The goal is to minimize the cost function, not necessarily to make it exactly zero.*

---

## 14. Polynomial Regression – Important Exam Points

| Question | Answer |
|:---|:---|
| What determines polynomial degree? | Highest power of $X$ |
| Degree 2 model? | $w_0 + w_1X + w_2X^2$ |
| Degree 3 model? | $w_0 + w_1X + w_2X^2 + w_3X^3$ |
| Why use polynomial regression? | To model curved relationships |
| Can Gradient Descent be used? | Yes |
| What is the cost function here? | $J(w) = \frac{1}{2n} \sum \text{Error}^2$ |
| What happens when degree increases? | Model becomes more flexible |
| Does higher degree always mean better model? | No; it may overfit |

---

## 15. Common Exam Mistakes

1. Confusing polynomial degree with the number of coefficients.
2. Forgetting the intercept ($w_0$).
3. Using $X$ instead of $X^2$ while calculating the gradient for $w_2$.
4. Forgetting to square the error in the cost function.
5. Using the old weights instead of updated weights when calculating new predictions.
6. Assuming lower training error always means better performance on unseen data.

---

## 16. Practice Question

### Given:
| $X$ | $Y$ |
|:---:|:---:|
| 1 | 2 |
| 2 | 5 |
| 3 | 10 |

Polynomial model: $\hat{y} = w_0 + w_1X + w_2X^2$  
Initial weights: $w_0 = w_1 = w_2 = 0$  
Learning rate: $\alpha = 0.01$

### Calculate:
1. Initial predictions.
2. Errors.
3. Initial cost.
4. Gradients for $w_0$, $w_1$, and $w_2$.
5. Updated weights after one iteration.

*Remember: Polynomial Regression uses polynomial features, and Gradient Descent finds the weights by repeatedly reducing the cost function.*
