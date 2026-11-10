# Logistic Regression – Complete Exam Notes

---

## 1. What is Logistic Regression?

Logistic Regression is a **Supervised Machine Learning algorithm** used for **classification**.  
It predicts the probability that an observation belongs to a particular class.

### Examples

| Input | Classification |
| :--- | :--- |
| Email | Spam / Not Spam |
| Transaction | Fraud / Not Fraud |
| Student data | Pass / Fail |
| Loan application | Approved / Rejected |

> **Important:** Although its name contains *Regression*, Logistic Regression is commonly used for **Classification**.

---

## 2. Linear Regression vs Logistic Regression

| Linear Regression | Logistic Regression |
| :--- | :--- |
| Used for Regression | Used for Classification |
| Predicts continuous values | Predicts class probabilities |
| Output can be any real number | Output is between $0$ and $1$ |
| **Example:** House price | **Example:** Spam detection |

---

## 3. Why Do We Need Sigmoid Function?

### Linear model:
$$z = w_0 + w_1 X$$

The output of this equation can be negative, positive, or greater than $1$. However, probability must strictly be between $0$ and $1$:

$$0 \leq P \leq 1$$

To convert the linear model output into a valid probability, Logistic Regression uses the **Sigmoid Function**.

---

## 4. Sigmoid Function

### Formula:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| Symbol | Meaning |
| :---: | :--- |
| $z$ | Linear model output |
| $e$ | Euler's number, approximately $2.71828$ |
| $\sigma(z)$ | Predicted probability |

### Complete Logistic Regression Equation:
$$P(Y=1|X) = \frac{1}{1 + e^{-(w_0 + w_1 X)}}$$

---

## 5. Numerical Example – Sigmoid Calculation

Assume:
$$w_0 = -4, \quad w_1 = 1$$

Therefore:
$$z = -4 + 1X$$

### Case 1: $X = 2$
1. **Calculate $z$:**
   $$z = -4 + (1 \times 2) = -2$$

2. **Apply Sigmoid:**
   $$P = \frac{1}{1 + e^{-(-2)}} = \frac{1}{1 + e^2} = \frac{1}{1 + 7.389} = \frac{1}{8.389}$$
   $$P = 0.1192 = 11.92\%$$

### Case 2: $X = 4$
1. **Calculate $z$:**
   $$z = -4 + (1 \times 4) = 0$$

2. **Apply Sigmoid:**
   $$P = \frac{1}{1 + e^0} = \frac{1}{1 + 1}$$
   $$P = 0.5 = 50\%$$

### Case 3: $X = 6$
1. **Calculate $z$:**
   $$z = -4 + (1 \times 6) = 2$$

2. **Apply Sigmoid:**
   $$P = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.1353} = \frac{1}{1.1353}$$
   $$P = 0.8808 = 88.08\%$$

### Summary of Sigmoid Results

| $X$ | $z$ | Probability |
| :---: | :---: | :---: |
| $2$ | $-2$ | $11.92\%$ |
| $4$ | $0$ | $50.00\%$ |
| $6$ | $2$ | $88.08\%$ |

### Important Observations
- **Negative $z$** gives a probability below $0.5$.
- **Zero $z$** gives a probability exactly $0.5$.
- **Positive $z$** gives a probability above $0.5$.
- Sigmoid output always lies strictly between $0$ and $1$.

---

## 6. Decision Boundary

After calculating probability, we need to decide the final class. Usually, we use a default threshold of $0.5$.

| Probability | Classification |
| :---: | :---: |
| $P \geq 0.5$ | Class 1 |
| $P < 0.5$ | Class 0 |

### Example:

| $X$ | Probability | Prediction |
| :---: | :---: | :--- |
| $2$ | $11.92\%$ | Class 0 – Fail |
| $4$ | $50.00\%$ | Class 1 – Pass |
| $6$ | $88.08\%$ | Class 1 – Pass |

For our numerical example, the decision boundary is:
$$X = 4$$

> **Definition:** **Decision Boundary** is the threshold line at which the model changes its predicted class according to the selected probability threshold.

---

## 7. How Does Logistic Regression Learn?

The model does not know the correct weights initially. It learns the optimal weights from training data using an optimization method such as **Gradient Descent**.

```
[Initialize Weights] ➔ [Calculate z] ➔ [Apply Sigmoid] ➔ [Predict Probability]
                                                                  │
[Repeat until Converged]  [Update Weights]  [Calculate Gradients]  [Calculate Log Loss]
```

---

## 8. Training Dataset

Consider the following student dataset:

| Student | Study Hours ($X$) | Actual Result ($Y$) |
| :---: | :---: | :---: |
| A | 1 | 0 |
| B | 2 | 0 |
| C | 3 | 0 |
| D | 4 | 1 |
| E | 5 | 1 |

*Where:* $0 = \text{Fail}$, $1 = \text{Pass}$

- **Model Equation:** $z = w_0 + w_1 X$
- **Sigmoid:** $p = \frac{1}{1 + e^{-z}}$
- **Initial Weights:** $w_0 = 0, \quad w_1 = 0$
- **Learning Rate:** $\alpha = 0.1$
- **Number of Observations:** $n = 5$

---

## 9. Calculate Initial Predictions

Using $z = w_0 + w_1 X$:
Since both initial weights are zero, $z = 0 + 0(X) = 0$.

Apply Sigmoid:
$$p = \frac{1}{1 + e^0} = \frac{1}{2} = 0.5$$

All students receive an initial predicted probability of $0.5$.

| Student | $X$ | Actual $Y$ | Initial Probability ($p$) |
| :---: | :---: | :---: | :---: |
| A | 1 | 0 | 0.5 |
| B | 2 | 0 | 0.5 |
| C | 3 | 0 | 0.5 |
| D | 4 | 1 | 0.5 |
| E | 5 | 1 | 0.5 |

---

## 10. Log Loss – Cost Function

Log Loss is used to measure how well the predicted probabilities match the actual class labels.

### For one observation:
$$L = - [y \ln(p) + (1 - y) \ln(1 - p)]$$

Where:
- $y$ = Actual class ($0$ or $1$)
- $p$ = Predicted probability of Class 1
- $\ln$ = Natural logarithm

### For $n$ observations (Average Log Loss):
$$J = -\frac{1}{n} \sum_{i=1}^{n} [y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i)]$$

### Two Important Cases:
- When actual $Y = 1$: $$L = -\ln(p)$$
- When actual $Y = 0$: $$L = -\ln(1 - p)$$

### Exam Shortcut:

| Actual $Y$ | Loss Formula |
| :---: | :---: |
| **1** | $-\ln(p)$ |
| **0** | $-\ln(1 - p)$ |

---

## 11. Log Loss Numerical Calculation

Using initial predictions ($p = 0.5$):

- **Student A ($Y=0$):** $L_A = -\ln(1 - 0.5) = -\ln(0.5) = 0.6931$
- **Student B ($Y=0$):** $L_B = -\ln(1 - 0.5) = 0.6931$
- **Student C ($Y=0$):** $L_C = -\ln(1 - 0.5) = 0.6931$
- **Student D ($Y=1$):** $L_D = -\ln(0.5) = 0.6931$
- **Student E ($Y=1$):** $L_E = -\ln(0.5) = 0.6931$

### Calculate Average Log Loss ($J$):
$$J = \frac{0.6931 + 0.6931 + 0.6931 + 0.6931 + 0.6931}{5} = \frac{3.4655}{5} = 0.6931$$

### Important Observations
- **Lower Log Loss** means better probability predictions.
- A **confident correct prediction** yields low loss.
- A **confident wrong prediction** yields a huge penalty (very high loss).
- **Perfect predictions** approach zero loss.

---

## 12. Gradient Descent for Logistic Regression

Gradient Descent updates weights to minimize Log Loss.

### Gradient Formula for Intercept ($w_0$):
$$\frac{\partial J}{\partial w_0} = \frac{1}{n} \sum (p - y)$$

### Gradient Formula for Weight ($w_1$):
$$\frac{\partial J}{\partial w_1} = \frac{1}{n} \sum (p - y)X$$

---

## 13. Calculate Gradient for $w_0$

Initial prediction $p = 0.5$. Calculate $(p - y)$ for each student:

| Student | $p$ | $y$ | $p - y$ |
| :---: | :---: | :---: | :---: |
| A | 0.5 | 0 | 0.5 |
| B | 0.5 | 0 | 0.5 |
| C | 0.5 | 0 | 0.5 |
| D | 0.5 | 1 | -0.5 |
| E | 0.5 | 1 | -0.5 |

$$\frac{\partial J}{\partial w_0} = \frac{0.5 + 0.5 + 0.5 - 0.5 - 0.5}{5} = \frac{0.5}{5} = 0.1$$

---

## 14. Calculate Gradient for $w_1$

| Student | $p - y$ | $X$ | $(p - y) \times X$ |
| :---: | :---: | :---: | :---: |
| A | 0.5 | 1 | 0.5 |
| B | 0.5 | 2 | 1.0 |
| C | 0.5 | 3 | 1.5 |
| D | -0.5 | 4 | -2.0 |
| E | -0.5 | 5 | -2.5 |

$$\text{Sum} = 0.5 + 1.0 + 1.5 - 2.0 - 2.5 = -1.5$$

$$\frac{\partial J}{\partial w_1} = \frac{-1.5}{5} = -0.3$$

---

## 15. Gradient Summary

| Weight | Gradient Value |
| :---: | :---: |
| $w_0$ | $0.1$ |
| $w_1$ | $-0.3$ |

---

## 16. Update Weights – Iteration 1

### Update Formula:
$$w_j^{\text{new}} = w_j^{\text{old}} - \alpha \times \text{Gradient}$$

Using Learning Rate $\alpha = 0.1$:

- **Update $w_0$:**
  $$w_0 = 0 - (0.1 \times 0.1) = -0.01$$

- **Update $w_1$:**
  $$w_1 = 0 - (0.1 \times -0.3) = 0 + 0.03 = 0.03$$

### Updated Linear Model:
$$z = -0.01 + 0.03X$$

---

## 17. Calculate Updated Predictions

Calculate $z = -0.01 + 0.03X$ and apply Sigmoid $p = \frac{1}{1 + e^{-z}}$:

| Student | $X$ | Calculation ($z$) | $z$ | Updated Probability ($p$) |
| :---: | :---: | :--- | :---: | :---: |
| A | 1 | $-0.01 + 0.03(1)$ | 0.02 | 0.5050 |
| B | 2 | $-0.01 + 0.03(2)$ | 0.05 | 0.5125 |
| C | 3 | $-0.01 + 0.03(3)$ | 0.08 | 0.5200 |
| D | 4 | $-0.01 + 0.03(4)$ | 0.11 | 0.5275 |
| E | 5 | $-0.01 + 0.03(5)$ | 0.14 | 0.5349 |

---

## 18. Calculate Updated Log Loss

| Student | Actual $Y$ | Probability ($p$) | Individual Loss |
| :---: | :---: | :---: | :---: |
| A | 0 | 0.5050 | 0.7032 |
| B | 0 | 0.5125 | 0.7185 |
| C | 0 | 0.5200 | 0.7339 |
| D | 1 | 0.5275 | 0.6397 |
| E | 1 | 0.5349 | 0.6256 |

### Average Log Loss:
$$J = \frac{0.7032 + 0.7185 + 0.7339 + 0.6397 + 0.6256}{5} \approx 0.68417$$

### Log Loss Comparison:

| Stage | Log Loss |
| :--- | :---: |
| Initial | 0.69310 |
| After Iteration 1 | 0.68417 |

> **Conclusion:** Log Loss **decreased**, proving the model improved after the first weight update!

---

## 19. When Does Gradient Descent Stop?

The process stops when:
1. Loss changes become negligible (convergence).
2. Predefined maximum number of iterations is reached.

---

## 20. Important Exam Questions

- **Q1. What is Logistic Regression?**  
  *A supervised learning algorithm used for classification by predicting class probabilities.*
- **Q2. Why do we use Sigmoid?**  
  *To map any real-valued number from the linear output into a probability between 0 and 1.*
- **Q3. Write the Sigmoid formula.**  
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Q4. What is a Decision Boundary?**  
  *The boundary line where predicted class changes based on a chosen threshold.*
- **Q5. What is the usual classification threshold?**  
  *$0.5$ (can be adjusted according to business requirements).*
- **Q6. What is Log Loss?**  
  *A cost function measuring error between predicted probabilities and actual target classes.*
- **Q7. Why do we use Log Loss?**  
  *It penalizes confident wrong predictions harshly.*
- **Q8. What is the Gradient for $w_0$?**  
  $$\frac{\partial J}{\partial w_0} = \frac{1}{n}\sum(p-y)$$
- **Q9. What is the Gradient for $w_1$?**  
  $$\frac{\partial J}{\partial w_1} = \frac{1}{n}\sum(p-y)X$$
- **Q10. What is the Gradient Descent update formula?**  
  $$w_j^{\text{new}} = w_j^{\text{old}} - \alpha \times \text{Gradient}$$

---

## 21. Common Exam Mistakes

1. ❌ Confusing Linear Regression with Logistic Regression.
2. ❌ Forgetting to apply Sigmoid after calculating $z$.
3. ❌ Using the wrong sign in the exponent (must be $-z$).
4. ❌ Confusing actual class $y$ with predicted probability $p$.
5. ❌ Using $-\ln(p)$ when actual $y = 0$.
6. ❌ Forgetting to average the total loss over all $n$ observations.
7. ❌ Forgetting to multiply $(p-y)$ by $X$ for $w_1$ gradient.
8. ❌ Adding the gradient instead of subtracting during weight update.
9. ❌ Using old weights when computing updated predictions.

---

## 22. Final Revision Cheat Sheet

| Concept | Formula |
| :--- | :--- |
| **Linear Model** | $z = w_0 + w_1 X$ |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ |
| **Log Loss ($Y=1$)** | $-\ln(p)$ |
| **Log Loss ($Y=0$)** | $-\ln(1 - p)$ |
| **Average Log Loss** | $-\frac{1}{n} \sum [y \ln(p) + (1-y)\ln(1-p)]$ |
| **Gradient $w_0$** | $\frac{1}{n} \sum (p - y)$ |
| **Gradient $w_1$** | $\frac{1}{n} \sum (p - y) X$ |
| **Weight Update** | $w_j^{\text{new}} = w_j^{\text{old}} - \alpha \times \text{Gradient}$ |

### Complete Flow Diagram

$$\text{Input} \longrightarrow \text{Linear Model } (z) \longrightarrow \text{Sigmoid } (p) \longrightarrow \text{Log Loss } (J) \longrightarrow \text{Gradient Descent} \longrightarrow \text{Updated Weights}$$