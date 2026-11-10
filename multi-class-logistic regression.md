# Multiclass Logistic Regression – Complete Notes

## 1. What is Multiclass Logistic Regression?

Multiclass Logistic Regression is a supervised machine learning algorithm used to classify data into more than two classes.

**Example:** *Student Result Prediction*  
We want to predict whether a student will **Fail**, get **Average**, or get **Excellent** based on study hours.

### Binary vs Multiclass Classification

| Feature | Binary Classification | Multiclass Classification |
| :--- | :--- | :--- |
| **Number of Classes** | Two classes | More than two classes |
| **Example** | Pass / Fail | Fail / Average / Excellent |
| **Possible Outcomes** | Two possible outcomes | Three or more possible outcomes |

Our problem has three classes, so we use **Multiclass Logistic Regression**.

---

## 2. Dataset Preparation

We will use the following dataset throughout this example.

| Student | Study Hours ($X$) | Result ($Y$) |
| :---: | :---: | :---: |
| A | 1 | Fail |
| B | 2 | Fail |
| C | 3 | Average |
| D | 4 | Average |
| E | 5 | Excellent |
| F | 6 | Excellent |

- **Input ($X$):** Study Hours  
- **Output ($Y$):** Student Result  

### 2.1 Class Encoding
Computers need numerical representations of class labels.

| Class | Encoded Value |
| :--- | :---: |
| Fail | `0` |
| Average | `1` |
| Excellent | `2` |

Therefore:
$$Y = [0, 0, 1, 1, 2, 2]$$

### 2.2 One-Hot Encoding
One-hot encoding represents each class using a binary vector.

| Class | Fail | Average | Excellent |
| :--- | :---: | :---: | :---: |
| **Fail** | 1 | 0 | 0 |
| **Average** | 0 | 1 | 0 |
| **Excellent** | 0 | 0 | 1 |

#### Complete Dataset with One-Hot Encoding

| Student | $X$ | Fail ($y_0$) | Average ($y_1$) | Excellent ($y_2$) |
| :---: | :---: | :---: | :---: | :---: |
| A | 1 | 1 | 0 | 0 |
| B | 2 | 1 | 0 | 0 |
| C | 3 | 0 | 1 | 0 |
| D | 4 | 0 | 1 | 0 |
| E | 5 | 0 | 0 | 1 |
| F | 6 | 0 | 0 | 1 |

**Example:** Student E's actual result is *Excellent*.
- One-hot encoding = $[0, 0, 1]$
- Only the actual class receives `1`, while all other classes receive `0`.

---

## 3. Model Architecture – Three Linear Equations

In Binary Logistic Regression, we use one linear equation:
$$z = w_0 + w_1 X$$

In Multiclass Logistic Regression, we have three classes. Therefore, we calculate three separate class scores.

- **Equation 1 – Fail:**
  $$z_0 = w_{00} + w_{01} X$$
- **Equation 2 – Average:**
  $$z_1 = w_{10} + w_{11} X$$
- **Equation 3 – Excellent:**
  $$z_2 = w_{20} + w_{21} X$$

**Where:**
- $X$ = Input (Study Hours)
- $w_0$ = Bias or intercept
- $w_1$ = Weight or coefficient
- $z$ = Raw model score (logit)

> **Important:** Each class has its own bias and weight. The model learns these parameters during training using **Gradient Descent**. We do not know the correct weights in advance.

---

## 4. Calculate Model Scores (Logits)

For understanding the prediction process, consider a new student who studies 4.5 hours ($X = 4.5$).

Assume the following illustrative weights have already been learned:

| Class | Bias ($w_0$) | Weight ($w_1$) |
| :--- | :---: | :---: |
| **Fail** | 3 | -1 |
| **Average** | 0 | 0.5 |
| **Excellent** | -3 | 1 |

### 4.1 Calculate Fail Score ($z_0$)
$$z_0 = 3 + (-1 \times 4.5) = 3 - 4.5 = -1.5$$

### 4.2 Calculate Average Score ($z_1$)
$$z_1 = 0 + (0.5 \times 4.5) = 2.25$$

### 4.3 Calculate Excellent Score ($z_2$)
$$z_2 = -3 + (1 \times 4.5) = -3 + 4.5 = 1.5$$

### 4.4 Score Comparison

| Class | Logit (Score $z$) |
| :--- | :---: |
| Fail | -1.50 |
| **Average** | **2.25** |
| Excellent | 1.50 |

The highest raw score is **2.25**, which belongs to **Average**.

> **Note:** Logits are raw scores, not probabilities. We cannot interpret them directly as percentages.

---

## 5. Softmax Function

Softmax converts raw class scores (logits) into normalized probability distributions.

### Formula
$$P(\text{class } k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**Where:**
- $z_k$ = Score of the target class
- $e$ = Euler's constant ($\approx 2.71828$)
- $\sum e^{z_j}$ = Sum of exponential values across all classes
- $K$ = Total number of classes

### 5.1 Calculate Exponential Values

Given logits: $z_0 = -1.5$, $z_1 = 2.25$, $z_2 = 1.5$

- **Fail:** $e^{-1.5} = 0.2231$
- **Average:** $e^{2.25} = 9.4877$
- **Excellent:** $e^{1.5} = 4.4817$

| Class | Logit ($z$) | Exponential ($e^z$) |
| :--- | :---: | :---: |
| Fail | -1.5 | 0.2231 |
| Average | 2.25 | 9.4877 |
| Excellent | 1.5 | 4.4817 |

### 5.2 Calculate Denominator (Sum of Exponentials)
$$\sum e^{z_j} = 0.2231 + 9.4877 + 4.4817 = 14.1925$$

### 5.3 Calculate Class Probabilities

- **Fail:**
  $$P(\text{Fail}) = \frac{0.2231}{14.1925} \approx 0.0157 \quad (1.57\%)$$
- **Average:**
  $$P(\text{Average}) = \frac{9.4877}{14.1925} \approx 0.6685 \quad (66.85\%)$$
- **Excellent:**
  $$P(\text{Excellent}) = \frac{4.4817}{14.1925} \approx 0.3158 \quad (31.58\%)$$

### 5.4 Final Probability Table

| Class | Probability | Percentage |
| :--- | :---: | :---: |
| Fail | 0.0157 | 1.57% |
| **Average** | **0.6685** | **66.85%** |
| Excellent | 0.3158 | 31.58% |
| **Total** | **1.0000** | **100%** |

Softmax ensures all outputs sum to 1 (or 100%).

---

## 6. Final Prediction – Argmax

After calculating probabilities, we select the class with the highest probability value using the **Argmax** operation:

$$\hat{y} = \text{argmax}\Big(P(\text{Fail}), P(\text{Average}), P(\text{Excellent})\Big)$$

$$\hat{y} = \text{argmax}(0.0157, 0.6685, 0.3158)$$

The highest probability is **0.6685**.

$$\mathbf{\text{Final Prediction} = \text{Average}}$$

### Key Takeaways
- **Logits:** Raw, unscaled linear outputs.
- **Softmax:** Maps logits to normalized probabilities.
- **Argmax:** Picks the discrete predicted class with maximum probability.

---

## 7. Cross Entropy Loss

Cross Entropy Loss measures the discrepancy between predicted probability distributions and true one-hot encoded targets. It severely penalizes confident wrong predictions.

### Formula
$$L = -\sum_{k=1}^{K} y_k \ln(p_k)$$

**Where:**
- $y_k$ = Ground truth one-hot vector value for class $k$
- $p_k$ = Predicted probability for class $k$
- $\ln$ = Natural logarithm

### 7.1 Example Calculation
For our sample student whose actual performance is **Average**:
- **Actual (One-Hot):** $Y = [0, 1, 0]$
- **Predicted:** $P = [0.0157, 0.6685, 0.3158]$

Substitute values:
$$L = - \big[0 \cdot \ln(0.0157) + 1 \cdot \ln(0.6685) + 0 \cdot \ln(0.3158)\big]$$

Terms multiplied by zero vanish:
$$L = -\ln(0.6685) \approx 0.4029$$

### 7.2 Loss Sensitivity Table

If the actual target is **Average**:

| Average Probability ($p$) | Cross Entropy Loss ($L$) |
| :---: | :---: |
| 90.00% | 0.1054 |
| **66.85%** | **0.4029** |
| 30.00% | 1.2040 |

- **Lower Loss** = Better predictions.
- Perfect prediction ($p = 1.0$) yields $L = 0$.

---

## 8. Gradient Descent

Gradient Descent optimizes model weights by iteratively minimizing the Cross Entropy Loss function.

### Main Workflow
1. Initialize weights and biases.
2. Calculate logits ($z$).
3. Apply Softmax ($p$).
4. Calculate Cross Entropy Loss ($L$).
5. Compute gradients ($\frac{\partial J}{\partial w}, \frac{\partial J}{\partial b}$).
6. Update parameters ($w, b$).
7. Repeat until convergence.

### 8.1 Initialization
For demonstration, set all parameters to zero:
$$w_{00} = w_{01} = w_{10} = w_{11} = w_{20} = w_{21} = 0$$

- **Learning Rate ($\alpha$):** $0.1$
- **Total Samples ($n$):** $6$

### 8.2 Calculate Initial Logits
Since all weights/biases are $0$:
$$z_0 = 0, \quad z_1 = 0, \quad z_2 = 0$$

### 8.3 Apply Softmax
For all classes:
$$P(\text{class}) = \frac{e^0}{e^0 + e^0 + e^0} = \frac{1}{3} \approx 0.3333$$

| Class | Initial Probability |
| :--- | :---: |
| Fail | 0.3333 |
| Average | 0.3333 |
| Excellent | 0.3333 |

The un-trained model assigns uniform probabilities across all classes.

### 8.4 Prediction Error Calculation
$$\text{Error} = (p - y)$$

For **Student A** (Actual = Fail, $Y = [1, 0, 0]$):

| Class | Predicted ($p$) | Actual ($y$) | Error ($p - y$) |
| :--- | :---: | :---: | :---: |
| Fail | 0.3333 | 1 | **-0.6667** |
| Average | 0.3333 | 0 | **0.3333** |
| Excellent | 0.3333 | 0 | **0.3333** |

### 8.5 Error Table Across All Training Samples

| Student | $X$ | Target Vector | Fail Error | Average Error | Excellent Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | 1 | $[1, 0, 0]$ | -0.6667 | 0.3333 | 0.3333 |
| **B** | 2 | $[1, 0, 0]$ | -0.6667 | 0.3333 | 0.3333 |
| **C** | 3 | $[0, 1, 0]$ | 0.3333 | -0.6667 | 0.3333 |
| **D** | 4 | $[0, 1, 0]$ | 0.3333 | -0.6667 | 0.3333
