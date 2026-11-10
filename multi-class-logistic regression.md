# Multiclass Logistic Regression – Complete Numerical Example

## Learning Objective
Build a Multiclass Logistic Regression model from scratch using one dataset. We will perform:
1. Dataset preparation
2. Label Encoding and One-Hot Encoding
3. Model initialization
4. Three linear equations
5. Logit calculation
6. Softmax probability calculation
7. Final class prediction
8. Cross Entropy Loss
9. Gradient calculation
10. Weight and bias updates
11. Repeated training iterations
12. Stopping criteria
13. Final prediction using trained weights
14. Model evaluation

> **Important:** We will not assume that the model already knows its weights. We will start with zero weights and learn them using Gradient Descent.

---

## 1. Understand the Problem

Suppose we want to predict a student's performance based on the number of hours they study. There are three possible results:
- **Fail**
- **Average**
- **Excellent**

This is a **Multiclass Classification** problem because the target has more than two possible classes.

### 1.1 Our Dataset

| Student | Study Hours ($X$) | Result ($Y$) |
| :---: | :---: | :---: |
| A | 1 | Fail |
| B | 2 | Fail |
| C | 3 | Fail |
| D | 3 | Average |
| E | 4 | Average |
| F | 5 | Average |
| G | 5 | Excellent |
| H | 6 | Excellent |
| I | 7 | Excellent |

We deliberately have some overlapping observations: students studying 3 hours can be *Fail* or *Average*, and students studying 5 hours can be *Average* or *Excellent*. This makes the training example more realistic than a perfectly separable dataset.

---

## 2. Prepare the Dataset

### 2.1 Label Encoding
The computer needs numerical values instead of text labels.

| Class | Encoded Value |
| :--- | :---: |
| Fail | `0` |
| Average | `1` |
| Excellent | `2` |

Therefore:
$$Y = [0, 0, 0, 1, 1, 1, 2, 2, 2]$$

### 2.2 One-Hot Encoding
We represent each actual class using a binary vector.

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
| C | 3 | 1 | 0 | 0 |
| D | 3 | 0 | 1 | 0 |
| E | 4 | 0 | 1 | 0 |
| F | 5 | 0 | 1 | 0 |
| G | 5 | 0 | 0 | 1 |
| H | 6 | 0 | 0 | 1 |
| I | 7 | 0 | 0 | 1 |

**Example:** Student E's actual class is *Average* ($Y = [0, 1, 0]$). Only the actual class receives `1`, while others receive `0`.

---

## 3. Initialize the Model

We have three classes, so we need three class-specific linear equations.

### 3.1 Linear Equations
- **Fail:** $z_0 = b_0 + w_0 X$
- **Average:** $z_1 = b_1 + w_1 X$
- **Excellent:** $z_2 = b_2 + w_2 X$

Each class has:
- One bias ($b$)
- One weight ($w$)
- Total parameters = $3 \text{ biases} + 3 \text{ weights} = 6$.

### 3.2 Initialize Weights and Biases
For our first training iteration, initialize everything to zero:

| Class | Initial Bias ($b$) | Initial Weight ($w$) |
| :--- | :---: | :---: |
| Fail | 0 | 0 |
| Average | 0 | 0 |
| Excellent | 0 | 0 |

- **Learning Rate ($\alpha$):** $0.1$
- **Number of training samples ($n$):** $9$

---

## 4. Calculate Initial Model Scores (Logits)

Using initial weights of zero for every student:
$$z_0 = 0 + (0 \times X) = 0$$
$$z_1 = 0 + (0 \times X) = 0$$
$$z_2 = 0 + (0 \times X) = 0$$

Therefore, every student initially receives raw scores: $Z = [0, 0, 0]$.

---

## 5. Apply Softmax

Softmax converts the three logits into normalized probabilities.

### 5.1 Formula
$$P_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

### 5.2 Calculate Initial Probabilities
Since all logits are zero ($e^0 = 1$ and denominator $= 1 + 1 + 1 = 3$):
- $P(\text{Fail}) = \frac{1}{3} \approx 33.33\%$
- $P(\text{Average}) = \frac{1}{3} \approx 33.33\%$
- $P(\text{Excellent}) = \frac{1}{3} \approx 33.33\%$

| Class | Initial Probability | Percentage |
| :--- | :---: | :---: |
| Fail | 0.3333 | 33.33% |
| Average | 0.3333 | 33.33% |
| Excellent | 0.3333 | 33.33% |
| **Total** | **1.0000** | **100%** |

---

## 6. Initial Prediction

Prediction rule: 
$$\hat{y} = \text{argmax}\big(P(\text{Fail}), P(\text{Average}), P(\text{Excellent})\big)$$

Since all probabilities are equal, the initial model has a tie. The model has not learned anything yet and requires training.

---

## 7. Calculate Initial Cross Entropy Loss

### 7.1 Formula
$$L = -\sum_{k=1}^{K} y_k \ln(p_k)$$

### 7.2 Calculate Loss for Student A
- $X = 1$, Actual one-hot: $Y = [1, 0, 0]$, Predicted: $P = [0.3333, 0.3333, 0.3333]$
- $L = -\ln(0.3333) \approx 1.0986$

Because every student has the same initial probability, **each student's initial loss = 1.0986**.

### 7.3 Average Initial Loss ($J$)
$$J = \frac{1}{n} \sum_{i=1}^{n} L_i = \frac{9 \times 1.0986}{9} = \mathbf{1.0986}$$

---

## 8. Gradient Descent – Learn the Weights

### 8.1 Gradient Formulas
- **Weight Gradient:** $\frac{\partial J}{\partial w_k} = \frac{1}{n} \sum (p_{ik} - y_{ik}) X_i$
- **Bias Gradient:** $\frac{\partial J}{\partial b_k} = \frac{1}{n} \sum (p_{ik} - y_{ik})$

### 8.2 Prediction Errors ($\text{Error} = p - y$)

| Student | $X$ | Target Vector ($Y$) | Fail Error | Average Error | Excellent Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| A | 1 | $[1, 0, 0]$ | -0.6667 | 0.3333 | 0.3333 |
| B | 2 | $[1, 0, 0]$ | -0.6667 | 0.3333 | 0.3333 |
| C | 3 | $[1, 0, 0]$ | -0.6667 | 0.3333 | 0.3333 |
| D | 3 | $[0, 1, 0]$ | 0.3333 | -0.6667 | 0.3333 |
| E | 4 | $[0, 1, 0]$ | 0.3333 | -0.6667 | 0.3333 |
| F | 5 | $[0, 1, 0]$ | 0.3333 | -0.6667 | 0.3333 |
| G | 5 | $[0, 0, 1]$ | 0.3333 | 0.3333 | -0.6667 |
| H | 6 | $[0, 0, 1]$ | 0.3333 | 0.3333 | -0.6667 |
| I | 7 | $[0, 0, 1]$ | 0.3333 | 0.3333 | -0.6667 |

### 8.3 Calculate Gradients
- **Fail Weight Gradient ($\frac{\partial J}{\partial w_0}$):** $\frac{6}{9} = \mathbf{0.6667}$
- **Average Weight Gradient ($\frac{\partial J}{\partial w_1}$):** $\frac{0}{9} = \mathbf{0.0000}$
- **Excellent Weight Gradient ($\frac{\partial J}{\partial w_2}$):** $\frac{-6}{9} = \mathbf{-0.6667}$
- **Bias Gradients ($\frac{\partial J}{\partial b_0}, \frac{\partial J}{\partial b_1}, \frac{\partial J}{\partial b_2}$):** All equal to **$0.0000$**.

---

## 9. Update the Weights

Update rules ($\alpha = 0.1$):
$$w_{\text{new}} = w_{\text{old}} - \alpha \times \text{Gradient}$$
$$b_{\text{new}} = b_{\text{old}} - \alpha \times \text{Bias Gradient}$$

- **Fail Weight:** $0 - (0.1 \times 0.6667) = \mathbf{-0.06667}$
- **Average Weight:** $0 - (0.1 \times 0) = \mathbf{0.00000}$
- **Excellent Weight:** $0 - (0.1 \times -0.6667) = \mathbf{+0.06667}$
- **Biases:** Remain **$0$** (since gradients are zero).

### Updated Parameters After Iteration 1

| Class | Updated Bias ($b$) | Updated Weight ($w$) |
| :--- | :---: | :---: |
| Fail | 0 | -0.06667 |
| Average | 0 | 0.00000 |
| Excellent | 0 | +0.06667 |

---

## 10. Repeat Training – Iteration 2 and Beyond

Using the updated parameters, compute new logits:
- $z_0 = -0.06667 X$
- $z_1 = 0$
- $z_2 = 0.06667 X$

Repeat the iterative loop (Logits $\rightarrow$ Softmax $\rightarrow$ Loss $\rightarrow$ Gradients $\rightarrow$ Parameter Updates).

### Training Loss Progress Table

| Iteration | Average Training Loss ($J$) |
| :---: | :---: |
| 0 | 1.0986 |
| 1 | 1.0380 |
| 2 | 1.0269 |
| 5 | 1.0135 |
| 10 | 0.9957 |
| 20 | 0.9628 |
| 50 | 0.8812 |
| 100 | 0.7863 |
| 200 | 0.6771 |
| 500 | 0.5460 |
| 1,000 | 0.4679 |
| 2,000 | 0.4094 |
| 5,000 | 0.3591 |

As iterations increase, training loss steadily decreases, confirming successful model optimization.

---

## 11. When Should We Stop Training?

We can stop Gradient Descent using one or more stopping conditions:

### Condition 1 – Maximum Iterations (Epochs)
Set a predetermined maximum number of training steps (e.g., 5,000 iterations). Once reached, training stops.

### Condition 2 – Loss Convergence (Tolerance)
Monitor the change in average loss ($\Delta J$) between consecutive iterations. If the improvement falls below a threshold (e.g., $\Delta J < 10^{-5}$) for several consecutive epochs, training terminates.

### Condition 3 – Early Stopping (Validation Loss)
Monitor performance on a separate validation dataset. If training loss continues to drop while validation loss starts increasing (indicating overfitting), training is halted early.

---

## 12. Final Prediction Using Trained Weights

Once optimal weights and biases are learned through training iterations (e.g., after 5,000 epochs):
1. For any new student study hours ($X_{\text{new}}$), compute the three class logits:  
   $$z_k = b_k + w_k X_{\text{new}}$$
2. Pass through Softmax to get probabilities $P_k$.
3. Use Argmax to choose the class with the highest probability:
   $$\hat{y} = \text{argmax}\big(P(\text{Fail}), P(\text{Average}), P(\text{Excellent})\big)$$

---

## 13. Model Evaluation

Evaluate the final model performance across the dataset using common metrics:
- **Accuracy:** Ratio of correct predictions to total samples.
- **Confusion Matrix:** $3 \times 3$ grid comparing actual classes vs predicted classes.
- **Precision, Recall, and F1-Score:** Evaluated per class to handle overlapping boundary cases effectively.
