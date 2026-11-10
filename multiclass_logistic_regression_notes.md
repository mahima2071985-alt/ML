# Multiclass Logistic Regression – Complete Step-by-Step Numerical Notes

## Learning Objective
Build a Multiclass Logistic Regression model from scratch using one dataset. We will start with zero weights and biases, train the model using Gradient Descent, and use the learned weights to predict a new student's result.

We will cover:
1. Understanding the problem
2. Dataset preparation
3. Label Encoding
4. One-Hot Encoding
5. Model initialization
6. Three linear equations
7. Logit calculation
8. Softmax
9. Initial prediction
10. Cross Entropy Loss
11. Gradient calculation
12. Weight and bias updates
13. Repeated training iterations
14. Stopping criteria
15. Final prediction using learned weights
16. Model evaluation

> **Important:** We will use the same dataset throughout the entire example. We will not assume that weights are already learned.

---

## 1. Understand the Problem

Multiclass Logistic Regression is a supervised machine learning algorithm used to classify data into more than two classes.

**Example:** *Student Result Prediction*  
We want to predict a student's performance based on the number of hours they study. There are three possible results:
- Fail
- Average
- Excellent

Since there are three possible classes, this is a **Multiclass Classification** problem.

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

Notice that some students have the same study hours but different results. For example:
- Student C studies 3 hours and gets Fail.
- Student D studies 3 hours and gets Average.

This makes the example more realistic because the model cannot perfectly separate every student using study hours alone.

### 1.2 Identify Input and Output
- **Input ($X$):** Study Hours
- **Output ($Y$):** Student Result

**Goal:** $\text{Input (Study Hours)} \rightarrow \text{Output (Predicted Result)}$  
For example, if a new student studies 4.5 hours, the model should predict Fail, Average, or Excellent.

---

## 2. Prepare the Dataset

Before training, we must convert the class labels into numerical representations.

### 2.1 Label Encoding
The dataset contains text labels (Fail, Average, Excellent). A machine learning model needs numerical representations. Therefore, we assign a number to each class:

| Class | Encoded Value |
| :--- | :---: |
| Fail | `0` |
| Average | `1` |
| Excellent | `2` |

- **Original Target:** `[Fail, Fail, Fail, Average, Average, Average, Excellent, Excellent, Excellent]`
- **After Label Encoding:** $Y = [0, 0, 0, 1, 1, 1, 2, 2, 2]$

> **Why do we do this?**  
> Computers perform mathematical operations using numbers. Label Encoding converts text labels into numbers.

### 2.2 One-Hot Encoding
One-Hot Encoding represents each class using a separate position vector. Since we have three classes, each vector contains three positions.

| Class | Fail | Average | Excellent |
| :--- | :---: | :---: | :---: |
| **Fail** | 1 | 0 | 0 |
| **Average** | 0 | 1 | 0 |
| **Excellent** | 0 | 0 | 1 |

#### Complete Dataset After One-Hot Encoding

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

> **Important Difference:**
> - Label Encoding gives **one number** per class.
> - One-Hot Encoding gives **one vector** per class.
> - In our Cross Entropy calculation, we use the one-hot representation of the actual class.

---

## 3. Initialize the Model

We have three classes: Fail, Average, and Excellent. Therefore, we need three separate linear equations.

### 3.1 Create Three Linear Equations
- **For Fail:** $z_0 = b_0 + w_0 X$
- **For Average:** $z_1 = b_1 + w_1 X$
- **For Excellent:** $z_2 = b_2 + w_2 X$

| Symbol | Meaning |
| :---: | :--- |
| $X$ | Study Hours |
| $b$ | Bias or intercept |
| $w$ | Weight or coefficient |
| $z$ | Raw model score (logit) |

Each class has its own weight and bias. Total parameters = $3 \text{ weights} + 3 \text{ biases} = 6 \text{ parameters}$.

### 3.2 Initialize Weights and Biases
At the beginning, the model has not learned anything. We initialize all parameters to zero:

| Class | Initial Bias ($b$) | Initial Weight ($w$) |
| :--- | :---: | :---: |
| Fail | 0 | 0 |
| Average | 0 | 0 |
| Excellent | 0 | 0 |

- **Learning Rate ($\alpha$):** $0.1$
- **Number of training samples ($n$):** $9$

> **Why initialize to zero?**  
> This gives us a simple starting point. Gradient Descent will update these values during training.

---

## 4. Calculate Initial Model Scores (Logits)

A logit is the raw score calculated by a linear equation:
$$z = b + w X$$

Since all initial weights and biases are zero, every student's three logits will initially be zero.

For **Student A** ($X = 1$):
- **Fail Score ($z_0$):** $z_0 = 0 + (0 \times 1) = 0$
- **Average Score ($z_1$):** $z_1 = 0 + (0 \times 1) = 0$
- **Excellent Score ($z_2$):** $z_2 = 0 + (0 \times 1) = 0$

Therefore: $z = [0, 0, 0]$. The same initial result applies to all nine students.

> **Important:** Logits are raw scores. They are not probabilities or percentages.

---

## 5. Apply Softmax Function

Softmax converts raw class scores into normalized probabilities.

### 5.1 Softmax Formula
$$P_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

- $P_k$ = Probability of class $k$
- $z_k$ = Logit of class $k$
- $e$ = Exponential constant ($\approx 2.71828$)
- $\sum$ = Sum of exponential values across all classes
- $K$ = Number of classes

For our three classes:
$$P(\text{Fail}) = \frac{e^{z_0}}{e^{z_0} + e^{z_1} + e^{z_2}}, \quad P(\text{Average}) = \frac{e^{z_1}}{e^{z_0} + e^{z_1} + e^{z_2}}, \quad P(\text{Excellent}) = \frac{e^{z_2}}{e^{z_0} + e^{z_1} + e^{z_2}}$$

### 5.2 Calculate Exponential Values
Logits $z = [0, 0, 0]$. Since $e^0 = 1$:

| Class | Logit ($z$) | Exponential ($e^z$) |
| :--- | :---: | :---: |
| Fail | 0 | 1 |
| Average | 0 | 1 |
| Excellent | 0 | 1 |

### 5.3 Calculate the Denominator
$$\text{Denominator} = e^0 + e^0 + e^0 = 1 + 1 + 1 = 3$$

### 5.4 Calculate Each Probability
- $P(\text{Fail}) = \frac{1}{3} \approx 0.3333 \quad (33.33\%)$
- $P(\text{Average}) = \frac{1}{3} \approx 0.3333 \quad (33.33\%)$
- $P(\text{Excellent}) = \frac{1}{3} \approx 0.3333 \quad (33.33\%)$

#### Initial Probability Table

| Class | Probability | Percentage |
| :--- | :---: | :---: |
| Fail | 0.3333 | 33.33% |
| Average | 0.3333 | 33.33% |
| Excellent | 0.3333 | 33.33% |
| **Total** | **1.0000** | **100%** |

> **Why did we get 0.3333?**  
> There are three classes, and all three logits are zero. So all exponential values are equal to 1. Each probability is $1 / 3 = 0.3333$. The initial model gives equal probability to every class because it has not learned anything yet.

---

## 6. Initial Prediction

After Softmax, we select the class with the highest probability:
$$\text{Prediction} = \text{argmax}\big(P(\text{Fail}), P(\text{Average}), P(\text{Excellent})\big)$$

Our probabilities: $[0.3333, 0.3333, 0.3333]$. All three probabilities are equal, resulting in a tie. A computer implementation may select the first class according to its class order, but this is not a meaningful learned prediction. We need to train the model to learn better weights.

---

## 7. Calculate Initial Cross Entropy Loss

### 7.1 Why Do We Calculate Cross Entropy Loss?
Our model produces probabilities. To evaluate them, we compare:
- Actual result of the student
- Predicted probability given to that actual result

Cross Entropy Loss measures the error between the actual class and predicted probabilities.  
**Main objective:** Reduce Cross Entropy Loss during training.

### 7.2 Cross Entropy Formula
$$L_i = -\sum_{k=1}^{K} y_{ik} \ln(p_{ik})$$

Expanded for three classes:
$$L_i = -\big[ y_{i1} \ln(p_{i1}) + y_{i2} \ln(p_{i2}) + y_{i3} \ln(p_{i3}) \big]$$

| Symbol | Meaning |
| :---: | :--- |
| $L_i$ | Individual student's loss |
| $y_{ik}$ | Actual one-hot value |
| $p_{ik}$ | Predicted probability |
| $\ln$ | Natural logarithm |
| $K$ | Number of classes |

### 7.3 Calculate Loss for Student A
- **Detail:** Student A, $X = 1$, Actual Result = Fail ($Y_A = [1, 0, 0]$)
- **Predicted Probabilities:** $P_A = [0.3333, 0.3333, 0.3333]$

Substitute into formula:
$$L_A = -\big[ 1 \times \ln(0.3333) + 0 \times \ln(0.3333) + 0 \times \ln(0.3333) \big]$$
$$L_A = -\ln(0.3333) \approx -(-1.0986) = \mathbf{1.0986}$$

The negative sign outside the logarithm changes the negative logarithm value into a positive loss.

### 7.4 Why Do All Nine Students Get the Same Initial Loss?
The actual classes are different, but all students initially receive the same predicted probabilities ($0.3333$).

#### Individual Loss Table

| Student | Actual Class | Probability of Actual Class | Loss |
| :---: | :--- | :---: | :---: |
| A | Fail | 0.3333 | 1.0986 |
| B | Fail | 0.3333 | 1.0986 |
| C | Fail | 0.3333 | 1.0986 |
| D | Average | 0.3333 | 1.0986 |
| E | Average | 0.3333 | 1.0986 |
| F | Average | 0.3333 | 1.0986 |
| G | Excellent | 0.3333 | 1.0986 |
| H | Excellent | 0.3333 | 1.0986 |
| I | Excellent | 0.3333 | 1.0986 |

### 7.5 Calculate Average Initial Loss ($J$)
We add all nine individual losses and divide by the number of students ($n = 9$):
$$J = \frac{1}{n} \sum_{i=1}^{n} L_i = \frac{9 \times 1.0986}{9} = \mathbf{1.0986}$$

> **Summary of Terms:**
> - **Individual Loss:** One student's prediction error
> - **Total Loss:** Sum of all students' losses
> - **Average Loss ($J$):** Total Loss divided by number of students

---

## 8. Gradient Descent – Learning the Weights

Now our model has initial weights $= 0$, initial biases $= 0$, and initial average loss $= 1.0986$. Gradient Descent calculates how parameters should change to reduce loss.

### 8.1 Gradient Formulas
- **Weight Gradient:** $\frac{\partial J}{\partial w_k} = \frac{1}{n} \sum_{i=1}^{n} (p_{ik} - y_{ik}) X_i$
- **Bias Gradient:** $\frac{\partial J}{\partial b_k} = \frac{1}{n} \sum_{i=1}^{n} (p_{ik} - y_{ik})$

> **Why do we use $p - y$?**  
> The difference between predicted probability and actual value indicates the direction in which the model parameters should change.

### 8.2 Prediction Errors ($\text{Error} = p - y$)

For Student A (Predicted: $[0.3333, 0.3333, 0.3333]$, Actual: $[1, 0, 0]$):
$$\text{Error} = [0.3333 - 1, 0.3333 - 0, 0.3333 - 0] = [-0.6667, 0.3333, 0.3333]$$

#### Complete Error Table

| Student | $X$ | Actual $Y$ | Fail Error | Average Error | Excellent Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| A | 1 | $[1,0,0]$ | -0.6667 | 0.3333 | 0.3333 |
| B | 2 | $[1,0,0]$ | -0.6667 | 0.3333 | 0.3333 |
| C | 3 | $[1,0,0]$ | -0.6667 | 0.3333 | 0.3333 |
| D | 3 | $[0,1,0]$ | 0.3333 | -0.6667 | 0.3333 |
| E | 4 | $[0,1,0]$ | 0.3333 | -0.6667 | 0.3333 |
| F | 5 | $[0,1,0]$ | 0.3333 | -0.6667 | 0.3333 |
| G | 5 | $[0,0,1]$ | 0.3333 | 0.3333 | -0.6667 |
| H | 6 | $[0,0,1]$ | 0.3333 | 0.3333 | -0.6667 |
| I | 7 | $[0,0,1]$ | 0.3333 | 0.3333 | -0.6667 |

---

## 9. Calculate Gradients

### 9.1 Fail Class Weight Gradient ($\frac{\partial J}{\partial w_0}$)
$$\frac{\partial J}{\partial w_0} = \frac{1}{9} \sum (\text{Error} \times X)$$
Using exact fractions ($-\frac{2}{3}$ and $+\frac{1}{3}$):
$$= \frac{1}{9} \left[ \left(-\frac{2}{3} \times 1\right) + \left(-\frac{2}{3} \times 2\right) + \left(-\frac{2}{3} \times 3\right) + \left(\frac{1}{3} \times 3\right) + \left(\frac{1}{3} \times 4\right) + \left(\frac{1}{3} \times 5\right) + \left(\frac{1}{3} \times 5\right) + \left(\frac{1}{3} \times 6\right) + \left(\frac{1}{3} \times 7\right) \right]$$
$$= \frac{1}{27} \left[ -2 - 4 - 6 + 3 + 4 + 5 + 5 + 6 + 7 \right] = \frac{1}{27} \left[ -12 + 30 \right] = \frac{18}{27} = \mathbf{0.6667}$$

### 9.2 Average Class Weight Gradient ($\frac{\partial J}{\partial w_1}$)
$$= \frac{1}{27} \left[ 1 + 2 + 3 - 6 - 8 - 10 + 5 + 6 + 7 \right] = \frac{1}{27} \left[ 24 - 24 \right] = \frac{0}{27} = \mathbf{0.0000}$$

### 9.3 Excellent Class Weight Gradient ($\frac{\partial J}{\partial w_2}$)
$$= \frac{1}{27} \left[ 1 + 2 + 3 + 3 + 4 + 5 - 10 - 12 - 14 \right] = \frac{1}{27} \left[ 18 - 36 \right] = -\frac{18}{27} = \mathbf{-0.6667}$$

### 9.4 Calculate Bias Gradients ($\frac{\partial J}{\partial b_k}$)
- **Fail Bias:** $\frac{1}{27} (-2 - 2 - 2 + 1 + 1 + 1 + 1 + 1 + 1) = \mathbf{0}$
- **Average Bias:** $\frac{1}{27} (1 + 1 + 1 - 2 - 2 - 2 + 1 + 1 + 1) = \mathbf{0}$
- **Excellent Bias:** $\frac{1}{27} (1 + 1 + 1 + 1 + 1 + 1 - 2 - 2 - 2) = \mathbf{0}$

### 9.5 Summary of All Gradients

| Class | Bias Gradient | Weight Gradient |
| :--- | :---: | :---: |
| Fail | 0 | 0.6667 |
| Average | 0 | 0.0000 |
| Excellent | 0 | -0.6667 |

---

## 10. Update Weights and Biases

Update rules ($\alpha = 0.1$):
$$w_{\text{new}} = w_{\text{old}} - \alpha \times \text{Gradient}$$
$$b_{\text{new}} = b_{\text{old}} - \alpha \times \text{Bias Gradient}$$

- **Fail Weight:** $w_{0(\text{new})} = 0 - (0.1 \times 0.6667) = \mathbf{-0.06667}$
- **Average Weight:** $w_{1(\text{new})} = 0 - (0.1 \times 0) = \mathbf{0.00000}$
- **Excellent Weight:** $w_{2(\text{new})} = 0 - (0.1 \times -0.6667) = \mathbf{+0.06667}$
- **Biases:** All updated biases remain **$0$**.

### Updated Parameters After First Iteration

| Class | Updated Bias | Updated Weight |
| :--- | :---: | :---: |
| Fail | 0 | -0.06667 |
| Average | 0 | 0.00000 |
| Excellent | 0 | +0.06667 |

---

## 11. Repeat Training – Second Iteration and Beyond

Using updated parameters:
- $z_0 = -0.06667 X$
- $z_1 = 0$
- $z_2 = +0.06667 X$

For **Student A** ($X = 1$), logits are $z = [-0.06667, 0, 0.06667]$.

### Softmax Probabilities for Student A (Iteration 2)
- $e^{-0.06667} \approx 0.9355$, $e^0 = 1$, $e^{0.06667} \approx 1.0689$
- $\text{Denominator} = 0.9355 + 1 + 1.0689 = 3.0044$
- $P(\text{Fail}) \approx 31.14\%$, $P(\text{Average}) \approx 33.29\%$, $P(\text{Excellent}) \approx 35.57\%$

### Training Loss Progress Table

| Iteration | Average Loss ($J$) |
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

---

## 12. When Should We Stop Training?

1. **Condition 1 – Maximum Iterations:** Stop after a fixed number of epochs (e.g., 5,000 iterations).
2. **Condition 2 – Loss Convergence:** Stop when loss improvement drops below a threshold ($\Delta J < \text{tolerance}$).
3. **Condition 3 – Early Stopping:** Stop when validation loss starts increasing (prevents overfitting).

---

## 13. Final Learned Weights

After 5,000 full-batch Gradient Descent updates:

| Class | Learned Bias ($b$) | Learned Weight ($w$) |
| :--- | :---: | :---: |
| Fail | 8.26515 | -2.33485 |
| Average | 0.78679 | 0.17422 |
| Excellent | -9.05194 | 2.16063 |

### Final Learned Equations
- **Fail:** $z_0 = 8.26515 - 2.33485 X$
- **Average:** $z_1 = 0.78679 + 0.17422 X$
- **Excellent:** $z_2 = -9.05194 + 2.16063 X$

---

## 14. Final Prediction Using Trained Weights

Predict result for a new student with **$X = 4.5$ hours**:

- **Fail Logit ($z_0$):** $8.26515 - (2.33485 \times 4.5) \approx \mathbf{-2.24167}$
- **Average Logit ($z_1$):** $0.78679 + (0.17422 \times 4.5) \approx \mathbf{1.57079}$
- **Excellent Logit ($z_2$):** $-9.05194 + (2.16063 \times 4.5) \approx \mathbf{0.67088}$

---

## 15. Calculate Final Softmax Probabilities

- $e^{-2.24167} \approx 0.1062$
- $e^{1.57079} \approx 4.8119$
- $e^{0.67088} \approx 1.9559$
- **Denominator:** $0.1062 + 4.8119 + 1.9559 \approx 6.8740$

#### Final Probability Table

| Class | Probability | Percentage |
| :--- | :---: | :---: |
| Fail | 0.0155 | 1.55% |
| **Average** | **0.6999** | **69.99%** |
| Excellent | 0.2846 | 28.46% |
| **Total** | **1.0000** | **100%** |

---

## 16. Final Prediction – Argmax

$$\text{Prediction} = \text{argmax}(0.0155, 0.6999, 0.2846) = \mathbf{\text{Average}}$$

The model predicts that a student who studies 4.5 hours belongs to the **Average** class.

---

## 17. Model Evaluation

### 17.1 Prediction Matrix Across Training Dataset

| Student | Actual Result | Predicted Result |
| :---: | :---: | :---: |
| A | Fail | Fail |
| B | Fail | Fail |
| C | Fail | Average |
| D | Average | Average |
| E | Average | Average |
| F | Average | Excellent |
| G | Excellent | Excellent |
| H | Excellent | Excellent |
| I | Excellent | Excellent |

- **Correct Predictions:** 7 / 9
- **Training Accuracy:** $\frac{7}{9} \approx \mathbf{77.78\%}$

### 17.3 Confusion Matrix

| Actual \ Predicted | Fail | Average | Excellent |
| :--- | :---: | :---: | :---: |
| **Fail** | 2 | 1 | 0 |
| **Average** | 0 | 2 | 1 |
| **Excellent** | 0 | 0 | 3 |

---

## 18. Complete Multiclass Logistic Regression Workflow

1. Collect the dataset.
2. Identify input $X$ and output $Y$.
3. Perform Label Encoding.
4. Perform One-Hot Encoding.
5. Initialize weights and biases.
6. Create one linear equation for each class.
7. Calculate logits.
8. Apply Softmax.
9. Calculate class probabilities.
10. Select predicted class using Argmax.
11. Calculate individual Cross Entropy Loss.
12. Calculate average Cross Entropy Loss.
13. Calculate gradients.
14. Update weights and biases.
15. Repeat training iterations.
16. Check stopping conditions.
17. Use final learned weights for new predictions.
18. Evaluate the model.

---

## 19. Important Exam Formula Sheet

| Concept | Formula |
| :--- | :--- |
| **Linear Score** | $z_k = b_k + w_k X$ |
| **Softmax** | $P_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$ |
| **Prediction** | $\hat{y} = \text{argmax}(P_k)$ |
| **Cross Entropy** | $L_i = -\sum_{k=1}^{K} y_{ik} \ln(p_{ik})$ |
| **Average Loss** | $J = \frac{1}{n}\sum_{i=1}^{n} L_i$ |
| **Weight Gradient** | $\frac{\partial J}{\partial w_k} = \frac{1}{n}\sum (p_{ik} - y_{ik}) X_i$ |
| **Bias Gradient** | $\frac{\partial J}{\partial b_k} = \frac{1}{n}\sum (p_{ik} - y_{ik})$ |
| **Weight Update** | $w_{\text{new}} = w_{\text{old}} - \alpha \times \text{Gradient}$ |
| **Bias Update** | $b_{\text{new}} = b_{\text{old}} - \alpha \times \text{Bias Gradient}$ |
| **Accuracy** | $\frac{\text{Correct Predictions}}{\text{Total Predictions}}$ |

---

## 20. Common Exam Mistakes

1. Confusing logits with probabilities.
2. Forgetting to include all classes in the Softmax denominator.
3. Forgetting that Softmax probabilities must sum to 1.
4. Confusing Label Encoding with One-Hot Encoding.
5. Using the wrong actual class in the Cross Entropy formula.
6. Forgetting the negative sign in Cross Entropy.
7. Forgetting to average the individual losses.
8. Forgetting to multiply the weight error by $X$ when calculating weight gradients.
9. Forgetting that bias gradients do not multiply by $X$.
10. Forgetting the minus sign in the Gradient Descent update formula.
11. Assuming one iteration is enough to train the model.
12. Assuming training loss must become exactly zero.
13. Confusing training accuracy with test accuracy.

---

## 21. Final Revision – Understand the Purpose of Each Step

| Component | Purpose |
| :--- | :--- |
| **Label Encoding** | Convert class names into numbers |
| **One-Hot Encoding** | Represent the actual class as a vector |
| **Linear Equations** | Calculate scores for each class |
| **Logits** | Represent raw class scores |
| **Softmax** | Convert scores into probabilities |
| **Argmax** | Select the class with the highest probability |
| **Cross Entropy** | Measure prediction error |
| **Gradient** | Calculate parameter changes |
| **Gradient Descent** | Update weights and biases |
| **Stopping Criteria** | Decide when to stop training |
| **Accuracy** | Measure correct predictions |