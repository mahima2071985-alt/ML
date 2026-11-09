# Linear Basis Function Models – Exam Notes

---

## 1. What is a Linear Basis Function Model?

A Linear Basis Function Model is a regression model that transforms the original input features into new features using mathematical functions called basis functions.

The purpose is to represent nonlinear patterns in the data.

### Normal Linear Regression

$$
\hat{y} = w_0 + w_1X
$$

This represents a straight-line relationship.

### Linear Basis Function Model

$$
\hat{y} = w_0 + w_1X + w_2X^2
$$

This can represent a curved relationship.

---

## 2. What is a Basis Function?

A basis function transforms an original input feature into a new feature.

For example:

$$
\phi_1(X) = X
$$

$$
\phi_2(X) = X^2
$$

Other possible basis functions:

| Basis function | Meaning |
| :--- | :--- |
| $\phi_1(X) = X$ | Original feature |
| $\phi_2(X) = X^2$ | Square |
| $\phi_3(X) = X^3$ | Cube |
| $\phi_4(X) = \sqrt{X}$ | Square root |
| $\phi_5(X) = \log X$ | Logarithm |
| $\phi_6(X) = e^X$ | Exponential |

We do not need to use all these functions. We select suitable functions based on the problem and data pattern.

---

## 3. Why Do We Use Basis Functions?

Consider this dataset:

| $X$ | $Y$ |
| :---: | :---: |
| 1 | 1 |
| 2 | 4 |
| 3 | 9 |
| 4 | 16 |
| 5 | 25 |

Observe:

$$
Y = X^2
$$

The relationship is curved, not a straight line.

A normal linear model:

$$
\hat{y} = w_0 + w_1X
$$

may not represent this pattern accurately.

So we introduce a new feature:

$$
\phi_2(X) = X^2
$$

The new model becomes:

$$
\boxed{\hat{y} = w_0 + w_1X + w_2X^2}
$$

This gives the model the ability to represent the quadratic pattern.

---

## 4. Transform the Dataset

Original dataset:

| $X$ | $Y$ |
| :---: | :---: |
| 1 | 1 |
| 2 | 4 |
| 3 | 9 |
| 4 | 16 |
| 5 | 25 |

Create two basis functions:

$$
\phi_1(X) = X
$$

$$
\phi_2(X) = X^2
$$

Transformed dataset:

| $X$ | $\phi_1(X) = X$ | $\phi_2(X) = X^2$ | $Y$ |
| :---: | :---: | :---: | :---: |
| 1 | 1 | 1 | 1 |
| 2 | 2 | 4 | 4 |
| 3 | 3 | 9 | 9 |
| 4 | 4 | 16 | 16 |
| 5 | 5 | 25 | 25 |

The model now has two transformed features and one intercept.

---

## 5. General Model Equation

$$
\boxed{\hat{y} = w_0 + w_1\phi_1(X) + w_2\phi_2(X)}
$$

For our quadratic example:

$$
\boxed{\hat{y} = w_0 + w_1X + w_2X^2}
$$

Where:

| Symbol | Meaning |
| :---: | :--- |
| $w_0$ | Intercept |
| $w_1$ | Weight for $X$ |
| $w_2$ | Weight for $X^2$ |
| $\phi_1(X)$ | First basis function |
| $\phi_2(X)$ | Second basis function |

---

## 6. Why Is It Still Called Linear?

This is an important exam question.

Consider:

$$
\hat{y} = w_0 + w_1X + w_2X^2
$$

The model is nonlinear with respect to the original input ($X$), because it contains $X^2$.

However, it is linear with respect to the parameters (weights):

$$
w_0, \quad w_1, \quad w_2
$$

Each weight is multiplied by a feature and then added.

Therefore:
It is nonlinear with respect to input $X$, but linear with respect to model parameters.
That is why it is called a **Linear Basis Function Model**.

---

## 7. Numerical Problem Using OLS

Dataset:

| $X$ | $Y$ |
| :---: | :---: |
| 1 | 1 |
| 2 | 4 |
| 3 | 9 |
| 4 | 16 |
| 5 | 25 |

Model:

$$
\hat{y} = w_0 + w_1X + w_2X^2
$$

### Step 1: Construct the design matrix

$$
X = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 4 \\ 1 & 3 & 9 \\ 1 & 4 & 16 \\ 1 & 5 & 25 \end{bmatrix}
$$

Target vector:

$$
Y = \begin{bmatrix} 1 \\ 4 \\ 9 \\ 16 \\ 25 \end{bmatrix}
$$

### Step 2: Use the OLS Normal Equation

$$
\beta = (X^TX)^{-1}X^TY
$$

Where:

$$
\beta = \begin{bmatrix} w_0 \\ w_1 \\ w_2 \end{bmatrix}
$$

### Step 3: Calculate $X^TX$

$$
X^TX = \begin{bmatrix} 5 & 15 & 55 \\ 15 & 55 & 225 \\ 55 & 225 & 979 \end{bmatrix}
$$

### Step 4: Calculate $X^TY$

$$
X^TY = \begin{bmatrix} 55 \\ 225 \\ 979 \end{bmatrix}
$$

### Step 5: Solve the normal equations

$$
X^TX\beta = X^TY
$$

The solution is:

$$
\boxed{w_0 = 0, \quad w_1 = 0, \quad w_2 = 1}
$$

### Step 6: Final model

$$
\hat{y} = 0 + 0X + 1X^2
$$

$$
\boxed{\hat{y} = X^2}
$$

### Step 7: Verify predictions

| $X$ | Actual $Y$ | Predicted $\hat{Y}$ |
| :---: | :---: | :---: |
| 1 | 1 | 1 |
| 2 | 4 | 4 |
| 3 | 9 | 9 |
| 4 | 16 | 16 |
| 5 | 25 | 25 |

All predictions match the actual values.

### Step 8: Calculate SSE and MSE

$$
\text{SSE} = \sum(\hat{y} - Y)^2
$$

$$
\text{SSE} = 0
$$

$$
\text{MSE} = \frac{\text{SSE}}{n} = \frac{0}{5}
$$

$$
\boxed{\text{MSE} = 0}
$$

The quadratic model perfectly represents this particular dataset.

---

## 8. Linear Basis Function Model Using Gradient Descent

The same model can also be trained using Gradient Descent.

Model:

$$
\hat{y} = w_0 + w_1X + w_2X^2
$$

### Step 1: Initialize weights

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

### Step 2: Calculate initial predictions

All predictions are zero.

### Step 3: Calculate errors

$$
\text{Error} = \hat{y} - Y
$$

$$
[-1, -4, -9, -16, -25]
$$

### Step 4: Calculate initial cost

$$
J(w) = \frac{1}{2n} \sum(\hat{y} - Y)^2
$$

$$
= \frac{1}{10}[1 + 16 + 81 + 256 + 625]
$$

$$
\boxed{J = 97.9}
$$

### Step 5: Calculate gradients

For $w_0$:

$$
\frac{\partial J}{\partial w_0} = \frac{1}{n} \sum \text{Error}
$$

$$
= \frac{-55}{5}
$$

$$
\boxed{-11}
$$

For $w_1$:

$$
\frac{\partial J}{\partial w_1} = \frac{1}{n} \sum(\text{Error} \times X)
$$

$$
= \frac{-225}{5}
$$

$$
\boxed{-45}
$$

For $w_2$:

$$
\frac{\partial J}{\partial w_2} = \frac{1}{n} \sum(\text{Error} \times X^2)
$$

$$
= \frac{-979}{5}
$$

$$
\boxed{-195.8}
$$

### Step 6: Update weights

Formula:

$$
w_j^{\text{new}} = w_j^{\text{old}} - \alpha \times \text{Gradient}
$$

Using $\alpha = 0.001$:

$$
w_0 = 0 - (0.001 \times -11) = \boxed{0.011}
$$

$$
w_1 = 0 - (0.001 \times -45) = \boxed{0.045}
$$

$$
w_2 = 0 - (0.001 \times -195.8) = \boxed{0.1958}
$$

New model:

$$
\boxed{\hat{y} = 0.011 + 0.045X + 0.1958X^2}
$$

### Step 7: Calculate new cost

New predictions:

$$
[0.2518, \ 0.8842, \ 1.9082, \ 3.3238, \ 5.1310]
$$

New cost:

$$
\boxed{J = 61.60248}
$$

Cost reduced from $97.9$ to $61.60248$.
This completes the first Gradient Descent update.

---

## 9. Multiple Original Features – Five Coefficients Example

Suppose we have two original input features:
- $X_1$ = House size
- $X_2$ = Number of bedrooms

We can create basis functions for both features.

For size:

$$
\phi_1(X_1) = X_1
$$

$$
\phi_2(X_1) = X_1^2
$$

For bedrooms:

$$
\phi_3(X_2) = X_2
$$

$$
\phi_4(X_2) = X_2^2
$$

Now we have four transformed features.

Transformed dataset:

| House | $X_1$ | $X_1^2$ | $X_2$ | $X_2^2$ | $Y$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| A | 1 | 1 | 1 | 1 | 10 |
| B | 2 | 4 | 1 | 1 | 15 |
| C | 3 | 9 | 2 | 4 | 20 |
| D | 4 | 16 | 2 | 4 | 25 |
| E | 5 | 25 | 3 | 9 | 30 |

Model equation:

$$
\boxed{\hat{y} = w_0 + w_1X_1 + w_2X_1^2 + w_3X_2 + w_4X_2^2}
$$

There are:
- Four transformed feature weights.
- One intercept.

Therefore:

$$
\boxed{\text{Total coefficients} = 5}
$$

The five coefficients are:

$$
w_0, \ w_1, \ w_2, \ w_3, \ w_4
$$

### Interaction features

We can add another basis function:

$$
\phi_5(X_1, X_2) = X_1X_2
$$

Then:

$$
\hat{y} = w_0 + w_1X_1 + w_2X_1^2 + w_3X_2 + w_4X_2^2 + w_5X_1X_2
$$

Now there are six coefficients, including the intercept.

---

## 10. General Formula

For one or more input features, the general Linear Basis Function Model can be written as:

$$
\boxed{\hat{y} = w_0 + \sum_{j=1}^{M} w_j\phi_j(X)}
$$

Where:

| Symbol | Meaning |
| :---: | :--- |
| $M$ | Number of transformed features |
| $w_0$ | Intercept |
| $w_j$ | Weight for the $j$-th feature |
| $\phi_j(X)$ | $j$-th basis function |

Total coefficients:

$$
\boxed{M + 1}
$$

---

## 11. Advantages and Limitations

### Advantages
1. Can represent nonlinear relationships.
2. Uses familiar linear regression techniques.
3. Can use OLS or Gradient Descent.
4. Flexible feature transformations can represent different patterns.

### Limitations
1. Too many basis functions can cause overfitting.
2. Higher powers can produce very large feature values.
3. Selecting suitable basis functions is important.
4. Gradient Descent may need feature scaling and learning-rate tuning.

---

## 12. Common Exam Questions

- **Q1. What is a basis function?**
  A basis function transforms an original input feature into a new feature to help the model represent different data patterns.

- **Q2. Why is it called a Linear Basis Function Model?**
  Because the model is linear with respect to its weights, even if the input features are transformed nonlinearly.

- **Q3. What is the purpose of $X^2$?**
  To help represent a quadratic or curved relationship.

- **Q4. Can we use multiple input features?**
  Yes. We can create basis functions from each original feature and also create interaction features.

- **Q5. What is an interaction feature?**
  A feature formed by combining two or more original features, such as $X_1X_2$.

- **Q6. What is the main risk of adding too many basis functions?**
  Overfitting.

---

## 13. Quick Exam Revision

| Concept | Remember |
| :--- | :--- |
| Normal Linear Regression | $w_0 + w_1X$ |
| Quadratic Basis Model | $w_0 + w_1X + w_2X^2$ |
| Basis Function | Transforms input into a new feature |
| OLS | Directly estimates weights |
| Gradient Descent | Iteratively updates weights |
| Interaction | $X_1X_2$ |
| Five-coefficient example | Four features + intercept |
| Main risk | Overfitting |

**Memory shortcut:**
$$
\text{Original features} \longrightarrow \text{Basis functions} \longrightarrow \text{Transformed dataset} \longrightarrow \text{Regression model} \longrightarrow \text{Estimate weights} \longrightarrow \text{Evaluate predictions}
$$