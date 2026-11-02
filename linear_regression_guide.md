# Simple Linear Regression: Mathematics, Gradient Descent, and Python Implementation

## Learning objectives

By the end of this guide, you will understand:

1. What Simple Linear Regression does.
2. How to calculate predictions, errors, SSE, and MSE.
3. How Gradient Descent updates slope and intercept.
4. How to implement Gradient Descent manually using NumPy.
5. How to build a real-world Linear Regression model using Scikit-learn.
6. How to evaluate, visualize, save, and reuse a trained model.

---

# Part 1: Understand Simple Linear Regression

Simple Linear Regression models the relationship between one input variable and one output variable.

Example:

- **X:** Hours studied
- **Y:** Exam marks

The model tries to find a straight line that predicts marks from study hours.

The equation is:

$$
\hat{y} = mx + c
$$

| Symbol | Meaning |
|---|---|
| $x$ | Input feature |
| $y$ | Actual target value |
| $\hat y$ | Predicted target value |
| $m$ | Slope (coefficient or weight) |
| $c$ | Intercept (bias) |

The slope controls how much the predicted value changes when X increases by one unit. The intercept is the predicted value when X is zero.

## Example dataset

We will first use this small dataset to understand every mathematical calculation.

| Student | Hours (X) | Actual marks (Y) |
|---|---:|---:|
| A | 1 | 3 |
| B | 2 | 5 |
| C | 3 | 7 |
| D | 4 | 9 |

Our objective is to find the best values of $m$ and $c$.

The relationship in this example is:

$$
y = 2x + 1
$$

We will pretend that we do not know this equation and learn how a model can find it.

---

# Part 2: Mathematical implementation using Gradient Descent

Gradient Descent is an optimization algorithm. It repeatedly adjusts the slope and intercept to reduce prediction error.

## Step 1: Initialize the parameters

At the beginning, assume:

$$
m = 0,\qquad c = 0
$$

The initial model is:

$$
\hat y = 0x + 0
$$

Calculate predictions:

| X | Actual Y | Predicted Y |
|---:|---:|---:|
| 1 | 3 | 0 |
| 2 | 5 | 0 |
| 3 | 7 | 0 |
| 4 | 9 | 0 |

All predictions are zero because both parameters are initialized to zero.

## Step 2: Calculate the errors

The prediction error is:

$$
Error = y - \hat y
$$

| X | Actual Y | Predicted Y | Error |
|---:|---:|---:|---:|
| 1 | 3 | 0 | 3 |
| 2 | 5 | 0 | 5 |
| 3 | 7 | 0 | 7 |
| 4 | 9 | 0 | 9 |

## Step 3: Calculate SSE

SSE means Sum of Squared Errors.

$$
SSE = \sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

Square each error and add them:

| X | Error | Squared error |
|---:|---:|---:|
| 1 | 3 | $3^2=9$ |
| 2 | 5 | $5^2=25$ |
| 3 | 7 | $7^2=49$ |
| 4 | 9 | $9^2=81$ |

$$
SSE = 9+25+49+81
$$

$$
\boxed{SSE=164}
$$

## Step 4: Calculate MSE (cost function)

MSE means Mean Squared Error. It is the average of the squared errors.

$$
J(m,c)=MSE=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2
$$

There are four data points, so:

$$
MSE=\frac{164}{4}
$$

$$
\boxed{MSE=41}
$$

The goal of Gradient Descent is to reduce this cost.

> SSE is the sum of squared errors. MSE is SSE divided by the number of observations.

## Step 5: Understand the Gradient Descent update rules

We need to calculate how the cost changes when we change the slope and intercept.

For the MSE cost function:

$$
\boxed{\frac{\partial J}{\partial m}=
\frac{2}{n}\sum_{i=1}^{n}(\hat y_i-y_i)x_i}
$$

$$
\boxed{\frac{\partial J}{\partial c}=
\frac{2}{n}\sum_{i=1}^{n}(\hat y_i-y_i)}
$$

The update rules are:

$$
\boxed{m_{\text{new}}=m-\alpha\frac{\partial J}{\partial m}}
$$

$$
\boxed{c_{\text{new}}=c-\alpha\frac{\partial J}{\partial c}}
$$

Where:

- $\alpha$ is the learning rate.
- $n$ is the number of observations.
- $\partial J/\partial m$ is the slope gradient.
- $\partial J/\partial c$ is the intercept gradient.

The learning rate controls the size of each update.

## Step 6: Calculate the slope gradient for iteration 1

Current values:

$$
m=0,\qquad c=0
$$

Predictions are all zero. Calculate $(\hat y-y)x$:

| X | Actual Y | Predicted Y | $(\hat y-y)x$ |
|---:|---:|---:|---:|
| 1 | 3 | 0 | $(0-3)\times1=-3$ |
| 2 | 5 | 0 | $(0-5)\times2=-10$ |
| 3 | 7 | 0 | $(0-7)\times3=-21$ |
| 4 | 9 | 0 | $(0-9)\times4=-36$ |
| | | | **-70** |

Therefore:

$$
\frac{\partial J}{\partial m}
=\frac{2}{4}(-70)
$$

$$
\boxed{\frac{\partial J}{\partial m}=-35}
$$

## Step 7: Calculate the intercept gradient for iteration 1

Add the prediction errors $(\hat y-y)$:

$$
(0-3)+(0-5)+(0-7)+(0-9)=-24
$$

Therefore:

$$
\frac{\partial J}{\partial c}
=\frac{2}{4}(-24)
$$

$$
\boxed{\frac{\partial J}{\partial c}=-12}
$$

## Step 8: Update the slope and intercept

Choose a learning rate:

$$
\alpha=0.01
$$

Update the slope:

$$
m_{\text{new}}=0-(0.01\times -35)
$$

$$
\boxed{m_{\text{new}}=0.35}
$$

Update the intercept:

$$
c_{\text{new}}=0-(0.01\times -12)
$$

$$
\boxed{c_{\text{new}}=0.12}
$$

The new model equation is:

$$
\boxed{\hat y=0.35x+0.12}
$$

## Step 9: Calculate new predictions

Use the updated equation:

$$
\hat y=0.35x+0.12
$$

| X | Actual Y | Calculation | Predicted Y |
|---:|---:|---|---:|
| 1 | 3 | $0.35(1)+0.12$ | 0.47 |
| 2 | 5 | $0.35(2)+0.12$ | 0.82 |
| 3 | 7 | $0.35(3)+0.12$ | 1.17 |
| 4 | 9 | $0.35(4)+0.12$ | 1.52 |

## Step 10: Calculate the new error and MSE

| X | Actual Y | Predicted Y | Error $y-\hat y$ | Squared error |
|---:|---:|---:|---:|---:|
| 1 | 3 | 0.47 | 2.53 | 6.4009 |
| 2 | 5 | 0.82 | 4.18 | 17.4724 |
| 3 | 7 | 1.17 | 5.83 | 33.9889 |
| 4 | 9 | 1.52 | 7.48 | 55.9504 |

$$
SSE=6.4009+17.4724+33.9889+55.9504
$$

$$
\boxed{SSE=113.8126}
$$

$$
MSE=\frac{113.8126}{4}
$$

$$
\boxed{MSE=28.45315}
$$

The initial MSE was 41. After one update it is approximately 28.45315. The error has decreased.

**Note:** The squared error for the fourth row is $7.48^2=55.9504$. Small arithmetic differences in manually rounded calculations can occur.

## Step 11: Repeat Gradient Descent

Repeat these operations:

1. Calculate predictions using the current slope and intercept.
2. Calculate the slope gradient.
3. Calculate the intercept gradient.
4. Update the slope.
5. Update the intercept.
6. Calculate the new MSE.
7. Continue until the cost converges or a stopping condition is reached.

For this dataset, Gradient Descent approaches:

$$
m\approx2,\qquad c\approx1
$$

The learned equation approaches:

$$
\boxed{\hat y=2x+1}
$$

Predictions:

| X | Actual Y | Predicted Y |
|---:|---:|---:|
| 1 | 3 | 3 |
| 2 | 5 | 5 |
| 3 | 7 | 7 |
| 4 | 9 | 9 |

The SSE and MSE approach zero for this perfectly linear example.

## When does Gradient Descent stop?

Typical stopping conditions include:

- The change in cost between iterations becomes very small.
- The gradients become sufficiently close to zero.
- A configured maximum number of iterations is reached.

Real-world data usually contains noise, so the error generally does not become exactly zero.

---

# Part 3: Implement Gradient Descent manually using Python

This section reproduces the mathematical process using NumPy. It is useful for understanding what happens inside an optimization algorithm.

## Step 1: Install the libraries

In Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter notebook
```

Create a notebook named:

`simple_linear_regression.ipynb`

## Step 2: Import libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Libraries imported successfully!")
```

## Step 3: Create the dataset

```python
X = np.array([1, 2, 3, 4])
y = np.array([3, 5, 7, 9])

df = pd.DataFrame({
    "Hours": X,
    "Actual_Marks": y
})

df
```

## Step 4: Visualize the data

```python
plt.scatter(X, y)
plt.xlabel("Hours Studied")
plt.ylabel("Exam Marks")
plt.title("Hours Studied vs Exam Marks")
plt.grid(True)
plt.show()
```

## Step 5: Initialize parameters

```python
m = 0.0
c = 0.0

learning_rate = 0.01
epochs = 5000

cost_history = []
n = len(X)
```

## Step 6: Run Gradient Descent

```python
for epoch in range(epochs):

    # 1. Calculate predictions
    y_pred = m * X + c

    # 2. Calculate gradients
    dm = (2 / n) * np.sum((y_pred - y) * X)
    dc = (2 / n) * np.sum(y_pred - y)

    # 3. Update slope and intercept
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    # 4. Calculate predictions after the update
    y_pred = m * X + c

    # 5. Calculate MSE
    mse = np.mean((y_pred - y) ** 2)

    # 6. Store cost
    cost_history.append(mse)

    if epoch % 500 == 0:
        print(
            f"Epoch: {epoch}, "
            f"Slope: {m:.4f}, "
            f"Intercept: {c:.4f}, "
            f"MSE: {mse:.6f}"
        )
```

## Step 7: Display the learned equation

```python
print("Final slope:", m)
print("Final intercept:", c)
print(f"Final equation: y = {m:.4f}x + {c:.4f}")
```

The values should be close to slope 2 and intercept 1.

## Step 8: Plot the regression line

```python
y_pred = m * X + c

plt.scatter(X, y, label="Actual data")
plt.plot(X, y_pred, label="Gradient Descent line")

plt.xlabel("Hours Studied")
plt.ylabel("Exam Marks")
plt.title("Linear Regression using Gradient Descent")

plt.legend()
plt.grid(True)
plt.show()
```

## Step 9: Plot cost reduction

```python
plt.plot(cost_history)

plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent Cost Reduction")

plt.grid(True)
plt.show()
```

The curve should decrease and flatten as the model converges.

## Step 10: Predict a new value

```python
new_hours = np.array([5])
predicted_marks = m * new_hours + c

print("Predicted marks:", predicted_marks[0])
```

The predicted result should be close to 11.

---

# Part 4: Real-world implementation using Scikit-learn

In practical Machine Learning projects, we usually use an existing library function instead of manually writing the Gradient Descent algorithm.

For ordinary Linear Regression, Scikit-learn provides:

```python
from sklearn.linear_model import LinearRegression
```

**Important mathematical distinction:** Scikit-learn's `LinearRegression` uses Ordinary Least Squares (OLS), not iterative Gradient Descent by default. OLS finds coefficients by solving the least-squares problem using numerical linear algebra. The resulting objective is the same squared-error objective, but the optimization method is different.

## Step 1: Create the project structure

```text
simple-linear-regression/
│
├── data/
│   └── student_marks.csv
│
├── models/
│
├── simple_linear_regression.ipynb
└── README.md
```

## Step 2: Import the required libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

print("Libraries imported successfully!")
```

## Step 3: Create and save a sample CSV dataset

For this practical example, create a dataset with ten records.

```python
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [15, 20, 30, 35, 45, 50, 60, 65, 75, 85]
}

df = pd.DataFrame(data)

df.to_csv("data/student_marks.csv", index=False)

df
```

This is sample data. In a real project, the CSV would normally come from an existing data source.

## Step 4: Load the dataset

```python
df = pd.read_csv("data/student_marks.csv")

print("Dataset shape:", df.shape)

display(df.head())
```

Inspect the data:

```python
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
display(df.describe())
```

## Step 5: Visualize the dataset

```python
plt.scatter(df["Hours"], df["Marks"])

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")

plt.grid(True)
plt.show()
```

## Step 6: Separate features and target

```python
X = df[["Hours"]]
y = df["Marks"]

print("X shape:", X.shape)
print("y shape:", y.shape)
```

- X contains the input feature.
- y contains the target that the model must predict.
- X is two-dimensional because Scikit-learn expects a feature matrix.

## Step 7: Split training and testing data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)
```

With ten records and a 20% test size, there will be eight training records and two test records.

`random_state=42` makes the split reproducible. It does not mean that 42 is a special ML value.

## Step 8: Create the Linear Regression model

```python
model = LinearRegression()
```

This creates the model object. It has not learned the dataset yet.

## Step 9: Train the model

```python
model.fit(X_train, y_train)
```

The `.fit()` method learns the coefficient and intercept from the training data.

For `LinearRegression`, Scikit-learn uses Ordinary Least Squares.

## Step 10: Extract the learned parameters

```python
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print(
    f"Regression equation: "
    f"y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}"
)
```

The equation is:

$$
\hat y = \text{coefficient}\times x+\text{intercept}
$$

## Step 11: Predict using test data

```python
y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Marks": y_test.values,
    "Predicted Marks": y_pred
})

display(results)
```

The `.predict()` method uses the trained model to generate predictions for unseen inputs.

## Step 12: Evaluate the model

### Mean Squared Error

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

```python
mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)
```

Lower MSE generally indicates smaller squared prediction errors.

### Mean Absolute Error

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
$$

```python
mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)
```

MAE tells us the average absolute prediction error in marks.

### R-squared

```python
r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)
```

R² measures how much variation in the target is explained by the model relative to predicting the test-set mean. It can be negative on test data if the model performs worse than that baseline.

### Display all evaluation metrics

```python
print("MODEL EVALUATION")
print("----------------")
print("MSE:", mse)
print("MAE:", mae)
print("R2 Score:", r2)
```

## Step 13: Plot the trained regression line

```python
plt.scatter(X, y, label="Actual data")

plt.plot(
    X,
    model.predict(X),
    color="red",
    label="Regression line"
)

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Simple Linear Regression using Scikit-learn")

plt.legend()
plt.grid(True)
plt.show()
```

## Step 14: Predict a completely new observation

```python
new_student = pd.DataFrame({
    "Hours": [5]
})

predicted_marks = model.predict(new_student)

print("Study hours:", new_student["Hours"].iloc[0])
print("Predicted marks:", predicted_marks[0])
```

Use the same feature name and structure that the model was trained with.

## Step 15: Save the trained model

```python
joblib.dump(model, "models/linear_regression_model.pkl")

print("Model saved successfully!")
```

The trained model is now saved as a file.

## Step 16: Load and reuse the saved model

```python
loaded_model = joblib.load(
    "models/linear_regression_model.pkl"
)

new_data = pd.DataFrame({
    "Hours": [6]
})

prediction = loaded_model.predict(new_data)

print("Predicted marks:", prediction[0])
```

This allows another Python program or application to load the trained model and make predictions without training it again.

> Only load model files from trusted sources. Serialized Python objects can execute code when loaded.

---

# Part 5: Understand the difference between both approaches

| Topic | Manual Gradient Descent | Scikit-learn LinearRegression |
|---|---|---|
| Model creation | Write the mathematical logic | `LinearRegression()` |
| Training | Implement update loop | `.fit()` |
| Optimization | Iterative Gradient Descent | Ordinary Least Squares |
| Slope | Calculate and update manually | `model.coef_` |
| Intercept | Calculate and update manually | `model.intercept_` |
| Prediction | `m * X + c` | `.predict()` |
| Error evaluation | Calculate manually | Scikit-learn metrics |
| Practical use | Learning and custom optimization | Standard regression projects |

## Complete real-world Machine Learning workflow

```text
1. Data collection
       |
       v
2. Data exploration and cleaning
       |
       v
3. Feature and target separation
       |
       v
4. Train-test split
       |
       v
5. Create model
   LinearRegression()
       |
       v
6. Train model
   model.fit(X_train, y_train)
       |
       v
7. Make predictions
   model.predict(X_test)
       |
       v
8. Evaluate model
   MSE, MAE, R2
       |
       v
9. Save trained model
   joblib.dump()
       |
       v
10. Load and reuse model
```

---

# Part 6: Important functions for revision

| Function / attribute | Purpose |
|---|---|
| `pd.read_csv()` | Load a CSV dataset |
| `df.head()` | View the first records |
| `df.info()` | Inspect columns and data types |
| `df.isnull().sum()` | Check missing values |
| `train_test_split()` | Split data into training and testing sets |
| `LinearRegression()` | Create a Linear Regression model |
| `model.fit()` | Train the model |
| `model.coef_` | Retrieve learned slope/coefficient |
| `model.intercept_` | Retrieve learned intercept |
| `model.predict()` | Predict target values |
| `mean_squared_error()` | Calculate MSE |
| `mean_absolute_error()` | Calculate MAE |
| `r2_score()` | Calculate R² |
| `joblib.dump()` | Save a trained model |
| `joblib.load()` | Load a saved model |

# Part 7: Exam revision — formulas

| Concept | Formula |
|---|---|
| Linear equation | $\hat y=mx+c$ |
| Prediction error | $e=y-\hat y$ |
| SSE | $\sum(y-\hat y)^2$ |
| MSE | $\frac{1}{n}\sum(y-\hat y)^2$ |
| Slope gradient | $\frac{2}{n}\sum(\hat y-y)x$ |
| Intercept gradient | $\frac{2}{n}\sum(\hat y-y)$ |
| Slope update | $m_{\text{new}}=m-\alpha\frac{\partial J}{\partial m}$ |
| Intercept update | $c_{\text{new}}=c-\alpha\frac{\partial J}{\partial c}$ |
| MAE | $\frac{1}{n}\sum|y-\hat y|$ |

# Part 8: Test your understanding

1. What is the purpose of Simple Linear Regression?
2. What is the difference between slope and intercept?
3. Why do we calculate SSE and MSE?
4. What does the learning rate control in Gradient Descent?
5. Why do we update both slope and intercept?
6. What is the difference between `.fit()` and `.predict()`?
7. Why do we split data into training and testing sets?
8. Does Scikit-learn's `LinearRegression()` use Gradient Descent by default?
9. What does R² tell us?
10. Why do we save a trained model?

## Final takeaway

- **Mathematics:** Understand how the line, errors, cost function, gradients, and parameter updates work.
- **Gradient Descent implementation:** Learn the optimization process by writing it with NumPy.
- **Practical ML development:** Use Scikit-learn's `LinearRegression()` with `.fit()` and `.predict()` to build a standard regression model.

Learn the mathematics first, understand the algorithm, and then use the library functions in practical projects.
