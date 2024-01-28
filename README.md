# Build Simple and Multiple Linear Regression from Scratch

In this notebook I will implement a linear regression model, and test it as a simple linear regression model in predicting sales given a TV marketing budget, and as a Multiple linear regression model in predicting house prices based on their size and quality.

# Objective
The objective is to solidify my understanding of linear regression model and their underhood details.

# Table of Contents

- [ 1 - Simple Linear Regression](#1)
  - [ 1.1 - Simple Linear Regression Model](#1.1)
  - [ 1.2 - Dataset](#1.2)
- [ 2 - Implementation of the Linear Regression Model](#2)
  - [ 2.1 - Defining the Linear Regression Model Structure (Number of Features)](#2.1)
  - [ 2.2 - Initialize the Model's Parameters](#2.2)
  - [ 2.3 - The Loop](#2.3)
      - [ 2.3.1 - Forward Propagation](#2.3.1)
      - [ 2.3.2 - Cost Function](#2.3.2)
      - [ 2.3.3 - Back Propagation](#2.3.3)
      - [ 2.3.4 - Update Parameters](#2.3.4)
  - [ 2.4 - Integrate parts 2.1, 2.2 and 2.3 in linear_regression() and make predictions](#2.4)
- [ 3 - Multiple Linear Regression](#3)
  - [ 3.1 - Multipe Linear Regression Model](#3.1)
  - [ 3.2 - Dataset](#3.2)
  - [ 3.3 - Performance of the Linear Regression Model for Multiple Linear Regression](#3.3)


 <a name='1'></a>
## 1 - Simple Linear Regression

<a name='1.1'></a>
### 1.1 - Simple Linear Regression Model

We can describe a simple linear regression model as

$$\hat{y} = wx + b,\tag{1}$$

where $\hat{y}$ is a prediction of dependent variable $y$ based on independent variable $x$ using a line equation with the slope $w$ and intercept $b$. 

Given a set of training data points $(x_1, y_1)$, ..., $(x_m, y_m)$, you will find the "best" fitting line - such parameters $w$ and $b$ that the differences between original values $y_i$ and predicted values $\hat{y}_i = wx_i + b$ are minimum.

**Weight** ($w$) and **bias** ($b$) are the parameters that will get updated when we **train** the model. They are initialized to some random values or set to 0 and updated as the training progresses.

For each training example $x^{(i)}$, the prediction $\hat{y}^{(i)}$ can be calculated as:

$$\hat{y}^{(i)} &=  w x^{(i)} + b,\$$

where $i = 1, \dots, m$.

We can organise all training examples as a vector $X$ of size ($1 \times m$) and perform scalar multiplication of $X$ ($1 \times m$) by a scalar $w$, adding $b$, which will be broadcasted to a vector of size ($1 \times m$):

\hat{Y} &=  w X + b,\\

This set of calculations is called **forward propagation**.

For each training example we can measure the difference between original values $y^{(i)}$ and predicted values $\hat{y}^{(i)}$ with the **loss function** $L\left(w, b\right)  = \frac{1}{2}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$. Division by $2$ is taken just for scaling purposes, To compare the resulting vector of the predictions $\hat{Y}$ ($1 \times m$) with the vector $Y$ of original values $y^{(i)}$, We can take an average of the loss function values for each of the training examples:

$$\mathcal{L}\left(w, b\right)  = \frac{1}{2m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.\tag{4}$$

This function is called the sum of squares **cost function**. The aim is to optimize the cost function during the training, which will minimize the differences between original values $y^{(i)}$ and predicted values $\hat{y}^{(i)}$.

When our weights were just initialized with some random values, and no training was done yet, you can't expect good results. We need to calculate the adjustments for the weight and bias, minimizing the cost function. This process is called **backward propagation**. 

According to the gradient descent algorithm, you can calculate partial derivatives as:

\begin{align}
\frac{\partial \mathcal{L} }{ \partial w } &= 
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)x^{(i)},\\
\frac{\partial \mathcal{L} }{ \partial b } &= 
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right).
\tag{5}\end{align}

We can see how the additional division by $2$ in the equation $(4)$ helped to simplify the results of the partial derivatives. Then update the parameters iteratively using the expressions

\begin{align}
w &= w - \alpha \frac{\partial \mathcal{L} }{ \partial w },\\
b &= b - \alpha \frac{\partial \mathcal{L} }{ \partial b },
\tag{6}\end{align}

where $\alpha$ is the learning rate. Then repeat the process until the cost function stops decreasing.

The general **methodology** to build a Linear regression model is to:
1. Initialize the model's parameters
2. Loop:
    - Implement forward propagation (calculate the model output),
    - Implement backward propagation (to get the required corrections for the parameters),
    - Update parameters.
3. Make predictions.


# Visualization of training process of simple linear regression model:
![fittheline](https://github.com/AbanoubYounan/Build-Linear-Regression-Model-from-Scratch/assets/73174478/863b0b51-35e3-47a9-a3d2-6cc02875e471)
