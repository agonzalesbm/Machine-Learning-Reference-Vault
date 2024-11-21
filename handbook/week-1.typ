#import "lib/template.typ": main
#import "lib/simpleTable.typ": simpleTable
#import "lib/codeBlock.typ": codeBlock
#show: doc => main(
  title: [
    Machine Learning
  ],
  version: "v0.1.",
  authors: (
    (name: "Andres Gonzales", email: "andresg.gonzales@fundacion-jala.org"),
  ),
  abstract: [
    This is a collection of notes and thoughts that I've been taking while learning about machine learning.
    It is based on the *"Machine Learning"* specialization from Coursera by _Andrew Ng_ as well as the lessons and labs from our course at *Fundación Jala*.
  ],
  doc,
)

= Supervised Learning

== Linear Regression Notes

To recap, simple linear regression with one variable is given by:

#simpleTable(
  columns: (1fr, 1fr),
  [*Attribute*], [*Formula*],
  [*Model*], [
    $ f_(w,b)(x) = w x + b $
  ],
  [*Parameters*], [
    $ w, b $
  ],
  [*Cost Function*], [
    $ J(w, b) = 1/(2m) sum_(i=1)^m ( f_(w,b)( x^(\(i\)) ) - y^(\(i\) ) )^2 $
  ],
  [*Objective*], [
    $ min_(w,b) J(w, b) $
  ],
)

This is the simplest form of linear regression, and we are given a dataset:

$ (X, Y) $

Where $X$ is a vector of features and $Y$ is a vector of labels.

#figure(
  image("./images/2024-10-30-simple-regression.png"),
  caption: [
    I made this diagram using _excalidraw_, that, in my head, represents what we are trying to do. $x^(\(i\))$ is the $i$-th training example, and $y^(\(i\))$ is the $i$-th training label.
  ]
)

=== Cost Function

One interesting observation is how the cost function $J(w)$ changes as we change the value of $w$. For example, the code below plots the cost for a simple target:

$ f_(w, b = 0) = w x $

Notice that for convenience, we are using $b = 0$, so our target is simply $f_(w) = w x$.

#codeBlock(
  ```python
  def plot_simple_error(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    w_range: NDArray[np.float64],
    x_marker_position: float,
  ) -> Tuple[Figure, Axes]:

    fig, ax = plt.subplots(figsize=(10, 6))
    errors = np.array([cost_function(y, simple_hypothesis(x, w)) for w in w_range])
    ax.plot(w_range, errors, color="blue", label="J(w)")
    ax.axvline(
        x=x_marker_position,
        color="red",
        linestyle="--",
        label=f"w = {x_marker_position}",
    )
    ax.set_xlabel("w")
    ax.set_ylabel("J(w)")
    ax.set_title("Cost as a function of w - J(w)")
    ax.legend()

    return fig, ax
  ```
)

This allow us to visualize the behavior of the cost function by using a *known model* and a range of sampling values for $w$. In the example below, we are using:

$ f_(w) =  (4 x) / 3 $

#codeBlock(
  ```python
  w: float = 4 / 3
  x_train = np.linspace(-5, 5, 100)
  y_train = simple_hypothesis(x_train, w)
  w_sample_range = np.linspace(-5, 8, 100)

  fig, ax = plot_simple_error(
      x=x_train, y=y_train, w_range=w_sample_range, x_marker_position=w
  )
  ```
)

@simple-cost shows the resulting plot. We can observe how the cost approaches a minimum as we change the value of $w$ from both sides, converging to a value close to $1.33$.

#figure(
  image("./images/cost-linear-reg-line.png"),
  caption: [
    Plot of the cost function $J(w)$ as a function of $w$ for the target $f_(w) =  (4 x) / 3$.
  ]
)<simple-cost>

A similar approach can be used to now introduce $b$ as a second target parameter. For example, using a target of the form:

$ f_(w, b) = (2 x) / 5 - 3 / 2  $

#codeBlock(
  ```python
  w = 2.5
  b = -1.5
  x_train = np.linspace(-5, 5, 100)
  y_train = complex_hypothesis(x_train, w, b)
  w_sample_range = np.linspace(-5, 5, 100)
  b_sample_range = np.linspace(-5, 5, 100)

  fig, ax = plot_complex_error_with_contour(
      x=x_train, y=y_train, w_range=w_sample_range, b_range=b_sample_range
  )
  ```
)

@complex-cost shows the resulting plot. We can observe how the cost function has a minimum at $(w, b) = (2.5, -1.5)$ but it is a bit more difficult to observe. As we increase the number of dimensions in the feature space, it becomes even more difficult to visualize the cost function.

But the main idea is that for linear regression, the cost function is *convex and will always have a global minimum.*

#figure(
  image("./images/cost-linear-reg-contour.png"),
  caption: [
    Plot of the cost function $J(w, b)$ as a function of $w$ and $b$ for the target $f_(w, b) = (2 x) / 5 - 3 / 2$.
  ]
)<complex-cost>

= Overview of Machine Learning
== What is Machine Learning?
Machine Learning enables computers to learn and improve from experience without beign explicity programmed. It powers many technologies and applications:

+ *Search engines*: Ranks web pages effectively for relevant search results.
+ *Social media and apps*: recognizes and tags friends in photos (e.g. Instragram).
+ *Recommendation systems*: Suggests content, like movies, based on user preferences.
+ *Voice recognition*: Powers features like voice-to-text, Siri, Google assistant.
+ *Spam detection*: identifies and filters spam emails.

== Applications beyond consumer use
Machine Learning is increasingly being applied in industries and solving critical problems, such as:

+ *Climate Change*: Optimizing wind turbine power generation.
+ *Healthcare*: Assisting doctors in accurate diagnosis.
+ *ManufacturinG*: Using computer vision to detect defects in assembly lines.

= Supervised vs. Unsupervised Machine Learning
== Machine Learning definition
Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. This concept was first introduced by Arthur Samuel, who demonstrated it with a checkers-playing program in the 1950s. The program improved by playing tens of thousands of games against itself, learning from patterns of success and failure, and eventually outperforming Samuel himself.

1. Learning Algorithms Improve with Experience
  - More data or learning algorithm opportunities has better performs.

2. Types of Machine Learning
  - Supervised Learnin, Unsupervised Learning.

== Supervised learning
Supervised learning refers to algorithms that learn input-to-output mappings $(x → y)$ using labeled data. The model learns by analyzing pairs of inputs $x$ and their corresponding correct outputs $y$. Once trained, the algorithm predicts the output $y$ for new, unseen inputs $x$.

=== Key Characteristics of Supervised Learning

1. *Labeled Data*
   Training involves providing input-output pairs where the correct labels $y$ are known.

2. *Examples of Applications*  
   - *Spam Filters*: Classify emails as "spam" or "not spam."  
   - *Speech Recognition*: Convert audio input to text.  
   - *Machine Translation*: Translate text between languages.  
   - *Online Advertising*: Predict the likelihood of users clicking on ads to drive revenue.  
   - *Self-Driving Cars*: Identify the positions of objects (e.g., other cars) for safe navigation.  
   - *Visual Inspection in Manufacturing*: Detect defects in products using image data.

3. *Two Main Types of Supervised Learning*
   - *Regression*: Predicts continuous values (e.g., house prices).  
   - *Classification*: Predicts discrete categories (e.g., spam vs. non-spam).

*Example: Predicting Housing Prices*

- *Task*: Predict house prices based on size.  
- *Data*: Input $x$ (house size) and output $y$ (price).  
- *Process*:  
  1. Plot house size vs. price.  
  2. Fit a model (e.g., a straight line or curve) to the data.  
  3. Use the model to predict prices for new house sizes (e.g., 750 square feet).

- *Regression in Action*:  
  Predicts continuous outcomes, such as \$150,000 or \$200,000, from infinitely many possibilities.

*Economic Impact*

- *Significance*:  
  Supervised learning accounts for 99% of the economic value created by machine learning today.  
  Especially impactful in industries like online advertising and manufacturing.

== Regression: Predicting Continuous Values
Regression is a supervised learning algorithm that predicts continuous values from infinitely many possible outputs. It is used when the target variable is a number.

=== Key Characteristics  
1. *Continuous Outputs*  
   - Outputs can take any value within a range, including decimals (e.g., house prices, stock values).
2. *Learning from Data*  
   - Algorithms analyze pairs of input $X$ and output $Y$ values to model their relationship.
3. *Visualization*  
   - A line or curve is fitted to the data points to make predictions for new inputs.

*Example: Predicting Housing Prices*  
- *Task*: Estimate house prices based on size.  
- *Process*: Input house size (e.g., in square feet) and output price (e.g., 150,000).  
- *Models*:  
  - Straight Line: Fits a linear relationship.  
  - Curved Models: Capture more complex patterns for better predictions.  

== Classification: Predicting Categories
Classification is a supervised learning algorithm that predicts categories or classes from a small, finite set of possibilities.

=== Key Characteristics 
1. *Discrete Outputs*  
   - Predicts a limited number of categories (e.g., 'benign' vs. 'malignant' or 'cat' vs. 'dog').
   - Categories can be non-numeric (e.g., names) or numeric (e.g., 0, 1, 2).  
2. *Visualization*  
   - Categories are often represented using symbols (e.g., circles and crosses).  
3. *Learning from Data*  
   - Algorithms identify boundaries or decision rules to classify new inputs.

*Example: Breast Cancer Detection*  
- *Task*: Diagnose whether a tumor is benign (0) or malignant (1).  
- *Process*: Inputs like tumor size and patient age are used to predict the tumor's class.  
- *Complexity*: Can handle multi-class outputs, such as different types of cancer.  

*Boundary Detection*  
In cases with two inputs (e.g., tumor size and patient age), the algorithm learns to draw a decision boundary that separates classes (e.g., benign vs. malignant).   

== Unsupervised Learning
Supervised learning is widely used in machine learning, but another key category is *unsupervised learning*. Unlike supervised learning, where data comes with labeled outputs, unsupervised learning deals with unlabeled data. This approach aims to discover patterns or structures in the data without predefined labels.

=== Key Concept
*Unsupervised learning* does not provide the algorithm with output labels (Y). Instead, the goal is to explore the data and identify hidden structures, clusters, or patterns.  

=== Example Scenario  
Imagine a dataset containing tumor size and patient age. Without labels indicating whether a tumor is benign or malignant, an unsupervised algorithm analyzes the data to find groups or clusters.  
== Types of Unsupervised Learning  

=== Clustering Algorithms  

Clustering involves grouping similar data points together based on patterns or shared characteristics. This type of unsupervised learning is widely used in various applications.

*Example 1: Google News * 
Google News uses clustering to group related news articles.  
- *Process*:  
    - The algorithm analyzes articles and identifies recurring keywords like "panda," "twins," or "zoo."  
    - Articles with similar words are grouped into clusters.  
- *Significance*:  
    - No human supervises the process.  
    - The algorithm adapts to new news topics daily.  

*Example 2: Genetic Data Clustering*
DNA microarrays contain information about genetic activity.  
- *Structure*:  
    - Each column represents one person's genetic data.  
    - Rows represent specific genes (e.g., for eye color or height).  
- *Clustering*:  
    - The algorithm identifies groups of individuals with similar genetic traits, assigning them to "type 1," "type 2," etc.  

*Example 3: Market Segmentation*
Businesses use clustering to categorize customers into distinct market segments.  
- *Process*:  
    - Analyze customer data to identify patterns in motivations, interests, or behaviors.  
- *Application*:  
    - Group customers for targeted marketing or personalized service.  

=== Applications Beyond Clustering  
Clustering is just one type of unsupervised learning. Other approaches explore different structures in the data, such as dimensionality reduction or anomaly detection, but all share the same unsupervised principle: discovering insights without predefined labels.

== Linear regression
- Supervised learning involves training a model with a *training set*.
  - Training set includes:
    - *Input features*: e.g., size of a house.
    - *Output targets*: e.g., price of a house.
  - Outputs the *function* (model) $f$.

=== Key Concepts
- *Function ($f$)*: Represents the model, maps input x to prediction y-hat.
  - y-hat: Estimated or predicted value.
  - y: Actual true value from the dataset.
- *Prediction (y-hat)*: Output of the model for given input x.

=== Model Representation
- Function f is often linear for simplicity:
  - $f(x) = w x + b$
    - *w*: Weight (slope of the line).
    - *b*: Bias (intercept).
  - Also written as $f_w,b(x)$ to explicitly denote parameters.

==== Visualizing Data and Predictions
- Input feature $x$ on the horizontal axis.
- Output target $y$ on the vertical axis.
- Training set points plotted; function $f(x)$ fits a line through the data.

=== Why Linear Functions?
- Linear functions are simple and foundational.
- They help transition to more complex, non-linear models (e.g., curves or parabolas).

=== Univariate Linear Regression
- One variable (input feature): *size of the house*.
- $f(x) = w x + b$:
  - *Linear regression*: Fits a straight line to data.
  - *Univariate*: Single input variable.
  - Example: Predicting house price based on size.

=== Multivariate Linear Regression (Future Scope)
- Incorporates multiple input variables:
  - E.g., size, number of bedrooms, location, etc.
  - Builds on univariate concepts.

== Cost Function Formula
- The *cost function* is a key step in implementing linear regression.
- It evaluates how well the model is performing and provides a basis to improve it.

=== Model 
- Training set contains:
  - *Input features*: x
  - *Output targets*: y
- Linear model

#figure(
  image("./images/2024-11-21-model-formula.png"),
  caption: [
    Linear model formula.
  ]
)

  - *Parameters* (adjustable): w, b
    - w: weight (determines slope of the line)
    - b: bias (y-intercept of the line)

=== Understanding Parameters
- Different values of w and b generate different lines.
  - Example 1: w = 0, b = 1.5
    - Horizontal line, f(x) = 1.5
  - Example 2: w = 0.5, b = 0
    - Line with slope = 0.5
  - Example 3: w = 0.5, b = 1
    - Line intersects y-axis at 1 with slope = 0.5
- Goal: Adjust w and b to fit the training set well.

=== Measuring Fit
- Training examples: (x^i, y^i)
  - *Prediction*: $y^hat = f_w,b(x^i) = w * x^i + b$
  - Aim: Minimize difference between y^hat and y^i for all examples.

== Constructing the Cost Function
=== Error Calculation
1. Compute the error: y^hat - y (difference between prediction and target).
2. Square the error to ensure all values are positive.

=== Squared error cost function
- Sum of squared errors for all examples:  

#figure(
  image("./images/2024-11-21-summing-errors-training.png"),
  caption: [
    Errors across training set.
  ]
)

== Intuition Behind J(w, b)
- J(w, b) measures how far predictions are from actual targets.