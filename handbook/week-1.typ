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

*Example 1: Google News*
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

==== Key Concepts
- *Function ($f$)*: Represents the model, maps input x to prediction y-hat.
  - y-hat: Estimated or predicted value.
  - y: Actual true value from the dataset.
- *Prediction (y-hat)*: Output of the model for given input x.

==== Model Representation
- Function f is often linear for simplicity:
  - $f(x) = w x + b$
    - *w*: Weight (slope of the line).
    - *b*: Bias (intercept).
  - Also written as $f_w,b(x)$ to explicitly denote parameters.

===== Visualizing Data and Predictions
- Input feature $x$ on the horizontal axis.
- Output target $y$ on the vertical axis.
- Training set points plotted; function $f(x)$ fits a line through the data.

==== Why Linear Functions?
- Linear functions are simple and foundational.
- They help transition to more complex, non-linear models (e.g., curves or parabolas).

==== Univariate Linear Regression
- One variable (input feature): *size of the house*.
- $f(x) = w x + b$:
  - *Linear regression*: Fits a straight line to data.
  - *Univariate*: Single input variable.
  - Example: Predicting house price based on size.

==== Multivariate Linear Regression (Future Scope)
- Incorporates multiple input variables:
  - E.g., size, number of bedrooms, location, etc.
  - Builds on univariate concepts.

=== Cost Function Formula
- The *cost function* is a key step in implementing linear regression.
- It evaluates how well the model is performing and provides a basis to improve it.

==== Model
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

==== Understanding Parameters
- Different values of w and b generate different lines.
  - Example 1: w = 0, b = 1.5
    - Horizontal line, f(x) = 1.5
  - Example 2: w = 0.5, b = 0
    - Line with slope = 0.5
  - Example 3: w = 0.5, b = 1
    - Line intersects y-axis at 1 with slope = 0.5
- Goal: Adjust w and b to fit the training set well.

==== Measuring Fit
- Training examples: (x^i, y^i)
  - *Prediction*: $y^hat = f_w,b(x^i) = w * x^i + b$
  - Aim: Minimize difference between y^hat and y^i for all examples.

=== Constructing the Cost Function
==== Error Calculation
1. Compute the error: y^hat - y (difference between prediction and target).
2. Square the error to ensure all values are positive.

==== Squared error cost function
- Sum of squared errors for all examples:

#figure(
  image("./images/2024-11-21-summing-errors-training.png"),
  caption: [
    Errors across training set.
  ]
)

=== Intuition Behind J(w, b)
- J(w, b) measures how far predictions are from actual targets.

== Cost function intuition
- In this section, we aim to build intuition about the cost function in linear regression.
- The goal is to find the best parameters $w$ and $b$ for the model by minimizing the cost function $J(w, b)$.

=== The Model
- The linear regression model is defined as:
  $f(w, b)(x) = w * x + b$
  - Parameters: $w$ (weight) and $b$ (bias).
  - Different choices for $w$ and $b$ yield different straight-line fits to the data.

=== The Cost Function
- The cost function $J(w, b)$ measures the difference between:
  - The model's predictions, $f(w, b)(x)$.
  - The true output values, $y$.
- Objective: Minimize $J(w, b)$ to find the best fit for the training data.

==== Simplified Case: Only $w$
- To better visualize the cost function:
  - Assume $b = 0$.
  - The model becomes: $f(w)(x) = w * x$.
  - Cost function simplifies to $J(w)$.
- Goal: Minimize $J(w)$ for the single parameter $w$.

=== Relationship Between $f(w)$ and $J(w)$
- Left graph: $f(w)(x)$, the model's predictions.
  - Horizontal axis: Input feature $x$.
  - Vertical axis: Output value $y$.
- Right graph: $J(w)$, the cost function.
  - Horizontal axis: Parameter $w$.
  - Vertical axis: Cost $J$.

=== Example: Training Data Points
- Training set: Points $(1, 1), (2, 2), (3, 3)$.

==== Case 1: $w = 1$
- $f(w)(x)$ is a line with slope 1.
- For each training example, $f(w)(x) = y$.
- Cost $J(1) = 0$ (perfect fit).
- Graph:
  - Left: Line passes through all data points.
  - Right: Point at $(w = 1, J(w) = 0)$.

==== Case 2: $w = 0.5$
- $f(w)(x)$ is a line with slope 0.5.
- Calculate squared errors:
  - Example 1: $(0.5 - 1)^2$.
  - Example 2: $(1 - 2)^2$.
  - Example 3: $(1.5 - 3)^2$.
- Cost $J(0.5) = 0.58$.
- Graph:
  - Left: Line deviates slightly from points.
  - Right: Point at $(w = 0.5, J(w) = 0.58)$.

==== Case 3: $w = 0$
- $f(w)(x)$ is a horizontal line on the x-axis.
- Cost $J(0) = 2.33$.
- Graph:
  - Left: Line far from points.
  - Right: Point at $(w = 0, J(w) = 2.33)$.

==== Case 4: $w = -0.5$
- $f(w)(x)$ is a downward-sloping line.
- Cost $J(-0.5) = 5.25$.
- Graph:
  - Left: Line farthest from points.
  - Right: Point at $(w = -0.5, J(w) = 5.25)$.

=== Observations
- Each $w$ value corresponds to:
  - A line on the $f(w)(x)$ graph.
  - A point on the $J(w)$ graph.
- The cost function $J(w)$ is minimized when $w = 1$ (best fit).

=== General Case: $w$ and $b$
- For models with both $w$ and $b$, the cost function $J(w, b)$ becomes more complex.
- Goal: Minimize $J(w, b)$.
- Visualization involves 3D plots for both parameters.

== Visualizing the cost function
=== Key Components
- *Model*: A linear regression model.
- *Parameters*: $w$ and $b$.
- *Cost Function*: $J(w, b)$.
- *Goal*: Minimize $J(w, b)$ over $w$ and $b$.

=== Revisiting Previous Visualization
- In the last video, $b$ was set to 0 to simplify the visualization of $J(w)$.
- We observed that $J(w)$ had a U-shaped curve (a "soup bowl" analogy).

=== Expanding to Two Parameters
- When both $w$ and $b$ are considered, the cost function $J(w, b)$ extends to three dimensions.
- The 3D surface of $J(w, b)$ resembles:
  - A soup bowl.
  - A curved dinner plate.
  - A hammock.

#figure(
  image("./images/2024-11-3d-model.png"),
  caption: [
    3D model.
  ]
)

=== Characteristics of the 3D Surface
- Axes represent $w$ and $b$.
- The height above any $(w, b)$ point represents the value of $J(w, b)$.

==== Example
- Suppose $w = 0.06$ and $b = 50$.
- The model function is $f(x) = 0.06 \cdot x + 50$, which underestimates housing prices.
- This choice corresponds to a specific point on the 3D surface of $J(w, b)$.

=== Contour Plots
- Contour plots provide an alternative visualization of $J(w, b)$ in 2D.
- Analogous to topographical maps, each ellipse (or oval) represents points with the same $J(w, b)$ value.

=== Constructing Contour Plots
- Start with the 3D surface.
- Slice the surface horizontally at specific heights.
- Each slice becomes an ellipse in the contour plot.

=== Features of Contour Plots
- The axes are $w$ (horizontal) and $b$ (vertical).
- The center of concentric ovals represents the minimum of $J(w, b)$.

=== Intuition
- Imagine the 3D bowl projected onto a 2D plane.
- The contour plot shows "height" information via ellipses.

=== Visualization Insights
- The bottom of the bowl (center of the smallest oval) is the minimum of $J(w, b)$.
- Different points on the contour plot correspond to specific linear functions $f(x)$, each with varying performance.

==== Example
- Points with equal $J(w, b)$ in the contour plot correspond to models with similar predictive quality.

== Gradient Descent
Gradient descent is an algorithm used to minimize a cost function $J(w, b)$.
- It systematically finds the values of $w$ and $b$ that minimize $J(w, b)$.
- It is widely used in machine learning, including in training neural networks (deep learning).

#figure(
  image("./images/2024-11-3d-model.png"),
)
=== Application Scope
Gradient descent is not limited to linear regression cost functions.
- It can minimize any differentiable function, including functions with multiple parameters.

=== How Gradient Descent Works
1. Start with initial guesses for parameters ($w$ and $b$).
   - Common choice: $w = 0$ and $b = 0$.
2. Adjust $w$ and $b$ iteratively to reduce $J(w, b)$.
3. Continue updating until $J(w, b)$ reaches or is near its minimum.

=== Surface Plot Example
Supose $J(w, b)$ represented as a surface plot:
- $w$ and $b$ define the horizontal axes.
- $J(w, b)$ defines the vertical axis (height of the surface).

=== Analogy: Descending a Hill
1. *Starting Point:*
   Supose standing on a hill. The height corresponds to $J(w, b)$ at that point.
2. *Choosing the Direction:*
   - Spin 360 degrees and determine the direction of steepest descent.
   - Take a small step in that direction.
3. *Repeat:*
   - After each step, reassess and take another step downhill.
   - Continue until you reach a local minimum.

== Key Properties of Gradient Descent

=== Local Minima
- *Definition:* A local minimum is a point where $J(w, b)$ is lower than at any nearby point.
- *Behavior:*
  - The algorithm converges to the local minimum nearest to the starting point.
  - Different starting points may lead to different local minima.

==== Example: Multiple Valleys
1. Start in one valley: Gradient descent will lead to the local minimum of that valley.
2. Start in another valley: Gradient descent will find the local minimum of this second valley.
   - This demonstrates the dependency on the starting point.

=== Gradient Descent Algorithm

#figure(
  image("./images/2024-11-21-gradiant-equation.png"),
)

==== Update Equation
- $w = w - alpha {partial}/{partial w} J(w, b)$
- This means the parameter $w$ is updated by subtracting $alpha$ times the derivative of the cost function $J(w, b)$ with respect to $w$.

=== Key Components
- *$alpha$ (Learning Rate)*:
  - A small positive number (e.g., 0.01).
  - Controls step size in the descent.
  - Larger $alpha$: aggressive steps.
  - Smaller $alpha$: baby steps.
- *Derivative Term*:
  - $frac{partial}{partial w} J(w, b)$ indicates the direction for updating $w$.
  - Combined with $alpha$, determines the magnitude of the step.

== Assignment Operator
- *Code Context*: $a = c$ stores the value of $c$ in $a$.
- *Math Context*: $a = c$ asserts equality.
- Programming uses $==$ for testing equality.

== Two Parameters: $w$ and $b$
- Both parameters are updated similarly:
  - $b = b - alpha frac{partial}{partial b} J(w, b)$

== Simultaneous Updates
- *Correct Implementation*:
  - Compute updates for both $w$ and $b$ simultaneously:
- *Incorrect Implementation*:
  - Update $w$ first, then use the new $w$ to compute $b$.
  - Leads to inconsistent updates as $b$ uses updated $w$.

== Convergence
- Repeat update steps until $w$ and $b$ stabilize.
- Convergence occurs when parameters no longer change significantly.

== Gradient Descent Details
- Simultaneous update is essential for proper gradient descent.
- Non-simultaneous updates resemble another algorithm with different properties.

== Next Steps
- Explore derivative terms (${partial}/{partial w}$ and ${partial}/{partial b}$) in detail.
- No prior knowledge of calculus required to implement gradient descent.

== Application Scope
Gradient descent is not limited to linear regression cost functions.
- It can minimize any differentiable function, including functions with multiple parameters.
- Example: $J(w_1, w_2, dots, w_n, b)$, where the goal is to minimize $J$ by adjusting $w_1$ through $w_n$ and $b$.

=== How Gradient Descent Works
1. Start with initial guesses for parameters ($w$ and $b$).
   - Common choice: $w = 0$ and $b = 0$.
2. Adjust $w$ and $b$ iteratively to reduce $J(w, b)$.
3. Continue updating until $J(w, b)$ reaches or is near its minimum.

=== Surface Plot Example
Supose $J(w, b)$ represented as a surface plot:
- $w$ and $b$ define the horizontal axes.
- $J(w, b)$ defines the vertical axis (height of the surface).

=== Analogy: Descending a Hill
1. *Starting Point:*
   Supose standing on a hill. The height corresponds to $J(w, b)$ at that point.
2. *Choosing the Direction:*
   - Spin 360 degrees and determine the direction of steepest descent.
   - Take a small step in that direction.
3. *Repeat:*
   - After each step, reassess and take another step downhill.
   - Continue until you reach a local minimum.

=== Local Minima
- *Definition:* A local minimum is a point where $J(w, b)$ is lower than at any nearby point.
- *Behavior:*
  - The algorithm converges to the local minimum nearest to the starting point.
  - Different starting points may lead to different local minima.

== The Role of Learning Rate

=== Understanding Alpha (Learning Rate)
- Alpha, the learning rate, significantly impacts gradient descent's efficiency.
- Poor choice of alpha may lead to:
  - Slow convergence
  - Divergence (failure to reach the minimum)

=== Gradient Descent Update Rule
- The update rule:
  $W -> W - alpha {partial J(W)}/{partial W}$
- Alpha controls the step size in each iteration.

=== Small Learning Rate
- Case: Alpha is very small (e.g., $alpha = 0.0000001$)
  - The derivative term is multiplied by a tiny number.
  - Results in *very small steps* toward the minimum.
  - Many iterations are needed, making the process inefficient.
- Graphically:
  - Slow descent along the curve $J(W)$ with minimal progress.

=== Large Learning Rate
- Case: Alpha is too large
  - Steps become *too big*, overshooting the minimum.
  - May lead to divergence:
    - Updates can cause $W$ to oscillate or move away from the minimum.
- Example:
  - Starting near the minimum but overshooting due to a large step size.
  - Cost $J(W)$ may increase instead of decrease.

== At the Local Minimum
- When $W$ reaches a local minimum:
  - The slope of the tangent line is zero:
    ${partial J(W)}/{partial W} = 0$
  - Update step becomes:
    $W -> W - alpha dot 0 = W$
  - Gradient descent stops changing $W$:
    - Parameters remain at the minimum.

== Automatically Smaller Steps Near Minimum
- As $W$ approaches the local minimum:
  - The derivative ${partial J(W)}/{partial W}$ decreases.
  - Step size automatically becomes smaller.
  - Gradient descent converges even with a fixed $alpha$.

== Gradient descent for linear regression
Train a linear regression model using the squared error cost function and gradient descent.

=== Key Components
=== Linear Regression Model
The hypothesis function:
$f(x; w, b) = w x + b$

=== Squared Error Cost Function
The cost function:

#figure(
  image("./images/2024-11-21-cost-function-gradient.png"),
)

=== Gradient Descent Algorithm
Update rules:
- $w := w - alpha {partial J(w, b)}/{partial w}$
- $b := b - alpha {partial J(w, b)}/{partial b}$

=== Properties of the Cost Function
==== Convexity
- The squared error cost function is *convex*, meaning it has a single global minimum.
- Gradient descent will always converge to the global minimum for appropriately chosen $alpha$.

// week 2

= Multiple features

#figure(
  image("images/2024-12-05-multiple-features.png"),
)

=== Single vs. Multiple Features
- *Original Linear Regression*:
  - A single feature $x$ (e.g., house size) predicts $y$ (e.g., house price).
  - Model: $f_{w,b}(x) = w x + b$.
- *Multiple Features*:
  - Predict using several features:
    - $x_1$: size of the house.
    - $x_2$: number of bedrooms.
    - $x_3$: number of floors.
    - $x_4$: age of the home (years).

=== Notation for Multiple Features
- *Feature Representation*:
  - $x_j$: the $j$-th feature.
  - $n$: total number of features (e.g., $n = 4$).
- *Training Examples*:
  - $x^{(i)}$: list (or vector) of features for the $i$-th example.
  - Example: $x^{(2)} = [1416, 3, 2, 40]$.
  - $x^{(i)}_j$: value of the $j$-th feature for the $i$-th example.
    - E.g., $x^{(2)}_3 = 2$ (number of floors in the second example).

=== Model for Multiple Features
- *General Model*:
  - $f_{w,b}(x) = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + b$.
- *Example*:
  - Housing price prediction:
    $f_{w,b}(x) = 0.1x_1 + 4x_2 + 10x_3 - 2x_4 + 80$.
  - Interpretation:
    - $b = 80$: base price ($80,000$).
    - $0.1$: price increases by $100$ for every additional square foot.
    - $4$: price increases by $4,000$ per bedroom.
    - $10$: price increases by $10,000$ per floor.
    - $-2$: price decreases by $2,000$ per year of age.

=== Vectorized Notation
- *Parameters and Features*:
  - $w$: vector of parameters $\[w_1, w_2, ..., w_n\]$.
  - $x$: vector of features $\[x_1, x_2, ..., x_n\]$.
  - $b$: single number (not a vector).
- *Model (Compact Form)*:
  - $f_{w,b}(x) = w \cdot x + b$, where $\cdot$ denotes the dot product.

=== Dot Product
- *Definition*:
  - $w \cdot x = w_1x_1 + w_2x_2 + ... + w_n x_n$.
  - Equivalent to $f_{w,b}(x) = w \cdot x + b$.
- *Purpose*: Simplifies notation and implementation.

=== Types of Regression
- *Univariate Regression*: Single feature.
- *Multiple Linear Regression*: Multiple features.
  - Note: Multivariate regression refers to something else not covered here.

= Vectorization

=== Example: Dot Product
- Suppose $w$ and $x$ are vectors with three numbers, and $b$ is a scalar:
  - $n = 3$, $w = \[w_1, w_2, w_3\]$, $x = \[x_1, x_2, x_3\]$.
- *Without Vectorization*:
  - Compute $f = w_1x_1 + w_2x_2 + w_3x_3 + b$ using a loop.
- *With Vectorization*:
  - Use the dot product: $f = w \cdot x + b$.
  - In Python: `f = np.dot(w, x) + b`.

=== Benefits of Vectorization
1. *Conciseness*: Code becomes shorter and easier to maintain.
2. *Efficiency*: Runs faster by leveraging parallel hardware.

=== Behind the Scenes
- *Non-Vectorized Implementation*:
  - Operations are sequential, one calculation at a time.
- *Vectorized Implementation*:
  - Parallel processing computes all operations (e.g., multiplications and additions) in a single step.

=== Practical Impact
- For large $n$ (e.g., thousands of features), vectorization significantly reduces computation time.
- Enables machine learning algorithms to scale effectively to large datasets.

=== Example: Gradient Descent Update
- *Non-Vectorized*:
#codeBlock(
  ```python
  for j in range(n):
      w[j] = w[j] - alpha * d[j]
  ```
)

= Gradient descent for multiple linear Regression

#figure(
  image("images/2024-12-05-gradiant-descent.png"),
)

=== Review of Multiple Linear Regression
- Parameters:
  - $w = \[w_1, w_2, ..., w_n\]$: a vector of length $n$.
  - $b$: a scalar parameter.
- Model in Vector Notation:
  - $f_{w,b}(x) = w \cdot x + b$, where $\cdot$ is the dot product.
- Cost Function:
  - $J(w, b)$ is now defined as a function of the parameter vector $w$ and scalar $b$.

=== Gradient Descent
- Update Rules:
  - $w_j := w_j - alpha {partial J}/{partial w_j}$ for $j = 1, ..., n$.
  - $b := b - alpha {partial J}/{partial b}$.
- For $n$ features:
  - Update each $w_j$ (for $j = 1$ to $n$) based on the gradient of $J$ with respect to $w_j$.
  - Update $b$ similarly as in the univariate case.

=== Vectorized Implementation
- Efficient updates using vectorized operations:
  - $w := w - alpha \cdot nabla_w J$
  - $b := b - alpha \cdot {partial J}/{partial b}$.

=== The Normal Equation
- An alternative to gradient descent:
  - Solves for $w$ and $b$ directly using advanced linear algebra.
  - Advantages:
    - No iterations required.
  - Disadvantages:
    - Limited to linear regression.
    - Computationally expensive for large $n$.
- Practical Use:
  - Rarely implemented manually.
  - Often used as a backend in some machine learning libraries.

= Feature Scaling

== *Mean Normalization*:
   - Center features around $0$ by subtracting the mean and dividing by the range.

#figure(
  image("images/2024-12-05-mean-normalization.png"),
)

== *Z-Score Normalization*:
   - Normalize using mean and standard deviation $sigma_j$.

#figure(
  image("images/2024-12-05-z-score-normalization.png"),
)

=== Benefits of Feature Scaling
- Scaled features ensure cost function contours are circular, enabling faster convergence.
- Gradient descent follows a direct path to the global minimum.

=== When to Scale
- Features should typically range from around $-1$ to $+1$, though small deviations are fine.
- Re-scaling is recommended for:
  - Very large ranges (e.g., $[-100, +100]$).
  - Very small ranges (e.g., $[-0.001, +0.001]$).
  - Moderate ranges (e.g., $[98.6, 105]$ degrees Fahrenheit).

= Choosing the learning rate

#figure(
  image("images/2024-12-05-choosing-learninig-rate.png"),
)

=== Importance of Learning Rate
- The learning rate alpha affects:
  - *Convergence speed*: Too small, and training is slow.
  - *Stability*: Too large, and training may not converge.
- Signs of incorrect alpha:
  - Cost $J$ oscillates or increases: alpha might be too large.
  - Cost $J$ consistently decreases: alpha is likely appropriate.

=== Debugging Gradient Descent
1. *Cost Goes Up and Down*:
   - Overshooting the minimum due to a large alpha.
   - Fix: Use a smaller alpha.
2. *Cost Consistently Increases*:
   - Possible bug: Ensure update is $w_1 := w_1 - alpha {partial J}/{partial w_1}$.

=== Strategy for Choosing alpha
- *Testing a Range of Values*:
  - Start with a small value, e.g., $alpha = 0.001$.
  - Increase alpha progressively (e.g., $3$): $0.001$, $0.003$, $0.01$, $0.03$, etc.
- *Optimal alpha*:
  - Choose the largest alpha that results in a steady decrease of $J$.

=== Practical Tips
- *Debugging Step*: Use a very small alpha temporarily to ensure $J$ decreases every iteration.
- *Efficient Training*: Avoid overly small alpha, as it slows convergence.
- *Visualization*:
  - Plot $J$ against iterations for different alpha values to observe trends.

=== Upcoming Lab
- Explore:
  - Feature scaling in code.
  - Impact of different alpha values on model training.
- Gain hands-on experience with alpha selection to develop intuition.
