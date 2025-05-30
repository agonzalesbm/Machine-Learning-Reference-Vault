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

= Feature engineering

#figure(
  image("images/2024-12-05-feature-engineering.png")
)

=== What is Feature Engineering?
- *Definition*: Using domain knowledge to create new features by transforming or combining existing ones.
- Benefits:
  - Enhances the learning algorithm’s ability to capture relationships in the data.
  - Provides a better representation of the problem for the model.

=== Importance of Features
- The choice of features significantly impacts a learning algorithm’s success.
- *Feature engineering*: Creating or modifying features to improve the model's performance.

=== Example: Predicting House Prices
- Original Features:
  - $x_1$: Width (frontage) of the lot.
  - $x_2$: Depth of the lot.
- Initial Model:
  - $f_{w,b}(x) = w_1x_1 + w_2x_2 + b$.
  - Treats frontage and depth as separate predictors.

=== Creating a New Feature
- Observation: Area ($x_3$) = $x_1 \cdot x_2$ might better predict house prices.
- Updated Model:
  - $f_{w,b}(x) = w_1x_1 + w_2x_2 + w_3x_3 + b$.
  - $w_3$ determines how much area contributes to the prediction, alongside frontage and depth.

= Polynomial regression

#figure(
  image("images/2024-12-05-polynomial-regression.png")
)

=== Example: Housing Prices
1. *Dataset*:
   - Feature $x$: size of the house in square feet.
   - Observations: A straight line does not fit the data well.
2. *Quadratic Model*:
   - Features: $x$ and $x^2$.
   - Adds a curve to the fit, but a quadratic model may decrease as $x$ increases, which might not align with intuition about housing prices.
3. *Cubic Model*:
   - Features: $x$, $x^2$, and $x^3$.
   - Produces a curve that increases as size increases, fitting the data better.

=== Feature Engineering in Polynomial Regression
- Polynomial terms:
  - $x^2$, $x^3$, etc., provide flexibility for modeling non-linear patterns.
- Alternative features:
  - Square root of $x$: Produces a curve that becomes less steep as $x$ increases, without flattening or decreasing.

=== Importance of Feature Scaling
- Polynomial terms create features with vastly different ranges:
  - $x$ ranges from $1$ to $1,000$.
  - $x^2$ ranges from $1$ to $1,000,000$.
  - $x^3$ ranges from $1$ to $1,000,000,000$.
- Feature scaling ensures these features have comparable ranges, improving gradient descent efficiency.

=== Choosing Features
- Wide range of feature choices:
  - Polynomial terms like $x^2$, $x^3$.
  - Non-linear transformations like $sqrt{x}$.
- How to decide:
  - Measure model performance with different feature combinations.
  - Later courses explore systematic methods for feature selection.

// week 3

= Motivations

#figure(
  image("images/2024-12-06-motivations.png")
)

== Classification
Predicting one of a small set of categories (e.g., spam or not spam).
- Contrast with regression:
  - Regression predicts a continuous value.
  - Classification predicts discrete categories.

=== Examples of Classification Problems
1. Email spam detection:
   - Output: No/Yes (0/1).
2. Fraud detection in financial transactions:
   - Output: No/Yes (0/1).
3. Tumor classification:
   - Output: Benign/Malignant (0/1).

=== Binary Classification
- *Definition*: Output variable $y$ has only two possible values.
- Terminology:
  - Classes: Categories being predicted (e.g., spam vs. not spam).
  - Labels:
    - Negative class: $y = 0$ (e.g., not spam, benign).
    - Positive class: $y = 1$ (e.g., spam, malignant).
- Note: Negative/Positive does not imply bad/good; it indicates absence/presence of the feature.

=== Linear Regression for Classification
- Attempting to use linear regression:
  1. Fit a straight line to the data.
  2. Apply a threshold (e.g., $0.5$) to predict:
     - $y = 0$ if $f(x) < 0.5$.
     - $y = 1$ if $f(x) \geq 0.5$.
- Limitations:
  - Sensitive to outliers:
    - A single extreme data point can shift the decision boundary, leading to poor predictions.
  - Outputs are not confined to $[0, 1]$, making it unsuitable for classification.

=== Need for Logistic Regression
- Logistic regression addresses the limitations of linear regression for classification:
  - Outputs are always in $[0, 1]$, representing probabilities.
  - Designed specifically for binary classification.

= Logistic regression

#figure(
  image("images/2024-12-06-sigmoid-function.png")
)

- Logistic regression is one of the most widely used classification algorithms.
- It outputs a probability that the label $y$ belongs to the positive class ($y = 1$).

=== The Sigmoid Function
- *Definition*: A mathematical function that maps any input $z$ to a value between $0$ and $1$.
- Formula: $g(z) = {1}/{1 + e^{-z}}$, where $e$ is Euler’s number ($approx 2.7$).
- Behavior:
  - At $z = 0$, $g(z) = 0.5$.

=== Logistic Regression Model
1. *Model Definition*:
   - Input: $f(x) = g(w \cdot x + b)$, where:
     - $w \cdot x$ is the dot product of parameters $w$ and features $x$.
     - $b$ is the bias term.
     - $g$ is the sigmoid function.
   - Output: A probability between $0$ and $1$.
2. *Interpretation*:
   - $f(x)$ represents $P(y = 1 \mid x)$, the probability that $y = 1$ given input $x$.
   - If $P(y = 1 \mid x) = 0.7$, there is a $70\%$ chance the label is $1$ and $30\%$ chance it is $0$.

=== Key Properties of Logistic Regression
- *Outputs*: Always between $0$ and $1$.
- *Prediction*: Threshold-based:
  - Predict $y = 1$ if $P(y = 1 \mid x) \geq 0.5$.
  - Predict $y = 0$ if $P(y = 1 \mid x) < 0.5$.

=== Practical Applications
- Widely used in applications like Internet advertising, where it determines probabilities and outputs binary predictions.

= Decision boundary

=== Threshold-Based Predictions
- To predict $y$:
  - If $f(x) \geq 0.5$, predict $y = 1$.
  - If $f(x) < 0.5$, predict $y = 0$.
- Since $g(z) \geq 0.5$ when $z \geq 0$, the model predicts:
  - $y = 1$ when $w \cdot x + b \geq 0$.
  - $y = 0$ otherwise.

=== Non-Linear Decision Boundaries
- By using polynomial features, the decision boundary can become non-linear.
- *Example with Polynomial Features*:
  - $z = w_1 x_1^2 + w_2 x_2^2 + b$.
  - If $w_1 = 1$, $w_2 = 1$, $b = -1$, the decision boundary is $x_1^2 + x_2^2 = 1$ (a circle).
  - Predictions:
    - $y = 1$ outside the circle.
    - $y = 0$ inside the circle.

=== More Complex Boundaries
- Higher-order polynomial features allow for more complex boundaries:
  - Example: $z = w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_1 x_2 + w_5 x_2^2$.
  - Possible boundaries: ellipses, intricate shapes, or more complex regions.
- Logistic regression can fit complex data with appropriate features.

= Cost function for logistic regression

- The cost function measures how well a logistic regression model fits the training data.
- Logistic regression requires a cost function that is convex to ensure reliable optimization with gradient descent.

=== Why Not Use Squared Error for Logistic Regression?
- Squared error cost function:
  - Works well for linear regression, producing a convex surface.
  - For logistic regression, results in a non-convex surface with many local minima.
  - Gradient descent may get stuck in local minima, failing to find optimal parameters.
- Solution: Define a new loss function specific to logistic regression.

=== Loss Function for Logistic Regression
1. Definition:
   - For $y = 1$: ${L o s s} = -log(f(x))$.
   - For $y = 0$: ${L o s s} = -log(1 - f(x))$.
   - $f(x) = {1}/{1 + e^{-z}}$, where $z = w \cdot x + b$.
2. Properties:
   - Encourages correct predictions:
     - When $y = 1$, small loss for $f(x) approx 1$.
     - When $y = 0$, small loss for $f(x) approx 0$.
   - Penalizes incorrect predictions:
     - Loss increases as $f(x)$ moves away from the true label.

=== Visualizing the Loss Function
1. *For $y = 1$*:
   - Loss approaches $0$ as $f(x) \to 1$.
   - Loss grows rapidly as $f(x) \to 0$.
2. *For $y = 0$*:
   - Loss approaches $0$ as $f(x) \to 0$.
   - Loss grows rapidly as $f(x) \to 1$.

=== Cost Function for Logistic Regression

#figure(
  image("images/2024-12-06-cost-logistic-regression.png")
)

Convexity:
   - With the logistic loss, $J(w, b)$ is convex.
   - Ensures gradient descent converges reliably to the global minimum.

=== Benefits of Logistic Loss
- Produces a smooth, convex cost surface without local minima.
- Ensures robust parameter optimization with gradient descent.

= Simplified Cost Function for Logistic Regression

#figure(
  image("images/2024-12-06-simplified-cost-function.png")
)
=== Simplified Loss Function
- The logistic regression loss function can be written compactly as:
  \[
  L(f(x), y) = -y \log(f(x)) - (1 - y) \log(1 - f(x))
  \]
- Explanation:
  - $y \in \{0, 1\}$ allows simplification:
    - If $y = 1$: $L = -log(f(x))$.
    - If $y = 0$: $L = -log(1 - f(x))$.
  - The simplified formula combines both cases into one expression.

= Gradient descent implementation

#figure(
  image("images/2024-12-06-gradient-descent-implementation.png")
)

- Goal: Minimize the cost function $J(w, b)$ to find the optimal parameters $w$ and $b$.
- Prediction: After training, the model can estimate $P(y = 1 \mid x)$ for a new input $x$.

=== Key Insights
- The gradient descent equations look similar for linear and logistic regression, but the difference lies in $f(x)$:
  - Linear regression: $f(x) = w \cdot x + b$.
  - Logistic regression: $f(x) = {1}/{1 + e^{-(w \cdot x + b)}}$.
- Gradient descent ensures convergence if the cost function is convex.

=== Enhancing Gradient Descent
- *Feature Scaling*:
  - Scaling features (e.g., to $[-1, 1]$) speeds up convergence, just as in linear regression.
- *Vectorized Implementation*:
  - Use vectorized operations to compute updates efficiently.
  - Examples provided in optional labs.

= The problem of overfitting

#figure(
  image("images/2024-12-06-problem-of-overfitting.png")
)

- *Overfitting*: The model fits the training data too well, capturing noise and failing to generalize to new data.
- *Underfitting*: The model does not fit the training data well, failing to capture the underlying patterns.

=== Examples with Regression
1. *Underfitting (High Bias)*:
   - Model: Linear regression with too few features.
   - Observation:
     - Poor fit to training data.
     - Fails to capture clear trends (e.g., housing prices leveling off for larger houses).
   - Cause: The model assumes a simplistic relationship between input and output.

2. *Good Fit (Just Right)*:
   - Model: Quadratic regression with features $x$ and $x^2$.
   - Observation:
     - Fits the training data reasonably well.
     - Likely to generalize to unseen examples.

3. *Overfitting (High Variance)*:
   - Model: Higher-order polynomial regression (e.g., $x$, $x^2$, $x^3$, $x^4$).
   - Observation:
     - Perfectly fits training data but results in a wiggly curve.
     - Poor generalization to new data.
   - Cause: The model captures noise or minor fluctuations in the training data.

=== Examples with Classification
1. *Underfitting (High Bias)*:
   - Model: Logistic regression with linear features.
   - Decision Boundary: A straight line.
   - Observation:
     - The decision boundary does not adequately separate positive and negative classes.

2. *Good Fit (Just Right)*:
   - Model: Logistic regression with quadratic features.
   - Decision Boundary: An ellipse or part of an ellipse.
   - Observation:
     - Fits the data well without overcomplicating the decision boundary.
     - Generalizes well to new data.

3. *Overfitting (High Variance)*:
   - Model: Logistic regression with many higher-order polynomial features.
   - Decision Boundary: Complex and highly contorted.
   - Observation:
     - Perfectly classifies training examples but is unlikely to generalize to unseen data.

=== Key Concepts
- *Generalization*:
  - A good model performs well on both training and unseen examples.
- *High Bias vs. High Variance*:
  - High Bias:
    - Assumes overly simplistic relationships.
    - Results in underfitting.
  - High Variance:
    - Captures noise and minor details.
    - Results in overfitting.

=== Finding the "Just Right" Model
- Use an appropriate number of features or complexity for the task.
- Balance bias and variance to achieve a model that generalizes well.

= Addressing overfitting

- Overfitting occurs when a model fits the training data too closely, capturing noise and failing to generalize.
- In this video, we discuss three key strategies to address overfitting:
  1. Collecting more data.
  2. Using fewer features (feature selection).
  3. Applying regularization.

=== Strategies to Address Overfitting

1. *Collect More Data*
   - Larger training sets help models generalize better by learning broader patterns.
   - Example: With more house size and price data, a high-order polynomial can fit smoother curves.
   - Limitation: Often, additional data may not be available.

2. *Use Fewer Features*
   - Removing less relevant features reduces model complexity.
   - Example:
     - Original features: $x, x^2, x^3, x^4, ...$.
     - Reduced features: $x, x^2$.
   - Feature selection:
     - Choose features intuitively or based on domain knowledge.
     - Risk: Discarding useful features may limit model performance.
   - Advanced Approaches:
     - Algorithms for automatic feature selection.

= Cost function with regularization

#figure(
  image("images/2024-12-06-regularization.png")
)

- Regularization modifies the cost function to penalize large parameter values.
- This discourages overly complex models and reduces overfitting.

=== How Regularization Works
- Penalizes large values of $w_j$, encouraging smaller parameters.
- Smaller $w_j$ reduces the influence of less important features, leading to a simpler, smoother model.
- Helps the model generalize better to unseen data.

=== Effects of lambda
1. *$lambda = 0$ (No Regularization)*:
   - Regularization term is ignored.
   - The model overfits, fitting a highly complex and wiggly curve.
2. *$lambda -> infinity$ (Very Large)*:
   - The regularization term dominates.
   - All $w_j \to 0$, resulting in an underfit (e.g., a flat line).
3. *Optimal lambda*:
   - Balances fitting the training data and keeping $w_j$ small.
   - Produces a model that generalizes well to new data.

=== Implementation Details
* Regularization is typically applied to $w_1, w_2, ..., w_n$.
* The bias parameter $b$ is usually excluded from regularization:
   - Regularizing $b$ has minimal practical impact and is often omitted.

= Regularized linear regression

#figure(
  image("images/2024-12-06-regularized-linear-regression.png")
)

- Regularized linear regression modifies the cost function to include a regularization term, penalizing large parameter values.
- Gradient descent can be adapted to minimize the regularized cost function effectively.

=== Regularized Cost Function
- First term: Mean squared error.
- Second term: Regularization term.
  - lambda: Regularization parameter, controlling the strength of regularization.
  - $w_j$: Model parameters (excluding $b$).

=== Gradient Descent Updates
All parameters are updated simultaneously after calculating their gradients.

=== Intuition Behind Regularization
Regularization term adds ${lambda}/{m} w_j$ to the gradient of $w_j$, shrinking $w_j$ on each iteration.
Effect:
  - Regularization reduces $w_j$ iteratively, preventing overfitting.
  - The regularization strength is controlled by lambda.

= Regularized logistic regression

=== Overfitting and Regularization

#figure(
  image("images/2024-12-06-cost-logistic-regression.png"),
)

- *Problem:* Logistic regression with high-order polynomial features can lead to overfitting.
  - The decision boundary becomes overly complex and does not generalize well to new examples.
- *Solution:* Use regularization to prevent overfitting, even with many features.
  - Add a regularization term to the cost function:
  - This penalizes large parameters $w_1, w_2, ..., w_n$.

=== Gradient Descent Update Rule
- *Gradient Descent:*
  - The update for regularized logistic regression is similar to linear regression.
  - The difference is in the derivative term: $w_j$ gets an additional regularization term:
  - No regularization is applied to $b$.

=== Logistic Regression vs Linear Regression
- *Logistic Function:* In logistic regression, the hypothesis function is the sigmoid function:
- The regularization term works similarly to regularized linear regression, but for logistic regression, $f$ is the sigmoid function, not a linear function.

= Neurons and the brain

#figure(
  image("images/2024-12-09-neurons-and-the-brain.png")
)

== Biological vs. Artificial Neural Networks
- *Biological neurons*:
  - Composed of dendrites (inputs), a nucleus (cell body), and an axon (output).
  - Function through electrical impulses and neuron-to-neuron connections.
- *Artificial neurons*:
  - Simplified mathematical models taking numerical inputs, performing computations, and producing outputs.
  - Organized into layers to process data collectively.

== Factors Behind Recent Success
1. *Data availability*:
   - Digitalization of society has provided vast datasets for applications.
2. *Algorithm scalability*:
   - Traditional algorithms like linear regression struggled to improve with more data.
   - Larger neural networks demonstrated increased performance with more data.

= Demand Prediction

#figure(
  image("images/2024-12-10-demand-prediction.png")
)

== Structure of a Neural Network:
    - *Input Layer*: Accepts a vector of features (data).
    - *Hidden Layer*: Processes the input vector and outputs a vector of activations (data like activators).
    - *Output Layer*: Takes the hidden layer's output and produces the final activation or prediction.
== Feature Representation:
    - Neural networks automatically determine relevant features during training.
    - No need to predefine features like affordability, awareness, or perceived quality explicitly.
    - The network learns and selects the most useful features for prediction.

= More complex neural networks

#figure(
  image("images/2024-12-17-more-complex-neural-networks.png")
)

1. *Introduction to Layers*
   - This example involves a neural network with *four layers* (excluding the input layer).
   - *Layer 0*: Input layer (not counted in the total layer count).
   - *Layers 1, 2, 3*: Hidden layers.
   - *Layer 4*: Output layer.

2. *Computation in Layer 3 (Hidden Layer)*
   - *Inputs*: Vector $a^{[2]}$, the output of Layer 2.
   - *Outputs*: Vector $a^{[3]}$, computed as:
     $a_j^{[3]} = g(w_j^{[3]} \cdot a^{[2]} + b_j^{[3]})$
     - $g$: Sigmoid activation function.
     - $w_j^{[3]}$: Weight vector for neuron $j$ in Layer 3.
     - $b_j^{[3]}$: Bias for neuron $j$ in Layer 3.

   - The result is a vector $a^{[3]}$, where each element corresponds to an activation value for a neuron in Layer 3.

3. *Notation Details*
   - Superscripts $[l]$: Indicate layer $l$.
   - Subscripts $j$: Refer to the $j^{t h}$ neuron in a layer.
   - Example:
     - $a^{[3]}_2$: Activation of the 2nd neuron in Layer 3.
     - $w^{[3]}_2$, $b^{[3]}_2$: Parameters for the same neuron.
   - Input vector $X$ is denoted as $a^{[0]}$, aligning with the general notation.

4. *General Computation for Any Layer*
   - For layer $l$ and neuron $j$:
     $$
     a_j^{[l]} = g(w_j^{[l]} . a^{[l-1]} + b_j^{[l]})
     $$
     - $a^{[l-1]}$: Activation vector from the previous layer.
     - $g$: Activation function (e.g., sigmoid, ReLU, etc.).

5. *Activation Functions*
   - Current focus: Sigmoid function $g(z) = {1}/{1 + e^{-z}}$.
   - *Role*: Outputs activation values for neurons in the network.
   - Future lessons may cover alternative activation functions.

6. *Recap of Activation Calculation*
   - To compute activations for any neuron in any layer:
     1. Dot-product the weight vector with the input activations from the previous layer.
     2. Add the corresponding bias.
     3. Apply the activation function.

= Inference making predictions (forward propagation)

#figure(
  image("images/2024-12-17-inference-making-predictions.png")
)

1. *Problem Overview*
   - Task: Binary classification to distinguish between handwritten digits '0' and '1'.
   - Input: 8x8 grayscale image (64 pixel intensity values ranging from 0 to 255).
   - Neural network architecture:
     - *Input layer*: 64 features.
     - *Hidden layer 1*: 25 neurons.
     - *Hidden layer 2*: 15 neurons.
     - *Output layer*: 1 neuron (predicted probability of digit being '1').

2. *Sequence of Computations*
   - Start with the input $X$ (treated as $a^[0]$ by convention).
   - Compute $a^[1]$:
     $a^[1]_j = g(w_j^[1] \cdot X + b_j^[1])$
     - $g$: Activation function.
     - $j$: Index of neuron (1 to 25).
   - Compute $a^[2]$:
     $a^[2]_j = g(w_j^[2] \cdot a^[1] + b_j^[2])$
     - Layer 2 has 15 neurons.
   - Compute $a^[3]$:
     $a^[3] = g(w^[3] \cdot a^[2] + b^[3])$
     - $a^[3]$: Scalar output (predicted probability).

3. *Binary Classification Output*
   - Optionally threshold $a^[3]$ at 0.5:
     - If $a^[3] \geq 0.5$: Predict '1'.
     - Otherwise: Predict '0'.

4. *Function Notation*
   - Neural network's output can be expressed as $f(X)$, analogous to linear or logistic regression.
   - $f(X)$: Function mapping input $X$ to output $a^[3]$.

5. *Algorithm Name*
   - *Forward Propagation*:
     - Propagates activations through the network, layer by layer, in a forward direction (from input to output).

6. *Architectural Insights*
   - Design: Hidden layers with decreasing neuron counts (e.g., 25 → 15 → 1) are typical.
   - Application: Use pre-trained parameters to perform inference on new data.

= Inference in Code

#figure(
  image("images/2024-12-17-inferrence-in-code.png")
)

1. *Problem Setup*
   - Input features $x$: [200°C, 17 minutes].
   - Neural network structure:
     - *Layer 1*: Dense layer with 3 hidden units using sigmoid activation.
     - *Layer 2*: Dense layer with 1 unit using sigmoid activation.

2. *Sequence of Computations*
   - Compute $a^[1]$:
     - Apply Layer 1 to $x$.
     - $a^[1]$ is a list of 3 values (e.g., [0.2, 0.7, 0.3]).
   - Compute $a^[2]$:
     - Apply Layer 2 to $a^[1]$.
     - $a^[2]$ is a single scalar value (e.g., 0.8).

3. *Binary Classification Output*
   - Threshold $a^[2]$ at 0.5:
     - If $a^[2] g = 0.5$: Predict $y = 1$.
     - Otherwise: Predict $y = 0$.

4. *Layer Details*
   - *Dense Layer*:
     - A layer type with learned parameters $w$ (weights) and $b$ (biases).
     - Uses the sigmoid function for activation in this example.

= Data in TensorFlow

#figure(
  image("images/2024-12-17-data-in-tensor-flow.png")
)

1. *Definition of $a^[1]$*
   - $a^[1]$ is computed as the output of applying *Layer 1* to $x$.
   - Result: A $1 "times" 3$ matrix.
   - Example: $a^[1] = "tf.tensor"([0.2, 0.7, 0.3])$ with shape $(1, 3)$.

2. *Details of Tensor Representation*
   - $a^[1]$ is stored as a TensorFlow tensor:
     - Data type: Float32 (32-bit floating-point numbers).
   - *Tensor Definition*:
     - A data type created by TensorFlow to efficiently store and compute on matrices.
     - Simplified understanding: Treat tensors as matrices for this course.

3. *Converting Between TensorFlow and NumPy*
   - TensorFlow tensors and NumPy arrays are two different matrix representations due to historical development.
   - Conversion example:
     - TensorFlow to NumPy use: .
      #codeBlock(
        ```python
          a^[1].numpy()
        ```
      )
     - Result: Converts the TensorFlow tensor to a NumPy array.

= Building a neural network

#figure(
  image("images/2024-12-17-building-neural-network.png")
)

1. *Specifying Layers Sequentially*
   - Layers (e.g., Layer 1, Layer 2, Layer 3) are defined and combined into a neural network using TensorFlow's `Sequential` function.
   - TensorFlow automatically strings the layers together into a cohesive model.
   - TensorFlow's `Sequential` function simplifies model creation by combining layers efficiently.

2. *Workflow for Neural Network Training*
   - Steps:
     - Define layers with `Sequential`.
     - Store data in a matrix.
     - Run `compile` to configure the model.
     - Use `fit` to train the model.
   - Example: This process mirrors earlier examples, like the coffee classification network.

3. *Inference and Predictions*
   - Use `model.predict(X_new)` to make predictions on new data.
   - The function generates predictions based on the trained model.

4. *Compact Code Using Sequential*
   - Instead of explicitly assigning layers (e.g., Layer 1, Layer 2, Layer 3), place them directly into the `Sequential` function.
   - Benefit: Produces more concise and readable code.
   - Example:
     - Explicit approach: Define and connect layers manually.
     - Compact approach: Use `Sequential` to define the network structure in one step.

#figure(
  image("images/2024-12-17-foward-prop-single-layer.png")
)

1. *Overview*
   - Forward propagation can be implemented manually in Python to understand the underlying mechanics of frameworks like TensorFlow and PyTorch.
   - This exercise helps gain intuition about computations and could inspire future innovations in neural network frameworks.

2. *Using 1D Arrays*
   - Vectors and parameters are represented as 1D arrays in Python (single square brackets), unlike 2D matrices (double square brackets).
   - Example: A parameter like $w^[2]_1$ is represented as $w 2_1$ in Python using underscores for subscripts.

3. *Step-by-Step Computation of $a 1$ (First Layer Activations)*
   - *Compute $a 1_1$*:
     - Parameters: $w 1_1 = [1, 2]$, $b 1_1 = -1$.
     - $z 1_1 = w 1_1 \cdot x + b 1_1$.
     - $a 1_1 = g(z 1_1)$, where $g$ is the sigmoid function.

   - *Compute $a 1_2$*:
     - Parameters: $w 1_2 = [-3, 4]$, $b 1_2 = 0$.
     - $z 1_2 = w 1_2 \cdot x + b 1_2$.
     - $a 1_2 = g(z 1_2)$.

   - *Compute $a 1_3$*:
     - Similar process to $a 1_1$ and $a 1_2$ with respective weights and biases.

   - *Group $a 1_1$, $a 1_2$, and $a 1_3$*:
     - Use `np.array` to combine these values into $a 1$, which represents the output of the first layer.

4. *Compute $a 2$ (Second Layer Output)*
   - Parameters: $w 2_1$ and $b 2_1$.
   - $z 2 = w 2_1 \cdot a 1 + b 2_1$.
   - $a 2 = g(z 2)$.

= General Implementation of forward propagation

#figure(
  image("images/2024-12-17-general-forward-propagation.png")
)

=== Dense Layer Function

- *Definition:* A dense layer is a single layer of a neural network that computes activations from the previous layer using weights $w$ and biases $b$.

==== Parameters
1. *Input Activation:* The activation from the previous layer, denoted as $a$ (e.g., $a_0 = x$ for input layer).
2. *Weights ($w$):* Parameters stacked in columns, e.g., $w_{1,1}, w_{1,2}, w_{1,3}$ form a $2  3$ matrix.
3. *Biases ($b$):* Parameters stacked into a 1D array, e.g., $[-1, 1, 2]$.

==== Dense Function Workflow
1. *Initialization:*
   - Compute the number of units: $"units" = W."shape"[1]$.
   - Initialize $a$ as an array of zeros with $"units"$ elements.
2. *Compute Activations:*
   - Loop through each unit $j$ in the range $[0, "units"-1]$:
     - Extract column $j$ of $w$ using $W[:, j]$.
     - Compute $z = w \cdot a + b[j]$.
     - Compute activation $a[j] = g(z)$, where $g$ is the sigmoid function.
3. *Return Output:* The array $a$ containing activations for the current layer.

#codeBlock(
  ```python
  def dense(a_prev, W, b):
      units = W.shape[1]
      a = np.zeros(units)
      for j in range(units):
          w = W[:, j]
          z = np.dot(w, a_prev) + b[j]
          a[j] = sigmoid(z)
      return a
  ```
)
