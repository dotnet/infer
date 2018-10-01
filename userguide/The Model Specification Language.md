---
layout: default 
--- 
[Infer.NET user guide](index.md)

## The Model Specification Language

**IMPORTANT NOTE:  Writing models directly in MSL is not supported - this page is a reference to give a better understanding of how Infer.NET works.**

The model specification language (MSL) is used by Infer.NET internally to define a probabilistic model. If you use [the Infer.NET modelling API](The Infer.NET modelling API.md), it internally constructs a model in MSL. The modelling API takes care of making sure that all language constraints are met and that the model is well formed. If desired, you can print out (to the console) the internally-generated MSL for your model, by setting **ShowMsl** to true in the [inference engine settings](inference engine settings.md). 

MSL is a subset of C# and is written as a special C# method, called a _model method_. The model method must be void, but can take any number of parameters. These parameters will allow quantities such as prior distributions, observed data and plate sizes to be passed to the model - they correspond to observed variables in the modelling API. Model methods can contain only the following types of statement:

*   **Variable declarations**  
    Used to declare constants or random variables, or arrays of these
*   **Static method calls**
    Used to define factors between random variables or constraints on random variables.
*   **For loops**  
    Used to process all elements of an array
*   **If statements**    
    Used to add gated factors, that is factors which may become active or inactive depending on the state of a random variable.

#### Example: A Gaussian model

The following method defines a model for inferring the mean and variance of an array of doubles.
```csharp
1 public void GaussianModel(double[] data)  
  {  
2 double mean = Factor.Random(new  Gaussian(0, 100));  
3   double precision = Factor.Random(new  Gamma(0, 1));  
4   for (int i = 0; i < data.Length; i++) {  
5     data[i] = Factor.Gaussian(mean, precision);  
    }  
6   InferNet.Infer(mean);  
7   InferNet.Infer(precision);  
  }
```
This method includes only variable declarations, static method calls and for loops, as required. Line 1 declares the method and indicates that it takes one parameter, the array of data. Lines 2 and 3 declare random variables for the mean and precision. with Gaussian and Gamma priors on these random variables. Line 4 loops over the data. Line 5 creates a Gaussian factor between the **i**th data element and the mean and precision variables. Finally, lines 6 and 7 indicate that the mean and precision are to be inferred, so that the inference engine knows to compute their posterior distributions.

### Declaring random variables

In MSL, the type of a random variable is represented by its distribution's sample type. For example, sampling from a Bernouilli distribution gives a `bool`, and sampling from a Gamma distribution gives a `double`. If the distribution is known and fixed, then the associated random variable can be defined using the static generic `Random` method in the `Factor` class. For example:

```csharp
  bool coin1Heads = Factor.Random(new  Bernoulli(0.5));  
  bool coin2Heads = Factor.Random(new Bernoulli(0.5));  
  int outcome = Factor.Random(new  Discrete(0.1, 0.3, 0.4, 0.2));
  double mean = Factor.Random(new  Gaussian(0, 100));
  double precision = Factor.Random(new  Gamma(0, 1));  
  VectorGaussian vg =new  VectorGaussian(Vector.Zero(2), PositiveDefiniteMatrix.Identity(2));
  Vector w = Factor.Random(vg);
```

The `Random` method, if called directly from C# code, does in fact return a sample from the distribution. However, as a line in model method it is interpreted by the model compiler as a random variable. This type of syntax encourages a generative viewpoint in which an observed datum is considered a sample from the overall graphical model. Because the `Random` method in the `Factor` class is a short-cut for calling the sampling method on a distribution, we can alternatively make an explicit call to the Sample method on the relevant distribution class. For example

```csharp
doublemean = Gaussian.Sample(0, 100);
```

 is equivalent to

```csharp
double mean = Factor.Random(new  Gaussian(0, 100));
```

#### Adding factors and constraints

Variables are also defined by combining existing variables in a factor. These are typically indicated by static methods on the `Factor` class, though factor methods may exist in other classes; for example if you define your own factor method - see [How to add a new factor](How to add a new factor and message operators.md). Also, you can refer to the [list of factors and constraints](list of factors and constraints.md) to see all of the built-in options. Here are some examples which make use of the code snippet in the previous section:

```csharp
  bool bothHeads = Factor.And(coin1Heads, coin2Heads);  
  double x = Factor.Gaussian(mean, precision);  
  double xminus1 = Factor.Difference(x, 1.0);
```

The above code defines new variables in terms of existing variables. Again, these can be thought of as samples in our generative view of the model. There are also a set of static methods on the `Constrain` class. For example:

```csharp
Constrain.Positive(xminus1);
```

This constains the variable `xminus1` to be positive, but, unlike a factor, does not create another variable in the model.
