---
layout: default 
--- 
 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## Gaussian Process classifier

This page describes an experimental feature that is likely to change in future releases

This example provides an introduction to Gaussian Process modelling in Infer.NET. You can run the code using the [**Examples Browser**](The examples browser.md). The goal is to build a non-linear Bayes point machine classifier by using a Gaussian Process to define the scoring function. To set up the problem, suppose we have the following data:

```csharp
// The data  
Vector[] inputs = new Vector[] {  
  Vector.FromArray(new double[2] {0, 0}),  
  Vector.FromArray(new double[2] {0, 1}),  
  Vector.FromArray(new double[2] {1, 0}),  
  Vector.FromArray(new double[2] {0, 0.5}),  
  Vector.FromArray(new double[2] {1.5, 0}),  
  Vector.FromArray(new double[2] {0.5, 1.0})  
};  
bool[] outputs = { true, true, false, true, false, false };
```

Each element of xdata is a vector of input values, and the corresponding element of ydata is the desired output (i.e. label or class) for that vector. As in the [linear Bayes point machine](Bayes Point Machine tutorial.md), we will first map the input into a real-valued score, then threshold the score to determine the output. The only difference is that the score will be an arbitrary non-linear function of the input vector.

### Random functions

A Gaussian Process is a distribution over functions. In Infer.NET, a function from `Vector` to `double` is denoted by the type `IFunction`. Therefore a random function has type `Variable<IFunction>`. Such variables can be given a Gaussian Process prior, and when you infer the variable, you get a Gaussian Process posterior. Infer.NET implements an efficient type of Gaussian Process called a _sparse Gaussian Process_ that allows you to control the cost of inference by specifying a basis on which the function will be represented. For the moment, we will skip the details of defining a sparse Gaussian Process and focus on creating and using random functions. Here is the code to create a random function:

```csharp
// Set up the GP prior, which will be filled in later  
Variable<SparseGP> prior = Variable.New<SparseGP>().Named("prior");  

// The sparse GP variable - a distribution over functions  
Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");
```

To use a random function, you provide an input vector and get a random output value. This is done with Variable.FunctionEvaluate(f,x). In our case, we want to evaluate the function _`f`_ at the locations provided by _inputs_, and threshold the output values:

```csharp
// The locations to evaluate the function  
VariableArray<Vector> x = Variable.Observed(inputs).Named("x");  
Range j = x.Range.Named("j");  

// The observation model  
VariableArray<bool> y = Variable.Observed(outputs, j).Named("y");  
Variable<double> score = Variable.FunctionEvaluate(f, x[j]);  
y[j] = (Variable.GaussianFromMeanAndVariance(score, 0.1) > 0);
```

Note that we have added some Gaussian noise to the score before thresholding it, to allow some noise in the labels. The Gaussian Process classification model can easily be changed into a Gaussian Process regression model or other likelihood model simply by changing the line of code that relates the data _`y`_ to the score.

### Gaussian Process distributions

A Gaussian Process distribution is defined by a mean function and a covariance function. The mean function maps Vector to double so it has type `IFunction`. The covariance function maps two Vectors to double so it has type `IKernelFunction`. For example, the following creates a Gaussian Process with zero mean function and squared exponential covariance function (length scale = exp(0) = 1):

```csharp
GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), new SquaredExponential(0));
```

Infer.NET provides a small set of commonly-used mean and covariance functions (see the [Kernels namespace](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Kernels.html)), and it is easy to define your own. You simply have to create a class that implements [IFunction](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.IFunction.html) or [IKernelFunction](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Kernels.IKernelFunction.html). 

To get a sparse Gaussian Process, we pair a GaussianProcess with a set of basis vectors. The basis vectors are intended to summarize the set of inputs into a smaller set. By changing the size of the basis, you control the cost of the inference. (For details, see the references at the end.)  If the basis set is exactly the set of inputs, then the distribution is equivalent to a full (non-sparse) Gaussian Process. A good strategy for computing the basis is to cluster the input vectors. Another approach is to use a random subset of the input vectors. Here for simplicity we will set them by hand to roughly partition the range of the inputs:

```csharp
// The basis  
Vector[] basis = new Vector[] {  
  Vector.FromArray(new double[2] {0.2, 0.2}),  
  Vector.FromArray(new double[2] {0.2, 0.8}),  
  Vector.FromArray(new double[2] {0.8, 0.2}),  
  Vector.FromArray(new double[2] {0.8, 0.8})  
};  
// Fill in the sparse GP prior  
prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));
```

Now we have all the pieces in place to infer the random function and make predictions:

```csharp
InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());  
// Infer the posterior Sparse GP  
SparseGP sgp = engine.Infer<SparseGP>(f);  

// Check that training set is classified correctly  
Console.WriteLine();  
Console.WriteLine("Predictions on training set:");  
for (int i = 0; i < outputs.Length; i++) {  
  Gaussian post = sgp.Marginal(inputs[i]);  
  double postMean = post.GetMean();  
  string comment = (outputs[i] == (postMean > 0.0)) ? "correct" : "incorrect";  
  Console.WriteLine("f({0}) = {1} ({2})", inputs[i], post, comment);
```

Note that we could have built a model for making predictions (as in the [Bayes Point Machine tutorial](Bayes Point Machine tutorial.md)) but here for simplicity we call the Marginal method on the [SparseGP](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.SparseGP.html) posterior to get the distribution of the score at a particular input. (It is also possible to get the joint distribution of the scores at multiple inputs.)

### Selecting the covariance function

An important issue in Gaussian Process modelling is choosing the appropriate covariance function (both its type and its parameters such as length scales). One approach is to treat each possible covariance function as a separate model and apply Bayesian model selection. In Infer.NET, it is straightforward to score a model as discussed in [Computing model evidence for model selection](Computing model evidence for model selection.md). For the Gaussian Process classifier, we just wrap all of the model code in an evidence block:

```csharp
// Open an evidence block to allow model scoring  
Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");  
IfBlock block = Variable.If(evidence);  

... all model code ...  

// Close the evidence block  
block.CloseBlock();
```

Now we can score different covariance functions against our data by setting prior.ObservedValue to various SparseGP priors. The example does this for 3 different possibilities, giving the following results:

```
Compiling model...done. 
Iterating:  
.........|.........|.........|.........|.........| 50  
SquaredExponential(0,0) evidence = -3.95  
Iterating:  
.........|.........|.........|.........|.........| 50  
SquaredExponential(-0.5,0) evidence = -3.881  
Iterating:  
.........|.........|.........|.........|.........| 50  
NNKernel(0 0,-1) evidence = -3.853  

Predictions on training set:  
f(0 0) = Gaussian(0.3254, 0.1141) (correct)  
f(0 1) = Gaussian(0.2812, 0.1001) (correct)  
f(1 0) = Gaussian(-0.5902, 0.1716) (correct)  
f(0 0.5) = Gaussian(0.3277, 0.07105) (correct)  
f(1.5 0) = Gaussian(-0.7789, 0.2367) (correct)  
f(0.5 1) = Gaussian(-0.1306, 0.09745) (correct)
```

Notice that the model is only compiled once, while inference is repeated three times (once for each prior). In this case, the neural net covariance function provides the best fit, and classifies all of the training data correctly.

### References for sparse Gaussian Processes

L. Csato, M. Opper. "[Sparse representation for Gaussian process models](http://academic.research.microsoft.com/Paper/560692.html)." In Advances in Neural Information Processing Systems 13. MIT Press, pp. 444-450, 2000. 

Yuan (Alan) Qi, Ahmed H. Abdel-Gawad, and Thomas P. Minka. ["Sparse-posterior Gaussian Processes for general likelihoods."](http://event.cwi.nl/uai2010/papers/UAI2010_0283.pdf) In Proceedings of the Twenty-Sixth Conference in Uncertainty in Artificial Intelligence, 2010.
