---
layout: default 
--- 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Computing derivatives of functions

The commonly used distributions like `Gaussian` and `Gamma` provide a `GetDerivatives` method that returns the derivative of the log-density at any point.  When you infer the marginal distribution of a variable, you are getting an approximation to the posterior.  Applying `GetDerivatives` to the posterior gives the derivative of the log-likelihood plus the derivative of the log-prior.  To get the derivative of the log-likelihood at a point, you can in principle make the prior narrow around that point, compute the derivative of the log-posterior, and subtract the derivative of the log-prior.
In practice, that approach suffers from large round-off errors.  

A better approach is to ask for `MarginalDividedByPrior`, which excludes the prior automatically.  That way the prior can be a point mass, and the derivative is exact.  

This technique is illustrated in the example below.  The example uses `Variable.ConstrainEqualRandom` to [increment the log density](Increment log density.md) by the expression whose derivative we want to take.

```csharp
// Compute the derivative of the function log(x) at the point x = 0.5
Variable<double> xPoint = Variable.Observed(0.5);
Variable<double> x = Variable.GammaFromMeanAndVariance(xPoint, 0.0);
x.AddAttribute(QueryTypes.MarginalDividedByPrior);
Variable<double> f = Variable.Log(x);
Variable.ConstrainEqualRandom(f, Gaussian.FromNatural(1, 0)); // increment the log density by f

InferenceEngine engine = new InferenceEngine();
var xPost = engine.Infer<Gaussian>(x, QueryTypes.MarginalDividedByPrior);
xPost.GetDerivatives(xPoint.ObservedValue, out double derivative, out _);
// derivative is 1 / xPoint
```
To compute the derivative at a different point, change `xPoint.ObservedValue` and call `engine.Infer` again.
Here is another example, using a Gaussian-distributed `x` and a Gamma-distributed `f`:
```csharp
// Compute the derivative of the function exp(x) at the point x = 0.5
Variable<double> xPoint = Variable.Observed(0.5);
Variable<double> x = Variable.GaussianFromMeanAndVariance(xPoint, 0.0);
x.AddAttribute(QueryTypes.MarginalDividedByPrior);
Variable<double> f = Variable.Exp(x);
Variable.ConstrainEqualRandom(f, Gamma.FromNatural(0, -1)); // increment the log density by f

InferenceEngine engine = new InferenceEngine();
var xPost = engine.Infer<Gaussian>(x, QueryTypes.MarginalDividedByPrior);
xPost.GetDerivatives(xPoint.ObservedValue, out double derivative, out _);
// derivative is exp(xPoint)
```
