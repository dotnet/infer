---
layout: default 
--- 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Increment log density

Sometimes it is convenient to specify parts of the model directly in terms of their log density instead of writing a sampler.  This can be done using `Variable.ConstrainEqualRandom`.  
When you write `Variable.ConstrainEqualRandom(x, dist)`, you are incrementing the log density by `dist.GetLogProb(x)`.  
If you just want to increment the log density by `x`, use `Variable.ConstrainEqualRandom(x, Gaussian.FromNatural(1,0))` or `Variable.ConstrainEqualRandom(x, Gamma.FromNatural(0,-1))` as appropriate.

For example, `Variable.ConstrainEqualRandom(x, new Gaussian(0, 1))` increments the log-density by `(new Gaussian(0,1)).GetLogProb(x)` which is equal to `-0.5*x*x - MMath.LnSqrt2PI`.  This is equivalent to the following:

```csharp
Variable<double> y = Variable.GaussianFromMeanAndVariance(x, 1);
y.ObservedValue = 0;
```

Unlike sampling, `ConstrainEqualRandom` works with improper distributions.  Improper distributions are unnormalized, which means `Gaussian.FromNatural(1,0).GetLogProb(x) == Gamma.FromNatural(0,-1).GetLogProb(x) == x`.