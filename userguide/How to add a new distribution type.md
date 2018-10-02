---
layout: default 
--- 
 
[Infer.NET user guide](index.md)

## How to add a new distribution type

The purpose of this section is to give you a flavour of what you need to do if you want to implement a distribution type which plugs into the Infer.NET framework. This is not an exhaustive account, and it assumes that you are familiar with writing C# classes and structs. Code for all of the built-in distributions is provided in the [src\\Runtime\\Distributions folder](https://github.com/dotnet/infer/tree/master/src/Runtime/Distributions) of your installation, so there is a lot of example code to guide you through if you decide to write your own distribution type. In addition, you can refer to the [code documentation for distributions](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.html).

Distribution types should be placed in the [**Microsoft.ML.Probabilistic.Distributions**](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.html) namespace, and should be marked with the \[Serializable\] attribute. Distribution types should also implement a copy constructor (you get this for free if you implement the SetTo method in the `SettableTo<>` interface).

#### Interfaces

Distribution types are the life-blood of Infer.NET - being used both in model definition, model output, and forming the messages that are updated by the inference algorithm as it executes its schedule. Because of this central role, distributions need to be able to provide the inference algorithms, and the message updates, with a way to query what type of operations the distribution allows. They do this by providing a subset of a standard set of interfaces. It is instructive to look at the main set of interfaces that the built-in Gaussian distribution implements:

```csharp
1 public struct Gaussian :  
2   IDistribution<double>,  
3   SettableTo<Gaussian>,  
4   SettableToProduct<Gaussian>,  
5   SettableToRatio<Gaussian>,  
6   SettableToPower<Gaussian>,  
7   SettableToWeightedSum<Gaussian>,  
8   Sampleable<double>,  
9   CanGetMean<double>,  
10  CanGetVariance<double>,  
11  CanGetMeanAndVarianceOut<double, double>,  
12  CanSetMeanAndVariance<double, double>,  
13  CanGetLogAverageOf<Gaussian>
14  {  
15      // Implementations  
16  }
```

This looks quite complicated - but these are all very simple interfaces (typically having just a single method) that need to be fleshed out for a given distribution. Visual Studio helps out in that it provides options to fill out all the boiler-plate code for the implementations. This set covers most (but not all) of the interfaces which a distribution might want to expose. Some distributions may only implement a few of these. Distributions should almost always inherit from `IDistribution<T>` as in line 2, where the type T should be specified as the sample type of the distribution.

Note that the **Gaussian** type is defined as a struct. This is reasonable because a Gaussian is parameterised by just two values. Defining it as a struct yields much more efficient array processing in compiled code as a struct is stored as a value type rather than a reference type. Many distributions such as **Beta**, **Gamma**, **Poisson**, and **Bernoulli** have a similar small footprint and are defined as structs. However some other distributions such as **VectorGaussian**, **Wishart**, **Discrete,** **Dirichlet**, and **SparseGP** have a larger footprint and need to be defined as classes.

One important general point to note is that messages in an inference algorithm are allowed to be improper (for example to have negative precision) provided the resulting marginals are proper; since the distribution types are used for the messages, your type should allow for such improper messages. We will touch on improper messages at a couple of points below.

#### IDistribution<T\>

The interface [IDistribution<T\>](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.IDistribution-1.html) should always be implemented. It inherits from several different interfaces

```csharp
public interface IDistribution<T> :  
    ICloneable,
    HasPoint<T>,
    Diffable,
    SettableToUniform,
    CanGetLogProb<T> {}
```

`ICloneable` is a standard .NET interface, and contains just a `Clone()` method which can be implemented using the copy constructor. The remaining interfaces are as follows:

```csharp
public interface HasPoint<T>  
{
  T Point { get; set; }
  bool IsPointMass { get; }  
}
```

[HasPoint<>](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.HasPoint-1.html) relates to whether the distribution can be a point mass. In the case of the Gaussian type, the answer is yes - the degenerate case having finite mean and zero variance (infinite precision) is supported by the type. The Point property in the interface allows a client to set or get the point. An important implementation requirement here is that a call to the get the Point property should succeed even if the state of the distribution is not a point. In the case of the Gaussian type, this returns the mean.

```csharp
public interface Diffable  
{  
   double MaxDiff(object that);
}
```

[Diffable](../apiguide/api/Microsoft.ML.Probabilistic.Math.Diffable.html) contains a single method which returns a measure of difference between two instances of the distribution (this and that). This can be any measure of difference you choose. For an exponential family distribution, this will typically be the maximum absolute difference between corresponding natural parameters. For a Gaussian, the absolute difference between the two precisions, and the absolute difference between the two precision times mean values are both calculated, and the maximum of the two is returned.

```csharp
public interface SettableToUniform  
{  
    void SetToUniform();  
    bool IsUniform();  
}
```

[SettableToUniform](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.SettableToUniform.html) relates to whether the distribution can be uniform. In the case of the Gaussian type, the answer is yes - the degenerate case having 0 precision (infinite variance) is supported by the type. The SetToUniform() method in the interface allows a client to set the distribution instance to this degenerate case, and the IsUniform() method allows client code to determine if the instance is in this degenerate state.

```csharp
public interface CanGetLogProb<T>  
{
  double GetLogProb(T value);  
}
```

[CanGetLogProb<T\>](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.CanGetLogProb-1.html) has one method GetLogProb which, given a sample value, returns the probability density

#### SettableTo... Interfaces

There are several interfaces in addition to SettableToUniform which relate to setting the parameters of the distribution through some calculation. For example SettableToProduct pertains to setting the instance as a product of two other instances of the same distribution type (modulo some normalisation term). Product and Ratio computations are widely used in an algorithm such as Expectation Propagation where factors are removed and inserted in turn in an overall factorisation. Here is the definition of the [SettableToProduct](../apiguide/api/Microsoft.ML.Probabilistic.Math.SettableToProduct-1.html) interface:

```csharp
public interface SettableToProduct<T>  
{  
    void SetToProduct(T a, T b);  
}
```

and here is the implementation of its one method for the in-built Gaussian type:

```csharp
public void SetToProduct(Gaussian a, Gaussian b)  
{ 
  if (a.IsPointMass) { 
    if (b.IsPointMass && !a.Point.Equals(b.Point))  
      throw new AllZeroException(); Point = a.Point;  
  } else if (b.IsPointMass) { 
     Point = b.Point;
  } else {  
    Precision = a.Precision + b.Precision;  
    MeanTimesPrecision = a.MeanTimesPrecision + b.MeanTimesPrecision;  
  }  
}
```

There are a couple of things to note here. First that the normalising factor is not calculated for the product. This can be calculated in logarithmic form by calling the `GetLogAverageOf` method of the `CanGetLogAverageOf` interface as described in the next subsection. The normalising factor is only needed for evidence calculations, and so is separated out into a separate interface. The second thing to note is that it is important in your implementation to deal with the degenerate cases.

The equivalent interface and method for ratios is very similar except for minus signs rather than plus signs. In this case the Precision and `MeanTimePrecision` can become negative giving rise to improper distributions. This is perfectly valid for the inference algorithm. Since such improper distributions cannot be normalised, we implicitly assume them to have a normalisation factor of 1.0 - this convention, which must be applied consistently, is important when we are dealing with calculating evidence.

Other related interfaces are (a) `SettableTo` which sets an existing instance to the state of another instance - this is widely used (b) `SettableToPower` which raises a distribution to a power - this is needed if your distribution participates in a gate or a `ShifAlpha` factor, and (c) `SettableToWeightedSum` which is also needed if your distribution participates in a gate.

#### CanGetLogAverageOf etc.

Like many of the distribution interfaces, [CanGetLogAverageOf](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.CanGetLogAverageOf-1.html) and its relatives each have a single method:
```csharp
public interface CanGetLogAverageOf<T>  
{  
 double GetLogAverageOf(T that);  
}
```
In this case the method calculates the log of the integral of the current instance with another instance of a distribution of the same type. This calculation represents the log of the probability that the two distributions would draw the same sample. Such a method is essential for any calculation involving evidence - such as having a gate in your model, or explicitly requesting evidence. So if you want to incorporate your distribution into such a model, you need to implement this set of interfaces.

An essential consideration for these methods is that one or more of the distributions may be improper as discussed in earlier subsections. So the implementation must take into account the different cases and use the appropriate normalisation factors. Refer to the source code for examples.

Related interfaces are (a) `CanGetLogAverageOfInverse` which is needed in similar circumstances as `CanGetLogAverageOf`, and (b) `CanGetLogPowerSum` which is needed for computing evidence when you are using Power EP.

#### Sampleable interface

The [Sampleable](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Sampleable-1.html) interface contains methods for sampling the distribution. The first method returns a random sample from the distribution. The second method is not relevant for distributions which are defined over value types and should just call the first method, ignoring the parameter. For distributions over a reference type, the second method allows the client code to pass down an existing instance to hold the result.

```csharp
public interface Sampleable<T>  
{  
    T Sample();  
    T Sample(T result);  
}
```

Sample methods should be marked with the \[Stochastic\] attribute, indicating that the return of the method is not completely determined by its arguments.

There is a fairly widespread assumption in Infer.NET that the `Sampleable` interface is implemented by a distribution, so you should try to provide an implementation even if it is approximate or incomplete, and even if you don't plan on sampling from the distribution. An example is the **SparseGP** distribution (Sparse Gaussian Process) - here the distribution is over functions, and it is not reasonable to provide sample functions over high dimensional input spaces. In this case, the `Sampleable` interface is implemented, but the implementation throws an exception for input spaces above dimension 1.

#### CanGetMean etc

The interfaces [CanGetMean](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.CanGetMean-1.html), [CanGetMeanAndVariance](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.CanGetMeanAndVariance-2.html) etc. are straightforward, and relate to which combinations of mean and variance can be got from or set in the distribution. For example, in the Gaussian case, we don't allow individual setting of the mean (which would require an implementation of `CanSetMean`, but we do allow joint setting of mean and variance (`CanSetMeanAndVariance` interface).

#### Static methods

It is recommended that a distribution type provide several methods for constructing an instance in standard ways. Two essential statics, which should normally be implemented for all distributions, are those which create a new point-mass instance and a new uniform instance. For Gaussian, these are:

```csharp
public static Gaussian PointMass(double mean);  
public static Gaussian Uniform();
```

Other examples provided by the Gaussian distribution are:

```csharp
public static Gaussian FromMeanAndVariance(double mean, double variance);
public static Gaussian FromMeanAndPrecision(double mean, double precision);
public static Gaussian FromNatural(double meanTimesPrecision, double precision);
```

The exact details will differ from distribution to distribution. Being static, these cannot be part of an interface and so are not required or queryable by the Infer.NET framework. However, as a courtesy to the users of your distribution, it is recommended that you cover the normal ways of parameterising your distribution type in these static construction methods.
