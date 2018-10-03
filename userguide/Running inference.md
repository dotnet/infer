---
layout: default 
--- 
 
[Infer.NET user guide](index.md)

## Running inference

Inference is concerned with the calculation of posterior probabilities based on one or more data observations. The word 'posterior' is used to indicate that the calculation occurs **after** the evidence of the data is taken into account; 'prior' probabilities refer to any initial uncertainty we have. In the section on [the Infer.NET modelling API](The Infer.NET modelling API.md), a simple model was introduced to learn a Gaussian of unknown mean and precision from data. The first two lines of the program showed our initial uncertainty in the **mean** and **precision** of our simple Gaussian model (i.e. the prior). Then we introduced some data. Now we would like to infer the marginal (see later) posterior probabilities for the **mean** and **precision**. The program below shows how to do this:

```csharp
// The model defined using the modelling API  
Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100);  
Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);  
VariableArray<double> data = Variable.Observed(new  double[] { 11, 5, 8, 9 });  
Range i = data.Range;  
data[i] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(i);  

// Create an inference engine for VMP  
InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());  
// Retrieve the posterior distributions  
Gaussian marginalMean = engine.Infer<Gaussian>(mean);  
Gamma marginalPrecision = engine.Infer<Gamma>(precision);  
Console.WriteLine("mean=" + marginalMean);  
Console.WriteLine("prec=" + marginalPrecision);
```

If you run this, you will get output as follows:

```
mean=Gaussian(8.165, 1.026)  
prec=Gamma(3, 0.08038)
```

The Gaussian distribution shows the mean and variance of the mean marginal, and the Gamma distribution shows the shape and scale of the precision marginal. The pattern inherent in the this simple program is common to many inference problems. We start with a model which has some structure defined by some unknown parameters (considered as variables), their prior uncertainties, and their relationships. We then provide some observed data. Finally we perform inference and retrieve the posterior marginal distributions of the variables we are interested in.

The term 'marginal' is a common term in statistics which refers to the operation of 'summing out' (in the case of discrete variables) or 'integrating out' (in the case of continuous variables) the uncertainty associated with all other variables in the problem. The inference methods in Infer.NET make heavy use of fully factorised approximations; i.e. approximations which are a product of functions of individual variables. Although the true posteriors may be joint distributions over the participating variables, the approximate posteriors in such models are naturally in the form of single variable marginal distributions. In the example the marginal posteriors, **marginalMean** and **marginalPrecision**, are retrieved. Based on the evidence of the data, these distributions are seen to be much tighter than the prior distributions for these variables. For further background reading on inference, marginal distributions and fully factorised approximations, plesae refer to [Resources and References](Resources and References.md).

### Creating an inference engine

All inferences in Infer.NET are achieved through the use of an [**InferenceEngine**](../apiguide/api/Microsoft.ML.Probabilistic.InferenceEngine.html) object. When you create this object, you can specify the inference algorithm you wish to use. For example, to create an inference engine that uses Variational Message Passing, you write: 

```csharp
InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
```

In this example above, we explicitly set the algorithm to Variational Message Passing (VMP). Other available algorithms are Expectation Propagation (EP) and Gibbs sampling. Between them, these three algorithms can solve a wide range of inference problems; they are discussed further in [working with different inference algorithms](Working with different inference algorithms.md). 

As well the inference algorithm, the engine has a number of other [settings](inference engine settings.md) which you can modify to affect how inference is performed and what is displayed during the inference process.

### Performing inference

Having created an inference engine, you can use it to infer marginal distributions for variables you are interested in, using the **Infer<TReturn\>()** method. When using this method, you should pass in the type of the distribution you want to be returned. For example, this code asks for the posterior distribution over a variable **x** as a Gaussian distribution.

```csharp
Gaussian xPosterior = engine.Infer<Gaussian>(x);
```

For array variables, you can specify an array type over distributions. Here are some examples, the first one showing inference on a 1-D random variable array where each element of the array has a Gaussian distribution, the second showing inference on a 1-D array of 2-D arrays of random variables where each element has a Gamma distribution.

```csharp
Gaussian[] x1dArrayPosterior = engine.Infer<Gaussian[]>(x1dArray);  
Gamma[][,] x1dArrOf2DArrPosterior = engine.Infer<Gamma[][,]>(x1dArrOf2DArr);
```

Note that this syntax is for the usual situation where the random variables have been defined with Variable.Array as described in the sections on [variable arrays](Arrays and ranges.md) and [jagged variable arrays](Jagged arrays.md). If you use .NET arrays for your variables, then you must call infer for each element of the array.

### Inferring constant and observed variables​

Besides random variables, you can call **Infer** on a constant variable or observed variable. For example, in the above Gaussian model you can Infer **data** as a Gaussian array:

```csharp
Gaussian[] dataPosterior = engine.Infer<Gaussian[]>(data);
```

The result here will be an array of Gaussians, each a point mass on the observed value. This is useful when you want to experiment with making some variables observed vs. random. For example, if you change the definition of **data** to be `Variable.Array<double>(new Range(4))` instead of Variable.Observed then the above call to Infer still works, and the result is an array of Gaussians which are not point masses. To minimize the overhead of this feature, there is a restriction on which  observed variables can be inferred. Specifically, you can only infer an observed variable that has a definition other than its observation. In other words, Infer.NET must be able to determine what distribution the variable would have if it had not been observed. In the model above, **data** satisfies this condition because of the line **data\[i\] = ...**  If this line was not present, then you would not be able to infer the distribution of **data**.

##### When does inference happen for a particular variable?

When you call `Infer()` the first time, the inference engine will collect all factors and variables related to the variable that you are inferring (i.e. the model), compile an inference algorithm, run the algorithm, and return the result. This is an expensive process, so the results are cached at multiple levels. If you call `Infer()` on the same variable again without changing anything, then it will immediately return the cached result from the last call. If you call `Infer()` on another variable in the same model, then it will return the cached result for that variable, if one is available. For instance, in the example above, when `Infer<Gaussian>(mean)` is called, the inference engine will compute both the posterior over the mean _and_ the posterior over the precision (since this requires no additional computation). Then when `Infer<Gamma>(precision)` is called, this cached precision is returned and no additional computation is performed. These cached results get invalidated when you change things. In particular, if you change observed values or minor settings on the inference engine such as the number of iterations, then the inference results are invalidated, but the compiled algorithm is kept. If you change the model itself, such as adding new variables or constraints, making a variable observed when it was previously un-observed, or you change major settings such as the choice of inference algorithm, then the compiled algorithm is also invalidated and the next call to `Infer()` will trigger a re-compile.

In some cases, you want more control over what variables are inferred when you call `Infer()`. To do this, set the `OptimiseForVariables` property on the inference engine, specifying the set of variables whose marginals should be calculated. For example:

```csharp
engine.OptimiseForVariables = new[] {x,y,z};
```

After setting this property, the next call to **Infer()** with one of the listed variables will compute (and cache) results for all and only the variables listed. To retrieve marginals for the other variables in the list, you call **Infer()** as normal and it is guaranteed to return straight away with the cached marginal value. If you call **Infer()** for a variable not in the list, you will get an error. To revert to normal inference engine behaviour, set **OptimiseForVariables** to null.

Reasons for using **OptimiseForVariables**:

*   You want to remove the (normally small) overhead of calculating additional marginals that will not be used. 
*   You want to control when inference happens e.g. to ensure a user interface remains responsive
*   You want to ensure that inference happens in one go, and not in a number of separate runs of the inference algorithm. 

### Alternative inference queries

So far, we have only considered how to use an inference engine to retrieve marginal distributions. However, some inference algorithms can perform other types of inference query to return other quantities (such as lists of samples from the marginal). To perform alternative inference queries, you pass a second argument to **Infer()** which specifies a **QueryType** object. The set of built-in query types at provided for convenience on the static class **QueryTypes**. When using GibbsSampling, you can retrieve a set of posterior samples for variable **x** via:

```csharp
IList<double> postSamples = engine.Infer<IList<double>>(x, QueryTypes.Samples);
```

When using QueryTypes, it is a good idea to tell the compiler in advance what QueryTypes you want to infer. Otherwise, inference may re-run on each query. To specify the QueryTypes in advance, attach them as attributes to the variable, like so:

```csharp
x.AddAttribute(QueryTypes.Marginal);  
x.AddAttribute(QueryTypes.Samples);
```


Notice that in this case, you need to explicitly specify whether you want to infer Marginals. If you attach this attribute to each variable you want to sample, then when you request samples for multiple variables, the sampler is only run once. The samples at the same position in the returned lists correspond to a consistent vector sample.
