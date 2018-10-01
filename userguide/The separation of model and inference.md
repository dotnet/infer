---
layout: default
---

[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## The separation of model and inference

The Infer.NET API provides a way of declaratively describing a model, and the Infer.NET compiler generates the code that runs inference on the model. The model encodes prior domain knowledge whereas the inference specifies computational constraints. There are some important reasons to keep these separate:

1. **Transparency**: The assumptions in the model can be clearly seen and are not embedded in complex inference code.
2. **Consistency**: You can create one model and query it in different ways - for example, in training you can query for parameters, in testing you can query for predictions.
3. **Flexibility**: Different inference algorithms can be run on the same model.
4. **Maintainability**: If you want to refine the assumptions encoded in the model, the clean separation makes it straightforward to update it.

This page illustrates each of these points. To make things concrete we will look at a simple example of modeling the time it takes to bike in to work. When we are biking into work the first time we may only have a vague idea of how long it might take, but as we record the timings each day we gradually get a better idea.

### 1. Transparency

Our model is as follows. We have a number of timings `N` - we make this a variable because we will want to run the model on several occasions. We want to be able to specifiy our prior belief in the average time it will take - this is `averageTimePrior` in the code below which is a distribution we can set at runtime. The average time (`averageTime`) is then a variable which we want to infer and, prior to seeing any data, it is distributed according to our prior belief `averageTimePrior`. Finally, we need to attach our observed `timings`, and we make an assumption that our observations are randomly scattered around the unknown `averageTime` with a known precision of 1.0 (precision is 1.0/variance and is a more natural parameter with which to describe a Gaussian distribution). We could think of this noise as reflecting various random changes in traffic, weather etc.

```csharp
var N = Variable.New<int>();
var n = new Range(N);
var averageTimePrior = Variable.New<Gaussian>();
var averageTime = Variable.Random<double, Gaussian>(averageTimePrior);
var noise = 1.0;
var timings = Variable.Array<double>(n);
timings[n] = Variable.GaussianFromMeanAndPrecision(averageTime, noise).ForEach(n);
```

Note that this makes very clear what the assumptions are, in this case, the data is generated from a Gaussian with fixed noise and random mean with an as yet unspecified Gaussian prior. At this point we also have not specified any data in the form of observed timings.

### 2. Consistency

Suppose we now are given some data and we want to infer the average time it takes to cycle into work. At this point, we need to specify our prior belief for the average time (our initial assumption, on the first line below, says that it will take 20 minutes, but we are very unsure about this so the variance is large). We also need to hook up the data:

```csharp
averageTimePrior.ObservedValue = Gaussian.FromMeanAndVariance(20, 100);
timings.ObservedValue = new double[] { 15, 17, 16, 19, 13 };
N.ObservedValue = data.ObservedValue.Length;
```

We can infer the posterior average time as follows:

```csharp
InferenceEngine engine = new InferenceEngine();
engine.Algorithm = new ExpectationPropagation();
Gaussian averageTimePosterior = engine.Infer<Gaussian>(averageTime);
```

This will give us a posterior Gaussian distribution which, based on the observations, updates our initial belief (20 minutes with large uncertainty) to an updated belief (16 minutes with a standard deviation of about 0.45). There is still uncertainty in the average time it takes to ride into work because we only have a few timings under our belt - the more timings we make, the more confident we'll be in the accuracy of the average time.

Now suppose we want to ask a slightly different question - namely when I ride in tomorrow, what is the probability that it will take less than 20 minutes? To answer this question we can add a 'tomorrowsTime' variable to the model and directly make that query of Infer.NET:

```csharp
Variable<double> tomorrowsTime = Variable.GaussianFromMeanAndPrecision(averageTime, noise);
double probLessThan20Mins = engine.Infer<Bernoulli>(tomorrowsTime < 20).GetProbTrue();
```

The answer can be found in the table at the end of this page.

### 3. Flexibility

The code above uses [Expectation Propagation](Expectation Propagation.md). However, Infer.NET supports [3 main algorithms](Working with different inference algorithms.md), the other two being [Variational Message Passing](Variational Message Passing.md) and [Block Gibbs Sampling](Gibbs sampling.md). As documented these all have different characteristics, for example, trading off accuracy against speed. Also, not all algorithms are applicable to each model - refer to [the list of factors and constraints](list of factors and constraints.md) for more guidance about this. For this simple model, all three of these algorithms are applicable. If we want to try a different algorithm, we don't need to change the model; instead we simply reset the engine's algorithm:

```csharp
engine.Algorithm = new VariationalMessagePassing();
engine.Algorithm = new GibbsSampling();
```

Each engine can be then run in turn and, depending on the model, will give slightly different results due the nature of the approximations. The Gibbs Sampling inference should be exact if enough iterations are run. Some comparative results are given at the end of this page.

### 4. Maintainability

In our initial model we made an assumption about the size of the noise representing the day-to-day variation. Suppose now we want to change our model so that we learn the noise rather than make an assumption about its value. This is easily done by making the noise a random variable (with some prior) rather than a fixed variable:

```csharp
var noise = Variable.GammaFromShapeAndRate(2, 2);
```

If the model code had been mingled with the inference code, this would have been much trickier. Putting all the above together, here are the results of running our second model using all three algorithms:

| Expectation Propagation | Variational Message Passing | Gibbs Sampling |
|------------------------------------------------------------------------|
| **AverageTime mean** | 16.02 | 16.02 | 16.03 |
| **AverageTime standard deviation** | 1.00 | 0.77 | 0.89 |
| **Noise mean** | 0.36 | 0.33 | 0.33 |
| **Noise standard deviation** | 0.24 | 0.16 | 0.17 |
| **Tomorrow's mean** | 16.02 | 16.02 | 15.98 |
| **Tomorrow's standard deviation** | 1.73 | 1.05 | 1.68 |
| **Prob. tomorrow's ride < 20 minutes** | 98.91% | 99.99% | 99.15% |

Now suppose that there are days where there are unforeseen events such as a puncture, or an accident which necessitates a detour. The distribution of timings on these days is different from our standard timings. We may want to improve our model to represent normal days and exceptional days. This can be done by incrementally extending the model to include [branches which are conditioned on random variables](Branching on variables to create mixture models.md). Alternatively, there are many other directions you could take to extend the model using the different modeling components of Infer.NET.