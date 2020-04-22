---
layout: default 
--- 
[Infer.NET user guide](index.md)

## Online learning

Bayesian inference makes it easy to update the parameters of a model as data arrives one at a time. In principle, you just use yesterday's posterior as today's prior. If the prior in your model has the same type as the posterior you are inferring, then this is straightforward. If the prior has a different type (for example it is specified using multiple factors and variables of its own), then you need a different approach. This document describes both methods.

As a concrete example, suppose you want to learn the mean of Gaussian as data arrives one at a time.  If all of the data were available together, then you could provide it all to Infer.NET and this would give the best results.  We will assume that your goal is to get results as close as possible to processing the data together.  To do that, we need to mimic the operations that Infer.NET would have performed if it were performing one iteration over the data.

If the prior for the mean is a fixed Gaussian distribution, then you are in the simple case. We’ll assume that the online learning is performed over a set of data batches, and the parameter _mean_ is shared across all of them. What you do after training on each data batch is to infer the posterior of _mean_ and plug it in as the new prior for the next batch. Thus, you can think of _meanPrior_ as the summary of what you’ve learnt “so far” (or “up to the current batch”).

When the prior is a more elaborate collection of factors and variables, you can no longer store your current knowledge in it in such a way. Suppose the variable _mean_ now has a Laplacian prior, i.e. a Gaussian distribution whose mean is zero but whose variance is drawn from a Gamma. The simple approach does not work here since the posterior will be projected onto a Gaussian but your desired prior is not Gaussian.  Since your goal is to get results as close as possible to processing the data together, you need to keep the form of the prior intact.

To process the batches sequentially, you need to add a new “accumulator” of your knowledge (an equivalent to what _meanPrior_ was used for in the non-hierarchical case). We’ll call this _meanMessage_. It stores the product of all messages sent to _mean_ by the previous batches. You attach this variable to _mean_ by using the special factor _ConstrainEqualRandom_. This has the effect of multiplying _meanMessage_ into the posterior of _mean_. The resulting factor graph is shown below.

![OnlineLearning.Png](OnlineLearning.Png)

Initially, _meanMessage_ should be _Uniform_, since there are no previous batches. After each batch, you need to store there the message sent upward to _mean_, which also happens to be the marginal of _mean_ divided by its prior. Infer.NET has a special QueryType that gives you this ratio directly:

```csharp
meanMessage.ObservedValue = engine.Infer<Gaussian>(mean, QueryTypes.MarginalDividedByPrior);
```

However, you need to give the compiler a hint that you’ll be doing this:

```csharp
mean.AddAttribute(QueryTypes.MarginalDividedByPrior);
```

Here’s the complete code for online learning of a Gaussian with a Laplacian mean:

```csharp
Variable<int> nItems = Variable.New<int>().Named("nItems");  
Range item = new Range(nItems).Named("item");  
Variable<double> variance = Variable.GammaFromShapeAndRate(1.0, 1.0).Named("variance");  
Variable<double> mean = Variable.GaussianFromMeanAndVariance(0.0, variance).Named("mean");  
VariableArray<double> x = Variable.Array<double>(item).Named("x");  
x[item] = Variable.GaussianFromMeanAndPrecision(mean, 1.0).ForEach(item);  

Variable<Gaussian> meanMessage = Variable.Observed<Gaussian>(Gaussian.Uniform()).Named("meanMessage");  
Variable.ConstrainEqualRandom(mean, meanMessage);  
mean.AddAttribute(QueryTypes.Marginal);  
mean.AddAttribute(QueryTypes.MarginalDividedByPrior);  
InferenceEngine engine = new InferenceEngine();  

// inference on a single batch  
double[] data = { 2, 3, 4, 5 };  
x.ObservedValue = data;  
nItems.ObservedValue = data.Length;  
Gaussian meanExpected = engine.Infer<Gaussian>(mean);  

// online learning in mini-batches  
int batchSize = 1;  
double[][] dataBatches = new double[data.Length / batchSize][];  
for (int batch = 0; batch < dataBatches.Length; batch++)  
{  
    dataBatches[batch] = data.Skip(batch * batchSize).Take(batchSize).ToArray();  
}  
Gaussian meanMarginal = Gaussian.Uniform();  
for (int batch = 0; batch < dataBatches.Length; batch++)  
{  
    nItems.ObservedValue = dataBatches[batch].Length;  
    x.ObservedValue = dataBatches[batch];  
    meanMarginal = engine.Infer<Gaussian>(mean);  
    Console.WriteLine("mean after batch {0} = {1}", batch, meanMarginal);  
    meanMessage.ObservedValue = engine.Infer<Gaussian>(mean, QueryTypes.MarginalDividedByPrior);  
}  
// the answers should be identical for this simple model  
Console.WriteLine("mean = {0} should be {1}", meanMarginal, meanExpected);
```

This approach generalizes to any number of shared parameters. For each parameter, you store its upward messages in an accumulator, and attach the accumulator to the parameter via _ConstrainEqualRandom_.  This approach is the closest you can get to processing the data together because it duplicates exactly what would happen internally if you ran EP for one iteration over the data.

The approach described here is a simplified version of what [Shared Variables](Sharing variables between models.md) do automatically.  You can use Shared Variables for online learning as well, but inference will be slightly less efficient since Shared Variables are designed for multiple passes through the data.
