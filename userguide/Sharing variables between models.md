---
layout: default 
--- 
 
[Infer.NET user guide](index.md)

## Sharing variables between models

On occasion you might want to split up your inference so that it runs on different sub-models (or different copies of the same model) where the participating sub-models have variables in common. Here are some common scenarios where you might want to do this:

*   You have very large amounts of data and a single model won't fit into memory.
*   Your model is of a type that needs to maintain large numbers of large messages, and won't fit into memory for even moderate amounts of data; this may happen with models that contain arrays of Discrete-distributed variables over large value ranges which switch on the values of those variables (such as LDA-type models).
*   You are doing online inference and the data is not all available ahead of time.
*   You want more control over the schedule by splitting your model into smaller chunks and scheduling them manually.
*   You want to parallelise your inference code.
*   You want to run different algorithms on different parts of your model.

All these scenarios have a common pattern. You will need to

1.  Run inference to convergence on one of the sub-models, say model A
2.  Extract the shared variable messages **output** from model A
3.  Initialise the next model, model B, say, by providing as **input** a product of all the output messages from all models except B

### Creating a shared variable

Infer.NET provides a [**SharedVariable**](../apiguide/api/Microsoft.ML.Probabilistic.Models.SharedVariable-1.html) class which makes it easy to implement this pattern. It supports sharing between models of different structures and also across multiple data batches. Also you can update all shared variables with a single function call. You can create shared variables and shared variable arrays by calling the Random function on a shared variable where the generic type parameter is set to the element type of the random variable. You must pass a prior as the first argument. There is an optional second argument (set to true by default) which determines whether messages to batches are calculated by division (`true` \- the default) or multiplication (`false`); the former is more efficient but may introduce round-off error.

The following example extends the simple Gaussian example so that the data is divided into two chunks. This first code fragment shows the model. In this fragment, the unknown mean and precision of the Gaussian are defined as shared variables. These are shared between the two chunks. The other variables and factors defined in the code are not shared.

Because both chunks share the same structure, a single **Model** object is created. Each **Model** object represents a model structure. In this case, we have two datasets ('chunks') using the same model structure. Notice that the data count for a given chunk is provided as a variable, since it will vary from chunk to chunk.

The shared variables are turned into ordinary variables in the context of a given **Model** by calling **GetCopyFor**. In this case, we create ordinary variables corresponding to the shared mean and precision, and use them to generate the data. 

```csharp
// The data  
double[][] dataSets = new double[][]  
{  
    new double[] { 11, 5, 8, 9 },  
    new double[] { -1, -3, 2, 3, -5 }  
};  
int numChunks = dataSets.Length;  
// The model  
Gaussian priorMean = Gaussian.FromMeanAndVariance(0, 100);  
Gamma priorPrec = Gamma.FromShapeAndScale(1, 1);  
SharedVariable<double> mean = SharedVariable<double>.Random(priorMean);  
SharedVariable<double> precision = SharedVariable<double>.Random(priorPrec);  
Model model = new Model(numChunks);  
Variable<int> dataCount = Variable.New<int>();  
Range item = new Range(dataCount);  
VariableArray<double> data = Variable.Array<double>(item);  
data[item] = Variable.GaussianFromMeanAndPrecision(  
    mean.GetCopyFor(model),  
    precision.GetCopyFor(model)).ForEach(item);
```

The above code uses the streaming syntax to fill in the data array. If a ForEach block were used instead, then the calls to GetCopyFor would have to be outside this block, because we only want one 'mean' array per model, not one for each item. In general, calls to GetCopyFor should be outside any ForEach blocks.

### Inference

This next code fragment shows the inference. During inference, we cycle through each chunk, setting the appropriate chunk-specific variables (in this case, data and dataCount). The data and dataCount are set using the **ObservedValue** property. We then infer the variables using the **InferShared** method on the **Model** instance which takes care of extracting output messages to the shared variables from one batch, and creating the correct input messages to the shared variables for the next batch. For convergence, we perform multiple cycles through the batches.

```csharp
// Set the inference algorithm  
InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());  
for (int pass = 0; pass < 5; pass++)  
{  
    // Run the inference on each data set for (int c = 0; c < numChunks; c++)  
    {  
        dataCount.ObservedValue = dataSets[c].Length;  
        data.ObservedValue = dataSets[c];  
        model.InferShared(engine, c);  
    }  
}
```

We get the final marginal distribution over the shared variables by using the **Marginal** method. Note this does not initiate any inference. It simply returns the stored result from the computations above.

```csharp
// Retrieve the posterior distributions  
Gaussian marginalMean = mean.Marginal<Gaussian>();  
Gamma marginalPrec = precision.Marginal<Gamma>();  
Console.WriteLine("mean=" + marginalMean);  
Console.WriteLine("prec=" + marginalPrec);
```

### Further topics

*   [Shared variable arrays](Shared variable arrays.md)
*   [Defining shared variables within a model](Shared variable definition.md)
*   [Jagged shared variable arrays](Jagged shared variable arrays.md)
*   [Computing evidence for models with shared variables](Shared variable evidence.md)
*   [Using shared variables to support hybrid algorithms](Shared variable hybrid.md)
