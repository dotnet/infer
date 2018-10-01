---
layout: default 
--- 
 
[Infer.NET user guide](index.md)

## Controlling how inference is performed

You can get fine-grained control of how inference is performed by getting hold of a compiled algorithm object. The following code shows, for the simple Gaussian example, how to do this. The starting point is to call GetCompiledInferenceAlgorithm on an InferenceEngine, passing in the complete set of variables that you want the compiled algorithm to be able to infer. 

```csharp
// Set the inference algorithm  
InferenceEngine engine = new  InferenceEngine(new  VariationalMessagePassing());  
// Get the compiled inference algorithm  
var ca = engine.GetCompiledInferenceAlgorithm(mean, precision);
```

Once you have a reference to the compiled algorithm, you can explicitly control the initialisation and updates of the algorithm. This allows, for example, implementation of a custom convergence criterion by monitoring the level of change in certain marginals of interest. The **IGeneratedAlgorithm** object that you get is the same that you would get if you [precompiled your inference algorithm](Using a precompiled inference algorithm.md).

The following code snippets illustrate all the ways in which you can use the compiled algorithm object:

#### Reset

This method sets all internal state to its initialised values. It is useful when you want to force the algorithm to restart its convergence.

```csharp
ca.Reset();
```

#### Update

The Update method performs a specified number of algorithm updates, starting from the current message state. Each update is an entire single pass through the schedule which updates all forward and backward messages. If the scheduler has determined that iteration is not necessary for inference, then this does nothing after the first iteration (unless observed values change).

```csharp
// Run through the schedule 10 times

ca.Update(10);
```

#### Execute

The execute method provides a short-cut for performing a reset followed by a series of updates. The following code snippet sets the number of iterations to 10 and gives equivalent results to ca.Reset() followed by ca.Update(10). However, Execute tries to minimize the amount of computation based on the existing message state. For example, if Execute(10) was just called and no observed values were changed, then calling Execute(10) again does nothing. Similarly, if Execute(9) was just called, then Execute(10) will do one additional update.

```csharp
// Execute 10 iterations of the algorithm  
ca.Execute(10);
```

#### Setting observed values

When using a compiled algorithm directly, you must use the **SetObservedValue(variableName, value)** method to ensure that the compiled algorithm is aware of the value of an observed variable.  You can also retrieve these values back using **GetObservedValue(variableName)**.  To get the name of a variable in the generated code, use the [NameInGeneratedCode](../apiguide/api/Microsoft.ML.Probabilistic.Models.Variable.html#Microsoft_ML_Probabilistic_Models_Variable_NameInGeneratedCode) property.  The compiled algorithm is automatically filled in with any observed values that were in place at the point when GetCompiledInferenceAlgorithm was called, so you only need to update the compiled algorithm when an observed value changes after that point.

#### Getting marginals

When using a compiled algorithm directly, you use the **Marginal(variableName)** method to retrieve marginal distributions (see below for an example). This can be done at any time, for example, so that the change in the marginal distribution during inference can be monitored to check for convergence. Unlike engine.Infer, this method does not cause inference to run; it simply returns whatever has already been computed by the compiled algorithm. 

#### Example

The following example illustrates these mechanisms by extending the simple Gaussian example in several ways. The purpose of this example is to perform inference using the same model for two different data sets, whilst avoiding compilation of the model between the two calls.

Firstly, the observed data changes for each call of the algorithm. We still create **data** as a variable array which is observed; however we defer setting what those observed values are until we are ready to call the algorithm; Note that we do need to set an initial dummy value to let the algorithm know that **data** is observed (in the code below it is just given a null value).

Because the length of **data** is now variable, we introduce a **dataCount** variable to indicate its length. Again, we will set the observed value of this variable when we are ready to call the inference, but we initialise it with a dummy value of 0.

The call to GetCompiledInferenceAlgorithm compiles the model and returns a **IGeneratedAlgorithm** object which can be then used to perform inference any number of times. The values for **data** and **dataCount** are changed in the body of the for loop, by calling **SetObservedValue**. The inference is then performed by calling the **Execute** method.

```csharp
// The model  
Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100);  
Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);  
Variable<int> dataCount = Variable.Observed(0);  
Range item = new  Range(dataCount);  
VariableArray<double> data = Variable.Observed<double>(null, item).Named("data");  
data[item] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(item);  

// The data  
double[][] dataSets = new  double[][]  
{
  new  double[] { 11, 5, 8, 9 },
  new  double[] { -1, -3, 2, 3, -5 }  
};  

// Set the inference algorithm  
InferenceEngine engine = new  InferenceEngine(new  VariationalMessagePassing());  

// Get the compiled inference algorithm  
var ca = engine.GetCompiledInferenceAlgorithm(mean, precision);  
// Run the inference on each data set  
for (int j = 0; j < dataSets.Length; j++)  
{
  // Set the data and the size of the range
  ca.SetObservedValue(dataCount.NameInGeneratedCode, dataSets[j].Length);  
  ca.SetObservedValue(data.NameInGeneratedCode, dataSets[j]);

  // Execute the inference, running 10 iterations
  ca.Execute(10);

  // Retrieve the posterior distributions
  Gaussian marginalMean = ca.Marginal<Gaussian>(mean.NameInGeneratedCode);
  Gamma marginalPrecision = ca.Marginal<Gamma>(precision.NameInGeneratedCode);
  Console.WriteLine("mean=" + marginalMean); 
  Console.WriteLine("prec=" + marginalPrecision);  
}
```

The output from the inference is:
```
mean=Gaussian(8.165, 1.026)  
prec=Gamma(3, 0.08038)  
mean=Gaussian(-0.7877, 1.532)  
prec=Gamma(3.5, 0.03672)
```
