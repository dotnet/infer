---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Controlling how inference is performed](Controlling how inference is performed.md)

## Using a precompiled inference algorithm

Internally Infer.NET compiles a model into a C# class for performing inference on that model. By default, the source of the generated class is placed at: **(Debug/Release)\\bin\\GeneratedSource\\\[ModelName\].cs**. and has a [standard structure](Structure of generated inference code.md). This page describes how this compiled inference algorithm can be directly included in a C# project. This might be useful for one of the following reasons:

*   **To remove the dependence on the compiler.** For standalone applications that make heavy use of a specific inference algorithm it might be desired to remove the dependence on the compiler.

*   **To use Infer.NET from Silverlight.** Security restrictions in Silverlight do not allow dynamic compilation, so you must precompile your model before including it in a Silverlight project.

*   **To run multi-threaded inference.** Including the compiled algorithm directly allows different instances of the compiled algorithm to be used in each thread separately without recompilation.

*   **Speed-up.** Inclusion of the compiled algorithm avoids compilation on first time usage.

*   **To manually edit and alter the created output code.** For advanced usage of Infer.Net it might be desired to alter the generated code. This mainly applies when specific features are not available in Infer.Net and you want to use the compiler to get a starting point for your own implementation of inference algorithms. This is only recommended if you know what you are doing!

#### Example

Here is a simple example of how to use pre-compiled code, based on the [Learning a Gaussian tutorial](Learning a Gaussian tutorial.md).

To ensure that the output code has a convenient interface, it is useful to follow a few guidelines:

*   Use 'observed' variables rather than 'constant' variables for any values which are fixed for a given call to the inference algorithm, but which may change between invocations (see [Creating Variables](Creating variables.md)).
*   Use the inline .Named("name") method for all variables you want to Infer or set as a given. Sensible choice of naming ensures a higher level of readability of the compiled output code.

```csharp
// Initial Data  
double[] dataSet = new double[100];  
for (int i = 0; i < dataSet.Length; i++)  
    dataSet[i] = Rand.Normal(0, 1);  

// Observed variables for data and data count  
Variable<int> dataCount = Variable.Observed(dataSet.Length).Named("dataCount");  
Range N = new Range(dataCount);  
VariableArray<double> data = Variable.Observed<double>(dataSet, N).Named("data");  

// Observations are assumed to be sampled from a Gaussian with unknown parameters  
Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");  
Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");  
data[N] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(N);  

// Create an inference engine for VMP  
InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());  

// Retrieve the posterior distributions  
Console.WriteLine("mean=" + engine.Infer(mean));Console.WriteLine("prec=" + engine.Infer(precision));
```

Run this example, take the generated output code (it will be named **Model_VMP.cs** unless you set the **ModelName** property on the engine) and add it to your project. This instance contains the compiled model class **Models.Model_VMP**.

There are two ways to call the compiled inference algorithm from your code. The easiest way to use this class is through its `IGeneratedAlgorithm` interface as described in [Controlling how inference is performed](Controlling how inference is performed.md). Alternatively you can use the strongly-typed model-specific properties and methods as documented below.

#### Calling the compiled model class directly

 The 4 steps are illustrated in the example code.

```csharp
// Run-time data  
double[] dataSet = new  double[10000];  
for (int i = 0; i < dataSet.Length; i++)  
    dataSet[i] = Rand.Normal(0, 1);  

// 1) Create an instance of the class  
Model_VMP model = new Model_VMP();  

// 2) Set the value of any observed variables e.g. data, priors  
model.data = dataSet;  
model.dataCount = dataSet.Length;  

// 3) Call the Execute() method, or Reset() followed by Update()  
model.Execute(20);  

// 4) Use the XXXMarginal() methods to retrieve posterior marginals  
//    for different variables. 
Gaussian inferredMean = model.MeanMarginal();  
Gamma inferredPrec = model.PrecisionMarginal();  

// Print out the results  
Console.WriteLine("mean=" + inferredMean);  
Console.WriteLine("prec=" + inferredPrec);
```
