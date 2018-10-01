---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Sharing variables between models](Sharing variables between models.md)

## Using shared variables to support hybrid algorithms

One use of shared variables is to use different algorithms in different sub-models. You might need to do this if some factors in your model are only supported by one algorithm, and other factors are only supported by a different algorithms. The following  example (illustrative but somewhat artificial) shows one model defining a mean variable, another defining a precision variable, and a third consisting of a Gaussian likelihood. The defining models then use Expectation Propagation, and the likelihood model uses Variational Message Passing. In this example we are interested in the hybrid model aspects of shared variables rather than the scalability aspects, so each model has a single chunk.

```csharp
double[] dataSet = new  double[] { 5, 5.1, 5.2, 4.9, -5.1, -5.2, -5.3, -4.9 };  
Range n = new  Range(data.Length);  
Model meanModel = new  Model(1);  
Model precModel = new  Model(1);  
Model dataModel = new  Model(1);  
var sharedMean = SharedVariable<double>.Random(Gaussian.Uniform());  
var sharedPrec = SharedVariable<double>.Random(Gamma.Uniform());  
var mean = Variable.GaussianFromMeanAndPrecision(0, 1);  
var prec = Variable.GammaFromShapeAndRate(10, 10);  
sharedMean.SetDefinitionTo(meanModel, mean);  
sharedPrec.SetDefinitionTo(precModel, prec);  
var x = Variable.Array<double>(n);  
x[n] = Variable.GaussianFromMeanAndPrecision(  
    sharedMean.GetCopyFor(dataModel), sharedPrec.GetCopyFor(dataModel)).ForEach(n);  
x.ObservedValue = data;  
InferenceEngine engine1 = new  InferenceEngine() { NumberOfIterations = 10 };  
InferenceEngine engine2 = new  InferenceEngine(new  VariationalMessagePassing())  
    { NumberOfIterations = 10 };  
for (int pass = 0; pass < 10; pass++)  
{  
    meanModel.InferShared(engine1, 0);  
    precModel.InferShared(engine1, 0);  
    dataModel.InferShared(engine2, 0);  
}  
meanPosterior = sharedMean.Marginal<Gaussian>();  
precPosterior = sharedPrec.Marginal<Gamma>();
```

The same approach allows running different parts of the model in parallel. For example, suppose we want to run meanModel and precModel on parallel threads. We ceate a separate inference engine for each thread and call Parallel.Invoke:

```csharp
InferenceEngine engine1b = new InferenceEngine()  
    { NumberOfIterations = 10 };  
for (int pass = 0; pass < 10; pass++)  
{  
    System.Threading.Tasks.Parallel.Invoke(  
        () => meanModel.InferShared(engine1, 0),  
        () => precModel.InferShared(engine1b, 0));  
    dataModel.InferShared(engine2, 0);  
}
```