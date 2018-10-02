---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Sharing variables between models](Sharing variables between models.md)

## Jagged shared variable arrays

To define a jagged shared variable array, it is useful first of all to create aliases for the jagged distribution array types that define your jagged random variable array. For example:

```csharp
using GaussianArray=DistributionStructArray<Gaussian, double>;  
using GaussianArrayArray=DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>
```

or

```csharp
using DirichletArray=DistributionRefArray<Dirichlet, Vector>;  
using DirichletArrayArray=DistributionRefArray<DistributionRefArray<Dirichlet, Vector>, Vector[]>;
```

Below is an example where we want to learn the mean of jagged array of Gaussian-distributed data using shared variables. The line defining the jagged shared variable array is highlighted:

```csharp
double[][][] data = new double[][][] {  
  new double[][] { new double[] { 1,2 }, new double[] { 3,4 } },   
  new double[][] { new double[] { 5,6 }, new double[] { 7,8 } }  
};  
int numChunks = data.Length;  
Range outer = new Range(data[0].Length);  
Range inner = new Range(data[0][0].Length);  
GaussianArrayArray priorW =  
  new GaussianArrayArray(new GaussianArray(new Gaussian(0, 1), inner.SizeAsInt), outer.SizeAsInt);  
  
var w = SharedVariable<double>.Random(Variable.Array<double>(inner), outer, priorW).Named("w");  
Model model = new Model(numChunks);  
var x = Variable.Array<double>(Variable.Array<double>(inner), outer).Named("x");  
x[outer][inner] = Variable.GaussianFromMeanAndPrecision(w.GetCopyFor(model)[outer][inner], 1.0);  
InferenceEngine engine = new InferenceEngine();  
  
for (int pass = 0; pass < 5; pass++)  
{  
  for (int c = 0; c < numChunks; c++)  
  {  
    x.ObservedValue = data[c];  
    model.InferShared(engine, c);  
  }  
}  
var posteriorW = w.Marginal<Gaussian[][]>();
```

If you want to make a jagged array with more than 2 brackets, you will need to specify the full array type as the type argument to SharedVariable. (This also works with 2 brackets, but isn't needed.) For example, to make an array with 3 brackets:  

```csharp
var x = SharedVariable<double[][][]>.Random(Variable.Array(Variable.Array<double>(innerinner), inner), outer, prior);
```