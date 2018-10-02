---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## How to handle missing data

It is often the case that there will be measurements missing from a data set. These could be, for example: unanswered questions in a questionnaire, diagnostic tests not performed on all participants in a clinical trial, or sensors which occasionally go offline. Infer.NET allows inference to be performed in a model even when there is missing data. In this example we will assume that the data is _missing at random_, however, Infer.NET can also be used where this is not the case.

#### Learning a Gaussian with missing data

To show how to handle missing data, we will extend the example from the [Learning a Gaussian tutorial](Learning a Gaussian tutorial.md). Without missing data, the code looks like this:

```csharp
// Sample data from standard Gaussian  
double[] data = new double[100];  
for (int i = 0; i < data.Length; i++) data[i] = Rand.Normal(0, 1);  

// Create mean and precision random variables  
Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");  
Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");  

Range dataRange = new Range(data.Length);  
VariableArray<double> x = Variable.Array<double>(dataRange);  
x[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(dataRange);

x.ObservedValue = data;  

// Create an inference engine for VMP  
InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());  
Console.WriteLine("mean=" + engine.Infer(mean));  
Console.WriteLine("prec=" + engine.Infer(precision));
```

Let's assume that some of the data is missing, as indicated by an array of 100 boolean values called **isMissing**, each element of which is true if the data point is missing and false if it is valid. For this example, we will assume that every other element is missing:

```csharp
bool[] isMissing = new bool[100];  
for (int i=0; i<isMissing.Length; i++) isMissing[i] = (i % 2)==0;
```

Now we turn this array into an observed Infer.NET VariableArray, indicating that the array is across the same range as the data:

```csharp
VariableArray<bool> isMissingVar = Variable.Observed(isMissing, dataRange);
```

We want to constrain the data to be drawn from the Gaussian only if the data is not missing, so we use an **IfNot** block. We replace the highlighted line above with the following:

```csharp
using (Variable.ForEach(dataRange))  
{
  using (Variable.IfNot(isMissingVar[dataRange]))  
  {  
    x[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision);  
  }  
}
```

Notice that we have changed the **ForEach** across **dataRange** from being inline, to be in block form - this is necessary since **isMissingVar** needs to be indexed by **dataRange**. If we run the new program, the expectation of the mean and precision are similar to before but the distributions have higher variance (are less confident), since they are estimated from half the amount of data.

### Handling special 'missing' values

It is common for data sets to use special values, like -1 or 99 to indicate that the data is missing. The method above can be used in this case, but it is often possible to use a more compact form. Supposing we change the definition of **data** above to be:

```csharp
double[] data = new double[] { -1, 5.0, -1, 7.0, -1 };
```

where negative values mean 'missing' and positive values are non-missing. We can now skip the creation of the **isMissing** array and instead write

```csharp
using (Variable.ForEach(dataRange))  
{  
  using (Variable.If(x[dataRange]>0))  
  {  
    x[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision);  
  }  
}
```

What we have done here is created a bool VariableArray on the fly using the expression **x\[dataRange\]>0**. This is possible because Infer.NET contains a factor which supports greater-than comparison. Had we instead written **x\[dataRange\]!=-1**, the compilation would have failed with the error 'No operator factor registered for 'Equal' with argument type System.Double'. If you get such an error message the longer method above must be used instead.
