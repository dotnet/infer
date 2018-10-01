---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Controlling how inference is performed](Controlling how inference is performed.md)

## Customising the algorithm initialisation

By default, Infer.NET initialises all messages to a uniform distribution. It it sometimes necessary to override this behaviour, for example to break symmetries in the model or to initialise based on a custom preprocessing of the data. Infer.NET supports overriding the initialisation of the algorithm by specifying an initial marginal for a variable or variables.

The syntax is:

```csharp
Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1);  
x.InitialiseTo(Gaussian.FromMeanAndVariance(0, 10));
```

when the algorithm is initialised, the initial marginal for **x** will be set to a Gaussian with zero mean and a variance of 10. 

If the InferenceEngine's Algorithm is set to Expectation Propagation, then `InitialiseTo` determines the first message that is sent to one child factor of **x** (according to the factor graph you get from `InferenceEngine.ShowFactorGraph`). If **x** has multiple child factors A and B, then only one of them (say A) will receive this initial distribution. Factor B will receive a distribution influenced by the message sent from A to **x**. The choice of which factor receives the initial distribution is arbitrary and may change as the Infer.NET compiler is updated. To guarantee that a particular factor receives an initial message, make a copy of the variable using `Variable.Copy`, initialize the copy, and use the copy when creating the factor.

Often you will need to initialise a variable array. If you want each element to have the same initialiser, then this is easily done by syntax similar to the above:

```csharp
Range r = new Range(10);  
VariableArray<double> y = Variable.Array<double>(r);  
y[r].InitialiseTo(Gaussian.FromMeanAndVariance(0, 1));
```

More commonly, you want each element to have its own initialiser. In this case, you must first create a .NET array of the initialisers. From here there are two options. The first option is to create a **VariableArray** of distribution objects and observe it to be the .NET array. Then you apply **InitialiseTo** to each element of the array, as follows:

```csharp
Gaussian[] inity = new Gaussian[r.SizeAsInt];  
for (int i = 0; i < inity.Length; i++)  
    inity[i] = Gaussian.FromMeanAndVariance(Rand.Normal(), 1);  
VariableArray<Gaussian> initVar = Variable.Observed(inity, r);  
y[r].InitialiseTo(initVar[r]);
```

The second option is create a single distribution object holding the initialisers. Use the static **Array** method in the Distribution class to create the correct **IDistribution** type needed by **InitialiseTo**. For example,

```csharp
y.InitialiseTo(Distribution<double>.Array(inity));
```

Here the initial marginals for the elements of y are set to be Gaussian with mean sampled from a standard Gaussian and with unit variance. This kind of initialisation is useful for breaking symmetry.

Important note: Initialising a variable will often change the inference schedule used. This is because the inference scheduler will use manually initialised messages in favour of automatically initialised ones, and so the order of operations may change. The scheduler will also warn if the initialisation is unnecessary, for example, if the initial values supplied are never used.
