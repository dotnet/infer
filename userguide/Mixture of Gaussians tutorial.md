---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## Tutorial 6: Mixture of Gaussians

This tutorial shows how to make a mixture of multivariate Gaussians and fit it to some data. It will introduce the _switch block_ which allows variable-sized mixtures to be constructed.

You can run the code in this tutorial either using the [Examples Browser](The examples browser.md) or by opening the Tutorials solution in Visual Studio and executing **MixtureOfGaussians.cs**.

### Parameters priors

This is going to be a Bayesian mixture of Gaussians, so we need to define priors on each of the parameters of our model. To do this, start by defining a range for the number of mixture components:

```csharp
Range k = new Range(2);
```

Using this range, create an array for the means of our mixture components and give them priors:

```csharp
VariableArray<Vector> means = Variable.Array<Vector>(k);  
means[k] = Variable.VectorGaussianFromMeanAndPrecision(
    Vector.FromArray(0.0,0.0), PositiveDefiniteMatrix.IdentityScaledBy(2,0.01)).ForEach(k);
```

The first line creates an array of Vector variables over range **k**. The second line specifies that each mean will be a separate draw from a multivariate Gaussian prior. Notice the use of **ForEach(k)** to duplicate the prior across the array.

The precisions of the mixture components are defined in a similar way. The precisions are positive-definite matrices, drawn from a Wishart prior:

```csharp
VariableArray<PositiveDefiniteMatrix> precs = Variable.Array<PositiveDefiniteMatrix>(k);  
precs[k] = Variable.WishartFromShapeAndScale(  
  100.0, PositiveDefiniteMatrix.IdentityScaledBy(2,0.01)).ForEach(k);
```

The final parameter is the vector of mixture weights, which has a Dirichlet prior.

```csharp
Variable<Vector> weights = Variable.Dirichlet(k, new double[] { 1, 1 });
```

Notice that we pass the range **k** as the first parameter. This tells Infer.NET that the size of the vector **weights** is given by the range **k**.

### Mixing it together 

Now that we have defined variables for each of the parameters of the model, we are ready to mix them to produce data. Create a VariableArray for the data:

```csharp
Range n = new Range(300);  
VariableArray<Vector> data = Variable.Array<Vector>(n);
```

For each data point, create a variable to indicate which mixture component the data point came from (the selector variables):

```csharp
VariableArray<int> z = Variable.Array<int>(n);
```

We use a **ForEach** block over **n** to repeatedly select a mixture component and draw from it. The selector **z** is drawn from the distribution given by the weights. Then we construct a Gaussian with the corresponding mean and precision, and draw from it. To fetch the chosen element out of the **means** and **precs** arrays, we use a **Switch** block.

_**See also:** [Branching on variables to create mixture models](Branching on variables to create mixture models.md)_

```csharp
using (Variable.ForEach(n)) {  
  z[n] = Variable.Discrete(weights);  
  using (Variable.Switch(z[n])) {  
    data[n] = Variable.VectorGaussianFromMeanAndPrecision(  
      means[z[n]], precs[z[n]]);  
  }  
}
```

And there you have it - a fully Bayesian multivariate Gaussian mixture model.

### Breaking symmetry 

We can now attach some data and run inference. For example data, we use a function which generates data from a known mixture of Gaussians model (see the tutorial file for the code of this function). We then use an inference engine to infer posterior distributions over the weights, means and precisions.

```csharp
// Attach some generated data  
data.ObservedValue = GenerateData(n.SizeAsInt); // The inference  
InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());  
Console.WriteLine("Dist over pi=" + ie.Infer(weights));  
Console.WriteLine("Dist over means=\n" + ie.Infer(means));  
Console.WriteLine("Dist over precs=\n" + ie.Infer(precs));
```

If you run this code, the result is:

```
Dist over pi=Dirichlet(151 151)  
Dist over means=  
[0] VectorGaussian(4.052 3.739, 0.02212  0.007127)  
                                0.007127 0.00796  
[1] VectorGaussian(4.052 3.739, 0.02212  0.007127)  
                                0.007127 0.00796  
Dist over precs=  
[0] Wishart(175, 0.00242   -0.002167)  
                 -0.002167 0.006726  
[1] Wishart(175, 0.00242   -0.002167)  
                 -0.002167 0.006726
```

This is not helpful - the two mixture components are identical! The reason for this is that the model is _symmetrical_ \- the mixture components are all defined the same way and the inference updates never distinguish them. 

We can break symmetry in this case by randomly initialising the messages used in message passing. Infer.NET does not do this automatically, because it normally guarantees to give the same result each time you perform inference - introducing random initialisation would break this contract. If you want to break symmetry, you must do it explicitly. In this example, we can achieve this using **InitialiseTo()** to randomise the initial message for **z**, as follows:

_**See also:** [Customising the algorithm initialisation](customising the algorithm initialisation.md)_

```csharp
Discrete[] zinit = new Discrete[n.SizeAsInt];  
for(int i = 0; i < zinit.Length; i++)  
  zinit[i]=Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt);  
(Distribution<int>.Array(zinit));
```

If 

you now run the inference again, you should get results similar to the following (modulo the randomness that we have introduced):

```
Dist over pi=Dirichlet(180 122)  
Dist over means=  
[0] VectorGaussian(2.033 2.897, 0.003775   -9.646e-05)  
                                -9.646e-05 0.004307  
[1] VectorGaussian(7.041 4.986, 0.006446   -6.828e-05)  
                                -6.828e-05 0.00586  
Dist over precs=  
[0] Wishart(189.2, 0.007827 0.0001753)  
                   0.0001753 0.00686  
[1] Wishart(160.2, 0.008003  9.324e-05)  
                   9.324e-05 0.008804
```

In this result, the two mixture components are different, which is a more useful outcome. However, note that the order of the results is arbitrary here - the components may come out flipped. Also, the exact marginal distributions of the mixture components are identical. The only reason that we are not getting identical marginals is because the inference is approximate.

​
