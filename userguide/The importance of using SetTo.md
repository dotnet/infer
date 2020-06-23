---
layout: default 
--- 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## The importance of using SetTo

The New/SetTo constructs are crucial when using branches. Suppose in the scalar mixture model on the [previous page](Branching on variables to create mixture models.md), we used ordinary C# assignments:

```csharp
Variable<int> c = Variable.Discrete(new double[] { 0.5, 0.5 });  
Variable<double> x = Variable.New<double>();  
using (Variable.Case(c,0))  
{  
  // incorrect!!
	x = Variable.GaussianFromMeanAndVariance(1,1);  
}  
using (Variable.Case(c,1))  
{  
  // incorrect!! - clobbers the assignment to x above
	x = Variable.GaussianFromMeanAndVariance(2,1);  
}
```

This code fails because of the imperative execution of C#. The second assignment to x destroys the random variable created in the first case. As the C# equality operator cannot be overridden, the Infer.NET API cannot intercept this assignment to provide different behaivour. Instead, Infer.NET provides New/SetTo to allow both definitions to exist in the model, with switching done at runtime. The correct code is:

```csharp
Variable<int> c = Variable.Discrete(new double[] { 0.5, 0.5 });  
Variable<double> x = Variable.New<double>();  
using (Variable.Case(c,0))  
{  
  // correct  
  x.SetTo(Variable.GaussianFromMeanAndVariance(1,1));  
}  
using (Variable.Case(c,1))  
{  
  // correct - the previous definition of x is retained  
  x.SetTo(Variable.GaussianFromMeanAndVariance(2,1));  
}
```

For more help with building mixture models, see the [Mixture of Gaussians tutorial](Mixture of Gaussians tutorial.md).

Another thing to note about assignment and `SetTo` is that you should never assign one variable directly to another, like so:

```csharp
// incorrect!! - two references to the same variable object  
observedLabel = label;  
// incorrect!! - two references to the same variable object  
observedLabel.SetTo(label);
```

Instead make a new variable using `Variable.Copy`:

```csharp
observedLabel.SetTo(Variable.Copy(label));
```
