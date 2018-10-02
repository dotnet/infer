---
layout: default 
--- 
[Infer.NET user guide](index.md) : [FSharp Wrapper](FSharp Wrapper.md) 

## Variable Arrays in F\#

Variable Arrays can be difficult to create and assign values to when dealing with large sets of complex data. This is especially true when 2-Dimensional and Jagged Arrays of data are involved. Therefore the module Variable provides various methods to make creation and assignment of such arrays both simpler and provide a more intuitive functional style of using such methods.

### Creating Arrays using a generator function

The `Variable.ArrayInit<'a>` function provides an easy way of creating and assigning the values of a VariableArray given a generator function to provide the values for assignment. The type signature for this method is : _`Range -> (Range -> Variable<'a>) -> VariableArray<'a>.`_ The following example shows how it can be used to create and initialise the mean and precision variableArrays used in the [Mixture of Gaussians tutorial](Mixture of Gaussians.md).

```fsharp
let means = Variable.ArrayInit  
              k (fun k -> Variable.VectorGaussianFromMeanAndPrecision(  
                            new Vector(2,0.0),  
                            PositiveDefiniteMatrix.IdentityScaledBy(2,0.01)))  
let precs = Variable.ArrayInit  
              k (fun k -> Variable.WishartFromShapeAndScale(  
                            100.0, PositiveDefiniteMatrix.IdentityScaledBy(2,0.01)))
```

### Assigning arrays from a generator function

The `Variable.AssignVariableArray` function provides an easy way of assigning the values of a VariableArray given a generator function to provide the values for assignment. The type signature for this method is : _`VariableArray<'a> -> Range -> (Range -> Variable<'a>) -> VariableArray<'a>`._ The following example shows how it can be used to assign values to the variableArrays used in the [Clinical trial tutorial](Clinical trial.md) using a generator function. Similarly, methods to assign values to 2-Dimensional and jagged `VariableArrays` are also provided in the form of Variable.AssignVariableArray2D with type signature : _`VariableArray2D<'a> -> Range -> (Range -> Variable<'a>) -> VariableArray2D<'a>`_ and

```fsharp
let controlGroup = Variable.Observed<bool>([|false; false; true; false; false|])  
let treatedGroup = Variable.Observed<bool>([|true; false; true; true; true |])  
let i = controlGroup.Range  
let j = treatedGroup.Range  
// Prior on being an effective treatment  
let isEffective = Variable.Bernoulli(0.5).Named("isEffective");  
let probIfTreated = ref (Variable.New<float>())  
let probIfControl = ref (Variable.New<float>())//rewrite If Block  
let f1 (vb1: Variable<bool>) =  
    probIfControl := Variable.Beta(1.0, 1.0).Named("probIfControl")  
    let controlGroup = Variable.AssignVariableArray controlGroup i (fun i ->Variable.Bernoulli(!probIfControl)) let treatedGroup = Variable.AssignVariableArray treatedGroup j (fun j ->Variable.Bernoulli(!probIfTreated))  
    ()
```
