---
layout: default 
--- 
[Infer.NET user guide](index.md) : [FSharp Wrapper](FSharp Wrapper.md)

 

## Imperative Statement Blocks in F\#

The module Variable contains various functions to replace the imperative statement blocks in Infer.NET, these include `ForeachBlock`, `SwitchBlock` and `IfBlock` methods.

### ForEach blocks

The `Variable.ForeachBlock` method which returns a `ForEachBlock` used for looping over a Range of values in a similar manner to a `foreach` loop in C#. This is  
 a function with the following type signature: _`Variable.ForEachBlock: RRange->(Range->unit)->unit`_

```fsharp
//using the Foreach Block  
let pixel = new Range(10)  
let bools = Variable.Array<bool>(pixel);  

Variable.ForeachBlock pixel ( fun pixel ->  
                bools.[pixel] <- Variable.Bernoulli(0.7) ||| Variable.Bernoulli(0.4)())
```

It should be noted that the `Variable.ArrayInit` method  or the `Variable.AssignVariableArray` method can also be used to assign the values of a `VariableArray` as shown in the section [Variable Arrays](Variable Arrays in FSharp.md)

### Switch blocks

The `Variable.SwitchBlock` method which returns a `SwitchBlock` used for branching on the values of a variable. This is a function with the following type signature: _`Variable.SwitchBlock: Variable<int>->(Variable<int>->unit)->unit`_

```fsharp
//using the Switch Block  

let mixtureSize = 2  
let k2 = new Range(mixtureSize)  
let c = Variable.Discrete(k2, [|0.5;0.5|])  
let means2 = Variable.Observed( [|1.0;2.0|], k2)
let x = Variable.New<double>();  
Variable.SwitchBlock c (fun _ -> 
    let _ = x.SetTo(Variable.GaussianFromMeanAndVariance(means2.[c], 1.0))  
    ()  
)
```

There is also another Switch construct which returns a `Variable<'a>` and can be used to initialise or set the values of `VariableArrays`. The function `Variable.SwitchExpr` has the following type signature: _`VVariable<int> -> (Variable<int>->Variable<'a>)->Variable<int>`_

```fsharp
//using the SwitchExpr Block  

let means = Variable.ArrayInit k (fun k ->  
                Variable.VectorGaussianFromMeanAndPrecision(  
                    new Vector(2,0.0), PositiveDefiniteMatrix.Identity(2)))  
let precs = Variable.ArrayInit k (fun k ->  
                Variable.WishartFromShapeAndScale(1.0,  
                    PositiveDefiniteMatrix.Identity(2)))  
let weights = Variable.Dirichlet(k,[|1.0; 1.0|])
let n = new Range(100)

let z = Variable.ArrayInit n (fun i -> Variable.Discrete(weights))  
let data = Variable.ArrayInit n (fun i ->  
               Variable.SwitchExpr (z.[i]) (fun zi ->  
                   Variable.VectorGaussianFromMeanAndPrecision(means.[zi], precs.[zi])))
```

### If Blocks

The `Variable.IfBlock` method returns an `IfBlock` for creating if then else statements which are dependant on the value of a `Variable<bool>`. This is a function with type signature: _`Variable.IfBlock: Variable<bool>->(Variable<bool>->unit)->Variable<bool>->unit)->unit`_ where, the first argument is a `boolean` variable, the second argument is a function to be applied if the Boolean Variable has value true and the third argument is a function to be applied if the Boolean Variable has value false. Here the If block is used in the [Clinical trial tutorial](Clinical trial.md) shown below

```fsharp
// Data from a clinical trial  
let controlGroup = Variable.Observed<bool>([|false; false; true; false; false|])  
let treatedGroup = Variable.Observed<bool>([|true; false; true; true; true |])  
let i = controlGroup.Range  
let j = treatedGroup.Range  
// Prior on being an effective treatment  
let isEffective = Variable.Bernoulli(0.5).Named("isEffective");  
let probIfTreated = ref (Variable.New<float>())  
let probIfControl = ref (Variable.New<float>())

// If Block function  
let f1 (vb1: Variable<bool>) =  
    probIfControl := Variable.Beta(1.0, 1.0).Named("probIfControl")  
    let controlGroup = Variable.AssignVariableArray controlGroup i (fun i ->
        Variable.Bernoulli(!probIfControl)) 
    let treatedGroup = Variable.AssignVariableArray treatedGroup j (fun j ->
        Variable.Bernoulli(!probIfTreated))  
        ()  

// IfNot Block function  
let f2 (vb2: Variable<bool>) =  
    let probAll = Variable.Beta(1.0, 1.0).Named("probAll")  
    let controlGroup = Variable.AssignVariableArray controlGroup i (fun i ->Variable.Bernoulli(probAll))  
    let treatedGroup = Variable.AssignVariableArray treatedGroup j (fun j ->Variable.Bernoulli(probAll))  
    ()  

// Call IfBlock  
let _ = Variable.IfBlock isEffective f1 f2
```
