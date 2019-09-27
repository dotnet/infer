---
layout: default 
--- 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Branching on variables to create mixture models

Infer.NET allows you to create arbitrary mixture models, such as [mixtures of Gaussians](Mixture of Gaussians tutorial.md), mixtures of factor analysers, and so on. To specify a mixture, you write a program that branches on a random variable, as if you were writing a sampler for that mixture. For example, the following code specifies that x's prior distribution is a mixture of two Gaussians:

```csharp
Variable<int> c = Variable.Discrete(0.5, 0.5);  
Variable<double> x = Variable.New<double>();  
using (Variable.Case(c,0))  
{  
  x.SetTo(Variable.GaussianFromMeanAndVariance(1,1));  
}  
using (Variable.Case(c,1))  
{  
  x.SetTo(Variable.GaussianFromMeanAndVariance(2,1));  
}
```

The basic idea is to define a random selector variable and then branch on the value of that variable. If the selector is 0, then x has distribution  Gaussian(1, 1); if the selector is 1, then x has distribution Gaussian(2,1). This example makes use of the constructs `Variable.New` and `Variable.Case`. The static method `Variable.New` is similar to `Variable.Array`, but for scalars. It creates a new random variable whose definition will be provided later using **SetTo**. (You may recall **SetTo** from the page on [working with arrays and ranges](Arrays and ranges.md).)  `Variable.Case` is more sophisticated, because it changes the state of the modelling API. All random variables and constraints defined within the lifetime of the `Variable.Case` object are tagged to only exist conditionally on the two arguments being equal.

You can put any modelling code you like inside a Case block, including other Case blocks, allowing you to define arbitrarily complex mixtures. 

Besides **Case**, Infer.NET also supports **If** and **Switch** blocks for creating mixtures. These are summarised in the table below

| **Purpose** | **Number of components** | **Examples of use** |
|--------------------------------------------------------------|
| **If blocks** | For parts of the model that are switched on or off according to a boolean random variable. **If blocks** can be thought of as providing a mixture of two models with and without the contents of the block. | 2 | \- A mixture of a Gaussian distribution and a broad noise distribution. \- For calculating the model evidence (marginal log likelihood). |
| **Case blocks** | For providing multi-way mixtures. **Case blocks** are switched on when an integer random variable takes a particular value. The number of case blocks in a mixture is fixed after compilation. | Many (fixed) | \- A mixture of two Gaussian distributions and a broad noise distribution. |
| **Switch blocks** | For providing variable-sized mixtures. **Switch blocks** are equivalent to a set of similar case blocks, one for each possible value of the integer random variable. The number of cases in a switch block can be varied after compilation using an observed value. | Many (variable) | \- A mixture of Gaussians where the number of components varies at runtime. |

#### Creating if blocks

As with the rest of the modelling API, if, switch and case blocks are created using static methods on the Variable class. You can create an if block which is switched on when a boolean random variable **b** is true using `Variable.If(b)`. The complementary if block, which is on when **b** is _false_, can be created using `Variable.IfNot(b)`.

Unlike the modelling API elements you have seen so far, if blocks contain other modelling elements. After an if block is created, any further modelling API commands create variables/factors inside the block, until the block is closed. **All blocks must be closed before inference is performed.** To make sure you close any blocks that you have created, it is recommended that you create blocks using the C# _using_ statement or equivalent in the language you are using. For example, the following introduces a constraint that a variable **x** must be positive, only if another variable **b** is true.

```csharp
Variable<bool> b = Variable.Bernoulli(0.5);  
using (Variable.If(b))  
{
  Variable.ConstrainPositive(x);  
} /// the block is now closed
```

The _using_ statement hides the fact that an IfBlock object has been created and then closed. In fact, the actual code that is being run is the code shown below, written without a using statement. The _using_ syntax is preferable since it ensures all blocks are closed. However, some .NET languages do not provide such a construct, and, whatever the language, you can always explicitly close blocks as follows:

```csharp
Variable<bool> b = Variable.Bernoulli(0.5);  
IfBlock ifb = Variable.If(b);  
Variable.ConstrainPositive(x);  
ifb.CloseBlock();
```

A danger of this method is that if an exception occurs before you close the block, then the block will remain open. You can use the 'try/finally' construct to ensure that the blocks are closed in this case. As a last resort, if you have caught an exception thrown by another function that opened a block, and don't have access to the block variable, you can call the function `Variable.CloseAllBlocks()` which will forceably close all open blocks and allow you to recover from the exception.

We can extend the above example to introduce a positivity constraint on a variable **y** if **b** is false. The result will be that only one of **x** or **y** will be constrained to be positive, depending on the value of **b**.

```csharp
using (Variable.IfNot(b))  
{
  Variable.ConstrainPositive(y);  
}
```

As well as adding constraints, variables can be created and factors added inside an if block. It is also possible to create a variable outside of an if block and assign it a prior inside.

#### Creating switch blocks

A switch block allows you to create mixtures of variable size. This is illustrated below:

```csharp
int mixtureSize = 2;  
Range k = new Range(mixtureSize);  
Variable<int> c = Variable.Discrete(k, new  double[] { 0.5, 0.5 });  
VariableArray<double> means = Variable.Observed(new double[] { 1, 2 }, k);  
Variable<double> x = Variable.New<double>();  
using (Variable.Switch(c))  
{  
  x.SetTo(Variable.GaussianFromMeanAndVariance(means[c], 1));  
}
```

This code defines the same model as the Case example above, but the size of the mixture is defined dynamically. The new element here is the `Variable.Switch` block. Inside this block, you can index arrays by the integer variable given to Switch. In this example, we are indexing the **means** array by the random variable **c**. The result is a variable whose value depends on **c**. Using such a variable in an expression creates a new variable whose value also depends on **c**. Finally, when you call [x.SetTo](The importance of using SetTo.md) with such a variable, you are providing multiple definitions of **x**, one for each value of **c**. 

The integer variable passed to **Variable.Switch** must be annotated with the range that the variable can be used to index, otherwise you will get an error that **c** cannot be used as an index of **means**. This annotation is passed to **Variable.Discrete** when constructing the integer variable, or **Variable.Dirichlet** when constructing the integer variable's parent. In this example, the variable **c** can take values from the range **k**, which is specified in the argument to **Variable.Discrete**. Occasionally you may need to explicitly provide this annotation by [adding a ValueRange attribute](Adding attributes to your model.md) to your switch variable; the model compiler will issue an error message about a 'missing ValueRange' if this is the case.

#### Branching on array elements

The examples above all branch on a single random variable. You can also branch on array elements, to create mixture models over arrays. To do so, you place the Case/If/Switch inside a [ForEach block](ForEach blocks.md), as follows:

```csharp
using (Variable.ForEach(range)) {
  using (Variable.Case(c[range], 0)) {  
    ...
  }  
  using (Variable.Case(c[range], 1)) {  
    ...
  }  
}
```

For example, the model at the beginning of this page can be defined for an array x as follows:

```csharp
Range i = new  Range(4);VariableArray<double> x = Variable.Array<double>(i);  
VariableArray<int> c = Variable.Array<int>(i);  
using (Variable.ForEach(i)) {
  c[i] = Variable.Discrete(0.5, 0.5);  
  using (Variable.Case(c[i], 0)) {
    x[i] = Variable.GaussianFromMeanAndVariance(1, 1);  
  }  
  using (Variable.Case(c[i], 1)) {
    x[i] = Variable.GaussianFromMeanAndVariance(2, 1);  
  }  
}
```

Note that VariableArrays do not require explicit SetTo, because the C# indexing notation x\[i\] is overridden to invoke SetTo.

### Local variables

If you create a variable inside a branch of an Case/If/Switch block, then the variable becomes a local variable of that branch. You cannot use a local variable in factors outside of the branch it was created in. However, you can infer the distribution of a local variable. What you get is the distribution of the local variable conditional on the branch being taken. For example:

```csharp
Variable<int> c = Variable.Discrete(new double[] { 0.5, 0.5 });  
Variable<double> x = Variable.New<double>();  
Variable<double> y;  
using (Variable.Case(c,0))  
{  
  y = Variable.GaussianFromMeanAndVariance(1,1));  
  x.SetTo(Variable.GaussianFromMeanAndVariance(y,1));  
}  
using (Variable.Case(c,1))  
{  
  x.SetTo(Variable.GaussianFromMeanAndVariance(2,1));  
}
```

Here _`y`_ is a local variable because it is created inside the Case for c=0. It cannot be used outside the Case c=0. Inferring _`y`_ will give its distribution conditional on c=0. Note the difference with how _`x`_ is created (_`x`_ is not a local variable and can be used in other cases).
