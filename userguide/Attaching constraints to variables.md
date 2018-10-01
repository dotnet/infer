---
layout: default 
--- 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Attaching constraints to variables

Infer.NET provides a variety of constraints that can be applied to variables. For example:

```csharp
Variable.ConstrainPositive(x);
```

constrains the random variable **x** to be positive. Built-in constraints include **Positive**, **True**, **False**, **Equal** and **EqualRandom** (see [the full list of built-in constraints](Constraints.md)). Constraints can be combined with factors to produce more complex constraints. Here are some examples:

```csharp
Variable.ConstrainTrue(a + b < x);  
Variable.ConstrainEqual(y - z, 5.0);
```

#### **ConstrainEqualRandom**

One of the most useful constraints is **ConstrainEqualRandom**. This constraint is an optimized version of **ConstrainEqual** that avoids the creation of an intermediate random variable. The lines

```csharp
Variable.ConstrainEqualRandom(x, new Gaussian(0,1));  
Variable.ConstrainEqualRandom(x, distribution);
```

are mathematically equivalent to

```csharp
Variable.ConstrainEqual(x, Variable.GaussianFromMeanAndVariance(0,1));  
Variable.ConstrainEqual(x, Variable.Random(distribution));
```

but the ConstrainEqualRandom line leads to more efficient code since no intermediate random variable is created. Furthermore, ConstrainEqualRandom can be used in cases where there is no corresponding method on Variable to create a random variable with the desired distribution. Therefore you should try to use **ConstrainEqualRandom** whenever you find yourself writing ConstrainEqual lines like the above. 

Similarly, if you find yourself writing

```csharp
Variable.ConstrainTrue(x == y);
```

then try to replace this with

```csharp
Variable.ConstrainEqual(x,y);
```

The second form avoids creating an intermediate random variable, making the code run faster and Infer.NET happier. 

#### **Under the hood**

Under the hood, a call to **Variable.ConstrainEqual** is converted into a call to **Variable.Constrain**, as follows:

```csharp
Variable.Constrain(Constrain.Equal, x, y);
```

Similar to factor functions, the most general way of adding a constraint is:

```csharp
Variable.Constrain(MethodName, arg1, arg2, ...);
```

which adds the constraint specified by MethodName to the arguments arg1, arg2 (which may be random variables, constants or parameters). Using this mechanism, you can [define your own constraints](How to add a new constraint.md).