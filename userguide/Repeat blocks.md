---
layout: default 
--- 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Repeat blocks

Repeat blocks provide an efficient way to repeat part of a model many times. A repeat block takes an argument **count** and contains the statements which are to be repeated:

```csharp
using (Variable.Repeat(count))  
{  
  // contained modelling statements  
}
```

For an integer count, a repeat block is equivalent to a ForEach block, in the sense that it repeats the contained statements **count** times. However, a repeat block is much more efficient. It achieves this efficiency by not providing any index variable---every repetition must be identical. Also, local variables of a repeat block cannot be used outside of the block (otherwise they would have to be treated as arrays of length **count**, which we want to avoid). As with [ForEach blocks](ForEach blocks.md), the using syntax is optional.

### Using repeat blocks to speed up inference

When parts of a model are repeated identically, repeat blocks can lead to substantial speed-ups. For example, suppose we wish to learn the probability of heads of a coin from 10,000,000 tosses. Let's say we are given a bool array **flips** containing these coin tosses. The standard approach in Infer.NET would be:

```csharp
var probHeads = Variable.Beta(1, 1); // The bias of the coin  
var N = new Range(flips.Length); // Range over the flips  
var flipArray = Variable.Array<bool>(N); // Array of flips  
// Model for the flips  
flipArray[N] = Variable.Bernoulli(probHeads).ForEach(N);  
flipArray.ObservedValue = flips;
```

Intuitively this seems very inefficient - we have to allocate and process large arrays. The code will take longer and longer as we make more and more coin tosses.

In this particular model, all tosses which give heads are equivalent, as are all tosses that give tails. So we can just record the counts of heads and tails instead of all individual coin tosses. Repeat blocks provide a consistent way to do this for any type of model.

Here is how we can write the same model using repeat blocks, assuming we now have counts of heads and tails in the variables **countTrue** and **countFalse**:

```csharp
using(Variable.Repeat(countTrue)) {  
  Variable.ConstrainTrue(Variable.Bernoulli(probHeads));  
}  
using(Variable.Repeat(countFalse)) {  
  Variable.ConstrainFalse(Variable.Bernoulli(probHeads));  
}
```

Inference of **probHeads** using this code takes the same amount of time no matter how many coin tosses were made. For example, for 10,000,000 tosses inference takes ~3 seconds using the original code but less than 1 millisecond when repeat blocks are used.

### Other uses of repeat blocks

_This section describes an experimental use of repeat blocks_

The count argument of a repeat block can be any real number, including fractions and negative numbers. This has no correspondence to traditional programming, but it does make sense in terms of factor graphs. Essentially the repeat block is raising the contained factors to a power. If the power is a positive integer **N**, it is as if the factors were simply repeated **N** times. Negative powers can be used to define [partition functions](http://en.wikipedia.org/wiki/Partition_function_%28mathematics%29) as part of an [undirected model](http://en.wikipedia.org/wiki/Markov_random_field). This use of repeat blocks will be explored further in future versions of this guide.