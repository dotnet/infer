---
layout: default 
--- 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Markov chains and grids

A wide variety of Markov chain and grid structured models can be created using VariableArrays. The basic idea is that when looping over a range with a ForEach block, you can access the loop counter i and use expressions of the form (i-k) or (i+k) where k is a constant integer. This is called "offset indexing" and allows you to link the array element at index i with index i-k of the same array or another array.

For example, suppose we want to simulate a Gaussian random walk where x\[t\] holds the position at time t. This can be written using a VariableArray as follows:

```csharp
Variable<int> numTimes = Variable.Observed(10);  
Range time = new Range(numTimes);  
VariableArray<double> x = Variable.Array<double>(time);  
using (var block = Variable.ForEach(time)) {  
  var t = block.Index;  
  using (Variable.If(t == 0)) {  
    x[t] = Variable.GaussianFromMeanAndVariance(0, 1);  
  }  
  using (Variable.If(t > 0)) {  
    x[t] = Variable.GaussianFromMeanAndVariance(x[t - 1], 1);  
  }  
}
```

 

The key expression here is x\[t-1\] where t is the loop counter. This creates the link between consecutive elements of the array. Note that in order to use this expression, we had to test for t>0, because a `ForEach` block always loops from 0 to the end of the range. In general, if you use (t-k) in a statement, you must surround the statement with If(t > k-1), and if you use (t+k) in a statement, you must surround the statement with If(t < length-k). For a forward loop, the base case must be defined using If(t == 0) or Case(t,0). For a backward loop, the base case must be defined using If(t == length-1) or Case(t,length-1). The base case must always precede the definition of the increment case.​

Because the model is defined using a `VariableArray` and not a C# array of Variable objects, the Infer.NET compiler can generate compact code that iterates over the array during inference. Also, the array length can be changed at runtime (by setting `numTimes.ObservedValue`).

If we wanted the position at time t to depend on the two previous times, we could write:

```csharp
using (var block = Variable.ForEach(time)) {  
  var t = block.Index;  
  using (Variable.If(t == 0)) {  
    x[t] = Variable.GaussianFromMeanAndVariance(0, 1);  
  }  
  using (Variable.If(t == 1)) {  
    x[t] = Variable.GaussianFromMeanAndVariance(x[t - 1], 1);  
  }  
  using (Variable.If(t > 1)) {  
    x[t] = Variable.GaussianFromMeanAndVariance(x[t - 1] + x[t - 2], 1);  
  }  
}
```

 Array elements can also be linked via constraints. For example, the definition of array x could be changed to:

```csharp
x[time] = Variable.GaussianFromMeanAndVariance(0, 1000).ForEach(time);  
using (var block = Variable.ForEach(time)) {  
  var t = block.Index;  
  using (Variable.If(t > 0)) {  
    Variable.ConstrainEqualRandom(x[t] - x[t - 1], new Gaussian(0, 1));  
  }  
}
```

Note this defines a somewhat different model to the above. 

#### Grids

A grid model is constructed by offset indexing on a two dimensional array (or jagged array).

```csharp
int length = 10;  
Range rows = new Range(length);  
Range cols = new Range(length);  
VariableArray2D<double> states = Variable.Array<double>(rows, cols);  
using (ForEachBlock rowBlock = Variable.ForEach(rows)) {
  var row = rowBlock.Index;  
  using (ForEachBlock colBlock = Variable.ForEach(cols)) {  
    var col = colBlock.Index;  
    using (Variable.If(row == 0)) {  
      states[row, col] = Variable.GaussianFromMeanAndVariance(0, 1);  
    }  
    using (Variable.If(row > 0)) {  
      states[row, col] = Variable.GaussianFromMeanAndVariance(states[row - 1, col], 1);  
    }  
    using (Variable.If(col > 0)) {  
      Variable.ConstrainEqualRandom(states[row, col] - states[row, col - 1], new Gaussian(0, 1));  
    }  
  }  
}
```

Here we have linked rows by parent factors and columns via constraints, to show both approaches. Since jagged arrays can have any depth, you can use this pattern to define higher-dimensional structures like cubes and hypercubes. There is no limitation on which elements can be connected---a node can be connected to 8 neighbors, for example.

#### Maximum of an array

Infer.NET provides a Max operation between two double variables. Using offset indexing, we can extend this to compute the Max of a VariableArray. The idea is to incrementally build up the partial maximums in an array, and then return the last element:

```csharp
public Variable<double> Max(VariableArray<double> array)  
{  
  Range n = array.Range;  
  var maxUpTo = Variable.Array<double>(n).Named("maxUpTo");  
  using (var fb = Variable.ForEach(n))  
  {  
    var i = fb.Index;  
    using (Variable.Case(i, 0))  
    {  
      maxUpTo[i] = Variable.Copy(array[i]);  
    }  
    using (Variable.If(i > 0))  
    {  
      maxUpTo[i] = Variable.Max(maxUpTo[i - 1], array[i]);  
    }  
  }  
  var max = Variable.Copy(maxUpTo[((Variable<int>)n.Size) - 1]);  
  return max;  
}
```

#### Counting example

The same idea can be used for counting. Infer.NET provides a CountTrue factor, but if it didn't, you could simulate one using offset indexing. The idea is to incrementally build up the partial sums in an array, and then return the last element:

```csharp
public Variable<int> CountTrue(VariableArray<bool> bools)  
{  
  Range n = bools.Range;  
  var sumUpTo = Variable.Array<int>(n).Named("sumUpTo");  
  var sizes = Variable.Array<int>(n).Named("sizes");  
  Range count = new Range(sizes[n]).Named("c");  
  sumUpTo.SetValueRange(count);  
  using (var block = Variable.ForEach(n)) {  
    var i = block.Index;  
    sizes[i] = 2 + i;  
    using (Variable.Case(i, 0)) {  
      using (Variable.If(bools[n])) {  
        sumUpTo[i] = 1;  
      }  
      using (Variable.IfNot(bools[n])) {  
        sumUpTo[i] = 0;  
      }  
    }  
    using (Variable.If(i > 0)) {  
      using (Variable.If(bools[n])) {  
        sumUpTo[i] = sumUpTo[i - 1] + 1;  
      }  
      using (Variable.IfNot(bools[n])) {  
        sumUpTo[i] = sumUpTo[i - 1] + 0;  
      }  
    }  
  }  
  var sum = Variable.Copy(sumUpTo[((Variable<int>) n.Size) - 1]);  
  return sum;  
}
```

The subtlety here is setting up the ValueRange for the sumUpTo array. Each element of sumUpTo ranges over a different number of values, so the size of its ValueRange must depend on n. A similar approach can be used to solve other counting problems, such as computing the probability of observing a run of H heads in a sequence of N coin flips.

See [Chess Analysis](Chess Analysis.md) for another example of offset indexing.

​
