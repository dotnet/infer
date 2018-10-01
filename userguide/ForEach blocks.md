---
layout: default 
--- 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## ForEach blocks

When array definitions get complicated, the ForEach(range) block comes to the rescue. Every variable created within a ForEach(range) block is automatically tagged with .ForEach(range). Thus the ForEach block works very similarly to a foreach loop in C# (except it is treated as a single object by the Infer.NET compiler, with corresponding efficiency gains during compilation). For example, if **bools** and **pixel** are defined as in [Working with arrays and ranges](Arrays and ranges.md), then:

```csharp
bools[pixel] = Variable.Bernoulli(0.7).ForEach(pixel) |  
               Variable.Bernoulli(0.4).ForEach(pixel);
```

is equivalent to:

```csharp
using(Variable.ForEach(pixel)) {  
  bools[pixel] = Variable.Bernoulli(0.7) | Variable.Bernoulli(0.4);  
}
```

ForEach blocks can also be nested, to define multi-dimensional arrays. So the following definition: 

```csharp
doubles2D[pixel,image] = Variable.GaussianFromMeanAndVariance(0,1).ForEach(pixel,image);
```

is equivalent to:

```csharp
using(Variable.ForEach(pixel)) {  
  using (Variable.ForEach(image)) {  
    doubles2D[pixel, image] = Variable.GaussianFromMeanAndVariance(0,1);  
  }  
}
```

and also equivalent to:

```csharp
using (Variable.ForEach(pixel)) {  
  doubles2D[pixel, image] = Variable.GaussianFromMeanAndVariance(0,1).ForEach(image);  
}
```

### Local variables

You can create local variables inside of ForEach blocks. For example:

```csharp
using(Variable.ForEach(pixel)) {  
  Variable<bool> x = Variable.Bernoulli(0.7);  
  Variable<bool> y = Variable.Bernoulli(0.4);  
  bools[pixel] = x | y;  
}
```

This is equivalent to the following:

```csharp
VariableArray<bool> x = Variable.Array<bool>(pixel);  
x[pixel] = Variable.Bernoulli(0.7);  
VariableArray<bool> y = Variable.Array<bool>(pixel);  
y[pixel] = Variable.Bernoulli(0.4);  
using(Variable.ForEach(pixel)) {  
  bools[pixel] = x[pixel] | y[pixel];  
}
```

Thus you can think of local variables as a shortcut to explicitly defining arrays, i.e. they are implicit arrays.

### The 'using' syntax is optional

The _using_ statement in C# hides the fact that a ForEachBlock object is being created and then disposed. The using statement is only a shortcut. We could equivalently write the definition of **doubles2D** as:

```csharp
ForEachBlock pixelBlock = Variable.ForEach(pixel);  
ForEachBlock imageBlock = Variable.ForEach(image);  
doubles2D[pixel, image] = Variable.GaussianFromMeanAndVariance(0,1);  
imageBlock.CloseBlock();  
pixelBlock.CloseBlock();
```

The _using_ syntax is preferable since it ensures all blocks are closed. However, some .NET languages do not provide such a construct, and, whatever the language, you can always explicitly close blocks as above. A danger of this method is that if an exception occurs before you close the block, then the block will remain open. You can use the 'try/finally' construct to ensure that the blocks are closed in this case. As a last resort, if you have caught an exception thrown by another function that opened a block, and don't have access to the block variable, you can call the function Variable.CloseAllBlocks() which will forceably close all open blocks and allow you to recover from the exception.

### Accessing the loop counter

Another advantage of ForEach blocks is that you can access the loop counter as a `Variable<int>` and use its value in factors or [conditional statements](Branching on variables to create mixture models.md). The loop counter is the 'Index' property of the ForEachBlock. For example, if you wanted the elements of doubles2D to have one definition for image<2 and another definition for image>=2, you could write:

```csharp
using(ForEachBlock pixelBlock = Variable.ForEach(pixel)) {  
using (ForEachBlock imageBlock = Variable.ForEach(image)) {  
    using (Variable.If(imageBlock.Index<2)) {  
      doubles2D[pixel, image] = Variable.GaussianFromMeanAndVariance(0,1);   
    }  
    using (Variable.IfNot(imageBlock.Index<2)) {  
      doubles2D[pixel, image] = Variable.GaussianFromMeanAndVariance(2,3);  
    }  
  }  
}
```

Because pixelBlock.Index is a `Variable<int>`, you can apply functions to it such as addition or subtraction. You can also use it to index arrays. In the above code, you could have indexed doubles2D by \[pixelBlock.Index, imageBlock.Index\], and this is equivalent to indexing by \[pixel,image\].
