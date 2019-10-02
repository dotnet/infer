---
layout: default 
--- 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Indexing arrays by observed variables

Besides ranges, VariableArrays can be indexed by integer variables. This allows you to create complex dependencies between array elements. For example, suppose you have an array of boolean variables, and you want to constrain one of them to be true. This can be accomplished as follows:

```csharp
Range item = new Range(4);  
VariableArray<bool> bools = Variable.Array<bool>(item);  
bools[item] = Variable.Bernoulli(0.7).ForEach(item);  
Variable<int> index = Variable.New<int>();  
Variable.ConstrainTrue(bools[index]);
```

The observed value of index can be set and changed at any time. This immediately changes the effect of the constraint. For example:

```csharp
InferenceEngine engine = new InferenceEngine();  
index.ObservedValue = 2;  
Console.WriteLine(engine.Infer(bools));  
// Result is:  
// [0] Bernoulli(0.7)  
// [1] Bernoulli(0.7)  
// [2] Bernoulli(1)  
// [3] Bernoulli(0.7)  
index.ObservedValue = 3;  
Console.WriteLine(engine.Infer(bools));  
// Result is:  
// [0] Bernoulli(0.7)  
// [1] Bernoulli(0.7)  
// [2] Bernoulli(0.7)  
// [3] Bernoulli(1)
```

Besides applying constraints, you can also use `bools[index]` in a factor, or basically anywhere you would use an ordinary variable.

To index by an unobserved (i.e. random) integer variable, use a [Switch block](Branching on variables to create mixture models.md).

#### Indexing by observed variable arrays

The index can be an array element, allowing you to compactly access many elements of a VariableArray. For example, suppose you want to constrain several elements of **bools** to be true. You can do this as follows:

```csharp
Range item = new Range(4);  
VariableArray<bool> bools = Variable.Array<bool>(item);  
bools[item] = Variable.Bernoulli(0.7).ForEach(item);  

Variable<int> numIndices = Variable.New<int>();  
Range indexed_item = new Range(numIndices);  
VariableArray<int> indices = Variable.Array<int>(indexed_item);  
Variable.ConstrainTrue(bools[indices[indexed_item]]);
```

Here we have made a variable (**numIndices**) for the size of the **indexed_item** range, so that we can vary the number of indices at runtime. For example:

```csharp
InferenceEngine engine = new InferenceEngine();  
numIndices.ObservedValue = 2;  
indices.ObservedValue = new int[] { 1, 2 };  
Console.WriteLine(engine.Infer(bools));  
// Result is:  
// [0] Bernoulli(0.7)  
// [1] Bernoulli(1)  
// [2] Bernoulli(1)  
// [3] Bernoulli(0.7)  
numIndices.ObservedValue = 3;  
indices.ObservedValue = new int[] { 1, 2, 3 };  
Console.WriteLine(engine.Infer(bools));  
// Result is:  
// [0] Bernoulli(0.7)  
// [1] Bernoulli(1)  
// [2] Bernoulli(1)  
// [3] Bernoulli(1)
```

#### Efficient access to subarrays

A speed-up is possible when you know that the indices are all distinct. In other words, you are extracting a subarray of the original variable array. The function Variable.Subarray handles this case efficiently, taking a VariableArray and an array of indices, and returning a smaller VariableArray. We can apply this transformation to the above example, to get:

```csharp
Range item = new Range(4);  
VariableArray<bool> bools = Variable.Array<bool>(item);  
bools[item] = Variable.Bernoulli(0.7).ForEach(item);  

Variable<int> numIndices = Variable.New<int>();  
Range indexed_item = new Range(numIndices);  
VariableArray<int> indices = Variable.Array<int>(indexed_item);  

VariableArray<bool> indexedBools = Variable.Subarray(bools, indices);  
// indexedBools automatically has range 'indexed_item'  
Variable.ConstrainTrue(indexedBools[indexed_item]);
```

During inference, this model gives the same answers as above, however it runs a bit faster since Infer.NET does not have to check for the case that two indices might be equal.

#### Indexing jagged arrays

When a jagged array is indexed by an observed variable, the ranges of its inner arrays automatically change to reflect the index. For example:

```csharp
Range item = new Range(4);  
VariableArray<int> sizes = Variable.Constant(new int[] { 3, 4 }, item);  
Range inner = new Range(sizes[item]);  
var bools = Variable.Array(Variable.Array<bool>(inner), item);  
bools[item][inner] = Variable.Bernoulli(0.7).ForEach(item,inner);  
Variable<int> index = Variable.New<int>();  
VariableArray<bool> boolsIndexed = bools[index];  
// boolsIndexed has length sizes[index]  
Range innerIndexed = boolsIndexed.Range;  
// boolsIndexed[inner] does not work; must use the newly constructed range  
Variable.ConstrainTrue(boolsIndexed[innerIndexed]);
```

The expression `bools[index]` returns a VariableArray whose length is `sizes[index]`. This is different from the length of **inner**, thus a new range is created for it. You can access this range via the Range property of the returned array.
