---
layout: default 
--- 
 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Working with arrays and ranges

When defining a large model, it is a good idea to avoid creating too many Variable objects. One way to do this is to use Infer.NET's built-in concept of a [**VariableArray**](../apiguide/api/Microsoft.ML.Probabilistic.Models.VariableArray-1.html). VariableArrays are processed as a unit during model compilation and inference, allowing dramatic speedups. This section covers single-dimensional and multi-dimensional arrays. [Jagged arrays](Jagged arrays.md) (arrays of arrays) are described in a later section.

#### Ranges

A [**Range**](../apiguide/api/Microsoft.ML.Probabilistic.Models.Range.html) object represents a stream of integers from 0 to N-1 inclusive. The following code creates a range from 0 to 9: 

```csharp
Range pixel = new Range(10);
```

It is also possible to create give a range a name to be used in the generated code, e.g.

```csharp
Rangeimage = new Range(4).Named("image");
```

The size of a range does not have to be fixed when defining your model, for example:

```csharp
Variable<int> nImages = Variable.New<int>();  
Range image = new Range(nImages);
```

You can then set the size of the range before you run inference:

```csharp
nImages.ObservedValue = 10;
```

#### Variable arrays

Having defined ranges, you can then declare a variable array using the syntax:

```csharp
Variable.Array<T>(Range r1,Range r2,...)
```

The provided ranges must be distinct. The dimensionality of the array is determined from the number of ranges. Currently, arrays of one or two dimensions are supported. More complex arrays can be created using [jagged arrays](Jagged arrays.md), and in fact, for large problems, it is usually better to use arrays of arrays in preference to 2D arrays, even if the underlying array is rectangular and not jagged.

The following code creates a 1D and then a 2D array:

```csharp
VariableArray<bool> bools = Variable.Array<bool>(pixel);  
VariableArray2D<double> doubles2D = Variable.Array<double>(pixel,image);
```

At this point, the arrays are declared but not initialized. You can fill an array in two ways: the **SetTo** method or using an indexer **\[\]**.

The **SetTo** method is used in conjunction with a factor that returns an array, e.g. **Factor.MatrixMultiply**. The **SetTo** method takes a factor along with factor arguments, and defines the array contents to be the output of the factor:

```csharp
doubles2D.SetTo(Factor.MatrixMultiply, someArray2D, anotherArray2D);
```

For built-in array factors like MatrixMultiply, there is also a shortcut method on Variable which creates and fills in the array in one go, inheriting its ranges from the arguments:

```csharp
VariableArray2D<double> doubles2D = Variable.MatrixMultiply(someArray2D, anotherArray2D);
```

The indexer approach is used in conjunction with a factor that returns a single item, e.g. **Variable.Bernoulli**. The right hand side of the indexer call must be a stream of variables, tagged with the same ranges used to create the VariableArray. The simplest way to create a stream of variables is with the `.ForEach` method:

```csharp
expression.ForEach(r1, r2,...)
```

Here r1 and r2 are Range objects. The result of this call is a stream of independent variables of type T, each having the same definition as the provided expression. For example, this code fills in the contents of the **bools** array declared above:

```csharp
bools[pixel] = Variable.Bernoulli(0.7).ForEach(pixel);
```

Each element of **bools** is now an independent Bernoulli variable with mean 0.7. When you pass a stream of variables to a factor, the result is also a stream, so the following is also valid:

```csharp
bools[pixel] = !(Variable.Bernoulli(0.7).ForEach(pixel));
```

In this case, each element of **bools** is an independent Bernoulli variable defined as the logical NOT of a Bernoulli variable of mean 0.7. You can also mix streams and individual variables, like so:

```csharp
bools[pixel] = Variable.Bernoulli(0.7).ForEach(pixel) | Variable.Bernoulli(0.9);
```

In this case, we take N independent Bernoulli variables with mean 0.7 and OR them with a single Bernoulli variable with mean 0.9, giving a new set of N variables. The resulting variables are not independent since they all depend on the outcome of the 0.9 variable. If that variable is true, then all of the results are true.

In general, the arguments to the factor on the right-hand side may be streams over different ranges, as long as the overall set of ranges on the right-hand side matches the indices on the left-hand side.

Besides `.ForEach`, you can create a stream of variables by indexing an array by its range object. For example, if we had an array of doubles called **allProbTrues**, then we could write:

```csharp
bools[pixel] = Variable.Bernoulli(allProbTrues[pixel]);
```

Because there is no loop, this notation allows large numbers of regularly-structured variables to be defined very compactly. 

_**See also:** [Indexing arrays by observed variables](Indexing arrays by observed variables.md)_

Arrays can be indexed by ranges, integer variables or integers. That is, **bools** can be indexed by **pixel,** a clone of **pixel,** a `Variable<int>`, or an `int`. For example, **bools\[0\]**, is equivalent to **bools\[Variable.Constant(0)\]**.

If your model has a lot of statements requiring the ForEach method to be appended for a given range, it may be more convenient to use [the ForEach block](ForEach blocks.md).

#### Streams of variables

This section describes in more detail the stream concept introduced above. The basic idea is that, besides arrays, Infer.NET has the concept of a **stream of variables.** To illustrate why this is needed, consider a factor like Factor.Bernoulli. This factor takes one scalar double. Passing it an array of doubles should be a type error. However, we also want to allow compact code such as:

```csharp
bools[pixel] = Variable.Bernoulli(allProbTrues[pixel]);
```

Infer.NET thus provides a stream syntax to allow code that is both compact and type-safe. Infer.NET VariableArrays are special in that they are indexed by **ranges** as well as integers. When a VariableArray is indexed by a range object, i.e. **varray\[range\]**, the result is a stream of variables tagged with that range. This stream can then be given to a factor which expects the variable type, returning a stream of result variables. The stream of result variables is then converted back into an array by assignment, again using the range as index. Notice the code is very similar to what you would write as the body of a loop, except no looping construct is needed. Conceptually, the syntax **allProbTrues\[pixel\]** is creating a stream of double variables, **Variable.Bernoulli** is then creating a stream of boolean variables, and **bools\[pixel\] = ...** is filling the **bools** array with that stream. In implementation, a stream is represented by a single object, with no looping involved. Thus the code is more efficient than an explicit C# loop that creates multiple Variable objects.

The result of **Variable.Bernoulli** above is a stream that can be passed through any number of factors before it is assigned to an array. Because streams are tagged with range objects, you can also combine multiple streams into one expression yielding a multidimensional VariableArray, like so:

```csharp
doubles2D[pixel,image] = y[pixel]+z[image];
```

#### Cloning ranges

Sometimes you want to access all pairs of array elements in an expression. This requires cloning the range which is discussed in more detail [here](Cloning ranges.md).

#### Constant arrays

A constant array is created using **Variable.Constant**, passing a .NET array and a set of ranges:

```csharp
VariableArray<double> data = Variable.Constant(  
    new double[] { 1, 2, 3, 4 }, image);  
VariableArray2D<double> data2D = Variable.Constant(  
    new double[,] { {5,6}, {7,9} }, range1, range2);
```

The resulting array can only be indexed by the ranges that it was created with. Constant arrays can be used just like random variable arrays. In particular, they can appear on the left-hand side of an indexed assignment. But the effect is to impose a constraint, rather than change the contents of the array. For example:

```csharp
data[image] = Variable.GaussianFromMeanAndPrecision(mean, 1).ForEach(image);
```

is equivalent to:

```csharp
Variable.ConstrainEqual(  
    data[image],  
    Variable.GaussianFromMeanAndPrecision(mean, 1).ForEach(image));
```

Constants are so named because they are embedded in the generated code as literal constants. For this reason, only objects taking a small amount of space should be made constants. For large arrays, it is usually more efficient, and better practice, to use observed variable arrays rather than constant arrays; these are described in the next section.

#### Arrays of observed variables

Observed variable arrays are similar to constant variable arrays but the values can be changed before running inference. They can be created using **Variable.Array** or **Variable.Observed**, the latter providing the additional convenience of providing an initial setting for the array.

```csharp
VariableArray<double> obs = Variable.Observed( new double[] { 1, 2, 3, 4 }, image);  
VariableArray2D<double> obs2D = Variable.Observed( new double[,] { {5,6}, {7,9} }, range1, range2);
```

Alternatively, you can define the observed variable arrays without specifying ranges, and define the ranges from the arrays:

```csharp
VariableArray<double> obs = Variable.Observed(new double[] { 1, 2, 3, 4 });  
VariableArray2D<double> obs2D = Variable.Observed(new double[,] { { 5, 6 }, { 7, 9 } });  
Range r = obs.Range;Range r0 = obs2D.Range0;Range r1 = obs2D.Range1;
```

If you use `Variable.Observed`, you must supply initial values. If you want to omit the initial observed values, you should use `Variable.Array` instead. Here is an example for a 2D array where the array sizes and the array itself are not specified when defining the model: 

```csharp
Variable<int> sizeX = Variable.New<int>();  
Variable<int> sizeY = Variable.New<int>();  
Range x = new Range(sizeX);  
Range y = new Range(sizeX);  
VariableArray2D<double> arrayXY = Variable.Array<double>(x, y);
```

Later, when you want to run inference, you can supply the observed value as follows. Infer.NET will expect the array to be the same size as the ranges:

```csharp
double[,] obs2DData = new double[,] { { 5, 6 }, { 7, 9 }, {6, 7} };  
sizeX.ObservedValue = obs2DData.GetLength(0);  
sizeY.ObservedValue = obs2DData.GetLength(1);  
arrayXY.ObservedValue = obs2DData;
```

#### Other array types

Observed variable arrays can have different type besides T\[\]. The options are:

```csharp
Variable.IList<T>(Range r)   
Variable.ISparseList<T>(Range r)  
Variable.IArray<T>(Range r)
```

​
