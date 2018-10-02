---
layout: default 
--- 
 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Jagged Arrays

In many models data will not be in the form of a flat single dimensional or a multi-dimension array. For example, items in a data array may have a variable number of features. In this case the observed data is represented as an array of arrays of values; in C# this is referred to as a jagged array, and Infer.NET has an API which allows creation of jagged variable arrays.

The simplest jagged array is a 1D array of 1D arrays. The outer array can be indexed by a standard range, but the inner array is variable in size, and the size itself is a function of the outer index. Here is an example the illustrates the syntax:

```csharp
int[] sizes = new int[] { 2, 3 };  
Range item = new Range(sizes.Length).Named("item");  
VariableArray<int> sizesVar = Variable.Constant(sizes, item).Named("sizes");  
Range feature = new Range(sizesVar[item]).Named("feature");  
var x = Variable.Array(Variable.Array<double>(feature), item).Named("x");
```

In this case, we specify the sizes as a constant variable array as shown in the first three lines. The inner range can then be defined using the natural expression for the size of the inner range. The jagged array is then created on the final line using the standard **Array** constructor where the first argument is the array element (created as variable array of doubles ranging over the inner range), and the second argument is the outer range. Here we have used implicit typing, but we could also be explicit about its type:

```csharp
VariableArray<VariableArray<double>,double[][]> x =  
    Variable.Array(Variable.Array<double>(feature), item).Named("x");
```

The .NET type of a jagged array in Infer.NET is a **[VariableArray](../apiguide/api/Microsoft.ML.Probabilistic.Models.VariableArray-2.html)** with two generic type parameters. The first parameter represents the type of element in the **VariableArray**. In this case we have a `VariableArray` of `VariableArray<double>`, so the first type argument is `VariableArray<double>`. The second type parameter is the .NET array type of the jagged array - in this case we have an array of array of doubles.

We can now use the ranges to index and loop over our jagged array in a natural manner:

```csharp
Gaussian xPrior = new Gaussian(1.2, 3.4);  
x[item][feature] = Variable.Random(xPrior).ForEach(item, feature);  
Variable.ConstrainPositive(x[item][feature]);
```

If you use **ForEach** blocks rather than the inline **ForEach** method, the outer **ForEach** block must always be over the outer **Range**.

#### Constant jagged arrays

You can define a constant jagged array initialised to its equivalent .NET array directly in the definition. For example:

```csharp
double[][] a = new double[][] { new double[] {1.1, 3.3}, new double[] {1.1, 2.2, 4.4} };  
int[] innerSizes = new int[a.Length];  
for (int i=0; i < a.Length; i++)  
    innerSizes[i] = a[i].Length;Range outer = new Range(a.Length).Named("outer");  
VariableArray<int> innerSizesVar = Variable.Constant(innerSizes, outer).Named("innerSizes");  
Range inner = new Range(innerSizesVar[outer]).Named("outer");  
var aConst = Variable.Constant(a, outer, inner);
```

Here we have used implicit typing again. The explicit type of aConst in this example is:

```csharp
VariableArray<VariableArray<double>, double[][]>
```

#### Observed jagged arrays

It is more efficient and flexible to use an observed jagged array rather than a constant jagged array. One way to do this is to change the last line in the code above to use `Variable.Observed` rather than `Variable.Constant`. Another approach is to use `Variable.Array` and then set the `ObservedValue` property. However, note that this example uses fixed ranges; if the observed array is going to change its size in any way then you should use ranges that are also variable. The following snippet of code shows how to do this.

```csharp
var outerSizeVar = Variable.New<int>();  
Range outer = new Range(outerSizeVar);  
var innerSizesVar = Variable.Array<int>(outer);  
Range inner = new Range(innerSizesVar[outer])  
var aObs = Variable.Array(Variable.Array<double>(inner), outer);
```

The observed values can then be set before inference, making sure that the jagged sizes, and the jagged array itself are consistently set:

```csharp
var a = new double[][] { new double[] {1.1, 3.3}, new double[] {1.1, 2.2, 4.4} };  
outerSizeVar.ObservedValue = a.Length;  
var innerSizes = new int[a.Length];  
for (int i = 0; i < a.Length; i++)  
  innerSizes[i] = a[i].Length;  
innerSizesVar.ObservedValue = innerSizes;  
aObs.ObservedValue = a;
```

#### More complex jagged arrays

The Infer.NET API also supports more complex jagged arrays such as a 2D array of arrays, or an array of array of arrays. Here is an example of a 2D array of arrays. In this example, the inner range is of fixed size which simplifies the definition.

```csharp
int[,] sizes2D = new int[,] { {2, 3}, {4, 2}, {3, 1} };  
Range rx = new Range(sizes2D.GetLength(0)).Named("rx");  
Range ry = new Range(sizes2D.GetLength(1)).Named("ry");  
VariableArray2D<int> sizes2DVar = Variable.Constant(sizes2D, rx, ry);  
Range rz = new Range(sizes2DVar[rx,ry]).Named("rz");  
var zVar = Variable.Array(Variable.Array<double>(rz), rx, ry).Named("zVar");
```

where the type of **zVar** is

```csharp
VariableArray2D<VariableArray<double>,double[,][]>
```

If, instead, we were to build an array of 2D arrays, the type would be

```csharp
VariableArray<VariableArray2D<double>, double[][,]>
```

This can be continued to any depth. This code creates a jagged array of depth 4:

```csharp
var a = Variable.Array<Vector>(new Range(1));  
var b = Variable.Array<VariableArray<Vector>, Vector[][]>(a, new Range(2));  
var c = Variable.Array<VariableArray<VariableArray<Vector>, Vector[][]>, Vector[][][]>(b, new Range(3));  
var d = Variable.Array<VariableArray<VariableArray<VariableArray<Vector>, Vector[][]>, Vector[][][]>, Vector[][][][]>(c, new Range(4));
```

Code like this can be simplified by using type aliases. For example:

```csharp
using VarVectArr2 = VariableArray<VariableArray<Vector>, Vector[][]>;
```
