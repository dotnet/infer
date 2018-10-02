---
layout: default 
--- 
 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Vector and Matrix types

Infer.NET includes types for representing vectors (1D arrays of double values) and matrices (2D rectangular arrays of double values).

#### The Vector type

The Vector class is a comprehensive class for doing efficient operations on vectors of `doubles`. The `Vector` class is actually an abstract base class that has four concrete implementations: namely `DenseVector`, `SparseVector`, `ApproximateSparseVector` and `PiecewiseVector`. However, most of the time you should interact with the Vector class which provides a common gateway for interacting with all the different vector types. The non-dense vector types are useful when you have vectors where most of the elements have the same value or approximately the same value. This is common, for example, when working with large `Dirichlet` and `Discrete` distributions, or distributions over characters. The sparsity is specified using the Sparsity class described in the next section.

#### Sparsity

Sparsity specifications, which are used in the Vector factory methods described in a later section, can be created using the [Sparsity class](../apiguide/api/Microsoft.ML.Probabilistic.Math.Sparsity.html). This class contains several static properties or methods for creating different sparsity specifications as shown in the following table. Once created, the properties of a sparsity specification cannot be changed.

| **Static property or method** | **Parameters** | **Explanation/Comments** |
|---------------------------------------------------------------------------|
| **Dense** _(static property)_ | _None_ | Creates the specification for dense vectors. This specification assumes that most elements of the vector will be different, so there is nothing to gain by having a sparse representation |
| **Sparse** _(static property)_ | _None_ | Creates the specification for exact sparse vectors. This specification assumes that most elements of the vector will be the same, and therefore computational and memory advantages can be gained by maintain a sparse representation of the Vector. The common value (i.e. the value shared by most of the Vector elements) is maintained by the Vector rather than the sparsity specification, and will be modified as needed by various vector operations. |
| **Piecewise** _(static property)_ | _None_ | Creates the specification for exact sparse piecewise vectors. This specification assumes that elements of the vector can be divided into a small number of groups (pieces) of consecutive elements having the same value. Specifying a common value is also supported, analogously to regular sparse vectors. |
| **ApproximateWithTolerance** _(static method)_ | **Tolerance**: The tolerance at which vector element values are considered equal to the common value. | The same as Sparse, except that after any operation, any values which are with the tolerance of the common value are considered equal to the common value. |
| **FromSpec** _(static method)_ | **StorageType**: Whether the specification should be dense, sparse or piecewise. **Tolerance**: The tolerance at which vector element values are considered equal to the common value. **CountTolerance**: The maximum allowed count of vector elements not set to the common value. | This method allows sparsity to be explicitly specified via the method parameters. The only additional option this gives is the count tolerance which is another form of approximation. After each vector operation, if the number of non-common values is greater than the count tolerance, then the elements with values closest to the common value are set to the common value so that the tolerance can be satisfied. If the count tolerance has a value of less than or equal to 0, then it is ignored. |

#### Factory Methods

To create a vector, use one of the following static methods on the Vector class. These include settings for the desired sparsity of the generated vector:

**Zero**: Creates a vector of a given length with all elements set to 0.0. 
```csharp
Vector v = Vector.Zero(2); // Dense by default  
Vector v = Vector.Zero(2, Sparsity.Dense);  
Vector v = Vector.Zero(2, Sparsity.Sparse);  
Vector v = Vector.Zero(2, Sparsity.Piecewise);  
Vector v = Vector.Zero(2, Sparsity.ApproximateWithTolerance(0.001));
```

**Constant**: Creates a vector of a given length with all elements are set to a given constant value.

```csharp
Vector v = Vector.Constant(2, 3.0); // Dense by default  
Vector v = Vector.Constant(2, 3.0, Sparsity.Dense);  
Vector v = Vector.Constant(2, 3.0, Sparsity.Sparse);  
Vector v = Vector.Constant(2, 3.0, Sparsity.Piecewise);  
Vector v = Vector.Constant(2, 3.0, Sparsity.ApproximateWithTolerance(0.001));
```

**FromArray**: Creates a vector from an array of doubles.

```csharp
Vector v = Vector.FromArray(doubleArray); // Dense by default  
Vector v = Vector.FromArray(doubleArray, Sparsity.Dense);  
Vector v = Vector.FromArray(doubleArray, Sparsity.Sparse);  
Vector v = Vector.FromArray(doubleArray, Sparsity.Piecewise);  
Vector v = Vector.FromArray(doubleArray, Sparsity.ApproximateWithTolerance(0.001));
```

**Copy**: Create a copy of an existing vector with a specified target sparsity.

```csharp
Vector v = Vector.Copy(v1); // Dense target by default  
Vector v = Vector.Copy(v1, Sparsity.Dense);  
Vector v = Vector.Copy(v1, Sparsity.Sparse);  
Vector v = Vector.Copy(v1, Sparsity.Piecewise);  
Vector v = Vector.Copy(v1, Sparsity.ApproximateWithTolerance(0.001));
```

There are many categories of functions on Vector instances which allow efficient processing. These are summarised below; for full details, see the [code documentation for the Vector class](../apiguide/api/Microsoft.ML.Probabilistic.Math.Vector.html).

#### Indexing

Indexing allows the getting and setting of individual elements of the vector. This is only recommended in exceptional circumstances as it is typically an inefficient way to process vectors. Other methods relating to indexing are **IndexOfAll**, **IndexOf**, **IndexOfMin**, and **IndexOfMax**.

#### Appending

Vectors can be created by appending one vector or scalar to another vector by means of the **Append** method. A static version of this, **Concat**, is also available.

#### SetTo

An existing Vector instance can be modified by setting its values to a set of given values. Instance methods **SetTo**, **SetToSubarray**, and **SetAllElements** provide this functionality. The **CopyTo** method provides the reverse functionality, allowing the current instance to be copied to another specified Vectorinstance.

#### SetToFunction

The Vector class has some methods to set the current instance to a combination of one or two other vectors. One general such method is **SetToFunction** which takes one or two vectors along with a delegate giving the element-wise calculation. Specific such methods include **SetToSum**, **SetToDifference**, **SetToProduct**, **SetToRatio** and **SetToPower**.

#### Operators

Operators +, - (unary and binary), *, / and ^ are all overridden to provide addition, subtraction/negation, product, division, and raising to a power. >, >=, <, <=, =, != are all overridden to provide vector comparison. These comparison operators make use of two general methods **All** and **Any** which take a condition delegate.

#### Reduce

This category of methods allows one or more vectors to be reduced to a scalar by means of a combining function. The most general method is **Reduce** which takes a general combining delegate as an argument. Specific versions include **Sum**, **SumI**, **SumISq**, **Max**, **Min**, and **Inner**.

#### Operations involving matrices

Methods such as **SetToProduct** (with a Matrix and Vector argument), **PredivideBy**, **Outer**, and **SetToDiagonal** are provided to support some fundamental operations involving vectors and matrices. Although these will work with any type of vector (sparse or dense), they will be quite inefficient for sparse vectors, and there is no support for sparse matrices in Infer.NET.

### The PositiveDefiniteMatrix type

The `PostiveDefiniteMatrix` class is used to represent a positive definite matrix which is the domain type for a Wishart distribution. Wishart distributions are used to represent uncertainty in the precision matrix of a `VectorGaussian` distribution, and many of the methods in the `PostiveDefiniteMatrix` class are geared towards this usage. For full details see the [code documentation for the PositiveDefiniteMatrix class](../apiguide/api/Microsoft.ML.Probabilistic.Math.PositiveDefiniteMatrix.html).
