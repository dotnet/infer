---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Factors and Constraints](Factors and Constraints.md)

## Miscellaneous Factors

This page lists the remaining built-in methods, which have not already been covered. In these methods, you can often pass in random variables as arguments e.g. `Variable<Vector>` instead of `Vector`. For compactness, this is not shown in the syntax below.

These methods provide a convenient short alternative to using `Variable<T>.Factor` and passing in the factor method, as described [on this page](Applying functions and operators to variables.md). 

| **Operation** | **Syntax** | **Description** |
|----------------------------------------------|
| _Subarray_ | `Variable.Subarray<T>(T[] array, int[] indices)` | Create a **T\[\]** random variable array by extracting elements of _array_ at the specified _indices_, which **cannot include duplicates**. Subarray should be used instead of `GetItems`, whenever you can be sure that there are no duplicates, since it is more efficient. |
| _GetItems_ | `Variable.GetItems<T>(T[] array, int[] indices)` | Create a **T\[\]** random variable array by extracting elements of _array_ at the specified _indices_, which may include duplicates. Some uses of GetItems can also be achieved through indexing. If there are no duplicate indices, Subarray should be used instead. |
| _Enum to int_ | `Variable.EnumToInt<TEnum>(TEnum enumVar)` | Create an **int** random variable corresponding to the supplied enum random variable. This allows enums to be used as arguments to **Variable.Case()** or **Variable.Switch()**. |
| _Vector from array_ | `Variable.Vector(double[] array)` | Create a vector random variable corresponding the supplied array of random doubles. |
| _Array from vector_ | `Variable.ArrayFromVector(Vector vector)` | Create an array of double random variables corresponding to the elements of the supplied random vector. This is more efficient than calling `GetItem` on each element. |
| _Concat_ | `Variable.Concat(Vector first, Vector second)` | Concatenates two random vectors. |
| _Subvector_ | `Variable.Subvector(Vector subVector, int startIndex, int count)` | Extract contiguous elements from a random vector. |
| _StringFromArray_ | `Variable.StringFromArray(char[] chars)` | Create a **string** random variable from a random character array. |
| _GetItem_ | `Variable.GetItem(Vector source, int index)` `Variable.GetItem(string str, int pos)` | Extract an element of a random vector or a string. |
| _FunctionEvaluate_ | `Variable.FunctionEvaluate(IFunction func, Vector x)` | Evaluate a random function at a point. Used to construct Gaussian Process models like [this one](Gaussian Process classifier.md). |
