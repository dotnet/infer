---
layout: default
---
[Factors and Constraints](Factors and Constraints.md)

## Array and list factors

This page lists the built-in methods and operators for creating random variables of array and list type. The most common way of creating array variables is through the modelling API using random variable [arrays and ranges](Arrays and ranges.md), and replicating factors across the range; many example of this can be found in the user guide. Sometimes arrays and lists can be defined directly as output of a factor, and it is those factors than are described here. Some of the factors involve variables of type `ISparseList<T>` where T is some domain (such as **bool** or **double**); a sparse list is one where just a few of the elements have values that differ from the common value shared by all the other elements. 

For both static methods and operators, you can often pass in random variables as arguments e.g. `Variable<ISparseList<double>>` instead of `ISparseList<double>`. For compactness, this is not shown in the syntax below. 

These methods provide a convenient short alternative to using `Variable<T>.Factor` and passing in the factor method, as described [on this page](Applying functions and operators to variables.md).

#### Distribution Factors

A distribution factor creates a random variable from a parameterised distribution.

| **Distribution** | **Syntax** | **Description** |
|-------------------------------------------------|
| _Bernoulli list_ | `Variable.BernoulliList(ISparseList<double> probTrue)` | Creates a random variable of type **`ISparseList<bool>`** with distribution parameterized by a sparse list of values representing probability of true. |
| _Bernoulli integer subset_ | `Variable.BernoulliIntegerSubset(ISparseList<double> probInSubset)` | Creates a random variable of type **`List<int>`** listing membership of a set with distribution parameterized by sparse list of values representing probability of set membership. |
| _Gaussian list_ | `Variable.GaussianListFromMeanAndPrecision(ISparseList<double> mean, ISparseList<double> precision)` | Creates a random variable of type **`ISparseList<double>`** with distribution parameterized by sparse lists of Gaussian means and precisions. |
| _Multinomial_ | `Variable.Multinomial(int trialCount, Vector probs)` | Creates an integer random variable array representing the counts of each index given a number of trials and the probability of each index. |
| _Multinomial list_ | `Variable.MultinomialList(int trialCount, Vector probs)` | As with _Multinomial_ except that the created random variable has type IList instead of array. |

#### Miscellaneous factors

| **Operation** | **Syntax** | **Description** |
|----------------------------------------------|
| _Subarray_ | `Variable.Subarray<T>(T[] array, int[] indices)` | Creates a **T\[\]** random variable array by extracting elements of _array_ at the specified _indices_, which **cannot include duplicates**. Subarray should be used instead of GetItems, whenever you can be sure that there are no duplicates, since it is more efficient. |
| _GetItems_ | `Variable.GetItems<T>(T[] array, int[] indices)` | Creates a **T\[\]** random variable array by extracting elements of _array_ at the specified _indices_, which may include duplicates. Some uses of GetItems can also be achieved through indexing. If there are no duplicate indices, Subarray should be used instead. |
| _ArrayFromVector_ | `Variable.ArrayFromVector(Vector vector)` | Create an array of double random variables corresponding to the elements of the supplied random vector. This is more efficient than calling GetItem on each element. |
| _Split_ | `Variable.Split<T>(T[] array, Range headRange, Range tailRange, out VariableArray<T> tail)` | Creates two arrays by splitting _array_ into disjoint parts. The sizes of the parts are given by _headRange_ and _tailRange_. |
| _SplitSubarray_ | `Variable.SplitSubarray<T>(T[] array, int[][] indices)` | Creates a jagged array, shaped according to _indices_, from disjoint elements of _array_. _indices_ cannot include duplicates. To allow duplicates, use JaggedSubarray. |
| _JaggedSubarray_ | `Variable.JaggedSubarray<T>(T[] array, int[][] indices)` | Creates a jagged array, shaped according to _indices_,  from elements of _array_. indices\[i\]\[j\] must be different for different j and same i, but can match for different i. If all indices are different, use SplitSubarray. |

​
