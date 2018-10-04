---
layout: default 
--- 
[Infer.NET development](index.md)

## Infer.NET Math library

This library provides numerical routines needed for machine learning applications. It includes linear algebra, special functions, and random number generation. It can be used separately from the rest of Infer.NET.

The library is pure .NET and Microsoft-owned. It is usable throughout Microsoft. 

The library can be configured to use Lapack (a native library for linear algebra) instead of the .NET implementation. This makes the code faster but less portable.

Current features:

*   Subclasses for PositiveDefiniteMatrix, LowerTriangularMatrix, UpperTriangularMatrix (for better type safety). 
*   LU decomposition. 
*   Cholesky decomposition. 
*   Solving triangular systems (Gaussian elimination and back-substitution). 
*   Can use Lapack (as a compilation option). 
*   Special functions (Gamma, Digamma, Trigamma, Erfc, ErfcInv) 
*   High-precision log-exp conversions (Log1Plus, Log1PlusExp, ExpMinus1, LogExpMinus1, LogSumExp, DiffLogSumExp) 
*   Non-uniform random number generation (Gaussian, Gamma, Wishart) 

Planned features:

*   More extensive Lapack binding. 

Wish list:

*   Sparse matrices. 

#### Rationale for the design 

A matrix library needs to provide routines for special matrix types and decompositions. One approach is to have methods such as InvertUpperTriangular which only apply to special matrices, with no explicit tagging on the matrix at all. This is the approach taken by Lapack. However, such method names are clumsy to use and lead to a messy class interface. They also provide no error checking, in case someone calls InvertUpperTriangular on a matrix which is not triangular.

The current design uses separate classes for special matrices and decompositions. Thus InvertUpperTriangular becomes Invert on the UpperTriangular class. This provides better organization of methods, shorter method names, clearer method signatures (since each argument is labeled with a matrix type), and type errors are caught at compile-time. However, it does prevent certain types of in-place operations, e.g. if a lower triangular matrix is transposed in place, it needs to become upper triangular. Since C# does not allow in-place type conversion, you must construct an upper triangular matrix to hold the result. Due to the cursor design (described below), the storage overhead of this is small, but it is still an annoyance. This approach can also lead to an explosion of subclasses, to handle each combination of features (e.g. a matrix can be positive definite and triangular). 

An alternative design is to have a single matrix class with Boolean fields indicating the matrix type. The Invert method does run-time dispatching based on these fields. This handles combinations of features quite naturally and allows in-place type conversions by modifying the type fields. However, you lose the type-checking and documentation advantages of subclassing.

Another issue in designing a matrix class is choosing the storage representation. The current design stores the matrix as a contiguous block of a larger one-dimensional array. This allows matrices to act as cursors, sharing the same array object as other data. This seems to add overhead when accessing elements of the matrix, but a decent compiler can eliminate this. 

Another design would have a private array for each matrix. This allows the representation to be a direct C# two-dimensional array. The storage requirements for both designs are the same. Indexing a private array might be (slightly) faster than a shared array. However, you lose the ability to function as a cursor.

A complete matrix library should offer several different storage options, each sharing a common interface. Infer.Net does not yet need this level of complexity, so it is not implemented. 

Efficiency considerations motivate the following:

*   Matrix and Vector sizes are immutable, similar to arrays in C#. 
*   Item accessors are non-virtual. 

As in Lapack, most matrix operations take a pre-allocated workspace and result storage as input. For example, `PositiveDefiniteMatrix.LogDeterminant` is overloaded to either take a workspace as input or allocate one dynamically.

Usually the target of the method receives the result, e.g. `L.SetToCholesky(A)`.

This is to mimic the equivalent Matlab statement `L = chol(A)`.

#### Example Program

```csharp
using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic;

class Program
{
  static void Main(string[] args)
  {
    Vector x = new Vector(new double[] { 0, 1 });
    Vector mean = new Vector(new double[] { 0.1, 1.1 });
    PositiveDefiniteMatrix variance = new PositiveDefiniteMatrix(new double[,] { { 0.1, 0.1 }, { 0.1, 0.5 } });
    int d = x.Count;
    double logProb;

    // evaluate the Gaussian density via direct matrix inverse and determinant.
    Vector dx = x - mean;
    logProb = -0.5 * variance.LogDeterminant() - d * MMath.LnSqrt2PI - 0.5 * dx.Inner(variance.Inverse() * dx);
    Console.WriteLine("log p(x|m,V) = {0}", logProb);

    // evaluate the Gaussian density using Cholesky decomposition.
    LowerTriangularMatrix varianceChol = new LowerTriangularMatrix(d, d);
    varianceChol.SetToCholesky(variance);
    dx.PredivideBy(varianceChol);
    // new dx = inv(chol(v))*dx so that
    // (new dx)'*(new dx) = dx'*inv(chol(v))'*inv(chol(v))*dx 
    // = dx'*inv(chol(v)*chol(v)')*dx
    // = dx'*inv(v)*dx
    logProb = -varianceChol.TraceLn() - d * MMath.LnSqrt2PI - 0.5 * dx.Inner(dx);
    Console.WriteLine("log p(x|m,V) = {0}", logProb);
    Console.ReadKey();
  }
}
```