---
layout: default 
--- 
[Infer.NET development](index.md)

## Estimating computational cost

This page describes the design of a compiler transform for estimating the computational cost of the code passed into the transform, in the form of an expression for the time and memory used.

In the following, x is of type T, y is an array with element type T and i and j are indices in loops of size I and J. 
This table doesn't take into account conditional execution (if statements) or sparse arrays.

| **Element** | **Code** | **Execution time** | **Memory (heap)** | **Notes** |
|-----------------------------------------------------------------------------|
| **Method call** | x = ​MyMethod(a,b,c) | timeof(MyMethod()) | sizeof(T) if T is a class or 0 otherwise |  |
|  | x = ​MyMethod(a,b,c, x) where x is 'result' parameter | timeof(MyMethod()) | 0 |  |
|  | y[i,j] = ​​​MyMethod(a,b,c) in loops of size I,J | ​I.J.timeof(MyMethod()) | I.J.​sizeof(T) |  |
|  | var x = ​​​MyMethod(a,b,c) in loops of size I,J | ​​I.J.timeof(MyMethod()) | sizeof(T) if T is a class or 0 otherwise | ​The effect of churn is ignored. |
| **Object creation** | x = T() | timeof(T()) | ​sizeof(T) if T is a class or 0 otherwise |  |
|  | var x = T() in loops of size I,J | I.J.timeof(T()) | sizeof(T) | The effect of churn is ignored. |
| **Array creation** | ​var y = new T[I,J] | I.J.timeof(T()) if T is a struct or 0 otherwise | I.J.​sizeof(T) if T is a struct I.J.sizeof(ref) if T is a class. | sizeof(ref) represents the size of a reference. |
|  | var y = new T[J] in loop of size I | I.J.timeof(T()) if T is a struct or 0 otherwise | J.sizeof(T) if T is a struct J.sizeof(ref) if T is a class |  |
|  | y[i] = new T[J] in loop of size I | I.J.timeof(T()) if T is a struct or 0 otherwise | I.J.sizeof(T) if T is a struct I.J.sizeof(ref) if T is a class |  |


The expressions for time and memory would be stored in a `Cost` object attached as an attribute on the expression and/or containing statements. The `Cost` object would:

*   have `Time` and `Memory` properties in the form of expressions. We would need to add a `timeof` operator and allow sizeof to be applied to classes (in C# it is only allowed to be applied to structs).

*   have helper methods/properties to interpret these expressions e.g. to provide `ActualTime()` and `ActualMemory()` in terms of seconds or bytes, given a cost estimator object that would provide (configurable) estimates of the actual values of timeof or sizeof expressions.

In addition, there would be an aggregator that would collect all costs for an entire method or program and highlight bottleneck expressions e.g. the top N expressions in terms of cost or memory.

Design questions:

*   What is the purpose of tracking memory usage - to look at max usage? to track churn? to estimate communication costs for parallel execution?

*   Depending on the above, we may need to track the lifetime of an object, to determine when memory is freed.

*   Does it make more sense to track memory allocation against variables rather than expressions? If the aim is to estimate communication costs, this may be necessary.

