---
layout: default
---
[Infer.NET development](../index.md) : [Infer.NET compiler design](../Compiler transforms.md)

## Indexing transform

Finds arrays indexed by constants or observed variables, such as a[0] or a[observedIndex], and replaces the expression with a new variable, such as a_item0.
The new variable is related to the original expression through `Factor.Copy`. If the replaced expression was on the left hand side of an assignment, then it is assigned `Factor.Copy(new variable)`, otherwise the new variable is assigned to `Factor.Copy(original expression)`.
 
Multiple identical expressions are replaced with the same variable e.g. multiple references to a[0] are all replaced with a_item0. Only stochastic expressions are transformed.
 
The new declaration and the `Factor.Copy` must be placed in the same loop context as the array declaration. In other words, if an array is declared outside a loop but the index expression is inside, then the new statement must be placed outside.
 
ISSUE: what to do if expression is used on both the left and right hand side of the assignment?

| Input | Output |
|----------------|
| `double[] x = new double[2];` <br /> `bool[] b = new double[3];` <br /> `for(int i=0;i<3;i++) {` <br /> `b[i] = Factor.IsPositive(x[0]);` <br /> `}` | `double[] x = new double[2];` <br /> `bool[] b = new double[3];` <br /> `double x_item0 = Factor.Copy(x[0]);` <br /> `for(int i=0;i<3;i++) {` <br /> `b[i] = Factor.IsPositive(x_item0);` <br /> `}` |
| `double[] x = new double[2];` <br /> `x[0] = Factor.Gaussian(0,1);` <br /> `x[1] = Factor.Gaussian(x[0],1);` | `double[] x = new double[2];` <br /> `double x_item0 = Factor.Gaussian(0,1);` <br /> `x[0] = Factor.Copy(x_item0);` <br /> `double x_item1 = Factor.Gaussian(x_item0,1);` <br /> `x[1] = Factor.Copy(x_item1);` |
