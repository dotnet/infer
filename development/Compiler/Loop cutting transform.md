---
layout: default
---
[Infer.NET development](../index.md) : [Compiler transforms](../Compiler transforms.md)

## Loop cutting transform

This transform cuts for loops, so that each statement in the loop ends up in a loop by itself.
It also converts variables declared inside loops into array variables declared at the top level. References to such variables are modified by adding indices appropriately.

| **Input** | **Output** |
|------------------------|
| `for(int i=0; i<10; i++) {` <br /> `double x;` <br /> `y[i] = Factor.Random(prior);` <br /> `z[i] = Factor.Gaussian(y[i],x);` <br /> `}` | `double[]] x = new double[10];` <br /> `for(int i=0; i<10; i++) {` <br /> `y[i] = Factor.Random(prior);` <br /> `}` <br /> `for(int i=0; i<10; i++) {` <br /> `z[i] = Factor.Gaussian(y[i],x[i]);` <br /> `}` |
