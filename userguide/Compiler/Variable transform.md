---
layout: default
---
[Infer.NET user guide](../index.md) : [Infer.NET development](../Infer.NET development.md) : [Infer.NET compiler design](../Infer.NET compiler design.md)

## Variable transform

Inserts a variable factor immediately after an assignment to a random variable. The factor clones the variable into definition, use, and marginal variables. All uses of the variable are changed to refer to the 'use' clone, and Infer() calls are changed to refer to the 'marginal' clone. The clone keeps the same indexing as the left-hand side of the assignment. For example:

| Input | Output |
|----------------|
| `bool firstCoin = Factor.Random<bool>(vBernoulli0);` <br /> `InferNet.Infer(firstCoin);` | `bool firstCoin_marginal;` <br /> `bool firstCoin_use;` <br /> `bool firstCoin = Factor.Random<bool>(vBernoulli0);` <br /> `firstCoin_use = Factor.Variable<bool>(firstCoin, firstCoin_marginal);` <br /> `InferNet.Infer(firstCoin_marginal);` |
| `double[] Alpha = new double[10];` <br /> `for(int d = 0; d<10; d++) {` <br /> `Alpha[d] = Factor.Random<double>(vGamma1);` <br /> `}` | `double[] Alpha = new double[10];` <br /> `double[] Alpha_marginal = new double[10];` <br /> `double[] Alpha_use = new double[10];` <br /> `for(int d = 0; d<10; d++) {` <br /> `Alpha[d] = Factor.Random<double>(vGamma1);` <br /> `Alpha_use[d] = Factor.Variable<double>(Alpha[d], Alpha_marginal[d]);` <br /> `}` |
| `double[,] Z = new double[400, 10];` <br /> `for(int d = 0; d<10; d++) {` <br /> `for(int N = 0; N<400; N++) {` <br /> `Z[N, d] = Factor.Random<double>(vGaussian1);` <br /> `}` <br /> `}` | `double[,] Z = new double[400, 10];` <br /> `double[,] Z_marginal = new double[400, 10];` <br /> `double[,] Z_use = new double[400, 10];` <br /> `for(int d = 0; d<10; d++) {` <br /> `for(int N = 0; N<400; N++) {` <br /> `Z[N, d] = Factor.Random<double>(vGaussian1);` <br /> `Z_use[N, d] = Factor.Variable<double>(Z[N, d], Z_marginal[N, d]);` <br /> `}` <br /> `}` |

As an optimization, a variable factor is not inserted if the variable is derived and not being inferred. Thus the helper variables created by previous transforms will not get variable factors.
 
The transform attaches ChannelInfo attributes to the clones, and an IsVariableFactor attribute to the variable factor expression.