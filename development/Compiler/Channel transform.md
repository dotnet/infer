---
layout: default
---
[Infer.NET development](../index.md) : [Infer.NET compiler design](../Infer.NET compiler design.md)

## Channel transform

The channel transform replicates stochastic variables into arrays so that each reference refers to a unique array element, corresponding to an edge in the factor graph. Deterministic variables and variables with only one use are left unchanged.
 
This transform first runs ChannelAnalysisTransform to count the number of uses of each stochastic variable. Then it inserts a Replicate statement at every place where a stochastic variable with multiple uses is assigned to. A uses array is created, and every use of the variable is replaced with a reference to the next element of the uses array. 
 
To minimize the amount of replication, elements of the uses array may be shared when appropriate. For example, if different elements of an array are being used, then these will share the same 'use' of the whole array. If a variable is being used in disjoint deterministic condition contexts, then these uses will be shared, since only one branch can be active during a given run.
 
This transform also attaches a MarginalPrototype attribute to every variable that does not yet have one, by applying some default rules.

**THE OLD BEHAVIOR IS DOCUMENTED BELOW**

#### Variable declarations

Variable declarations are copied to make a definition and a marginal channel. An array for the _N_ uses of the variable is also declared. The three arrays are tied together with a `UsesEqualDef` factor. The examples below assume that the variable is marked as stochastic in each case.
All generated declarations are given `ChannelInfo` attributes which record information like the variable they were generated from, the type of the channel, the use count (for use channels) etc.
 
The `UsesEqualsDef` method call is marked with the `DeterministicVariable` attribute if the variable is defined by a deterministic factor.

| Input | Output |
|----------------|
| double x; | double x; <br /> double x_marginal; <br /> double[] x_uses = new double[N]; <br /> x_uses = Factor.UsesEqualDef<double\>(x, x_marginal); |
| bool[] barray; | bool[] barray; <br /> bool[] barray_marginal; <br /> bool[][] barray_uses = new bool[N][]; <br /> barray_uses = Factor.UsesEqualDef<bool[]>(barray, barray_marginal); |
| bool[][] jarray; | bool[][] jarray; <br /> bool[][] jarray_marginal; <br /> bool[][][] jarray_uses = new bool[N][][]; <br /> jarray_uses = Factor.UsesEqualDef<bool[][]>(jarray, jarray_marginal); |

Infer statements are modified to refer to the marginal channel variable i.e. Infer(a) transforms to Infer(a_marginal). 

#### Variable assignments

Assignments which allocate new arrays are duplicated to create corresponding arrays for the marginal and uses channels. This provides support for jagged arrays. The uses channel allocations are placed in a loop over the number of uses.

| Input | Output |
|----------------|
| x = Factor.Gaussian(0,1); | x = Factor.Gaussian(0,1); |
| barray = new bool[2]; | barray = new bool[2]; <br /> barray_marginal = new bool[2]; <br /> for (int _ind = 0; _ind < barray_uses.Length; _ind++) <br /> { <br />  barray_uses[_ind] = new bool[2]; <br /> } |
| jarray = new bool[2][]; | jarray = new bool[2][]; <br /> jarray_marginal = new bool[2][]; <br /> for (int _ind = 0; _ind < jarray_uses.Length; _ind++) <br /> { <br />  jarray_uses[_ind][] = new bool[2][]; <br /> } |
| jarray[i] = new bool[sizes[i]]; | jarray[i] = new bool[sizes[i]]; <br /> jarray_marginal[i] = new bool[sizes[i]]; <br /> for (int _ind = 0; _ind < jarray_uses.Length; _ind++) <br /> { <br />  jarray_uses[_ind][i] = new bool[sizes[i]]; <br /> } |

#### Variable references

A stochastic variable reference on the LHS of an assignment is left unchanged, but a check is made to ensure that this only happens once (i.e. there is no mutation). 
 
All other variable references are replaced by a different element of the variable's uses array. The literal indexing transform ensures that, where literal indexing occurs, there is only one level of indexing and only one reference to any array element.
 
The examples below assume the declarations given above:

| RHS reference | In loop over | Output | Effect on _N_ |
|-------------------------------------------------------|
| x | - | x_uses[n] | Increases by 1 | 
| barray | - | barray_uses[n] | Increases by 1 |
| barray[i] | [i] | barray_uses[n][i] | Increases by 1 |
| jarray[i][j] | [i,j] | jarray_uses[n][i][j] | Increases by 1 |
| barray[0] | - | barray_usesn[0][n_lit] | If _n___lit_ is not yet defined, set it to _N_ and increase _N_ by 1. Otherwise use the existing value of n_lit. |

