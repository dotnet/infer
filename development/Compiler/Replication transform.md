---
layout: default
---
[Infer.NET development](../index.md) : [Compiler transforms](../Compiler transforms.md)

## Replication transform

Ensures that random variable expressions in loops are fully-indexed i.e. they are indexed by the loop variables of all containing loops.
This is achieved by replicating any expression which is not fully-indexed across all unindexed loops. For example:

| Input | Output |
|----------------|
| `for(int i=0;i<10;i++) {` <br /> `data[i] = Factor.Gaussian(mean, precision);` <br /> `}` | `for(int i=0;i<10;i++) {` <br /> `data[i] = Factor.Gaussian(mean_rep[i], precision_rep[i]);` <br /> `}` |

The transformation ensures that no two iterations of the loop refer to the same variable. Thus they can all be processed in parallel by subsequent transforms.
 
If the variable being referenced was declared inside loop i, then it is implicitly indexed by i already and does not need to be replicated over i. 
 
The variable 'means' in the expression means[indices[i]] is not considered to be indexed by i. This is because different values of i might lead to the same element of means being selected. Thus to be safe, we need to replicate means over i. In this particular case, we can use Factor.GetItems(means, indices).
 
When the transform encounters a variable reference or array indexer expression, it proceeds as follows:

1. Collect the set of loops that the expression appears in, excluding loops that the variable was declared in. This reduced set of loops is called the "reference loop context". 
2. If a loop variable appears as the left hand side in a parent condition, exclude the loop from replication.
3. If a loop variable appears as an index in the expression, exclude the loop from replication.
4. Replicate the expression over the remaining reference loops, starting from the outermost. Let the current reference loop have loop variable i. Replicating the expression over i involves 3 steps:
    1. We know that i does not appear as an index in the expression, but it may appear in some other form. For example, the expression may be means[indices[i]]. Split the expression into the largest target that does not involve i, and then the remaining indices. For example, means[indices[i]] splits into "means" and "[indices[i]]". If loop j is outside of i, then means[j][indices[i]][k] would split into "means[j]" and "[indices[i]][k]" (regardless of whether k is outside of i). If loop j was inside of i, then means[j][indices[i]][k] would split into "means" and "[j][indices[i]][k]".
    2. Replicate the target of the split expression, by inserting a statement such as: means_j_rep_i = Replicate(means[j], size_i);
    3. Replace the target of the split expression with the replicated array, indexed by i. Thus means[j][indices[i]][k] becomes means_j_rep_i[i][indices[i]][k]. This becomes the new expression to replicate in the remaining reference loops.

| RHS reference | In loop over | Output | Replication |
|-----------------------------------------------------|
| x | [i][j] or [i,j] | x_rep_i[j] | `x_rep = Replicate<double>(x);` <br /> `x_rep_i = Replicate<double>(x_rep[i]);` |
| barray | [i][j] | barray_rep_i[j] | `barray_rep = Replicate<bool[]>(barray);` <br /> `barray_rep_i = Replicate<bool[]>(barray_rep[i]);` |
| barray[0] | [i] | barray_0_rep[i] | `barray_0_rep = Replicate<bool>(barray[0]);` |
| barray[i] | [i,j] or [j,i] | barray_i_rep[j] | `barray_i_rep = Replicate<bool>(barray[i]);` | 
| barray[i] | [i][j] | barray_i_rep[j] | `barray_i_rep = Replicate<bool>(barray[i]);` |
| jarray | [i] | jarray_rep[i] | `jarray_rep = Replicate<bool[][]>(jarray);` |
| jarray[i] | [i,j] | jarray_i_rep[j] | `jarray_i_rep = Replicate<bool[]>(jarray[i]);` |
| jarray[i][0] | [i,j] | jarray_i_0_rep[j] | `jarray_i_0_rep = Replicate<bool>(jarray[i][0]);` |
| jarray[j][i] | [i,j] | jarray[j][i] | _none_ |
| jarray[i][j] | [i][j,k] | jarray_i_j_rep[k] | `jarray_i_j_rep = Replicate<bool>(jarray[i][j]);` |
| jarray[i][k] | [i][j,k] | jarray_i_rep[j][k] | `jarray_i_rep = Replicate<bool[]>(jarray[i]);` |
| jarray[k][i] | [i,j,k] | jarray_rep[j][k][i] | `jarray_rep = Replicate<bool[][]>(jarray);` |
| matrix[i,k] | [i,j,k] | matrix_rep[j][i,k] | `matrix_rep = Replicate<bool[,]>(matrix);` |
| matrix[i,k] | [i][j,k] | matrix_rep[j] | `matrix_rep = Replicate(matrix[i,k]);` |
| matrix[i,l] | [i,j][k,l] | matrix_rep_rep[k][i,l] | `matrix_rep = Replicate<bool[,]>(matrix);` <br /> `matrix_rep_rep = Replicate<bool[,](matrix_rep[j]);` |
| matrix[i,l] | [i][j][k,l] | matrix_rep_rep[k] | `matrix_rep = Replicate<bool>(matrix[i,l]);` <br /> `matrix_rep_rep = Replicate<bool>(matrix_rep[j]);` |
| jarray[match1]...[matchLast][const] | [match1]...[matchLast,extra1][extra2] | jarray_match_const_rep[extra1] <br /> jarray_extra1_rep[extra2] | `jarray_match_const_rep = Replicate<bool[unmatched]>(jarray[match1]...[matchLast][const]);` <br /> `jarray_extra1_rep = Replicate<bool[][unmatched]>(jarray_match_const_rep[extra1]);` |

#### Longer Example (all rows are part of the same program)

```csharp
double precision = 1.0;
int arrayLength = 2;
double scalar = Gaussian.Sample(0.0, precision);
double[] array = new double[arrayLength];
double[] items = new double[scalarGiven];
items = Factor.GetItems(array, arrayGiven);
double[,] array2D = new double[arrayLength,scalarGiven];
```

| Input | Output |
|----------------|
| `for (int i = 0; i < array.Length; i++) {` <br /> `array[i] = Gaussian.Sample(scalar, precision);` <br /> `}` | `double[] scalar_rep = new double[array.Length];` <br /> `scalar_rep = Factor.Replicate<double>(scalar);` <br /> `for (int i = 0; i < array.Length; i++) {` <br /> `array[i] = Gaussian.Sample(scalar_rep[i], precision);` <br /> `}` |
| `for (int j = 0; j < array2D.GetLength(0); j++) {` <br /> `for (int k = 0; k < scalarGiven; k++) {` <br /> ` array2D[j,k] = Factor.Sum(scalar, items[k]);` <br /> `}` <br /> `}` | `double[,] scalar_rep = new double[array2D.GetLength(0), scalarGiven];` <br /> `scalar_rep = Factor.ReplicateNd<double>(scalar);` <br /> `double[][] items_rep = new double[array2D.GetLength(0)][];` <br /> `items_rep = Factor.Replicate<double[]>(items);` <br /> `for (int j = 0; j < array2D.GetLength(0); j++) {` <br /> `for (int k = 0; k < scalarGiven; k++) {` <br /> ` array2D[j, k] = Factor.Sum(scalar_rep[j,k], items_rep[j][k]);` <br /> `}` <br /> `}` |
