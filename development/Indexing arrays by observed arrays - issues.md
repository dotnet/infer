---
layout: default 
--- 
[Infer.NET development](index.md)

## Why special handling is needed

Consider the following MSL:

```csharp
double[] a = new double[N]; // random
int[] b = new int[M]; // observed
for(int j = 0; j < M; j++) {
  Constrain.Positive(a[b[j]]);
}
```

Without any special treatment, this would generate the following message-passing code:

```csharp
a_uses_F[0] = new Gaussian[N];
a_uses_B[0] = new Gaussian[N];
for(int j = 0; j < M; j++) {
  a_uses_B[0][b[j]] = op( a_uses_F[0][b[j]] );
}
```

To decide if this code is correct, consider the following cases:

*   **Permutation**  The elements of the b array are all distinct and M=N, i.e. b is a permutation of the numbers 1 to N.
*   **Subarray**  The elements of b are all distinct, but M<N.
*   **Duplicated**  Some indices appear multiple times in b (the most general case).

If b is a permutation, then the message-passing code is correct as it is.
If b is a subarray, and a_uses_B[0] is initialized to uniform, then the code does produce the correct posterior distribution for a, but it has two other problems:

1. The evidence for the model will be wrong because the evidence contribution of 'a' will add up all N elements of a_uses_B[0], including the uniform ones, while it should only add the M elements that were actually given a message.
2. Memory is wasted since a_uses_B[0] allocates memory for N elements while only M elements are used.

If b is duplicated, then the code has two additional problems:

1. Messages with the same b[j] will overwrite each other in the a_uses_B[0] array. Instead they should be multiplied together (and the resulting scale factor added to the evidence). 
2. For EP, the message from 'a' to the factor should multiply in the other messages received by the factor for the same array element. For example, suppose b[j]=0 for all j. Then a[0] is being used M times, and will receive M messages. The message from a[0] to a factor instance should include all the other M-1 messages.

#### Current implementation

Currently, the subarray case is handled by inserting an intermediate factor Factor.Subarray. This factor doesn't change any of the messages but simply provides a correction to the evidence based on the lengths M and N. This factor is inserted even for permutations, where it provides no evidence correction. The code above is transformed into:

```csharp
double[] a_b = new double[M];
a_b = Factor.Subarray(a, b);
for(int j = 0; j < M; j++) {
  Constrain.Positive(a_b[j]);
}
```

The duplicated case is handled by inserting an intermediate factor Factor.GetItems, in place of Factor.Subarray. This factor takes care of multiplying the messages for duplicate indices, in both directions. It is currently implemented using an O(N+M) algorithm but it could be changed to run in O(M).

Even with these intermediate factors, memory is wasted since the a_uses_B[0] array is always length N.

The transformation is currently done in the API as the model is being constructed. However, it would be better to do this as a compiler transform, because the expression a[b[j]] could arise in various ways that are hard to detect in the API. For example, suppose instead of observing b[j], we switch on b[j]. It would also allow more errors to be detected in the API.

#### More complex indexing

The inefficiency of the current implementation grows as the indexing gets more complex. Consider the following MSL (indexing by a jagged array):

```csharp
double[] a = new double[N]; // random
int[][] b = new int[M][L]; // observed
for(int j = 0; j < M; j++) {
  for(int k = 0; k < L; k++) {
    Constrain.Positive(a[b[j][k]]);
  }
}
```

This gets transformed into:

```csharp
double[][] a_b = new double[M][L];
for(int j = 0; j < M; j++) {
  a_b[j] = Factor.GetItems(a, b[j]);  
}
for(int j = 0; j < M; j++) {
  for(int k = 0; k < L; k++) {
    Constrain.Positive(a_b[j][k]);
  }
}
```

This code gives the correct answer. However, the original code has cost O(N+ML). In the transformed code, the entire array 'a' is used by M GetItems factors, increasing the cost to O(NM+ML). For example, if L=1 and M=N, this increases the cost from O(N) to O(N^2).

Here is another problem case (indexing a jagged array, can make a similar example for indexing a 2D array):

```csharp
double[][] a = new double[N][P]; // random
int[] b = new int[M]; // observed
int[] c = new int[M]; // observed
for(int j = 0; j < M; j++) {
   Constrain.Positive(a[b[j]][c[j]]);
}
```

This gets transformed into:

```csharp
double[][] a_b = new double[M][P];
a_b = Factor.GetItems(a, b);
double[] a_b_c = new double[M];
for(int j = 0; j < M; j++) {
  a_b_c[j] = Factor.GetItem(a_b[j], c[j]);
}
for(int j = 0; j < M; j++) {
  Constrain.Positive(a_b_c[j]);
}
```

The original code has cost O(NP+M). The transformed code has cost O(NP+MP), which is worse if M > N.

Putting these together, we can index a jagged array by jagged indices:

```csharp
double[][] a = new double[N][P]; // random
int[][] b = new int[M][K]; // observed
int[][] c = new int[M][L]; // observed
for(int j = 0; j < M; j++) {
  for(int k = 0; k < K; k++) {
    for(int l = 0; l < L; l++) {
      Constrain.Positive(a[b[j][k]][c[j][l]]);
    }
  }
}
```

This gets transformed into:

```csharp
double[][][] a_b = new double[M][K][P];
for(int j = 0; j < M; j++) {
  a_b[j] = Factor.GetItems(a, b[j]);
}
double[][][] a_b_c = new double[M][K][L];
for(int j = 0; j < M; j++) {
  for(int k = 0; k < K; k++) {
    a_b_c[j][k] = Factor.GetItems(a_b[j][k], c[j]);
  }
}
for(int j = 0; j < M; j++) {
  for(int k = 0; k < K; k++) {
    for(int l = 0; l < L; l++) {
      Constrain.Positive(a_b_c[j][k][l]);
    }
  }
}
```

The original code has cost O(NP+MKL). The transformed code has cost O(NPM+MKP+MKL).

#### Alternative design

Create a new DistributionSubarray type, which represents a distribution over an array of length N but only stores M elements, the rest being implicitly uniform. (There could be a specialized version for M=1.) Change a_uses_B[0] and a_uses_F[0] to use this type. Using this message type automatically fixes the evidence computation, so Factor.Subarray is never needed. We still need Factor.GetItems to deal with duplication. It should be changed to use an O(M) algorithm.

The case of indexing by a jagged array is automatically efficient since Factor.GetItems(a, b[j]) will use DistributionSubarrays of size L, making the cost O(N+ML).

The case of indexing a jagged array would be handled by a new GetItemJagged factor, which works as follows:

```csharp
GetItemJagged(a, b, c) = a[b][c]
```

So the transformed code would be:

```csharp
double[] a_b_c = new double[M];
for(int j = 0; j < M; j++) {
  a_b_c[j] = Factor.GetItemJagged(a, b[j], c[j]);
}
for(int j = 0; j < M; j++) {
  Constrain.Positive(a_b_c[j]);
}
```

`GetItemJagged` will use DistributionSubarrays of size 1, so the cost is O(NP+M).
The case of indexing a jagged array by jagged indices would be handled by a new GetItemsJagged factor, which works as follows:

```csharp
GetItemsJagged(a, b, c)[k][l] = a[b[k]][c[l]]
```

So the transformed code would be:

```csharp
double[][][] a_b_c = new double[M][K][L];
for(int j = 0; j < M; j++) {
  a_b_c[j] = Factor.GetItemsJagged(a, b[j], c[j]);
}
for(int j = 0; j < M; j++) {
  for(int k = 0; k < K; k++) {
    for(int l = 0; l < L; l++) {
      Constrain.Positive(a_b_c[j][k][l]);
    }
  }
}
```

`GetItemsJagged` will use jagged DistributionSubarrays of size KL, so the cost is O(NP+MKL).