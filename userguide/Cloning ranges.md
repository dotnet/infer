---
layout: default 
--- 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Cloning ranges

Sometimes you want to access all pairs of array elements in an expression. For example, suppose you wanted to define a two-dimensional array **outerProduct** where outerProduct\[i,j\] = y\[i\]*y\[j\]. How do you declare ranges i and j? You might try the following:

```csharp
Range i = new Range(3);
Range j = new Range(3);
VariableArray<double> y = Variable.Array<double>(i);
y.ObservedValue = new double[] { 1, 2, 3 };
VariableArray2D<double> outerProduct = Variable.Array<double>(i, j);
outerProduct[i, j] = y[i]*y[j];
```
But this is not accepted by Infer.NET, because y was declared with range i and cannot be indexed by range j, even though j has the same size. In order to tell Infer.NET that range j is compatible with range i, you create it by cloning range i:

```csharp
Range i = new Range(3);
Range j = i.Clone();
VariableArray<double> y = Variable.Array<double>(i);
y.ObservedValue = new double[] { 1, 2, 3 };
VariableArray2D<double> outerProduct = Variable.Array<double>(i, j);
outerProduct[i, j] = y[i]*y[j];
```
This version works as desired.

### **Mixed Membership Stochastic Blockmodel**

Another good example of where you need to clone a Range is in the **'Mixed Membership Stochastic Blockmodel**' of Airoldi et al. which attempts to model relational information among nodes in a network(for example individuals in an social network). Given **N** nodes, **K** blocks and a binary matrix of known relationships between nodes (**YObs** in the code below), the aim is to learn the block membership mixture **pi** for each node at the same time as learning the pairwise relationship **B** between different blocks. Because this is a model of relationships, nodes are processed in pairs (initiator and receiver) and these nodes both index the same array variable **pi**. However the nested `Variable.ForEach` statements require different ranges; the solution is to clone the range as highlighted below. Similarly the link matrix **B** requires different ranges for the nested `Variable.Switch` statements.

```csharp
// Observed interaction matrix  
var YObs = new bool[5][];  
YObs[0] = new bool[] { false, true, true, false, false };  
YObs[1] = new bool[] { true, false, true, false, false };  
YObs[2] = new bool[] { true, true, false, false, false };  
YObs[3] = new bool[] { false, false, false, false, true };  
YObs[4] = new bool[] { false, false, false, true, false };int K = 2; // Number of blocks  
int N = YObs.Length; // Number of nodes  

// Ranges  
Range p = new Range(N).Named("p"); // Range for initiator  
Range q = p.Clone().Named("q"); // Range for receiver  
Range kp = new Range(K).Named("kp"); // Range for initiator block membership  
Range kq = kp.Clone().Named("kq"); // Range for receiver block membership  

// The model  
var Y = Variable.Array(Variable.Array<bool>(q), p); // Interaction matrix  
var pi = Variable.Array<Vector>(p).Named("pi"); // Block-membership probability vector  
pi[p] = Variable.DirichletUniform(kp).ForEach(p);  
var B = Variable.Array<double>(kp, kq).Named("B"); // Link probability matrix  
B[kp, kq] = Variable.Beta(1, 1).ForEach(kp, kq);  

using (Variable.ForEach(p)) { 
    using (Variable.ForEach(q)) { 
        var z1 = Variable.Discrete(pi[p]).Named("z1"); // Draw initiator membership indicator 
        var z2 = Variable.Discrete(pi[q]).Named("z2"); // Draw receiver membership indicator 
        z2.SetValueRange(kq); 
        using (Variable.Switch(z1)) 
            using (Variable.Switch(z2))  
                Y[p][q] = Variable.Bernoulli(B[z1, z2]); // Sample interaction value 
    }  
}  

// Initialise to break symmetry  
var piInit = new Dirichlet[N];  
for (int i = 0; i < N; i++) { 
    Vector v = Vector.Zero(K); 
    for (int j = 0; j < K; j++) v[j] = 10 * Rand.Double();  
    piInit[i] = new Dirichlet(v);  
}  

// Hook up the data  
Y.ObservedValue = YObs;  

// Infer  
var engine = new InferenceEngine(new VariationalMessagePassing());  
pi.InitialiseTo(Distribution<Vector>.Array(piInit));  
var posteriorPi = engine.Infer<Dirichlet[]>(pi);  
var posteriorB = engine.Infer<Beta[,]>(B);
```
