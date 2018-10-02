---
layout: default 
--- 
 
[Infer.NET user guide](index.md) : [Sharing variables between models](Sharing variables between models.md)

## Shared variable arrays

The [**SharedVariable**](../apiguide/api/Microsoft.ML.Probabilistic.Models.SharedVariable-1.html) class provides support for array and jagged array variables as well as scalar variables. This section gives an example of how to define an array shared variable. A detailed example (which shows many aspects of shared variables and other scalability mechanisms) is given in the [LDA example](Latent Dirichlet Allocation.md), and this section shows some simplified snippets from that example. The following snippet shows the shared variable definition of a Dirichlet-distributed Vector random variable _Phi_.

```csharp
int numTopics = 5;  
int sizeVocab = 10;  
Range T = new Range(numTopics);SharedVariableArray<Vector> Phi = SharedVariable<Vector>.Random  
    (T, CreateUniformDirichletArray(numTopics, sizeVocab);
```

Whenever you define a shared variable, you need to call the Random method on the `SharedVariable<>` class where the generic parameter is the domain type - Vector in this case. This method requires (for a 1-D array) the [range](Arrays and ranges.md) of the random variable array (in this case T), and a prior. The form of the prior is in the form of a distribution array. The `Dirichlet` distribution in Infer.NET is a class rather than a struct, and as such, the type of a `Dirichlet` distribution array is shown by the following alias which you can place in your code (struct distributions such as Gaussian are of type `DistributionStructArray<,>`):

```csharp
using DirichletArray=DistributionRefArray<Dirichlet, Vector>;
```

Now we need to create a prior of this type. Here is the code which will create a uniform prior; it first of all creates a .NET array of uniform Dirichlet distributions and then calls the Array<> method on the Distribution<Vector\> class to obtain the correct type for the prior:

```csharp
private static DirichletArray CreateUniformDirichletArray(int length, int valueLength Sparsity sparsity)  
{  
    Dirichlet[] result = new Dirichlet[length];  
    for (int i=0; i < length; i++)  
        result[i] = Dirichlet.Uniform(valueLength, sparsity);  
    return (DirichletArray)Distribution<Vector>.Array<Dirichlet>(result);  
}
```

The variable **sparsity** is a class member of type Sparsity specifying a particular sparsity scheme (dense, sparse or sparse with tolerance). Once you have defined your shared variable, you can get a copy of it for each model. For example:

```csharp
Model DocModel = new Model(numBatches);  
VariableArray<Vector> PhiDoc = Phi.GetCopyFor(DocModel);
```

Inference is done as for the [scalar case](Sharing variables between models.md) by doing several passes over the data, and by calling InferShared on the model.

Finally, note that marginal will be in the form of a Infer.NET distribution array rather than an array of distributions. Internally Infer.NET uses distribution arrays (classes which support distribution interfaces as well as array interfaces) to facilitate efficient treatment of message passing operations. If you want a .NET array, you will need to explicitly convert it:

```csharp
postPhi = Distribution.ToArray<Dirichlet[]>(Phi.Marginal<IDistribution<Vector[]>>());
```

### Initialisation

In some models, for example LDA or factor analysis, you will need to initialise the inference in order to break symmetry. For shared variable implementations of a model, initialisation is done on the copy. In the following example, an initialiser is set up as an observed variable so we can change it at run time without triggering a model recompile (this is important because we will want to change it from pass to pass). Here is the definition:

```csharp
Variable<IDistribution<Vector[]>> PhiInit = Variable.New<IDistribution<Vector[]>>();  
PhiBeta.InitialiseTo(PhiInit);
```

Then at run time, we set the value. It is recommended that after the first pass of the data, the initialiser is just set to the current marginal:

```csharp
if (pass == 0)  
    PhiInit.ObservedValue = phiInit;else PhiInit.ObservedValue = Phi.Marginal<IDistribution<Vector[]>>();
```

Full details can be found in the code for the [LDA shared variable example](Shared variable LDA.md).
