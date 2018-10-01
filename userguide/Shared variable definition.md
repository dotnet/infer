---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Sharing variables between models](Sharing variables between models.md)

## Defining shared variables within a model

The standard definition of a shared variable requires that you specify a fixed prior for the variable. In some situations, for example when defining a hierarchical model, the variable that you want to share is defined in terms of other variables. In the simplest case (shown below) the variable may be defined in terms of a prior which which you want to set at run time. These situations are handled using the SetDefinitionTo method on a shared variable instance. This method allows the shared variable to be defined in terms of a non-shared variable. In this scenario, at least two models are defined. The first model defines the shared variable (and there must be only one copy of this model), and the other models use the shared variable (there will typically be many copies of the other models).

Here is a simple example from the [Shared variable LDA](Shared variable LDA.md). This snippet shows the shared variable _Phi_ being defined in terms of a non-shared variable _PhiPrior._ This is the defining model. Note that (a) the batch count for the defining model is required to be 1, and (b) the Random method on the SharedVariable instance requires a uniform prior - otherwise SetDefinitionTo will raise an exception.

```csharp
Model PhiDefModel = new  Model(1);  
SharedVariableArray<Vector> Phi = SharedVariable<Vector>.Random(  
    T, CreateUniformDirichletArray(numTopics, sizeVocab, ...))  
PhiPrior = Variable.Array<Dirichlet>(T);  
PhiDef[T] = Variable<Vector>.Random(PhiPrior[T]);  
Phi.SetDefinitionTo(PhiDefModel, PhiDef);
```

The shared variable _Phi_ can then be used as usual in any other models by using the GetCopyFor method:

```csharp
Model DocModel = new  Model(numBatches);  
VariableArray<Vector> PhiDoc = Phi.GetCopyFor(DocModel);
```