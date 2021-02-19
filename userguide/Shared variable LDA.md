---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md) : [Latent Dirichlet Allocation](Latent Dirichlet Allocation.md)

## LDA shared variable implementation

[Page 1](Standard LDA.md) \| [Page 2](LDA Inference and Prediction.md) \| Page 3

### Description

This page describes the shared variable implementation of LDA (with fixed Dirichlet priors) in Infer.NET. For background information, refer to [the main page for the LDA example](Latent Dirichlet Allocation.md). The implementation described on this page uses Infer.NET's [shared variable mechanism](Sharing variables between models.md) to split the overall model into a bunch of smaller sub-models each one of which requires much less memory than the whole; this partitioning of the model largely corresponds to splitting the data set into many small chunks. The per-topic word mixture random variables are shared amongst all copies, and the Infer.NET shared variable mechanism hides the calculation that is needed to flange together all the copies.

It is assumed that you have read through the description of the [standard LDA implementation](Standard LDA.md). Both `LDAModel` and `LDAShared` are classes in the LDA project in [src/Examples/LDA](https://github.com/dotnet/infer/tree/master/src/Examples/LDA). The next section highlights aspects of the  `LDAShared` model definition, concentrating on those aspects which differ from the standard implementation. A later section highlights the differences between between the `LDAShared` and `LDAModel` class with respect to inference.

### LDAShared Model Definition

The model is defined in the constructor method for the `LDAShared` class. The constructor method has exactly the same arguments as the `LDAMode`l class, but with the addition of one more argument:

| **Argument** | **Type** | **Description** |
|-------------------------------------------|
| numBatches | `int` | The number of batches. A good value to set this to is the square root of the number of documents. |

However, the model construction code differs a little bit due to the presence of the shared variables. There are two variables in the model which are shared across all copies - these are as follows. The evidence variable requires special treatment in a shared variable environment and must be marked as such:

```csharp
Evidence = SharedVariable<bool>.Random(new Bernoulli(0.5));  
Phi = SharedVariable<Vector>.Random(T, CreateUniformDirichletArray(  
    numTopics, sizeVocab, phiSparsity));  
Evidence.IsEvidenceVariable = true;
```

Note that a prior for a shared variable must be specified when the shared variable is defined, and the prior cannot be a variable itself. As the domain of _Phi_ is a `Vector` array, the prior must be in the form of a distribution over a Vector array - the code shows the details. We set this to uniform because, in fact, we will be using the shared variable `SetDefinition` mechanism to define _Phi_ in terms of a prior that we can change at runtime. _Phi_ is defined in its own very simple model which we will call _PhiDefModel_. There will just be one copy of this model as opposed to the multiple copies of the document part of the model. This first model is defined as follows:

```csharp
PhiDefModel = new  Model(1).Named("PhiDefModel");  
IfBlock evidencePhiDefBlock = null;  
EvidencePhiDef = Evidence.GetCopyFor(PhiDefModel).Named("EvidencePhiDef");  
evidencePhiDefBlock = Variable.If(EvidencePhiDef);  
PhiDef = Variable.Array<Vector>(T).Named("PhiDef");  
PhiDef.SetSparsity(PhiSparsity);  
PhiDef.SetValueRange(W);  
PhiPrior = Variable.Array<Dirichlet>(T).Named("PhiPrior");  
PhiDef[T] = Variable<Vector>.Random(PhiPrior[T]);  
Phi.SetDefinitionTo(PhiDefModel, PhiDef);  
evidencePhiDefBlock.CloseBlock();
```

The first line instantiates this sub-model and says how many copies there will be. As this is the model which defines _Phi,_ there will just be one copy (a defining model for a shared variable must always have one copy only). The remainder of the code is identical to the corresponding code for the `LDAModel` constructor except for the lines shown in bold: (a) we need to get a copy of the _Evidence_ variable for this model, as _Evidence_ is a shared variable, and (b) we use the `SetDefinition` method on the _Phi_ shared variable in order to give its definition in terms of non-shared variables. 

The second sub-model represents that part of the full model which relates to the data. This is defined as follows (refer to the source code for the more efficient power plate version of this model):

```csharp
DocModel = new  Model(numBatches).Named("DocModel");  
IfBlock evidenceDocBlock = null;  
EvidenceDoc = Evidence.GetCopyFor(DocModel).Named("EvidenceDoc");  
evidenceDocBlock = Variable.If(EvidenceDoc);  
Theta = Variable.Array<Vector>(D).Named("Theta");  
Theta.SetSparsity(ThetaSparsity);  
Theta.SetValueRange(T);  
ThetaPrior = Variable.Array<Dirichlet>(D).Named("ThetaPrior");  
Theta[D] = Variable<Vector>.Random(ThetaPrior[D]);  
PhiDoc = Phi.GetCopyFor(DocModel);  
PhiDoc.SetSparsity(PhiSparsity);  
PhiDoc.SetValueRange(W);  
Words = Variable.Array(Variable.Array<int>(WInD), D).Named("Words");  
using (Variable.ForEach(D))  
{
  using (Variable.ForEach(WInD))  
  {  
      using (Variable.Repeat(WordCounts[D][WInD]))  
      {
        Variable<int> topic = Variable.Discrete(Theta[D]).Named("topic");
        using (Variable.Switch(topic))  
          Words[D][WInD] = Variable.Discrete(PhiDoc[topic]);  
      }  
  }  
}  
evidenceDocBlock.CloseBlock();
```

The first line instantiates this sub-model and specifies how many copies there will be - this is set to the number of batches specified in the call to the constructor. The remainder of the code is identical to the corresponding code for the `LDAModel` constructor, except for the highlighted lines: (a) we need to get a copy of the _Evidence_ variable for this model, as _Evidence_ is a shared variable, and (b) we use the `GetCopyFor` method on the _Phi_ shared variable to get a copy of the variable for each copy of the sub-model. 

### Inference

Refer to the [Page 2](LDA Inference and Prediction.md) for inference and prediction usage. Here we just concentrate on the difference in implementation between the `LDAModel` and `LDASparse` classes. The implementation of this method differs from the `LDAModel` class in that the calls to the different model copies are scheduled explicitly. The general schedule is of the form of a number of outer passes, and, within each pass, a pass through all the data batches:

```csharp
for (int pass = 0; pass < NumPasses; pass++)  
{  
    engine.NumberOfIterations = ... 
    for (int batch = 0; batch < NumBatches; batch++)  
    {  
        // Set up the observed values ... 
        if (pass == 0)  
            PhiInit.ObservedValue = phiInit;  
        else PhiInit.ObservedValue = SharedPhi.Marginal<IDistribution<Vector[]>>();  
        SharedModel.InferShared(engine, batch);  
        // Store the theta posteriors (not shared across batches)  
        var postThetaBatch = engine.Infer<Dirichlet[]>(Theta);  
        for (int i = 0, j=startDoc; j < endDoc; i++, j++)  
            postTheta[j] = postThetaBatch[i];  
    }  
}
```

Note that during the first pass, the symmetry-breaking _Theta_ initialisation messages are set to random values, whereas in subsequent passes, the initialisation is more efficiently set to the current _Theta_ posterior marginal. Note also that the _Theta_ random variables are not shared across batches, and so are not handled by the `InferShared` method; instead we must retrieve these from the model using the standard inference call on the engine.

The number of internal iterations done for each batch can be explicitly set via the `IterationsPerPass` property as shown by the following example:

```csharp
var model = new LDAShared(batchCount, sizeVocab, numTopics, thetaSparsity, phiSparsity);  
model.IterationsPerPass = Enumerable.Repeat(10, 5).ToArray();
```

This specifies 5 passes of 10 iterations.

Once the passes are complete, the marginal posterior distributions of the shared variables can be retrieved as follows. Note the special treatment for the evidence variable:

```csharp
postPhi = Phi.Marginal<Dirichlet[]>();
return Model.GetEvidenceForAll(BetaModel, DocModel);
```

<br/>
[Page 1](Standard LDA.md) \| [Page 2](LDA Inference and Prediction.md) \| Page 3
