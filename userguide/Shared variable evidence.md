---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Sharing variables between models](Sharing variables between models.md)

## Computing evidence for models with shared variables

Computing evidence in a shared variable model requires special treatment. In [the non-shared variable case](Computing model evidence for model selection.md) we just created a `Variable<bool>` evidence variable and wrapped the whole model in an `IfBlock`. The log evidence of the model was then given by the log odds of the evidence variable's marginal.

The following snippets show how to deal with evidence in the shared variable case, using the [LDA shared variable example](Shared variable LDA.md). You must first explicitly mark the evidence variable as such by setting the shared variable's `IsEvidenceVariable` property to true:

```csharp
SharedVariable<bool> Evidence = SharedVariable<bool>.Random(new Bernoulli(0.5));  
Evidence.IsEvidenceVariable = true;
```

Then wrap each model with an IfBlock conditioned on the model's copy of the evidence variable:

```csharp
Model PhiDefModel = new Model(1);  
Variable<bool> EvidencePhiDef = Evidence.GetCopyFor(PhiDefModel);  
IfBlock evidencePhiDefBlock = Variable.If(EvidencePhiDef);  
... 
evidencePhiDefBlock.CloseBlock();  

Model DocModel = new Model(numBatches);  
Variable<bool> EvidenceDoc = Evidence.GetCopyFor(DocModel);  
IfBlock evidenceDocBlock = Variable.If(EvidenceDoc);  
... 
evidenceDocBlock.CloseBlock();
```

Finally, retrieve the log evidence by call the GetEvidenceForAll method, and passing in all models:

```csharp
double logEvidence = Model.GetEvidenceForAll(PhiDefModel, DocModel);
```
