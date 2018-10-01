---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md) : [Latent Dirichlet Allocation](Latent Dirichlet Allocation.md)

## LDA implementation

Page 1 \| [Page 2](LDA Inference and Prediction.md) \| [Page 3](Shared variable LDA.md)

### Description

This page describes a simple, reference implementation of LDA (with fixed Dirichlet priors) in Infer.NET. For background information, refer to [the main page for the LDA example](Latent Dirichlet Allocation.md). The implementation described on this page assumes that the full graphical model along with all the inference messages can be stored in memory at once. The large number of these messages coupled with their large size means that only moderately sized problems can be solved with this implementation. This is partially alleviated by an inference option that supports sparse representations of these messages, and this allows much larger size problems to be handled in cases where topic distributions over words are truly expected to be sparse. For an implementation which also scales to support large data sets, refer to [the shared variable version of LDA](Shared variable LDA.md).

Both LDAModel and LDAShared are classes in the LDA project in [src/Examples/LDA](https://github.com/dotnet/infer/tree/master/src/Examples/LDA). The next section highlights aspects of the model definition in the LDAModel class. [Page 2](LDA Inference and Prediction.md) shows inference and prediction capabilities. [Page 3](Shared variable LDA.md) describes the LDAShared class.

### LDA Model Constructor

The model is defined in the constructor method for the `LDAModel` class which has the following arguments:  

| **Argument** | **Type** | **Description** | **Examples** |
|----------------------------------------------------------|
| sizeVocab | `int` | The size of the vocabulary | 10000 |
| numTopics | `int` | The number of latent topics | 20 |

### LDA Definition

First, a [Range](Arrays and ranges.md) is defined for each plate in the graphical model. For words in documents, note that different documents will have different numbers of observed words, so in that case we need a jagged Range WInD. The `NumWordsInDoc` array is specified at inference time.

```csharp
Range D = new Range(NumDocuments).Named("D");  
Range W = new Range(SizeVocab).Named("W");  
Range T = new Range(NumTopics).Named("T");  
NumWordsInDoc = Variable.Array<int>(D).Named("NumWordsInDoc");  
Range WInD = new Range(NumWordsInDoc[D]).Named("WInD");
```

Once we have defined the ranges, we can define the variables in the model. The main variables are the _Theta_ (the mixture parameters for the per-document distributions over topics), and the _Phi_ (the mixture parameters for the per-topic distributions over words); these are drawn respectively from Dirichlet priors _ThetaPrior_ and _PhiPrior_ which are set at run time (when training, these are set by specifying an _alpha_ and _beta_ which are the parameters of symmetric Dirichlet priors). The word observations are in the form of arrays of arrays of integer variables (indices into the vocabulary for each document). An additional jagged array variable gives the corresponding word counts. Indices and counts are observed at inference time.

```csharp
Theta = Variable.Array<Vector>(D);  
Theta.SetSparsity(ThetaSparsity);  
Theta.SetValueRange(T);  
ThetaPrior = Variable.Array<Dirichlet>(D).Named("ThetaPrior");  
Theta[D] = Variable<Vector>.Random(ThetaPrior[D]);  
Phi = Variable.Array<Vector>(T);  
Phi.SetSparsity(PhiSparsity);  
Phi.SetValueRange(W);  
PhiPrior = Variable.Array<Dirichlet>(T).Named("PhiPrior");  
Phi[T] = Variable<Vector>.Random(PhiPrior[T]);  
Words = Variable.Array(Variable.Array<int>(WInD), D).Named("Words");  
WordCounts = Variable.Array(Variable.Array<double>(WInD), D).Named("WordCounts");
```

The remainder of the model, defines the generative process whereby each observed word in a document is considered as a sample from a topic's word distribution, where the topic is a sample from the document's topic distribution, as described on [the main LDA page.](Latent Dirichlet Allocation.md)

```csharp
using (Variable.ForEach(D))  
{  
    using (Variable.ForEach(WInD))  
    {  
        using (Variable.Repeat(WordCounts[D][WInD]))  
        {  
            var topic = Variable.Discrete(Theta[D]);  
            using (Variable.Switch(topic))  
                Words[D][WInD] = Variable.Discrete(Phi[topic]);  
        }  
    }  
}
```

This completes the main definition of the model, but there are a couple of other things we need to set up before doing inference. Firstly note that we need to break symmetry when we're doing inference because the model is symmetric with respect to the latent topics. We will do this by initialising the messages out of _Theta_. We will do this in a general way by defining a variable _ThetaInit_ which we will observe at inference time to be a random message in the form of a distribution over a Vector\[\] domain. The advantage of making _ThetaInit_ a variable is that we can modify it without causing a model recompilation.

```csharp
ThetaInit = Variable.New<IDistribution<Vector[]>>();  
Theta.InitialiseTo(ThetaInit);
```

Lastly, we would like to retrieve the evidence from the model so that we can compare models - for example we might want to optimise the _alpha_ and _beta_ hyper-parameters. We can compute [model evidence](Computing model evidence for model selection.md) by creating a mixture of the model with an empty model as follows:

```csharp
var Evidence = Variable.Bernoulli(0.5).Named("Evidence");  
VariableIfBlock evidenceBlock = Variable.If(Evidence);  
    ... 
    main model code  
    ... 
evidenceBlock.CloseBlock();
```

Go to [Page 2](LDA Inference and Prediction.md) for inference and prediction usage. Go to [Page 3](Shared variable LDA.md) for a scalable shared variable version of this model.

<br/>
Page 1 \| [Page 2](LDA Inference and Prediction.md) \| [Page 3](Shared variable LDA.md)
