---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md) : [Latent Dirichlet Allocation](Latent Dirichlet Allocation.md)

## Inference and Prediction

[Page 1](Standard LDA.md) \| Page 2 \| [Page 3](Shared variable LDA.md)

This page shows to perform inference and prediction with your LDA model. Example test code for each of the usages is available in the project's [TestLDA project](LDA test program.md).

### Inference

Both the `LDAModel` class and the `LDAShared` class derive from the `ILDA` interface which has an inference method and also an Engine property of type `InferenceEngine`. To learn the mixture distributions in the LDA model, call the Infer method on an LDA instance. The code can be seen in `LDAModel.cs` and `LDAShared.cs`. In this usage, we observe the words in the documents, and want to infer the distributions over topic and word mixtures. You will need to provide the following arguments to the Infer method.

| **Argument** | **Type** | **Description** | **Example** |
|---------------------------------------------------------|
| wordsInDoc | `Dictionary<int,int>[]` | Array of dictionaries giving word counts for each document. Each dictionary is keyed by word index and the corresponding value is the document count for the word. | See the [LDA test program](LDA test program.md) for an example. |
| alpha | `double` | Parameterises _ThetaPrior_. | 150 / numTopics |
| beta | `double` | Parameterises _PhiPrior._ | 0.1 |

On return, the posteriors are available as follows:

| **Result** | **Type** | **Description** | **Comments** |
|--------------------------------------------------------|
| Log evidence | `double` | The log of the model evidence. This is the return value of the method, and can be used to tune hyper-parameters, and to compare models | The log evidence values can vary quite widely across data sets. You may want to normalise this by dividing by the sum of document lengths as shown in the example program. |
| postTheta | `Dirichlet[]` | The per-document posterior marginal distributions over topic mixtures | This can be used along with postPhi to create a predictive distribution for words in each of these documents. |
| postPhi | `Dirichlet[]` | The per-topic posterior marginal distributions over word mixtures | This can be used with the LDATopicModelInference class to infer the topic distribution of a new document. |

If you are running inference on an `LDAModel` instance, you can set the `NumberOfIterations` (and other parameters) on the Engine property of the class as normal. However, if you are running inference on an `LDAShared`  instance, the `NumberOfIterations` is ignored. Instead there is a special property "IterationsPerPass" which is of type `int[]`, and allows the number of passes, and the number of iterations per pass to be set. Refer to [Page 3](Shared variable LDA.md) for details on setting this, and on other differences between `LDAModel` and `LDAShared`.

### Prediction

One way of testing your model is to hold out some data for each training document, and see how likely the held-out words are - you can do this using evaluation measures such as [Perplexity](http://en.wikipedia.org/wiki/Perplexity) which can also be used to compare different models. Prediction on an LDA model is achieved by running the Predict method on an instance of the `LDAPredictionModel` class. This returns the predictive distributions over words that you need to calculate Perplexity. In this usage we observe the mixture distributions over topics and words (as inferred by the Infer method on the `LDAModel` or `LDAShared` class) and infer the predictive distributions over words. Internally the calculation is done one document at a time; this has the effect of keeping the per-topic word mixture distributions fixed, and also provides scalability with the number of documents.

You will need to provide the following arguments to the Predict method:

| **Argument** | **TType** | **Description** | **Example** |
|----------------------------------------------------------|
| postTheta | `Dirichlet[]` | The posterior marginal distributions over topic mixtures for each document. | These are available from the call to the Infer method on `LDAModel`  or `LDAShared` instances. |
| postPhi | `Dirichlet[]` | The posterior marginal distributions over word mixtures for each topic. | These are available from the call to the Infer method on `LDAModel`  or `LDAShared` instances. |

There is a single output from the Predict method:

| **RResult** | **Type** | **Description** | **Comments** |
|---------------------------------------------------------|
| Word distributions | `Discrete[]` | The predictive distributions over words for each document. | This can be used to calculate perplexity. |

### Topic Inference

The prediction method only operates on documents for which we know the posterior distributions over topics. Another mode of operation is to infer topics for a new document whilst fixing the learnt distributions for the per-topic word mixtures.. This is provided by the `InferTopics` method on an instance of the `LDATopicModelInference` class. In this usage we observe the per-topic word mixture distributions, and the words in the test documents, and infer the topic mixture distributions for each test document.

You will need to provide the following arguments to the `InferTopics` method.

| **Argument** | **Type** | **Description**  | **Example** |
|----------------------------------------------------------|
| alpha | `double` | Parameterises _ThetaPrior._ | 150 / numTopics |
| postPhi | `Dirichlet[]` | The posterior marginal distributions over word mixtures for each topic. | These are available from the call to the Infer method on LDAModel  or LDAShared instances. |
| wordsInDoc | `int[][]` | Array of dictionaries giving word counts for each document. Each dictionary is keyed by word index and the corresponding value is the document count for the word. | See the [LDA test program](LDA test program.md) for an example. |

There is a single output from the `InferTopics` method:

| **Result** | **Type** | **Description** | **Comments** |
|--------------------------------------------------------|
| Topic mixture distributions | `Dirichlet[]` | The distributions over topic mixtures for each document. | Relating latent topics to true topics may not be straightforward. |

<br/>
[Page 1](Standard LDA.md) \| Page 2 \| [Page 3](Shared variable LDA.md)
