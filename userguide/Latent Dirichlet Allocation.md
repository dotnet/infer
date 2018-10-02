---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## Latent Dirichlet Allocation (LDA)

### Background

An [LDA model](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) (Blei, Ng, and Jordan 2003) is a generative model originally proposed for doing topic modeling. There have been many papers which have cited and extended the original work, applying it to new areas, and extending the basic model.

In LDA, the data is in the form of a collection of documents, where each document is considered as a collection of words. It is assumed that each document is represented as a mixture of latent topics, and each topic is represented as a mixture over words. These mixture distributions are assumed to be Dirichlet-distributed random variables which must be inferred from the data. The generative process can be described as follows:

*   For each topic, sample a distribution over words from a Dirichlet prior
*   For each document, sample a distribution over topics from a Dirichlet prior
*   For each word in the document
    *   Sample a topic from the document's topic distribution
    *   Sample a word from the topic's word distribution
    *   Observe the word

Inference can be done in several different ways, each providing a different cost/accuracy trade-off. In the literature, you can find inference via Variational Message Passing, Expectation Propagation, and Gibbs sampling.

### LDA in Infer.NET

The Visual Studio solution for the LDA example can be found in the [src\\Examples\\LDA folder](https://github.com/dotnet/infer/tree/master/src/Examples/LDA). In this example, we will deviate from the Blei paper in that we will not learn the parameters of the Dirichlet priors - they will be fixed in advance. (Learning the priors requires operations that are not yet included in Infer.NET.) With the Dirichlet priors fixed, the rest of the model is straightforward to implement in Infer.NET - in fact the model itself is only a few lines of code. Making it scalable to large vocabularies, and to very large data sets is a bit more tricky. By default, Infer.NET keeps the full factor graph in memory, and for an LDA this corresponds to a large number of very large Dirichlet and Discrete messages which rapidly consume available memory.

There are several ways to achieve scalability. The first is to maintain sparse messages rather than the full dense messages. In particular, the 'upward' messages that flow from the observations to the parameters are always sparse, the downward messages may or may not be sparse depending on the number of words in the topic (this is automatically handled by the Dirichlet factor). A second general mechanism is to partition the factor graph into many copies of a small factor graph corresponding to a partitioning of the data; each copy will have parameter variables shared with other copies which must be appropriately processed, and this calculation is handled by Infer.NET's shared variable mechanism. The scalability in this case is achieved by having one partition in memory at a time. Finally, words that occur multiple times in a document can be handled more efficiently; the likelihood factors for repeated words are equivalent to raising the factor for a single word to the appropriate power. This piece of factor graph is called a 'power plate', and Infer.NET provides a [Repeat block](Repeat blocks.md) to express this; the corresponding generated inference code is more efficient than treating each factor separately.

This LDA example provides a packaged and tested LDA implementation that you can use on your own data. It also illustrates the different scalability mechanisms available in Infer.NET. The example provides the following functionality:

*   Two model choices (click the links for further details)
    *   [Standard LDA](Standard LDA.md)
    *   [Shared variable LDA](Shared variable LDA.md) (scales to large data sets)
*   Predictive word distributions for new words in documents with known (learnt) topic distributions
*   Inference of topics for unseen documents
*   Model evidence

This example also comes with a separate test program which shows usage for all functionality.