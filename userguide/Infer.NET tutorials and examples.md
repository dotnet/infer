---
layout: default
---
[Infer.NET user guide](index.md)

## Tutorials & Examples

### Tutorials

The following tutorials provide a step-by-step introduction to Infer.NET. Can be viewed through the **[Examples Browser](The examples browser.md)**.

1.  **[Two coins](Two coins tutorial.md)** \- a first tutorial, introducing **the basics** of Infer.NET.
2.  **[Truncated Gaussian](Truncated Gaussian tutorial.md)** \- using **variables and observed values** to avoid unnecessary compilation.
3.  **[Learning a Gaussian](Learning a Gaussian tutorial.md)** \- using ranges to handle **large arrays** of data; **visualising** your model.
4.  **[Bayes Point Machine](Bayes Point Machine tutorial.md)** \- demonstrating how to **train and test** a Bayes point machine classifer.
5.  **[Clinical trial](Clinical trial tutorial.md)** \- using **if blocks** for **model selection** to determine if a new medical treatment is effective.
6.  **[Mixture of Gaussians](Mixture of Gaussians tutorial.md)** \- constructing a **multivariate mixture** of Gaussians.

### String Tutorials

The following tutorials provide an introduction to an experimental Infer.NET feature: inference over string variables. The first two tutorials can be viewed through the **[Examples Browser](The examples browser.md)**, and the third one is available as a separate project.

1.  **[Hello, Strings!](Hello, Strings!.md)** \- introduces **the basics** of performing inference over string variables in Infer.NET.
2.  **[StringFormat Operation](StringFormat operation.md)** \- demonstrates a powerful string operation supported in Infer.NET, **StringFormat**.
3.  **[Motif Finder](Motif Finder.md)** \- defining a **complex model** combining string, arrays, integer arithmetic and control flow statements.

### Short Examples

Short examples of using Infer.NET to solve a variety of different problems. Can be viewed through the **[Examples Browser](The examples browser.md)**.

*   **[Bayesian PCA and Factor Analysis](Bayesian PCA and Factor Analysis.md)** \- how to build a low dimensional representation of some data by linearly mapping it into a low dimensional manifold.
*   **[Rats example from BUGS](Rats example from BUGS.md)** \- a hierarchical normal model, used to illustrate Gibbs sampling.
*   **[Click model](Click model example.md)** \- an information retrieval example which builds a model to reconcile document click counts and human relevance judgements of documents.
*   **[Difficulty versus ability](Difficulty versus ability.md)** \- a model of multiple-choice tests and crowdsourcing.
*   **[Gaussian Process classifier](Gaussian Process classifier.md)** \- a Bayes point machine that uses kernel functions to do nonlinear discrimination.
*   **[Recommender System](Recommender System.md)** \- a matrix factorization model for collaborative filtering.
*   **[Student skills](Student skills.md)** \- cognitive assessment models for inferring the skills of a test-taker.
*   **[Chess Analysis](Chess Analysis.md)** \- comparing the strength of chess players over time.
*   **[Discrete Bayesian network](Discrete Bayesian network.md)** \- uses Kevin Murphy's Wet Grass/Sprinkler/Rain example to illustrate how to construct a discrete Bayesian network, and how to do parameter learning within such a model.

### Longer Examples

*   [**Latent Dirichlet Allocation**](Latent Dirichlet Allocation.md) \- this example provides Infer.NET implementations of the popular LDA model for topic modeling. The implementations pay special attention to scalability with respect to vocabulary size, and with respect to the number of documents. As such, they provide good examples for how to scale Infer.NET models in general.
*   **[Mixed Membership Stochastic Block Model](Cloning ranges.md)** \- models relational information among objects (for example individuals in an social network).
*   **[Click through model](Click through model sample.md)** \- a web search example where you convert a sequence of clicks by the user into inferences about the relevance of documents.
*   **[Image classifier example](Image classifier example.md)** \- an image search example where you classify tagged images by example.
*   **[Clinical trial](Clinical trial UI.md)** \- the [clinical trial tutorial example](Clinical trial tutorial.md) with an interactive user interface.
*   **[Monty Hall problem](Monty Hall problem.md)** \- an Infer.NET implementation of the [Monty Hall problem](http://en.wikipedia.org/wiki/Monty_Hall_problem), along with a graphical user interface.
*   **[Conference reviewer model](Calibrating reviews of conference submissions.md)** \- for estimating submission quality in the light of noisy, biased or uncertain reviewers.
*   **[The separation of model and inference](The separation of model and inference.md)**
*   **[Aggregation models for Crowdsourcing](Community-Based Bayesian Classifier Combination.md)**
*   **[BCCWords: Bayesian Text Sentiment Analysis using Crowdsourced Annotations](BCCWords.md)**
*   [**Multinomial Logistic Regression**](The softmax factor.md)
*   From the forums:
    *   **[Bayesian Linear Regression](http://social.microsoft.com/Forums/en-US/infer.net/thread/3fed94a3-f0da-4dc7-993f-71d9b571d278)**
    *   **[Discrete Markov network](http://social.microsoft.com/Forums/en-US/infer.net/thread/14af2e98-ff05-4d9a-9ffe-78d4a9b08623)**, [more](http://social.microsoft.com/Forums/en-US/infer.net/thread/589ba1d6-d4ee-4b97-828f-18c325319008)
    *   [**Hidden Markov Model**](https://github.com/oliparson/infer-hmm)

### How-to Guides

How to achieve various general tasks in Infer.NET.

*   **[How to handle missing data](How to handle missing data.md)** \- how to cope with situations where some observations of a variable are missing.
*   **[How to build scalable applications](How to build scalable applications.md)** \- how to scale your applications to cope with large data sets and large value ranges.
*   **[How to represent large irregular graphs](How to represent large irregular graphs.md)** \- how to represent large graph structures efficiently.
*   **[How to save distributions to disk](How to save distributions to disk.md)** \- how to serialize the distribution classes.
*   [**How to do causal inference**](Causal inference with Infer.NET.md)
