---
layout: default
---
## Infer.NET User Guide

*   **Introduction**

    *   [What is Probabilistic Programming?](../InferNet_Intro.pdf)
    *   [What is Infer.NET?](What is Infer.NET.md)
    *   [A simple example](a simple example.md)
    *   [How Infer.NET works](how Infer.NET works.md)
    *   [Frequently Asked Questions](Frequently Asked Questions.md)
    *   [Resources and References](Resources and References.md)
    *   [Infer.NET 101 paper](../InferNet101.pdf), a detailed sample-based introduction to programming with Infer.NET.

*   **Building probabilistic models** 

    *   Conceptual background: [Model-Based Machine Learning Book](http://mbmlbook.com/)
    *   [The Infer.NET modelling API](The Infer.NET modelling API.md)
        *   [Creating variables](Creating variables.md)
            *   [Variable types and their distributions](Variable types and their distributions.md)
            *   [Vector and Matrix types](Vector and matrix types.md)
            *   [Applying functions and operators to variables](Applying functions and operators to variables.md)        
        *   [Attaching constraints to variables](Attaching constraints to variables.md)
        *   Working with arrays and ranges
            *   [Arrays and ranges](Arrays and ranges.md)
            *   [ForEach blocks](ForEach blocks.md)
            *   [Markov chains and grids](Markov chains and grids.md)
            *   [Jagged arrays](Jagged arrays.md)
            *   [Indexing arrays by observed variables](Indexing arrays by observed variables.md)
            *   [Cloning ranges](Cloning ranges.md)
        *   [Branching on variables to create mixture models](Branching on variables to create mixture models.md)
            *   [The importance of using SetTo](The importance of using SetTo.md)
        *   [Repeat blocks](Repeat blocks.md)
        *   [Computing model evidence for model selection](Computing model evidence for model selection.md)
        *   [Increment log density](Increment log density.md)

    *   [Factors and constraints](Factors and Constraints.md)
        *   [Boolean factors](Bool factors.md)
        *   [Integer and enum factors](Int factors.md)
        *   [Double factors](Double factors.md)
        *   [String and char factors](String and char factors.md)
        *   [Array and list factors](Array and list factors.md)
        *   [Vector and matrix factors](Vector and matrix factors.md)
        *   [Generic factors](Generic factors.md)
        *   [Constraints](Constraints.md)
        *   [Undirected factors](Undirected factors.md)
        *   [List of factors and constraints](list of factors and constraints.md)

    *   Advanced model building
        *   [Adding attributes to your model](Adding attributes to your model.md)
        *   [The Model Specification Language](The Model Specification Language.md)
        *   [Transform browser](Transform browser.md)

*   **Running inference on your model**

    *   [Running inference](Running inference.md)
        *   [Inference engine settings](inference engine settings.md)
        *   [Working with different inference algorithms](Working with different inference algorithms.md)
            *   [Expectation Propagation](Expectation Propagation.md)
            *   [Variational Message Passing](Variational Message Passing.md)
            *   [Gibbs sampling](Gibbs sampling.md)
        *   [Monitoring the progress of inference](Monitoring the progress of inference.md)
        *   [Quality bands](Quality bands.md)
    *   [Controlling how inference is performed](Controlling how inference is performed.md)
        *   [Customising the algorithm initialisation](customising the algorithm initialisation.md)
        *   [Using a precompiled inference algorithm](Using a precompiled inference algorithm.md)
        *   [Structure of generated inference code](Structure of generated inference code.md)
    *   [Debugging inference](Debugging inference.md)
    *   [Improving the speed and accuracy of inference](improving the speed and accuracy of inference.md)
        *   [Using sparse messages](using sparse messages.md)
    *   [Sharing variables between models](Sharing variables between models.md)
        *   [Shared variable arrays](Shared variable arrays.md)
        *   [Defining shared variables within a model](Shared variable definition.md)
        *   [Jagged shared variable arrays](Jagged shared variable arrays.md)
        *   [Computing evidence for models with shared variables](Shared variable evidence.md)
        *   [Using shared variables to support hybrid algorithms](Shared variable hybrid.md)
    *   [Online learning](Online learning.md)
    *   [Distributed inference](Distributed inference.md)
    *   [Computing derivatives of functions](Computing derivatives of functions.md)

*   **Extending Infer.NET**

    *   [Infer.NET component architecture](Infer.NET component architecture.md)
    *   [Adding a new distribution type](How to add a new distribution type.md)
    *   [Adding a new factor and its message operators](How to add a new factor and message operators.md)
    *   [Adding a new constraint](How to add a new constraint.md)
    *   [Modifying the operator search path](Modifying the operator search path.md)
    *   Automatically computing EP messages: [KJIT](https://github.com/wittawatj/kernel-ep)

*   **Calling Infer.NET from other languages**

    *   [F#](FSharp Wrapper.md)
        *   [Variable Arrays in F#](Variable Arrays in FSharp.md)
        *   [Inference in F#](Inference in FSharp.md)
        *   [Operator Overloading in F#](Operator Overloading in FSharp.md)
        *   [Imperative Statement Blocks in F#](Imperative Statement Blocks in FSharp.md)
    *   [C++/CLI](CPlusPlus.md)
    *   [Mono](Running with Mono.md)

*   **[Learners](Infer.NET Learners.md)**

    *   [Bayes Point Machine classifiers](Learners/Bayes Point Machine classifiers.md)
    *   [Matchbox recommender](Learners/Matchbox recommender.md)

*   **[Tutorials and examples](Infer.NET tutorials and examples.md)**

    *   [The Examples Browser](The examples browser.md)
    *   Tutorials: [Two coins](Two coins tutorial.md), [Clinical trial](Clinical trial tutorial.md), [Mixture of Gaussians](Mixture of Gaussians tutorial.md), [more...](Infer.NET tutorials and examples.md)
    *   String tutorials: [Hello, Strings!](Hello, Strings!.md), [StringFormat operation](StringFormat operation.md), and [Motif finder](Motif Finder.md).
    *   Examples: [Latent Dirichlet Allocation](Latent Dirichlet Allocation.md), [Recommender System](Recommender System.md), [Bayesian PCA](Bayesian PCA and Factor Analysis.md), [Discrete Bayesian network](Discrete Bayesian network.md), [more...](Infer.NET tutorials and examples.md)
    *   How-to guides: [How to handle missing data](How to handle missing data.md), [How to build scalable applications](How to build scalable applications.md), [more...](Infer.NET tutorials and examples.md)


*   [Infer.NET development](../development/index.md)
*   [**Code documentation**](../apiguide/api/index.html)
*   [Release change history](Release change history.md)
