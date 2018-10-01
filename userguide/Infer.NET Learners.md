---
layout: default
---
[Infer.NET user guide](index.md)

## Learners

A "learner" is a complete solution for a machine learning application such as classification or recommendation. Learners can be invoked directly from the command-line or a .NET program without having to learn the Infer.NET API. Each learner includes training. prediction, and evaluation capabilities. The source code for learners is included in the Infer.NET release to serve as comprehensive examples of how to build robust machine learning functionality using Infer.NET.

*   ### [Bayes Point Machine classifiers](Learners/Bayes Point Machine classifiers.md)


    *   [Introduction](Learners/Bayes Point Machine classifiers/Introduction.md)
    *   [Learner API](Learners/Bayes Point Machine classifiers/API.md)
        *   [Mappings](Learners/Bayes Point Machine classifiers/API/Mappings.md)
            *   [Standard data format mapping](Learners/Bayes Point Machine classifiers/API/Mappings/Standard Data Format Mapping.md)
            *   [Native data format mapping](Learners/Bayes Point Machine classifiers/API/Mappings/Native Data Format Mapping.md)
            *   [Evaluation data format mapping](Learners/Bayes Point Machine classifiers/API/Mappings/Evaluation Data Format Mapping.md)
        *   [Creation & serialization](Learners/Bayes Point Machine classifiers/API/Creation.md)
        *   [Settings & events](Learners/Bayes Point Machine classifiers/API/Settings.md)
        *   [Training](Learners/Bayes Point Machine classifiers/API/Training.md)
        *   [Prediction](Learners/Bayes Point Machine classifiers/API/Prediction.md)
        *   [Evaluation](Learners/Bayes Point Machine classifiers/API/Evaluation.md)
    *   [Model](Learners/Bayes Point Machine classifiers/Modelling.md)
    *   [Command-line runners](Learners/Bayes Point Machine classifiers/Runners.md)
        *   [Training](Learners/Bayes Point Machine classifiers/Runners/Train.md)
        *   [Incremental training](Learners/Bayes Point Machine classifiers/Runners/TrainIncremental.md)
        *   [Prediction](Learners/Bayes Point Machine classifiers/Runners/Predict.md)
        *   [Cross-validation](Learners/Bayes Point Machine classifiers/Runners/CrossValidate.md)
        *   [Convergence diagnosis](Learners/Bayes Point Machine classifiers/Runners/DiagnoseTrain.md)
        *   [Sampling weights](Learners/Bayes Point Machine classifiers/Runners/SampleWeights.md)
        *   [Evaluation](Learners/Bayes Point Machine classifiers/Runners/Evaluate.md)

*   ### [Matchbox recommender](Learners/Matchbox recommender.md)

    *   [Introduction](Learners/Matchbox recommender/Introduction.md)
    *   [Learner API](Learners/Matchbox recommender/Learner API.md)
        *   [Data mappings](Learners/Matchbox/API/Data mappings.md)
            *   [Standard data format mapping](Learners/Matchbox/API/Mappings/Standard.md)
            *   [Native data format mapping](Learners/Matchbox/API/Mappings/Native.md)
            *   [Data splitting mapping](Learners/Matchbox/API/Mappings/Splitting.md)
            *   [Negative data generation mapping](Learners/Matchbox/API/Mappings/Negative.md)
        *   [Setting up a recommender](Learners/Matchbox/API/Setting up a recommender.md)
        *   [Training](Learners/Matchbox/API/Training.md)
        *   [Prediction](Learners/Matchbox/API/Prediction.md)
        *   [Evaluation](Learners/Matchbox/API/Evaluation.md)
            *   [Rating prediction](Learners/Matchbox/API/Evaluation/Rating prediction.md)
            *   [Item recommendation](Learners/Matchbox/API/Evaluation/Item recommendation.md)
            *   [Related users](Learners/Matchbox/API/Evaluation/Related users.md)
            *   [Related items](Learners/Matchbox/API/Evaluation/Related items.md)
        *   [Serialization](Learners/Matchbox/API/Serialization.md)
    *   [Model](Learners/Matchbox recommender/Model.md)
    *   [Command-line runners](Learners/Matchbox recommender/Command-line runners.md)
        *   [Negative data generator](Learners/Matchbox/Runners/Negative data generator.md)
        *   [Data splitter](Learners/Matchbox/Runners/Data splitter.md)
        *   [Trainer](Learners/Matchbox/Runners/Trainer.md)
        *   [Predictors](Learners/Matchbox/Runners/Predictors.md)
        *   [Evaluators](Learners/Matchbox/Runners/Evaluators.md)
