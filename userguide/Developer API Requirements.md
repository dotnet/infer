---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Infer.NET development](Infer.NET development.md)

## Developer API Requirements

The list of requirements below came out of a meeting where Andy Gordon presented his ideas on where FUN was headed. These are listed in no particular order.

#### Inference Requirements

*   Easy to discover, get and use pre-defined models
*   Queries to be easy:
    *   Named e.g. "Classify", "Train", "Predict"
    *   Strongly typed
    *   Hides inference engine (Infer.NET)
    *   Can be standardized/sharable across models
    *   Cannot require compilation (Infer.NET)
    *   Query syntax should be identical in compiled and non-compiled cases. (Infer.NET)
    *   Should not assume the existence of a model (e.g. could work with Infer.NET as well as TLC) (Infer.NET)
    *   Discoverable
    *   Options should be consistent across different models, e.g. via option classes or interfaces
*   Option to pre-compile inference code  (Infer.NET)
*   Cross-language (.NET, Excel, C++, R, Matlab)
*   Online/incremental (as an option)
*   Active learning (as an option) - label can be provided by a callback.
*   Model Configuration/Training/Classification can happen in a different place.
*   Trained parameters can be saved and restored.
*   Inference settings can be saved and restored.
*   It should be possible to save/restore the minimal state needed for a particular query. (optional)
*   Inference code can be distributed in a hard to read form (license)
*   Evidence computation is easy.
*   Forwards sampling is easy.
*   Diagnostics (convergence, performance, accuracy, etc.)
*   Common tasks are easy, difficult tasks are possible. Target rising star developer.
*   Inference settings are discoverable with documentation at coding time.
*   Queries can accept data in native form (no conversion is necessary during the query) -> Efficiency.
*   Data can be in a sparse form.
*   Queries for a model family can accept data in a standard form -> Ease of use
    *   The standard form should ideally be statically strongly typed
*   Consider accepting URI for data input.
*   Configurable models can publish their capabilities.

#### Model Composition Requirements

*   Models can be composed:

    *   Mixture as a function
    *   Array as a function
    *   Model selection as a function

#### Plug-in Requirements

- discoverable parameters/options so that a GUI can be built on the fly to control the algorithm

#### Action Items

- Competitive API survey (TLC, Weka, Google Predict, Mahout, "R", etc) -> jbronsk