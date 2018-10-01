---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Infer.NET development](Infer.NET development.md)

## Infer.NET inference API design

This document describes the design of the Infer.NET inference API.
 
The inference API has five main components:

*   **the probabilistic program:** consisting of the model, the model name and the set of inference queries (as dictated by the list of variables to infer, query type and any observed values).
*   **the inference algorithm:** determines the set of fixed points which are possible solutions to a given probabilistic program e.g. the logical algorithm (e.g. EP/VMP) and any parameters that affect the set of fixed points e.g. grouping or approximating family to use.
*   **the compilation settings:** changes the generated code but not the set of fixed points.
*   **the inference runtime settings:** number of iterations, resume last run etc. which don't change the set of fixed points or the generated code but may affect which fixed point you get (or whether the algorithm converges).
*   **inference monitoring settings:** changes what feedback is given to the user during the inference process e.g. showing the browser, timing information, errors and warnings, progress etc.

These components are related to Infer.NET classes as follows:

*   **InferenceEngine:** inference monitoring and runtime settings. Currently has the model name. Holds the algorithm and compiler objects. Also holds algorithm settings that are common to multiple algorithms, such as grouping. Currently also stores the inference queries.
*   **Compiler:** holds compilation settings only
*   **Algorithm:** holds algorithm-specific algorithm settings. Also holds the default query type for the algorithm.
*   **Variable:** used to define the model. Currently variables also hold any variable-specific setting e.g. the approximating family (through marginal prototype attributes) and custom query types.