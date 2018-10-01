---
layout: default 
--- 
[Infer.NET user guide](../../../../index.md) : [Learners](../../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../../Bayes Point Machine classifiers.md) : [The Learner API](../../API.md) : [Mappings](../Mappings.md)

## Native Data Format Mapping

The `IBayesPointMachineClassifierMapping` interface supplies input data in a format that the inference algorithms run by the Bayes Point Machine classifier during training and prediction can consume without any conversion. The supplied format is therefore referred to as the _native_ data format.

The native data format permits two distinct feature representations: _dense_ and _sparse_. Whereas the dense representation stores all features associated with a single instance in an array of `double` values, the sparse representation consists of an array of all non-zero feature values and an array of the corresponding feature indices. Both representations produce the same results, only at different computational costs. Training and prediction on data with many zero-valued features is typically faster in a sparse representation.

In addition to the features representations, the native format also fixes the type of the labels. In binary classification, labels must be Boolean, in multi-class classification labels must be provided as zero-based, consecutive integers (i.e. 0,1,...,![K](../../../BPM/ClassCount.png) \- 1, where ![K](../../../BPM/ClassCount.png) is the number of classes).

Delivering data in native format requires an implementation of 8 methods:

1.  **IsSparse**:

    ```csharp
    bool IsSparse(TInstanceSource instanceSource);
    ```

    `IsSparse` indicates whether the features are specified in a sparse or a dense representation. Note that the feature representation must not be changed. Neither between instances nor between different instances subsets, such as training and test sets.



2.  **GetFeatureCount**:

    ```csharp
    int GetFeatureCount(TInstanceSource instanceSource);
    ```

    `GetFeatureCount` states how many features the classification data contains. It is needed to set up the inference algorithms (it determines the number of weights, for instance) when the features have a sparse representation (as one cannot infer the total number of features in this representation).



3.  **GetClassCount**:

    ```csharp
    int GetClassCount(  
        TInstanceSource instanceSource = default(TInstanceSource),   
        TLabelSource labelSource = default(TLabelSource));
    ```

    `GetClassCount` returns the total number of class labels present in instances. 

All aforementioned three methods are used to set up the Bayes Point Machine classifier and verify the validity of both features and labels. They are called during training as well as prediction. `IsSparse` and `GetFeatureCount` partially determine the expected return values of `GetFeatureValues` and `GetFeatureIndexes` (see below).

The next two methods deliver feature values and indices for single instances and are only used during [singleton prediction](../Prediction.md). Their return values will _not_ be cached.

4.  **GetFeatureValues** _(single-instance)_:

    ```csharp
    double[] GetFeatureValues(  
        TInstance instance,   
        TInstanceSource instanceSource = default(TInstanceSource));
    ```

    `GetFeatureValues` returns an array of feature values for the specified instance. If the feature representation is sparse (i.e. `IsSparse` returns true), this method only returns all _non-zero_ feature values associated with the given instance. If the representation is dense, `GetFeatureValues` returns all feature values whether they're zero or not.

5.  **GetFeatureIndexes** _(single-instance)_:

    ```csharp
    int[] GetFeatureIndexes(  
        TInstance instance,   
        TInstanceSource instanceSource = default(TInstanceSource));
    ```

    GetFeatureIndexes returns null if features are expected to be in a dense representation. Otherwise it returns an array of feature indices corresponding to the non-zero feature values returned by the single-instance GetFeatureValues method. 


Then, there are two methods which provide feature values and indices of multiple instances to both the training and prediction algorithms. During training, the set of instances may be further split into subsets, referred to as _batches_, which makes it possible to learn from data that does not fit into memory (more on this in [Batched training](../Training.md)). Bulk prediction, which also calls into the following two methods, does _not_ work with batches.

6.  **GetFeatureValues** _(multi-instance)_:

    ```csharp
    double[][] GetFeatureValues(  
        TInstanceSource instanceSource, int batchNumber = 0);
    ```

    `GetFeatureValues` returns an array of feature values for each instance associated with the specified batch of instances. By default, it is assumed that the features and labels of all instances may live in a single batch. If this is not possible, perhaps because this requires too much memory, the Bayes Point Machine classifier mapping allows to split the training data into a number of batches (specified by the Bayes Point Machine classifier [setting](../Settings.md) `BatchCount`). Instance batch indexes run from 0 to `BatchCount` \- 1\. Note also that a single training run passes several times over the all batches (see `IterationCount` [setting](../Settings.md)).

    Again, the feature representation may be sparse or dense, dependent on the return value of `IsSparse`. 

7.  **GetFeatureIndexes** _(multi-instance)_:

    ```csharp
    int[][] GetFeatureIndexes(  
        TInstanceSource instanceSource, int batchNumber = 0);
    ```

    `GetFeatureIndexes` returns null if features are expected to be in a dense representation. Otherwise it returns the feature indices of all instances in the specified batch of instances. These must correspond with the feature values returned by `GetFeatureValues`, given the same batch of instances. 

The final method in `IBayesPointMachineClassifierMapping` provides ground truth labels during training:

8.  **GetLabels**:

    ```csharp
    TLabel[] GetLabels(  
        TInstanceSource instanceSource,   
        TLabelSource labelSource = default(TLabelSource),   
        int batchNumber = 0);
    ```  
    `GetLabels` provides all ground truth labels for a given batch of instances from an instance or label source. Note that `TLabel` is bound to `bool` in binary classification and to `int` in multi-class classification. `GetLabels` is _not_ called during prediction. 

The `NativeClassifierMapping` class and its subclasses in the `Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal` namespace are examples of implementations of the `IBayesPointMachineClassifierMapping` interface. These classes wrap `IClassifierMapping` and show how data in standard format is converted to native format. They also show how to cache data batches during training.
