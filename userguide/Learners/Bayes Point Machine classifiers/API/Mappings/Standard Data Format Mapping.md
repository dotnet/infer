---
layout: default 
--- 
[Infer.NET user guide](../../../../index.md) : [Learners](../../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../../Bayes Point Machine classifiers.md) : [The Learner API](../../API.md) : [Mappings](../Mappings.md)

## Standard Data Format Mapping

Let's start with closer look at the `IClassifierMapping` interface. It declares 4 methods:

1.  **GetInstances**:

    ```csharp
    IEnumerable<TInstance> GetInstances(TInstanceSource instanceSource);
    ```  
    `GetInstances` provides the set of instances of the sample at hand (training or test set) to the learner. It is used during both training and prediction. Moreover, both generic types, `TInstance` and `TInstanceSource`, may be freely chosen. For example, `TInstance` may be bound to a class which provides both features and labels; it may also be a reference to or an index into an object of such a class.

    To allow for caching, the Bayes Point Machine classifier assumes that the same instance source always provides the same instances. 

2.  **GetFeatures**:

    ```csharp
    TFeatures GetFeatures(  
        TInstance instance,   
        TInstanceSource instanceSource = default(TInstanceSource));
    ```
    `GetFeatures` provides all feature values (and indices) for the specified instance of the sample. This method is equally used during training and prediction. Note that it is not necessary to specify an instance source if the instance itself contains the corresponding features values.

    The Bayes Point Machine classifier requires `TFeatures` to be bound to `Microsoft.ML.Probabilistic.Math.Vector`. This is not a significant restriction since Infer.NET Vector supports both sparse and dense representations (via `SparseVector` and `DenseVector`, respectively). Note also that the chosen representation in standard format determines the representation in native format, i.e. a `SparseVector` will be converted to a native feature representation which is sparse and a `DenseVector` will be converted to a native feature representation which is dense.

    For caching purposes, `GetFeatures` is assumed to always provide the same features values (and indices) for the same instance.

    You may want to _add a constant feature_ to all instances - i.e. add a feature whose value is always equal to 1. This will make the classifier invariant to shifting the feature values (by adding a bias for each class). In addition, the less correlation exists among the features, the better (highly correlated features may lead to slow convergence of the training algorithm). 

3.  **GetLabel**:

    ```csharp
    TLabel GetLabel(  
        TInstance instance,   
        TInstanceSource instanceSource = default(TInstanceSource),   
        TLabelSource labelSource = default(TLabelSource));
    ```

    `GetLabel` provides the ground truth label for the specified instance of interest. This method is only called during training.

    The possibility of a label source separate from the instance source provides more flexibility, and is perhaps primarily of interest in a scenario where features and labels live in distinct data sources. If, however, the instance itself already contains the label, both instance and label source may be omitted.

    To be able to cache labels and avoid superfluous conversions into the native data format, the Bayes Point Machine classifier assumes that every call into `GetLabel` provides the same label for the same instance. 

4.  **GetClassLabels**:

    ```csharp
    IEnumerable<TLabel> GetClassLabels(  
        TInstanceSource instanceSource = default(TInstanceSource),   
        TLabelSource labelSource = default(TLabelSource));
    ```

    `GetClassLabels` provides the set of labels for all classes present in the classification data (population). Not only does it determine how many classes the classification task consists of, but also what the corresponding class labels are. In the simplest case, when `TLabel` is bound to `bool`, `GetClassLabels` would return { true, false }. `GetClassLabels` hence guarantees the validity of the labels encountered during the lifetime of the classifier.

    `IClassifierMapping` requires this method because it is generally impossible to infer the entire set of class labels from a population sample. That is, the training or test sets may not always contain all class labels.

Because it is possible to freely choose suitable types for `TInstance`, `TInstanceSource`, `TLabelSource` and `TLabel` (`TFeatures` needs to be of type `Vector`), it is generally straightforward to provide a simple and efficient implementation for the `IClassifierMapping` interface. Providing data in the standard format therefore is a very convenient way to communicate with a classifier.

As previously mentioned, the Bayes Point Machine classifiers ultimately need to convert the data supplied in standard format into the native format understood by the Infer.NET inference algorithms. In many situations, however, the programming costs of directly delivering data in native format (i.e. providing a careful implementation of the `IBayesPointMachineClassifierMapping` as described in the [following subsection](Native Data Format Mapping.md)) outweigh the computational costs associated with a conversion from standard to native format.

A concrete example of an implementation of `IClassifierMapping` can be found in the [tutorial introduction on gender prediction.](../../Introduction.md)
