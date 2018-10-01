---
layout: default
---
[Learners](../../../../Infer.NET Learners.md) : [Matchbox recommender](../../../Matchbox recommender.md) : [Learner API](../../../Matchbox recommender/Learner API.md) : [Data mappings](../../Data mappings.md)

## Data splitting mapping

In order to evaluate the accuracy of a learner, we need to split the available data into a training and a test set. In most cases a random split is sufficient, but for a good evaluation of a recommender system a more sophisticated approach is required. In particular,  we need to be able to control the fraction of cold users and items in the test set, and also to split the ratings for each user proportionally. That is, the data instances are not independent, and this needs to be treated with care. Note that user and item features are not split. These need to be provided during training if available, while in test we only require the features of the cold users or items.

The data splitting algorithm is implemented in the `TrainTestSplittingRecommenderMapping` class. Just like the standard data format mapping holds internally a native data format mapping, the data splitting mapping internally holds a standard data format mapping. We call this concept _mapping nesting_. Since you once defined how your data is accessed in a standard way, you can use this to instantiate a splitting mapping. This can be done by using the `SplitToTrainTest` extension method:

### Creating a data splitting mapping
```csharp
var mapping = new  StandardMapping();  
var splittingMapping = mapping.SplitToTrainTest(  
    trainingOnlyUserFraction: 0.5,  
    testUserRatingTrainingFraction: 0.25,  
    coldUserFraction: 0.1,  
    coldItemFraction: 0.1,  
    ignoredUserFraction: 0.1,  
    ignoredItemFraction: 0.1,   
    removeOccasionalColdItems: true);
```
Here is an explanation of the method parameters and the way the splitting algorithm works:

1.  The requested fraction of ignored items is removed, together with all associated instances.
2.  The requested fraction of cold items is moved to the test set, together with all the associated instances.
3.  The requested fraction of ignored users remaining after the first two steps is removed, together with all associated instances.
4.  The requested fraction of cold users remaining after the first two steps is moved to the test set, together with all associated instances.
5.  The requested fraction of training-only users remaining after the first two steps is moved to the training set, together with all associated instances.
6.  For each user remaining after all previous steps, the requested fraction of test user ratings for training is moved to the training set, while the rest is moved to the test set. At least one instance is always moved to the training set for each user.
7.  If requested, instances associated with the occasionally produced cold items can be removed from the test set. An item is said to be 'occasionally cold' if it is covered only by the test set but wasn't explicitly chosen as cold. Such items can be produced by steps (4) and (6). The anticipated use of this option is when the requested number of cold users and items is set to zero, in order to ensure that all entities in the test set are covered by the training set.

The sum of the user-related fractions cannot exceed 1, and the sum of the item-related fractions cannot exceed 1:
```
ignoredUserFraction + coldUserFraction + trainingOnlyUserFraction <= 1.0  
ignoredItemFraction + coldItemFraction <= 1.0 
```
Note that although this procedure produces a random split, it is always deterministic. The shuffling algorithm uses the Infer.NET `Rand` class, so one can reset the random seed setting `Rand.Restart(seed)`.

### Splitting data

Once the mapping is instantiated, it can provide a training and a test set using the specified parameters. These are obtained through the `SplitInstanceSource.Training` and `SplitInstanceSource.Test` static methods:
```csharp
var trainingDataset = splittingMapping.GetInstances(SplitInstanceSource.Training(dataset));  
var testDataset = splittingMapping.GetInstances(SplitInstanceSource.Test(dataset));
```

### Examples

Here is a short example of splitting a sample dataset. Consider the following dataset:
```
u1, i1  
u1, i2  
u2, i1  
u2, i2  
u2, i3  
u2, i4  
u3, i1  
u4, i1
```
Ratings are irrelevant, since they are always preserved for the corresponding user-item pair. It can be easier to think of this dataset as a sparse representation of rated items for each user:
```
u1: i1, i2  
u2: i1, i2, i3, i4  
u3: i1  
u4: i1
```
Let's assume that we do not want any cold users or items in our test set, and also that we want to use the entire data. In addition, we want all instances associated with half of the users to go the training set, and half of the ratings of the rest of the users to go to the test set. This is roughly equivalent to a 75:25 split. The corresponding mapping has the following parameters:
```
trainingOnlyUserFraction: 0.5  
testUserRatingTrainingFraction: 0.5  
coldUserFraction: 0.0  
coldItemFraction: 0.0  
ignoredUserFraction: 0.0  
ignoredItemFraction: 0.0  
removeOccasionalColdItems: true
```
One possible split is to firstly select users u3 and u4 for the training set only, and then split in half the ratings of users u1 and u2. Also, let's assume that item i2 is selected for training from the ratings of user u1, and items i3 and i4 are selected for training from the ratings of user u2. This will produce the following split:

Training:
```
u1, i2  
u2, i3  
u2, i4  
u3, i1  
u4, i1
```
Test:
```
u1, i1  
u2, i1  
u2, i2
```
Let's consider another possibility though. What could happen is to firstly select users u1 and u3 for training only, and then split in half the ratings of users u2 and u4. Also, for user u2, let's randomly pick items i1 and i3 for training, and items i2 and i4 for test.

This set-up shows two corner cases. First, selecting the pair u2-i4 for test produces an unintentional cold item - i4. This is explained in point 7 from the splitting algorithm. And since removeOccasionalColdItems is set to true, this instance will be moved to the training set. Second, user u4 participates in only one instance. As point 6 of the splitting algorithm states: "At least one instance is always moved to the training set for each user". This means that the only instance for user u4 will be moved to training. Thus, the final split will be:

Training:
```
u1, i1  
u1, i2  
u2, i1  
u2, i3  
u2, i4  
u3, i1  
u4, i1
```
Test:
```
u2, i1
```
This is obviously a pathological case which arises due to the very small size of the dataset, but it is worth mentioning that such a split is possible. Note that the first split produced a 50:30 training-test split ratio, while the second one resulted in a 70:10 ratio. With sufficient data, the split will approach a 75:25 ratio.
