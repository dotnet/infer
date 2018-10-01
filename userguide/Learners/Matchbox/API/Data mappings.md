---
layout: default
---
[Learners](../../../Infer.NET Learners.md) : [Matchbox recommender](../../Matchbox recommender.md) : [Learner API](../../Matchbox recommender/Learner API.md)

## Data mappings

In order to feed data into the learner, the Infer.NET Recommender API requires you to provide the recommender with an object that tells the system how to read the data. This approach avoids unnecessary data conversions. Here is a simple example.

Consider the standard input to the recommender which consists of a collection of instances, where each instance is a triple of a user, an item and a star-rating value. You are required to implement an interface and provide the recommender with an object, which knows how to obtain the instances, and for each instance how to obtain the user, item and rating. These methods are entirely generic with regard to their input parameters, so you can specify whatever types you have the data stored in - anything from in-memory data collections, through to files stored on disc, to databases.
```csharp
IEnumerable<TInstance> GetInstances(TInstanceSource instanceSource);  
TUser GetUser(TInstanceSource instanceSource, TInstance instance);  
... 
```
The internal implementation of the recommender, however, operates on fixed data types. For various reasons these differ from the standardly accepted instances of user-item-rating triples. The internal representation works with an array of users, an array of items and an array of ratings. In addition, all of these have to be zero-based consecutive integers. If the training data is already in this format, you can use the so called native data format mapping, and then no data conversion will happen: 
```csharp
IList<int> GetUserIds(TInstanceSource instanceSource, int batchNumber = 0);
IList<int> GetUserIds(TInstanceSource instanceSource, int batchNumber = 0);  
...
```
The standard mapping serves as a wrapper around the native mapping. Thus, it will always be more efficient to use the native mapping. However, it is unlikely that your data is already in this format, and therefore the standard mapping is more commonly used.

Data mappings: [Standard data format](Mappings/Standard.md) | [Native data format](Mappings/Native.md) | [Data splitting](Mappings/Splitting.md) | [Negative data generation](Mappings/Negative.md)
