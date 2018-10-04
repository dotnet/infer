---
layout: default
---
[Learners](../../Infer.NET Learners.md) : [Matchbox recommender](../Matchbox recommender.md)

## Learner API

There are two ways in which you can use the Infer.NET Matchbox recommender. The first one, which is also the simplest, is through the command line. This is explained in detail in the [Command-line runners](Command-line runners.md) section. Here we will cover the other approach, which makes use of the developer API. For this to work you need to reference the `Microsoft.ML.Probabilistic.dll`, `Microsoft.ML.Probabilistic.Learners.dll`, `Microsoft.ML.Probabilistic.Learners.Recommender.dll` and include the `Microsoft.ML.Probabilistic.Learners` and `Microsoft.ML.Probabilistic.Learners.Mappings` namespaces. In this section we will cover an overview example of the recommender API, while more of the detail will be filled in the subsections.

Before a recommender is created, a data mapping needs to be instantiated. This is simply an interface implementation which tells the recommender how to read data. This approach was preferred to passing fixed parameters to the system in order to avoid unnecessary data conversions. The mapping defines methods which the recommender will call during training or bulk prediction. Here is a sample implementation of a mapping which provides training instances from a comma-separated file; a single line of this file contains the rating that a user has given to an item - for example, "Person Name, Movie Name, 5".
```csharp
[Serializable]
class  CsvMapping :  IStarRatingRecommenderMapping
 <string, Tuple<string, string, int>, string, string, int, NoFeatureSource, Vector>
{
 public  IEnumerable<Tuple<string, string, int>> GetInstances(string instanceSource)
 {
   foreach (string line in  File.ReadLines(instanceSource))
   {
     string[] split = line.Split(new [] { ',' });
     yield  return  Tuple.Create(split[0], split[1], Convert.ToInt32(split[2]));
   }
 }

 public  string  GetUser(string instanceSource, Tuple<string, string, int> instance)
 { return instance.Item1; }

 public  string  GetItem(string instanceSource, Tuple<string, string, int> instance)
 { return instance.Item2; }

 public  int  GetRating(string instanceSource, Tuple<string, string, int> instance)
 { return instance.Item3; }

 public  IStarRatingInfo<int> GetRatingInfo(string instanceSource)
 { return  new  StarRatingInfo(0, 5); }

 public  Vector  GetUserFeatures(NoFeatureSource featureSource, string user)
 { throw  new  NotImplementedException(); }

 public  Vector  GetItemFeatures(NoFeatureSource featureSource, string item)
 { throw  new  NotImplementedException(); }
}
```
The `GetInstances` method will be invoked by the recommender to read the user-item-rating triples used for training. Then for each instance the recommender will obtain the corresponding object using the `GetUser`, `GetItem`, and `GetRating` methods. `GetRatingInfo` tells the system what the minimum and maximum rating values are. This is considered to be data dependent, and therefore was not designed as a setting. Finally, user and item features are not used in this simple example.

Once we have the data mapping, we can create a recommender, set relevant settings, and train the system:
```csharp
var dataMapping = new  CsvMapping();
var recommender = MatchboxRecommender.Create(dataMapping);
recommender.Settings.Training.TraitCount = 5;
recommender.Settings.Training.IterationCount = 20;
recommender.Train("Ratings.csv");
```
The recommender is instantiated using the `MatchboxRecommender.Create` factory method, which takes in the data mapping. The next line sets the number of traits. These were discussed in the [Introduction](Introduction.md) and typically vary between 1 and 20. We then set the number of iterations. This should be in the range between 1 and 100, typically greater than 10. Both of these parameters depend on the data and should be tuned. Finally, the recommender is trained on the Ratings.csv file using the Train method. The system knows how to parse the input, because this is specified in the mapping. 

Once training is completed, the recommender can optionally be serialized using the Save method and then deserialized using the MatchboxRecommender.Load static method:
```csharp
recommender.Save("TrainedModel.bin");
// ...
var recommender = MatchboxRecommender.Load<string, string, string, NoFeatureSource>( "TrainedModel.bin");
```
And finally, recommendations are made using the Recommend method. It takes as input the user to make recommendations to and the number of items to recommend:
```csharp
recommender.Recommend("Person 1", 10);
```
Subsections: [Data mappings](../Matchbox/API/Data mappings.md) | [Setting up](../Matchbox/API/Setting up a recommender.md) | [Training](../Matchbox/API/Training.md) | [Prediction](../Matchbox/API/Prediction.md) | [Evaluation](../Matchbox/API/Evaluation.md) | [Serialization](../Matchbox/API/Serialization.md)
