---
layout: default
---
[Learners](../../../Infer.NET Learners.md) : [Matchbox recommender](../../Matchbox recommender.md) : [Learner API](../../Matchbox recommender/Learner API.md)

## Prediction

The recommender supports four modes of prediction - rating prediction, item recommendation, find related items, and find related users. In addition, each of these has a singleton and a bulk version. The latter three methods will search for results amongst all users or items by default, but these subsets can be limited if required. Predictions can be made for cold-start users and items - ones which did not participate in training. This is all covered in detail below.

### Rating prediction

A distribution over the ratings that a user will give to an item can be obtained by using the `PredictDistribution` method. It will return an instance of the Infer.NET  `Discrete` distribution class, which maps each possible rating value to a probability. The first element of the distribution (the one with index zero) will correspond to the smallest rating, and the last element of the distribution will correspond to the largest rating value.
```csharp
Discrete ratingDistribution = recommender.PredictDistribution("Person 1", "Movie Title");
```
If you are not interested in the distribution over the possible ratings, but rather in a integer valued rating, then you can use the Predict method. It will internally call `PredictDistribution`, and it will convert the obtained result into a point estimate by using a given loss function. The loss function can be specified by the `SetPredictionLossFunction` setting. By default the mean of the discrete distribution is taken.
```csharp
int rating = recommender.Predict("Person 1", "Movie Title");
```
Both `PredictDistribution` and `Predict` have corresponding bulk overloads allowing you to make predictions for multiple users and items in a single call. An instance dataset needs to be passed to these methods, and internally the users and the items will be obtained by calling the corresponding data mapping functions. No rating-related methods will be called. The return type is a dictionary which maps each user to a collection of item-rating pairs (also stored in a dictionary).
```csharp
IDictionary<string, IDictionary<string, int>> ratings = recommender.Predict(instances);
```
### Item recommendation

A list of recommended items can be obtained by using the Recommend method. It takes as input the user to make recommendations to and the number of recommended items to return. The way this method works is by iterating over the items and computing a score for each one of them, and then returning the top ones. 
```csharp
IEnumerable<string> recommendations = recommender.Recommend("Person 1", 10);
```
The corresponding bulk method takes in a list of users and the number of recommendations to make to each user. It returns a dictionary which maps each input user to a list of recommended items.
```csharp
IDictionary<string, IEnumerable<string>> recommendations = recommender.Recommend(users, 10);
```
### Related items

The recommender can find a list of items related to a given item. Two items are said to be "related" if they are likely to be rated similarly by a random user. Note that this is not entirely the same as the commonly accepted "users who liked this item also liked…", but the algorithm will in addition include "users who did not like this item also did not like…". The way this method works is by iterating over the items and computing a similarity score for each of them, then selecting the top ones.
```csharp
IEnumerable<string> relatedItems = recommender.GetRelatedItems("Movie Title", 10);
```
The corresponding bulk method takes in a list of items and the number of related items to find for each one. It returns a dictionary which maps each input item to a list of related items.
```csharp
IDictionary<string, IEnumerable<string>> relatedItems = GetRelatedItems(items, 10);
```
### Related users

Two users are considered to be related if they have the same preferences for a random item. That is, they like similar items and they dislike similar items. Related users to a given user can be obtained by calling the `GetRelatedUsers` method, which takes in a user and an integer which specifies how many related users to return. Similarly to the `GetRelatedItems` method, this method internally iterates over the users and computes a similarity score for each one of them, then returns the top ones.
```csharp
IEnumerable<string> relatedUsers = recommender.GetRelatedUsers("Person 1", 10); 
```
The corresponding bulk method takes in a list of users and the number of related users to find for each one. It returns a dictionary which maps each input users to a list of related users.
```csharp
IDictionary<string, IEnumerable<string>> relatedUsers = GetRelatedUsers(users, 10);
```
### Cold-start predictions

The Matchbox recommender supports cold-start predictions. This means predictions for users or items which did not participate in training. If no features are used, then the predictions for each user or item will be same. They will be strongly dominated by the entities with the highest bias. These results can be improved with the use of explicit features. Therefore, each of the methods listed above has yet another overload which takes in a feature source as a last parameter. For example:
```csharp
IEnumerable<string> relatedItems =   
    recommender.GetRelatedItems("Movie Title", 10, featureSource);  
IDictionary<string, IEnumerable<string>> recommendations =   
    recommender.Recommend(users, 10, featureSource); 
```
The feature source will be only accessed (through the data mapping) if training was performed with features _and_ the entity to make predictions for is cold. Otherwise, if the user or item to make predictions for participated in training, the features of this entity will not be accessed.

### User and item prediction subsets

By default the recommender will search among all items for recommendations, among all items for related items, and among all users for related users. However, the sets of entities to search among can be explicitly narrowed down. This is done by specifying the `ItemSubset` property for item recommendation and related items, and the `UserSubset` property for related users.

Here is an example for item recommendation. Consider that the items on which the recommender was trained is `i1 - i20`. Also, consider that user u1 likes these items in decreasing order of their index. Thus, `recommender.Recommend(u1, 5)` returns `[i1, i2, i3, i4, i5]`. But we can then limit the prediction item subset to only the items with even indices:
```csharp
recommender.ItemSubset = new[] { i2, i4, i6, i8, i10, i12, i14, i16, i18, i20 };
```
Then, `recommender.Recommend(u1, 5)` will return `[i2, i4, i6, i8, i10]`. This technique is particularly useful when we don't want the recommender to return items that a particular user has already rated. Since the system cannot afford to store the entire training data, it will eventually recommend items that were already rated. This in practice means that if a user buys a given product, they might see this product again in the proposed recommendations. Using ItemSubset will prevent this from happening.
