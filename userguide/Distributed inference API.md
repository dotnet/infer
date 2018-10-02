---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Infer.NET development](Infer.NET development.md)

## Distributed Inference API

This document describes the new chunking API that replaces the existing SharedVariable API and to allow automatic distributed/parallel inference. The API is based on the concept of partitioned ranges and partitioned arrays. It is more flexible than SharedVariables in terms of chunking though it does not provide any support for hybrid inference or custom schedules. This document does not describe how the inference will actually be carried out, but the expectation is that inference will follow the guidelines in [Distributed inference](Distributed inference_2.md).

#### Partitioned Ranges

Consider the following non-partitioned model, based on the [Learning a Gaussian tutorial](Learning a Gaussian tutorial.md):

```csharp
Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100);
Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
Range dataRange = new Range(data.Length);
VariableArray<double> x = Variable.Array<double>(dataRange);
x[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(dataRange);
VariableArray<double> y = Variable.Array<double>(dataRange);
y[dataRange] = Variable.GaussianFromMeanAndPrecision(x[dataRange], 1.0);
y.ObservedValue = data;
 
InferenceEngine engine = new InferenceEngine();
Console.WriteLine("mean=" + engine.Infer(mean));
Console.WriteLine("prec=" + engine.Infer(precision));
```

To scale this model to large data, we would previously have had to:

1. Turn mean and precision into shared variables.
2. Change dataRange to range over a block of data instead of the whole data array.
3. Choose block sizes.
4. Divide the data into blocks.
5. Explicitly loop over data blocks and iterations of inference.

In the new scheme, steps 1 and 5 are removed. The user still does steps 2-4, in order to tell the compiler how the model should be divided into blocks and what the block sizes are. The user also provides a delegate to load each data block. The partitioned version of the model becomes:

```csharp
Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100);
Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
Range block = new Range(numBlocks);
VariableArray<int> sizeOfBlock = Variable.IArray<int>(block); 
Range dataRange = new Range(sizeOfBlock[block]);
var x = Variable.Array(Variable.Array<double>(dataRange), block);
x[block][dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(block, dataRange);
var y = Variable.IArray(Variable.Array<double>(dataRange), block);
y[block][dataRange] = Variable.GaussianFromMeanAndPrecision(x[block][dataRange], 1.0);
 
sizeOfBlock.ObservedValue = Util.IArrayFromFunc(numBlocks, b => ...);
y.ObservedValue = Util.IArrayFromFunc(numBlocks, b => ...);
 
InferenceEngine engine = new InferenceEngine();
engine.SetAttribute(block, new Partitioned());
Console.WriteLine("mean=" + engine.Infer(mean));
Console.WriteLine("prec=" + engine.Infer(precision));
```

The new API elements here are the `Partitioned` attribute, `Variable.IArray`, and `Util.IArrayFromFunc`. The `Partitioned` attribute simply marks a range as being partitioned. Any array that is indexed by a partitioned range automatically becomes a partitioned array. Intuitively, this marking instructs the compiler to have only one element of the array in memory at once (per thread). In this case, `sizeOfBlock`, `x`, and `y` are partitioned arrays, so if `b` is the current block number, then only the `bth` element of these arrays will be in memory. The reason that this is always one element at a time (instead of a user-defined number like 10 or 20) is that, in general, blocks will need to be different sizes depending on characteristics of the data so it would not particularly help to have any number other than 1 as a constant parameter on the range.
 
Partitioned ranges are sufficient to capture all of the chunking functionality of `SharedVariables` and `SharedVariableArrays`. The compiler can automatically determine which variables are shared by simply looking at which variables are indexed by `block` or are local variables of a loop over `block`. For example, the compiler knows that mean and precision are shared variables in this model since they are not indexed by `block` nor are they local to a loop over `block`. The generated code will always process a single block at a time per thread. If there is one thread, this is just sequential chunking. If there are multiple threads, this is parallel chunking.
 
Observing a partitioned array requires a special approach. If `y` had type `Variable<double[]>`, then we could not assign to `y.ObservedValue`, because the data array may not fit in memory. Instead, `y` has type `Variable<IArray<double>>`, via the helper function `Variable.IArray`. (You can also use `Variable.IList`, which creates a `Variable<IList<double>>`.) `Util.IArrayFromFunc` creates an implicit array that calls a delegate whenever an element is requested.

Another way to solve this problem is to use constraints and delegates: 
 
```csharp
Variable<Func<int,double[]>> LoadDataBlock = Variable.New<Func<int, double[]>>();
y.ConstrainFromIndex(b => Variable.Evaluate(LoadDataBlock, b));
Variable<Func<int,int>> GetSizeOfBlock = Variable.New<Func<int, int>>();
sizeOfBlock.SetFromIndex(b => Variable.Evaluate(GetSizeOfBlock, b));
LoadDataBlock.ObservedValue = b => (...);
GetSizeOfBlock.ObservedValue = b => (...);
```

The shortcut `ConstrainFromIndex` takes a delegate mapping a `Variable<int>` to a `Variable<double[]>`, and constrains `y[block]` to equal the output of the delegate. The user-defined delegate `LoadDataBlock` maps an index into an array of doubles. For illustration, `LoadDataBlock` is also a variable in the model, allowing the delegate to be changed at runtime. The user does not need to use `ConstrainFromIndex`: they could set up the same constraints manually. `SetFromIndex` works similarly, filling in an array from a delegate. Note that similar API functions already exist in our `FSharpWrapper` library.

#### Inferring a partitioned array

In some problems, we need to infer a partitioned array. For example, suppose we wanted to infer the array x above. What should `engine.Infer(x)` return? It cannot explicitly return the entire distribution, since this may not fit in memory. However, it can return an _implicit_ array of distributions. An implicit array is an integer count together with a delegate that maps an index into a value. By looping over all indices and calling the delegate, we can access all elements of the array. For example:

```csharp
object xDist = engine.Infer(x);
Console.WriteLine("x block 0 = " + xDist[0]);
```

The delegate can itself return an implicit array, allowing us to represent jagged implicit arrays. This is the same representation used above for observing a partitioned array. However, fetching elements one at a time is not efficient, so we would also need some LINQ-style query operators that could act on this array.
 
Another way that we could have solved this problem is to tell the engine where to put the distribution of each block of x. This could be done by a method `engine.InferByItem`:

```csharp
engine.InferByItem(x, b => StoreMarginalOfBlock(b));
Console.WriteLine("mean=" + engine.Infer(mean));
Console.WriteLine("prec=" + engine.Infer(precision));
```

Notice that engine.InferByItem returns nothing, nor does it do any inference. It only tells the engine how to store the marginals of x, once inference takes place. This is needed because if there are multiple partitioned arrays that we want to infer, we need to call `InferByItem` on each one before running inference. The advantage of this approach over implicit arrays is that the engine does not need to store the marginals after inference is complete. However, it conflicts with the existing behavior of `engine.Infer`.

#### Partitioning a model

The constraint that one element of a partitioned array is in memory at a time implies constraints on how the model is written. For example, you cannot pass an entire partitioned array to a factor. Whenever you use a partitioned array, it must be indexed, and the index must be a loop counter. You cannot index a partitioned array by a random variable or an observed value. The next example illustrates the consequences of these constraints.

#### Matchbox Example

Matchbox provides a more complicated example with multiple partitioned ranges. Here is the non-partitioned version:

```csharp
Range user = new Range(numUsers);
Range item = new Range(numItems);
Range trait = new Range(numTraits);
Range obs = new Range(numObservations);
VariableArray<int> userOfObs = Variable.Array<int>(obs);
VariableArray<int> itemOfObs = Variable.Array<int>(obs);
VariableArray<bool> ratingOfObs = Variable.Array<bool>(obs);
var userTraits = Variable.Array(Variable.Array<double>(trait), user);
userTraits[user][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(user, trait);
var itemTraits = Variable.Array(Variable.Array<double>(trait), item);
itemTraits[item][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(item, trait);
using (Variable.ForEach(obs)) {
  VariableArray<double> products = Variable.Array<double>(trait);
  products[trait] = userTraits[userOfObs[obs]][trait] * itemTraits[itemOfObs[obs]][trait];
  Variable<double> affinity = Variable.Sum(products);
  Variable<double> affinityNoisy = Variable.GaussianFromMeanAndVariance(affinity, affinityNoiseVar);
  ratingOfObs[obs] = (affinityNoisy > 0);
}
userOfObs.ObservedValue = ...;
itemOfObs.ObservedValue = ...;
ratingOfObs.ObservedValue = ...;
```

Notice that it uses observed arrays of indices to fetch the appropriate user and item traits for each observation. In the partitioned version, we want to handle a large number of users and items, so we partition both the users and the items. This partitioning induces a 2D tile structure on the observations, as explained in [Distributed inference](Distributed inference_2.md). Thus in the partitioned model, each (userBlock,itemBlock) pair has separate observation arrays:

```csharp
Range userBlock = new Range(numUserBlocks);
var sizeOfUserBlock = Variable.IArray<int>(userBlock);
Range user = new Range(sizeOfUserBlock[userBlock]);
Range itemBlock = new Range(numItemBlocks);
var sizeOfItemBlock = Variable.IArray<int>(itemBlock);
Range item = new Range(sizeOfItemBlock[itemBlock]);
Range trait = new Range(numTraits);
var numObservationsInBlock = Variable.IArray<int>(userBlock, itemBlock);
Range obs = new Range(numObservationsInBlock[userBlock, itemBlock]);
var userOfObs = Variable.IArray(Variable.Array<int>(obs), userBlock, itemBlock);
var itemOfObs = Variable.IArray(Variable.Array<int>(obs), userBlock, itemBlock);
var ratingOfObs = Variable.IArray(Variable.Array<bool>(obs), userBlock, itemBlock);
var userTraits = Variable.Array(Variable.Array(Variable.Array<double>(trait), user), userBlock);
userTraits[userBlock][user][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(userBlock, user, trait);
var itemTraits = Variable.Array(Variable.Array(Variable.Array<double>(trait), item), itemBlock);
itemTraits[itemBlock][item][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(itemBlock, item, trait);
using (Variable.ForEach(userBlock))
using (Variable.ForEach(itemBlock))
using (Variable.ForEach(obs)) {
  VariableArray<double> products = Variable.Array<double>(trait);
  products[trait] = userTraits[userBlock][userOfObs[userBlock,itemBlock][obs]][trait] * itemTraits[itemBlock][itemOfObs[userBlock,itemBlock][obs]][trait];
  Variable<double> affinity = Variable.Sum(products);
  Variable<double> affinityNoisy = Variable.GaussianFromMeanAndVariance(affinity, affinityNoiseVar);
  ratingOfObs[userBlock,itemBlock][obs] = (affinityNoisy > 0);
}
InferenceEngine engine = new InferenceEngine();
engine.SetAttribute(userBlock, new Partitioned());
engine.SetAttribute(itemBlock, new Partitioned());
sizeOfUserBlock.ObservedValue = Util.IArrayFromFunc(numUserBlocks, ub => 1);
sizeOfItemBlock.ObservedValue = Util.IArrayFromFunc(numItemBlocks, ib => 1);
numObservationsInBlock.ObservedValue = Util.IArrayFromFunc(numUserBlocks, numItemBlocks, (ub, ib) => 1);
userOfObs.ObservedValue = Util.IArrayFromFunc(numUserBlocks, numItemBlocks, (ub, ib) => new int[1]);
itemOfObs.ObservedValue = Util.IArrayFromFunc(numUserBlocks, numItemBlocks, (ub, ib) => new int[1]);
ratingOfObs.ObservedValue = Util.IArrayFromFunc(numUserBlocks, numItemBlocks, (ub, ib) => new bool[1]);
```

Notice that the observed indices, besides being arranged differently, must also have different values. For example, userOfObs now refers to the nth user within a given userBlock. The indices are now local to each block rather than global. An important property of this transformation is that we never refer to a different block than the one we are currently processing. By defining the partitioned model in this way, we can automatically derive the algorithm described in [Distributed inference](Distributed inference_2.md). Also notice that the obs range is dependent on (userBlock,itemBlock), and not marked as partitioned. If the number of observations was so large that even a single (userBlock,itemBlock) chunk cannot fit in memory, then we could partition obs as well.
 
If you think that the above code is too verbose, keep in mind that you can always define aliases. Here is the same model using aliases:

```csharp
Range userBlock = new Range(numUserBlocks);
VariableArray<int> sizeOfUserBlock = Variable.Array<int>(userBlock);
Range user = new Range(sizeOfUserBlock[userBlock]);
Range itemBlock = new Range(numItemBlocks);
VariableArray<int> sizeOfItemBlock = Variable.Array<int>(itemBlock);
Range item = new Range(sizeOfItemBlock[itemBlock]);
Range trait = new Range(numTraits);
VariableArray2D<int> numObservationsInBlock = Variable.Array<int>(userBlock, itemBlock);
Range obs = new Range(numObservationsInBlock[userBlock, itemBlock]);
var userOfObsB = Variable.Array(Variable.Array<int>(obs), userBlock, itemBlock);
var userOfObs = userOfObsB[userBlock,itemBlock];
var itemOfObsB = Variable.Array(Variable.Array<int>(obs), userBlock, itemBlock);
var itemOfObs = itemOfObsB[userBlock,itemBlock];
var ratingOfObsB = Variable.Array(Variable.Array<bool>(obs), userBlock, itemBlock);
var ratingOfObs = ratingOfObsB[userBlock,itemBlock];
var userTraitsB = Variable.Array(Variable.Array(Variable.Array<double>(trait), user), userBlock);
var userTraits = userTraitsB[userBlock];
userTraits[user][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(userBlock, user, trait);
var itemTraitsB = Variable.Array(Variable.Array(Variable.Array<double>(trait), item), itemBlock);
var itemTraits = itemTraitsB[itemBlock];
itemTraits[item][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(itemBlock, item, trait);
using (Variable.ForEach(userBlock))
using (Variable.ForEach(itemBlock))
using (Variable.ForEach(obs)) {
  VariableArray<double> products = Variable.Array<double>(trait);
  products[trait] = userTraits[userOfObs[obs]][trait] * itemTraits[itemOfObs[obs]][trait];
  Variable<double> affinity = Variable.Sum(products);
  Variable<double> affinityNoisy = Variable.GaussianFromMeanAndVariance(affinity, affinityNoiseVar);
  ratingOfObs[obs] = (affinityNoisy > 0);
}
```

Using aliases, the inner loop now looks exactly like the non-partitioned version.

#### TrueSkill Example

A partitioned version of the two-player TrueSkill model ends up quite similar to Matchbox. Recall the [compact TrueSkill model](How to represent large irregular graphs.md):

```csharp
Range player = new Range(nPlayers);
VariableArray<double> skill = Variable.Array<double>(player);
skill[player] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(player);
Range game = new Range(nGames);
VariableArray<int> winnerVar = Variable.Observed(winner, game);
VariableArray<int> loserVar = Variable.Observed(loser, game);
Variable<double> winner_performance = Variable.GaussianFromMeanAndVariance(skill[winnerVar[game]], 1);
Variable<double> loser_performance = Variable.GaussianFromMeanAndVariance(skill[loserVar[game]], 1);
Variable.ConstrainTrue(winner_performance > loser_performance);
```

Partitioning over games is straightforward and similar to the Learning a Gaussian example above. Suppose we want to partition over players. The skill array turns into a jagged array `skill[block][playerInBlock]`. The indexing expression `skill[winnerVar[game]]` needs to turn into a block-local indexing expression `skill[block][winnerInBlock[block][game]]`. For the indexing expression `skill[loserVar[game]]`, we cannot use the same block, since the loser could be any player, in any block. Thus we need to iterate over two player blocks. These two blocks induce a 2D game partition. Technically this violates the constraint that only one block is in memory at a time, but it still satisfies the syntactic constraint that a partitioned array can only be indexed by loop counters, and thus it can still be distributed efficiently.

```csharp
Range block = new Range(numBlocks);
VariableArray<int> sizeOfBlock = Variable.IArray<int>(block);
Range player = new Range(sizeOfBlock[block]);
Range block2 = block.Clone();
Range player2 = new Range(sizeOfBlock[block2]);
var skill = Variable.Array(Variable.Array<double>(player), block);
skill[block][player] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(block,player);
var numGames = Variable.IArray<int>(block, block2);
Range game = new Range(numGames[block,block2]);
var winnerVar = Variable.IArray(Variable.Array<int>(game), block, block2);
var loserVar = Variable.IArray(Variable.Array<int>(game), block, block2);
using (Variable.ForEach(block))
using (Variable.ForEach(block2))
using (Variable.ForEach(game)) {
  Variable<double> winner_performance = Variable.GaussianFromMeanAndVariance(skill[block][winnerVar[block, block2][game]], 1);
  Variable<double> loser_performance = Variable.GaussianFromMeanAndVariance(skill[block2][loserVar[block, block2][game]], 1);
  Variable.ConstrainTrue(winner_performance > loser_performance);
}
InferenceEngine engine = new InferenceEngine();
engine.SetAttribute(block, new Partitioned());
engine.SetAttribute(block2, new Partitioned());
```

Notice that (winnerVar,loserVar) are now local indices within a player block. The game range is now dependent on (block,block2) and not marked as partitioned. If the number of games was so large that a (block,block2) chunk could not fit in memory, then we could partition _game_ as well. An interesting question is how to handle games with a variable number of players, such as team games. The solution is similar to the sparse factorized Bayes Point Machine, discussed next.

#### Sparse Factorized Bayes Point Machine

In a sparse factorized Bayes Point Machine, we have an array of weights, and each observation involves an arbitrary subset of the weights. The non-partitioned model is:

```csharp
Range obs = new Range(numObservations);
Range feature = new Range(numFeatures);
VariableArray<double> w = Variable.Array<double>(feature);
w[feature] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(feature);
VariableArray<int> xValueCount = Variable.Array<int>(obs);
Range featureOfObs = new Range(xValueCount[obs]);
var xValues = Variable.Array(Variable.Array<double>(featureOfObs), obs);
var xIndices = Variable.Array(Variable.Array<int>(featureOfObs), obs);
VariableArray<bool> y = Variable.Array<bool>(obs);
using (Variable.ForEach(obs)) {
  VariableArray<double> product = Variable.Array<double>(featureOfObs);
  VariableArray<double> wSparse = Variable.Subarray(w, xIndices[obs]);
  product[featureOfObs] = xValues[obs][featureOfObs] * wSparse[featureOfObs];
  Variable<double> score = Variable.Sum(product);
  y[obs] = (score > 0);
}
```

Suppose we have a large number of features, so that we want to partition both the observations and the features. Following the same recipe as previous models, arrays indexed by `[obs]` now need to be indexed by `[obsBlock][obs]`, and arrays indexed by `[feature]` now need to be indexed by `[featureBlock][feature]`. Arrays such as `xValues` which are indexed by both obs and feature now need to have 4 indices. The only tricky bit is the sum. The product array becomes a jagged array, so we need to sum twice: over the features in each block and then over blocks. Here is one approach:

```csharp
Range obsBlock = new Range(numObsBlocks);
var sizeOfObsBlock = Variable.IArray<int>(obsBlock);
Range obs = new Range(sizeOfObsBlock[obsBlock]);
Range featureBlock = new Range(numFeatureBlocks);
var sizeOfFeatureBlock = Variable.IArray<int>(featureBlock);
Range feature = new Range(sizeOfFeatureBlock[featureBlock]);
var w = Variable.Array(Variable.Array<double>(feature), featureBlock);
w[featureBlock][feature] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(featureBlock,feature);
var xValueCount = Variable.IArray(Variable.Array(Variable.Array<int>(featureBlock), obs), obsBlock);
Range featureOfObs = new Range(xValueCount[obsBlock][obs][featureBlock]);
var xValues = Variable.IArray(Variable.Array(Variable.Array(Variable.Array<double>(featureOfObs), featureBlock), obs), obsBlock);
var xIndices = Variable.IArray(Variable.Array(Variable.Array(Variable.Array<int>(featureOfObs), featureBlock), obs), obsBlock);
var y = Variable.IArray(Variable.Array<bool>(obs), obsBlock);
using (Variable.ForEach(obsBlock))
using (Variable.ForEach(obs)) {
  VariableArray<double> sumOfBlock = Variable.Array<double>(featureBlock);
  using (Variable.ForEach(featureBlock)) {
    VariableArray<double> product = Variable.Array<double>(featureOfObs);
    VariableArray<double> wSparse = Variable.Subarray(w[featureBlock], xIndices[obsBlock][obs][featureBlock]);
    product[featureOfObs] = xValues[obsBlock][obs][featureBlock][featureOfObs] * wSparse[featureOfObs];
    sumOfBlock[featureBlock] = Variable.Sum(product);
  }
  Variable<double> score = Variable.Sum(sumOfBlock);
  y[obsBlock][obs] = (score > 0); 
}
InferenceEngine engine = new InferenceEngine();
engine.SetAttribute(obsBlock, new Partitioned());
engine.SetAttribute(featureBlock, new Partitioned());
```

Notice the meaning of xIndices has changed. It is now the local index of a feature within a feature block. A key operation here is `Variable.Sum(sumOfBlock)`. This is taking the sum of a distributed array. This violates the earlier syntactic constraint that we cannot pass a partitioned array to a factor. In order to support this, we need a message operator for `Sum` that can accept a distributed array and perform the necessary communication between machines to compute the result. We need similar support for all factors that take arrays as input or output, such as `AllTrue` and `Replicate`.
 
A potential problem with the above approach is that it involves a double loop over all observations and all feature blocks. If the data is very sparse, then only a fraction of the feature blocks may end up being used by any single observation. We'd like to loop over only the feature blocks that are actually used by the observation. One way to do this is to make a 2D tiling of (obsBlock,featureBlock) pairs, and loop over the observations involved in each tile.

```csharp
Range obsBlock = new Range(numObsBlocks);
var sizeOfObsBlock = Variable.IArray<int>(obsBlock);
Range obs = new Range(sizeOfObsBlock[obsBlock]);
Range featureBlock = new Range(numFeatureBlocks);
var sizeOfFeatureBlock = Variable.IArray<int>(featureBlock);
Range feature = new Range(sizeOfFeatureBlock[featureBlock]);
var w = Variable.Array(Variable.Array<double>(feature), featureBlock);
w[featureBlock][feature] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(featureBlock, feature);
var xValueCount = Variable.IArray(Variable.Array(Variable.Array<int>(featureBlock), obs), obsBlock);
Range featureOfObs = new Range(xValueCount[obsBlock][obs][featureBlock]).Named("userFeature");
var y = Variable.IArray(Variable.Array<bool>(obs), obsBlock);
// changes start here...
var numObsInFeatureBlock = Variable.IArray(Variable.Array<int>(featureBlock), obsBlock);
Range obsInFeatureBlock = new Range(numObsInFeatureBlock[obsBlock][featureBlock]);
var obsIndex = Variable.IArray(Variable.Array(Variable.Array<int>(obsInFeatureBlock), featureBlock), obsBlock);
var xValues = Variable.IArray(Variable.Array(Variable.Array(Variable.Array<double>(featureOfObs), obsInFeatureBlock), featureBlock), obsBlock);
var xIndices = Variable.IArray(Variable.Array(Variable.Array(Variable.Array<int>(featureOfObs), obsInFeatureBlock), featureBlock), obsBlock);
var numFeatureBlocksOfObs = Variable.IArray(Variable.Array<int>(obs), obsBlock);
Range featureBlockOfObs = new Range(numFeatureBlocksOfObs[obsBlock][obs]); // not partitioned
using (Variable.ForEach(obsBlock)) {
  var sumOfBlock = Variable.Array(Variable.Array<double>(obsInFeatureBlock), featureBlock);
  using (Variable.ForEach(featureBlock))
  using (Variable.ForEach(obsInFeatureBlock)) {
    VariableArray<double> wSparse = Variable.Subarray(w[featureBlock], xIndices[obsBlock][featureBlock][obsInFeatureBlock]);
    VariableArray<double> product = Variable.Array<double>(featureOfObs);
    product[featureOfObs] = xValues[obsBlock][featureBlock][obsInFeatureBlock][featureOfObs] * wSparse[featureOfObs];
    sumOfBlock[featureBlock][obsInFeatureBlock] = Variable.Sum(product);
  }
  // sumOfBlockTransposed is indexed by [obs][featureBlockOfObs]
  var sumOfBlockTransposed = Variable.Transpose(sumOfBlock, obsIndex[obsBlock], obs, featureBlockOfObs);
  using (Variable.ForEach(obs)) {
    Variable<double> score = Variable.Sum(sumOfBlockTransposed[obs]);
    y[obsBlock][obs] = (score > 0);
  }
} 
InferenceEngine engine = new InferenceEngine();
engine.SetAttribute(obsBlock, new Partitioned());
engine.SetAttribute(featureBlock, new Partitioned());
```

This approach requires a new factor `Variable.Transpose` that transposes a jagged array, where in this case the first dimension of the input array is distributed.

#### Partitioning a model without arrays

Since Infer.NET allows branching on a loop index, it is possible to partition a model without using an array. This is useful for partitioning a model into dissimilar parts. Consider the Learning a Gaussian model without arrays:

```csharp
Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100);
Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1); 
for (int i = 0; i < data.Length; i++) {
  Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision);
  x.ObservedValue = data[i];
}
```

Suppose we want to process each data point in parallel, without putting them into an array. We can create a partitioned range and then branch on it:

```csharp
Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100);
Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
Range block = new Range(data.Length);
using (ForEachBlock fb = Variable.ForEach(block)) {
  for (int i = 0; i < data.Length; i++) {
    using (Variable.Case(fb.Index, i)) {
      Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision);
      x.ObservedValue = data[i];
    }
  }
}
InferenceEngine engine = new InferenceEngine();
engine.SetAttribute(block, new Partitioned());
```

Notice that `block` is never used here by any array. It is only used to specify how the model should be divided into blocks. Each block is defined by a `Variable.Case` which can contain arbitrary modelling elements.

#### Dependent Partitioned Ranges

In Infer.NET, a _dependent_ range is a range whose size depends on another range. If a dependent range is marked as partitioned, then the range it depends on must also be partitioned. To see why, consider a hierarchical Gaussian model, similar to "Learning a Gaussian" but with multiple rows of data, each having a separate mean:

```csharp
Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
Range row = new Range(numRows);
VariableArray<double> mean = Variable.Array<double>(row);
mean[row] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(row);
VariableArray<int> sizeOfRow = Variable.Array<int>(row);
Range item = new Range(sizeOfRow[row]);
var x = Variable.Array(Variable.Array<double>(item), row);
x[row][item] = Variable.GaussianFromMeanAndPrecision(mean[row], precision).ForEach(item);
```

In this model, `item` is a dependent range, since each row can have a different amount of data. Suppose some rows are very large, so we want to partition them by introducing an `itemBlock` range. A first attempt might look like this:

```csharp
Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
Range row = new Range(numRows);
VariableArray<double> mean = Variable.Array<double>(row);
mean[row] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(row);
Range itemBlock = new Range(numItemBlocks[row]); // problem!
var sizeOfRow = Variable.Array(Variable.Array<int>(itemBlock), row);
Range item = new Range(sizeOfRow[row][itemBlock]).Named("n");
var x = Variable.Array(Variable.Array(Variable.Array<double>(item), itemBlock), row).Named("x");
x[row][itemBlock][item] = Variable.GaussianFromMeanAndPrecision(mean[row], precision).ForEach(itemBlock, item);InferenceEngine engine = new InferenceEngine();
engine.SetAttribute(itemBlock, new Partitioned());
```

However, this is not valid. Recall that the meaning of a partitioned range is that the nth element of all arrays indexed by that range is in memory at any time. In this model, this means that `x[row][n]` is in memory at any time, for all rows. But since the number of item blocks depends on the row, this isn't possible. This conflict arises because `itemBlock` is a dependent range and marked partitioned, but the range it depends on (`row`) is not partitioned. (You might argue that if there is no block numbered `n`, then we could just treat it as a block of size zero. But that is equivalent to saying that each row had the same number of blocks in the first place.)

We can resolve this conflict in several ways. One way is to mark `row` as partitioned. This requires that only one row is in memory at a time, which may not be what we want. Another approach is to use the same number of blocks in every row, removing the dependence between `row` and `itemBlock`:

```csharp
Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
Range row = new Range(numRows);
VariableArray<double> mean = Variable.Array<double>(row);
mean[row] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(row);
Range itemBlock = new Range(numItemBlocks); // fixed
var sizeOfRow = Variable.Array(Variable.Array<int>(row), itemBlock);
Range item = new Range(sizeOfRow[itemBlock][row]).Named("n");
var x = Variable.Array(Variable.Array(Variable.Array<double>(item), row), itemBlock).Named("x");
using (Variable.ForEach(itemBlock)) {
  x[itemBlock][row][item] = Variable.GaussianFromMeanAndPrecision(mean[row], precision).ForEach(item);
}
InferenceEngine engine = new InferenceEngine();
engine.SetAttribute(itemBlock, new Partitioned());
```

Notice that item is still a dependent range but `itemBlock` is not. Because `itemBlock` does not depend on `row`, we can make it the first index of the arrays. If a row doesn't have enough items, we could pad it with empty blocks, or use fewer items per block. A third approach to resolve the conflict is to introduce a `rowBlock` partition, and make `itemBlock` depend on `rowBlock`:

```csharp
Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
Range rowBlock = new Range(numRowBlocks);
VariableArray<int> sizeOfRowBlock = Variable.Array<int>(rowBlock);
Range row = new Range(sizeOfRowBlock[rowBlock]);
var mean = Variable.Array(Variable.Array<double>(row), rowBlock);
mean[rowBlock][row] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(rowBlock, row);
VariableArray<int> numItemBlocks = Variable.Array<int>(rowBlock);
Range itemBlock = new Range(numItemBlocks[rowBlock]);
var sizeOfRow = Variable.Array(Variable.Array(Variable.Array<int>(row), itemBlock), rowBlock);
Range item = new Range(sizeOfRow[rowBlock][itemBlock][row]).Named("n");
var x = Variable.Array(Variable.Array(Variable.Array(Variable.Array<double>(item), row), itemBlock), rowBlock).Named("x");
using (Variable.ForEach(rowBlock))
using (Variable.ForEach(itemBlock)) {
  x[rowBlock][itemBlock][row][item] = Variable.GaussianFromMeanAndPrecision(mean[rowBlock][row], precision).ForEach(item);
}
 
 
InferenceEngine engine = new InferenceEngine(); 
engine.SetAttribute(rowBlock, new Partitioned());
engine.SetAttribute(itemBlock, new Partitioned());
```

In this approach, we could place rows of similar length into the same row block, so that it is reasonable to give them the same number of item blocks.