---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## How to represent large irregular graphs

In some problems, you have a large set of interacting entities, where the interactions are irregular. This interaction graph can be represented efficiently in Infer.NET by making use of the ability to [index arrays by observed variables](Indexing arrays by observed variables.md). 

For illustration, let's consider the problem of inferring the skill levels of players given a set of game outcomes. Suppose there are 4 players (numbered 0,1,2,3) and the game outcomes are:

```
[game 0] player 2 beat player 0  
[game 1] player 2 beat player 1  
[game 2] player 3 beat player 0  
[game 3] player 3 beat player 2
```

The outcomes define a graph of interactions amongst the player performances. That is, we know that in game 0, player 2 performed better than player 0. To get the skill of each player, we assume that his performance was Gaussian-distributed around his skill. The constraints on the performances then gives information about the skills. (This is the model used by [Microsoft TrueSkill](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/).) We will consider two different ways of writing this model in Infer.NET. The first uses a separate Variable object for each player. The second uses a VariableArray of players, and runs much faster.

### Big, slow model: A variable for each player

An explicit encoding of the game outcomes into an Infer.NET model would be as follows. The game outcomes are represented by two arrays, giving the index of the player that won or lost:

```csharp
int nPlayers = 4;
int nGames = 4;
int[] winner = new int[] { 2, 2, 3, 3 };  
int[] loser = new int[] { 0, 1, 0, 2 };
```

We create a C# array holding a Variable object for each player's skill:

```csharp
Variable<double>[] skill = new Variable<double>[nPlayers];  
for (int player = 0; player < nPlayers; player++) {  
  skill[player] = Variable.GaussianFromMeanAndVariance(0, 100);  
}
```

Now each game outcome can be written as a constraint between the winning and losing performance, like so:

```csharp
for(int game = 0; game < nGames; game++) {
  Variable<double> winner_performance =         
      Variable.GaussianFromMeanAndVariance(skill[winner[game]], 1);  
  Variable<double> loser_performance =
      Variable.GaussianFromMeanAndVariance(skill[loser[game]], 1);  
  Variable.ConstrainTrue(winner_performance > loser_performance);  
}  

InferenceEngine engine = new InferenceEngine();  
for (int player = 0; player < nPlayers; player++) {  
  Console.WriteLine("player {0} skill = {1}", player, engine.Infer(skill[player]));  
}
```

When run, this code prints out:

```
player 0 skill = Gaussian(-6.983, 51.46)  
player 1 skill = Gaussian(-6.544, 56.45)  
player 2 skill = Gaussian(3.009, 35.28)  
player 3 skill = Gaussian(10.52, 45.54)
```

As expected, player 3 is inferred to have the highest skill.

We call this approach 'unrolling the graph', similar to unrolling a loop in C#. In this approach, the structure of the game graph is baked into the model definition, allowing a specialized inference schedule to be computed. However, if there are many players or games, a large amount of code will be generated, making Infer.NET run slowly. (See the [Learning a Gaussian tutorial](Learning a Gaussian tutorial.md) for a simple example of this.) Furthermore, unrolling prevents the game graph from changing at runtime.

### Small, fast model: Indexing a VariableArray

Indexing a VariableArray provides both a more efficient model definition and the ability to change the game graph at runtime. The idea is to represent the player variables with VariableArrays:

```csharp
Range player = new Range(nPlayers).Named("player");  
VariableArray<double> skill = Variable.Array<double>(player).Named("skill");  
skill[player] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(player);
```

The game outcomes are copied into two VariableArrays, giving the index of the player that won or lost:

```csharp
Range game = new Range(nGames).Named("game");  
VariableArray<int> winnerVar = Variable.Observed(winner, game).Named("winner");  
VariableArray<int> loserVar = Variable.Observed(loser, game).Named("loser");
```

Note these variables can be set and changed at any time, even after the model is compiled. 

For each game, we want to make a constraint that the winner had a better performance than the loser. This is accomplished with similar code as before, but excluding the C# loop (the constraint is automatically applied across all games):

```csharp
Variable<double> winner_performance = Variable.GaussianFromMeanAndVariance(skill[winnerVar[game]], 1);  
Variable<double> loser_performance = Variable.GaussianFromMeanAndVariance(skill[loserVar[game]], 1);  
Variable.ConstrainTrue(winner_performance > loser_performance);
```

Finally, we infer the entire array of player skills:

```csharp
InferenceEngine engine = new InferenceEngine();  
Console.WriteLine(engine.Infer(skill));
```

The result is: 

```
[0] Gaussian(-6.983, 51.46)  
[1] Gaussian(-6.544, 56.45)  
[2] Gaussian(3.009, 35.28)  
[3] Gaussian(10.52, 45.54)
```

At this point, you can change the observed values of the winner and loser arrays and get new results quickly.
