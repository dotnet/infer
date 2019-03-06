---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## Chess Analysis

This example is a model of the outcomes of chess games. The model is similar to TrueSkill except each player is described by two latent variables instead of one. Each player is assumed to have a real-valued latent skill, representing their ability to win, and a real-valued latent draw margin, representing their ability to force a draw. A player wins only when their performance exceeds the performance plus draw margin of their opponent. The latent variables are allowed to vary over time. This model was used in the paper "[TrueSkill Through Time: Revisiting the History of Chess](https://www.microsoft.com/en-us/research/wp-content/uploads/2008/01/NIPS2007_0931.pdf)". You can run this example in the [Examples Browser](The examples browser.md).  The source code is in [ChessAnalysis.cs](https://github.com/dotnet/infer/blob/master/src/Tutorials/ChessAnalysis.cs).

The model includes several parameters, such as the performance variance and the expected change in skill from year to year. In the paper, these parameters were tuned by maximizing model evidence. In the example, these parameters are learned automatically as part of model training (if you enable the **inferParameters** option).

The example code starts by generating values for the parameters, latent variables, and game outcomes, by sampling from the model. Then it runs inference and compares the estimated values to the true ones. The player variables over multiple years are stored in one big jagged array. The variables for successive years are linked via a Markov chain, implemented by [offset indexing](Markov chains and grids.md) in Infer.NET.
