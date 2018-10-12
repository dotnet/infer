---
layout: default 
--- 
[Infer.NET user guide](index.md)

## The two coins example in Matlab

For a description of this tutorial and the C# code please see [the two coins tutorial](Two coins tutorial.md).

### Matlab script

```matlab
dllFolder = 'c:\Program Files\Infer.NET\Bin';  
NET.addAssembly(fullfile(dllFolder,'Microsoft.ML.Probabilistic.Compiler.dll'));  
NET.addAssembly(fullfile(dllFolder,'Microsoft.ML.Probabilistic.dll'));  
import Microsoft.ML.Probabilistic.Distributions.*  
import Microsoft.ML.Probabilistic.Models.*  
import Microsoft.ML.Probabilistic.*  
import System.*  

% The model  
b = Bernoulli(0.5);  
firstCoin = Variable.Bernoulli(0.5);  
secondCoin = Variable.Bernoulli(0.5);  
% methods(firstCoin) gives all operations that can be performed  
bothHeads = firstCoin.op_BitwiseAnd(firstCoin, secondCoin);  

% The inference  
engine = InferenceEngine();  
disp(['Probability both coins are heads: ' char(engine.Infer(bothHeads).ToString)]);  
bothHeads.ObservedValue = false;  
disp(['Probability distribution over firstCoin: ' char(engine.Infer(firstCoin).ToString)]);
```
