---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Calling Infer.NET from IronPython](Calling Infer.NET from IronPython.md)

## The two coins example in IronPython

For a description of this tutorial and the C# code please see [the two coins tutorial](Two coins tutorial.md).

### IronPython script
```python
#-----------------------------------------------------------------------------------  
# Infer.NET IronPython example: Two Coins  
#-----------------------------------------------------------------------------------  

import InferNetWrapper  
from InferNetWrapper import *  

# two coins example  
def two_coins() :  
 print("\n\n------------------ Infer.NET Two Coins example ------------------\n");  

 # The model  
 b = Microsoft.ML.Probabilistic.Distributions.Bernoulli(0.5)  
 firstCoin = Variable.Bernoulli(0.5)  
 secondCoin = Variable.Bernoulli(0.5)  
 bothHeads = firstCoin & secondCoin  

 # The inference  
 engine = InferenceEngine()  
 print "Probability both coins are heads:", engine.Infer(bothHeads)  
 bothHeads.ObservedValue = False  
 print "Probability distribution over firstCoin:", engine.Infer(firstCoin)
 ```
