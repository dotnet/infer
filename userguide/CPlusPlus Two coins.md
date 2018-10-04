---
layout: default 
--- 
[Infer.NET user guide](index.md) : [C++/CLI](CPlusPlus.md)

## The two coins example in C++/CLI

For a description of this tutorial and the C# code please see [the two coins tutorial](Two coins tutorial.md).

```c++
#include "stdafx.h"  
using namespace System;  
using Microsoft::ML::Probabilistic::Models;

void TwoCoins()  
{  
  // The model  
  Variable<bool>^ firstCoin = Variable::Bernoulli(0.5)->Named("firstCoin");  
  Variable<bool>^ secondCoin = Variable::Bernoulli(0.5)->Named("secondCoin");  
  Variable<bool>^ bothHeads = (firstCoin & secondCoin)->Named("bothHeads");  

  // The inference  
  InferenceEngine^ ie = gcnew InferenceEngine();  
  Console::WriteLine("Probability both coins are heads: " + ie->Infer(bothHeads));  
  bothHeads->ObservedValue = false;  
  Console::WriteLine("Probability distribution over first coin: " + ie->Infer(firstCoin));  
}  

int main()  
{  
  TwoCoins();  
  return 0;  
}
```
