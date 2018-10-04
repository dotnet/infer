---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Calling Infer.NET from IronPython](Calling Infer.NET from IronPython.md)

## Running Infer.NET from IronPython

### Setting the IRONPYTHONPATH Environment Variable

You can set the IRONPYTHONPATH environment variable to point to the folders where your Iron Python modules are; multiple folders should be separated by semi-colons.

### Importing Modules

Typically the following system namespaces will typically be imported:

```python
import System  
from System import Console  
from System import Array  
from System import IO  
import clr  
import sys
```

The path to the Infer.NET DLLs can optionally be set via `sys.path.append(...)`. Alternatively the DLLs can be copied to the current folder. The Infer.NET DLLs must then be referenced as follows:

```python
#sys.path.append(r"<pathname of Infer.NET dlls>")  
clr.AddReferenceToFile("Microsoft.ML.Probabilistic.Compiler.dll")  
clr.AddReferenceToFile("Microsoft.ML.Probabilistic.dll")
```

The following shows the typical Infer.NET namespaces that need to be imported. The classes within a module can be accessed without using the namespace prefix by using the statement from module import *

```python
# Import all namespaces  
import Microsoft.ML.Probabilistic  
import Microsoft.ML.Probabilistic.Models  
import Microsoft.ML.Probabilistic.Distributions  
import Microsoft.ML.Probabilistic.Factors  
import Microsoft.ML.Probabilistic.Math

# Import all classes from above namespaces  
from Microsoft.ML.Probabilistic import *  
from Microsoft.ML.Probabilistic.Distributions import *  
from Microsoft.ML.Probabilistic.Models import *  
from Microsoft.ML.Probabilistic.Factors import *  
from Microsoft.ML.Probabilistic.Math import *

# Optionally import the IronPythonWrapper and associated classes  
import IronPythonWrapper  
from IronPythonWrapper import *
```

Alternatively, you can make use of the package folder provided as part the Infer.NET installation.

### InferNet Package

The InferNetWrapper package folder can be found in the [src\\IronPythonWrapper](https://github.com/dotnet/infer/tree/master/src/IronPythonWrapper).

For IronPython, you can copy this folder into the Lib\\site-packages folder in the IronPython installation. For [Sho](http://research.microsoft.com/sho/), you can copy this folder into the Packages folder in the Sho installation folder. You will need administrator privileges to do these copies. Once they have been copied, you can just import the package to start working with Infer.NET. For example:

```shell
>>> import InferNetWrapper  
>>> import InferNetWrapper as inet  
>>> firstCoin = inet.Variable.Bernoulli(0.5)  
>>> secondCoin = inet.Variable.Bernoulli(0.5)  
>>> bothHeads = firstCoin & secondCoin  
>>> bothHeads.ObservedValue = False  
>>> ie = inet.InferenceEngine()  
>>> result = ie.Infer(firstCoin)  
Compiling model...done. 
>>> result  
<Microsoft.ML.Probabilistic.Distributions.Bernoulli object at 0x000000000000002B [Bernoulli(0.3333)]>  
>>>
```

Note that when you call the Infer method it will, by default, write the generated inference code to disk, so you must have write permission in your current working directory.

### Running the Tutorial Examples from the Console

Assuming you have imported the modules or installed the package as above, it is straightforward to run the tutorial examples. Navigate to the folder containing the examples (or put that folder into your IronPython path), then import the tutorial modules and run. For example:

```shell
import TwoCoins  
>>> TwoCoins.two_coins()  


------------------ Infer.NET Two Coins example ------------------  

Probability both coins are heads:Compiling model...done. 
Bernoulli(0.25)  
Probability distribution over firstCoin:Compiling model...done. 
Bernoulli(0.3333)  
>>>
```
