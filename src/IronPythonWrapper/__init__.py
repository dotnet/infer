# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
import System
from System import Console
from System import Array
from System import IO
import clr
import sys

infernetpackagedir = System.IO.Path.GetDirectoryName(System.IO.FileInfo(__file__).FullName)

#add path to infer.net dlls .................................
clr.AddReferenceToFileAndPath(infernetpackagedir+"\\Microsoft.ML.Probabilistic.dll")
clr.AddReferenceToFileAndPath(infernetpackagedir+"\\Microsoft.ML.Probabilistic.Compiler.dll")

#---import all namespaces----------------
import Microsoft.ML.Probabilistic
import Microsoft.ML.Probabilistic.Models
import Microsoft.ML.Probabilistic.Distributions
import Microsoft.ML.Probabilistic.Collections
import Microsoft.ML.Probabilistic.Factors
import Microsoft.ML.Probabilistic.Math
from Microsoft.ML.Probabilistic import * 

#---------import all classes and methods from above namespaces-----------
from Microsoft.ML.Probabilistic.Distributions import *
from Microsoft.ML.Probabilistic.Models import *
from Microsoft.ML.Probabilistic.Collections import *
from Microsoft.ML.Probabilistic.Factors import *
from Microsoft.ML.Probabilistic.Math import *

#---import wrapper---
import InferNetWrapper
