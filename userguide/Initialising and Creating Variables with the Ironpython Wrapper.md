---
layout: default
---
[Infer.NET User Guide](index.md) : [Calling Infer.NET from IronPython](Calling Infer.NET from IronPython.md)

## The IronPython Wrapper

The syntax for creating [VariableArrays](Arrays and ranges.md) of various depths and dimensions can be quite complicated. In addition, when the `VariableArrays` need to be initialised for inference (typically for latent variable models which require initialisation to break symmetry), the initialisers are in the form of `DistributionArray` objects rather than arrays of Distribution objects. Current versions of IronPython have difficult in dealing with the Infer.NET signatures for these operations. The Infer.NET IronPython wrapper defines two classes (**CreateArray** and **InitArray**) which have static methods to deal with these situations. **CreateArray** has static methods for creating jagged variable arrays and **InitArray** has static methods for initialising variable arrays. This wrapper also serves as an example for other unanticipated scenarios which may arise.

### Importing the Wrapper

Use of the Infer.NET IronPython wrapper is optional. If you do want to make use of it, you will need to import it:

```IronPythonWrapper
import IronPythonWrapper  
from IronPythonWrapper import *
```

The wrapper can be found in the [src\\IronPythonWrapper](https://github.com/dotnet/infer/tree/master/src/IronPythonWrapper) folder.

### Examples

As an example, the method ```csharp
InitArray.init_var_array(var_array, init_func, length)
``` takes 3 parameters: a `VariableArray`, a function for initialising each element of the array, and the length of the `VariableArray`. The example below shows how this function can be used to break symmetries in the Mixture of Gaussians Tutorial:
```python
#Define an initialisation function to break symmetries  
def init_func() :
  return Discrete.PointMass(Rand.Int(2), 2)  

# Create latent indicator variable for each data point  
length = 300  
n = Range(length)  
z = Variable.Array[int](n).Named("z")  
InitArray.init_var_arr(z,InitFunc,length)
```

The domain type of the distribution returned by the initialisation function must be the same as the type of the Variable to be initialised, in this case integer.

When jagged and multidimensional `VariableArrays` are to be initialised the size of each dimension and the sizes of the jagged elements are expressed using IronPython integer Lists as parameters. For example the method `InitArray.init_jagged_var_array(var_array, init_func, jagged_lengths)` takes 3 parameters: a `VariableArray`, an initialisation function, and an IronPython List of lengths of the inner jagged array elements.

The example below creates a jagged array of 2-Dimensional elements initialised in a similar fashion to the Mixture of Gaussians Tutorial shown above, excapt that (a) it uses the helper method `CreateArray.create_var_arr_of_2D_arr` to create the variable array, and (b) it uses the corresponding 2D version of the initialiser method.

```python
#Define an initialisation function to break symmetries
def init_func() : return Discrete.PointMass(Rand.Int(2), 2)  

# Create IronPython List for jagged 2D sizes  
jagged_2D_sizes = [ [1,1],[2,2],[3,3],[4,4],[5,5] ]  

# Create latent indicator variable for each data point  
z = CreateArray.create_var_arr_of_2D_arr(jagged_2D_sizes, int)  

#Initialise the variable array  
InitArray.init_jagged_2D_var_arr(z, init_func, jagged_2D_sizes)
```
