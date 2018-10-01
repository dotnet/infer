---
layout: default
---

[Infer.NET user guide](index.md)

## Defining variables that are observed

In this section, we provide examples of defining variables that are observed. For convenience, we assume the element type is Vector, while they can be replaced with any other element type.

| **Data definition** | **Variable Definition** |
|---------------------------------------------|
| `Vector data = new Vector();` | `Variable<Vector> dataGiven = Variable.New<Vector>();` <br /> `dataGiven.ObservedValue = data;` |
| `Vector[] data = new Vector[nItems];` | `VariableArray<Vector> dataGiven = Variable.Array<Vector>(item);` <br /> `dataGiven.ObservedValue= data;` |
| `Vector[][] data = new Vector[nGroups][];` <br /> `for (int g = 0; g < nGroups; g++)` <br /> `{` <br /> `data[g] = new Vector[nItems];` <br /> `}` | `Range item = new Range(nItems);` <br /> `VariableArray<Vector>[] dataGiven = new VariableArray<Vector>[nGroups];` <br /> `for (int g = 0; g < nGroups; g++)` <br /> `{` <br /> `dataGiven[g]= Variable.Array<Vector>(item);` <br /> `}` <br /> `for (int g = 0; g < nGroups; g++)` <br /> `{` <br /> `dataGiven[g].ObservedValue = data[g];` <br /> `}`|
| `Vector[,] data = new Vector[nItems, nFeatures];` | `Range item = new Range(nItems);` <br /> `Range feature = new Range(nFeatures);` <br /> `VariableArray2D<Vector> dataGiven = Variable.Array<Vector>(item, feature);` <br /> `dataGiven.ObservedValue = data;` |
| `Vector[][,] data = new Vector[nGroups][,];` <br /> `for (int g = 0; g < nGroups; g++)` <br /> `{` <br /> `data[g] = new Vector[nItems, nFeatures];` <br /> `}` | `Range item = new Range(nItems);` <br /> `Range feature = new Range(nFeatures);` <br /> `VariableArray2D<Vector>[] dataGiven = new VariableArray2D<Vector>[nGroups];` <br /> `for (int g = 0; g < nGroups; g++)` <br /> `{` <br /> `dataGiven[g] = Variable.Array<Vector>(item, feature);` <br /> `}` <br /> `for (int g = 0; g < nGroups; g++)` <br /> `{` <br />  `dataGiven[g].ObservedValue = data[g];` <br /> `}` |
| `Vector[,][] data = new Vector[nPlaces, nGroups][];` <br /> `for (int p = 0; p < nPlaces; p++) {` <br /> `for (int g = 0; g < nGroups; g++) {` <br /> `data[p, g] = new Vector[nItems];` <br /> `}` <br /> `}` | `Range item = new Range(nItems);` <br /> `VariableArray<Vector>[,] dataGiven = new VariableArray<Vector>[nPlaces, nGroups];` <br /> `for (int p = 0; p < nPlaces; p++) {` <br /> `for (int g = 0; g < nGroups; g++) {` <br /> `dataGiven[p, g] = Variable.Array<Vector>(item);` <br /> `}` <br /> `}` <br /><br /> `for (int p = 0; p < nPlaces; p++) {` <br /> `for (int g = 0; g < nGroups; g++) {` <br /> `dataGiven[p, g].ObservedValue = data[p, g];` <br /> `}` <br /> `}` |