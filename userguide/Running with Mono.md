---
layout: default
---
[Infer.NET User Guide](index.md)

## Mono Support

This version of Infer.NET has been tested with Mono version 5.0.

The following examples will not build or run on Mono because they require Windows Presentation Foundation (WPF)

*   ClinicalTrial
*   Image_Classifier
*   MontyHall

**Note on building the F# samples in MonoDevelop**

If MonoDevelop cannot resolve the reference to FSharp.Core when trying to build the F# samples, try adding a <HintPath> in the TestFSharp.fsproj file which points to the instance of this library on your machine for .NET Framework 4.5.

For example:  

```xml
<Reference Include="FSharp.Core, Version=4.3.0.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a">  
    <HintPath>\usr\lib\mono\4.5\FSharp.Core.dll</HintPath>  
    <Private>True</Private>  
</Reference>  
```
