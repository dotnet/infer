---
layout: default
---
[Learners](../../../Infer.NET Learners.md) : [Matchbox recommender](../../Matchbox recommender.md) : [Command-line runners](../../Matchbox recommender/Command-line runners.md)

## Negative data generator

The concept of positive-only data and the need to generate negative data is explained in the [corresponding data mapping section](../API/Mappings/Negative.md).

The negative data generator can be called by running "`Learner Recommender GenerateNegativeData`". It takes two input parameters - the existing positive-only data file name and the name of the file to store the output to. For example: 
```
Learner Recommender GenerateNegativeData --input-data PositiveData.dat   
                                         --output-data StarRatingData.dat
```
The input file, although containing only positive data, has to follow the guidelines for the runner's data formatting. That is, it has to provide a rating descriptor, as well as a rating value. For example:
```
R,1,1  
u2,i3,1  
u2,i1,1  
u1,i1,1  
u3,i2,1  
u4,i4,1
```
The corresponding output will be:
```
R,0,1  
u2,i3,1  
u2,i1,1  
u2,i4,0  
u2,i2,0  
u1,i1,1  
u1,i3,0  
u3,i2,1  
u3,i1,0  
u4,i4,1  
u4,i1,0
```
Data in this format can now be fed into the recommender for training.
