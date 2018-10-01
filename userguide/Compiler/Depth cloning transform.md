---
layout: default
---
[Infer.NET user guide](../index.md) : [Infer.NET development](../Infer.NET development.md) : [Infer.NET compiler design](../Infer.NET compiler design.md)

## Depth cloning transform

Clones variables to ensure that each is used at a fixed indexing depth. First, DepthAnalysisTransform collects information about the indexing depths of each variable, including the depth of its definition, the number of uses, and for each usage depth, the context in which that depth was used. Second, each usage whose depth does not match the definition depth is transformed. For example, suppose array means is defined via means[i][j] = (...). Then its definition depth is 2. If we encounter a usage at depth 1, we replace it with a new array means_depth1 defined by means_depth1[i][j] = Copy(means[i][j]) (note means_depth1 has definition depth 2). If we encounter a usage at depth 3, i.e. means[i][j][k], then we replace it with means_depth3[i][j][k], where means_depth3 is a new array defined by means_depth3[i][j][k] = Copy(means[i][j][k]).