---
layout: default
---
[Infer.NET user guide](../index.md) : [Infer.NET development](../Infer.NET development.md) : [Infer.NET compiler design](../Infer.NET compiler design.md)

## Message optimisation transform

Removes duplicate messages or redundant message operations in order to reduce the memory/computation needed for inference.

| Optimisation | Description | Action | Saving |
|----------------------------------------------|
| Collapse identical composite message arrays | Composite message arrays are unrolled and a ResultIndex supplied to the operator. For some operators e.g. VMP UsesEqualsDef, the returned message does not depend on the result index i.e. all messages are the same. <br /> _Note: no need for new annotation_ | THIS WILL BE DONE BY THE BELOW IF WE USE THE BUFFER TRANSFORM. <br /> We can remove duplicates by collapsing the message array - make the array length 1 and setting all indices of the array to 0. <br /> _Arrays to be collapsed could be marked in MessageTransform, but the collapsing would be achieved here._ | Memory saving of ~50% for VMP algorithms. | 
| Remove redundant operations | Some operators (e.g. copy operators) return one of their arguments directly as the return value. These arguments should be annotated with [IsReturned]. | Any reference to the result of such an operator should be replaced with the argument expression instead (copying across dependencies as appropriate). <br /> _The operators could be marked in MessageTransform and the replacement done here._ | Memory overhead and small computation overhead of copy operations. Cleaner generated code. |
| Collapse unit length arrays | to be done? makes most sense for composite arrays e.g. uses | - | Small memory savings, but mainly cleaner generated code. |