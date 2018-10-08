---
layout: default
---
[Infer.NET development](index.md)

## Compiler-generated variable names

| **Gate** | **_cases** - Used to represent an integer or boolean gate selector via a 1-of-N encoding. |
| **Depth cloning** | **_depthN** - Created when a jagged array is used at different indexing depths. |
| **Replication** | **_rep\[\]** - When a variable is used inside a (range) loop, it is replicated for each iteration. |
| **Variable** | **\_use & \_marginal** - The Variable factor is introduced for each variable that needs to be inferred. The output edge of this factor is suffixed by *_use*. For instance, if we want to infer the variable _mean_, the input edge to the introduced factor will be called _mean_, and the output edge will be called *mean_use*. The marginal distribution is stored in a variable suffixed by *_marginal*. |
| **Channel 2** | **_uses\[\]** - When a variable is used multiple times in the model, its uses are disambiguated by adding the `_uses[]` suffix (one element of this array for each use of the variable). This achieves a 1:1 mapping from variables to edges. An exception to this rule may be made when the same variable is used within gates. |
| **Message** | **_F & _B** - Each edge becomes a message. The forward messages are suffixed by *_F*, and the backward messages are suffixed by _B. <br /> **_toDef & _marginal** - When the product of the children of replicate has to be stored in a buffer, a variable suffixed by *_toDef* is used. This allows to optimize the computation of the forward message by dividing the value in the *_toDef* variable by the backward message. Similarly, if the marginal needs to be buffered, it is stored in a variable suffixed by *_marginal*. Note that this variable is different from the one introduced by the `Variable` transform. It differs in that the one here will be called `*_F/B_marginal`, while the one from Variable will be called `*_marginal_F/B`.  However, they should hold exactly the same value (once both are updated). |
| **Hoisting** | **_hoist** - When the value of a variable is computed multiple times inside a loop, but the computations performed are the same for each iteration, then this computation is taken outside the loop. The temporary variable created is called *_hoist*. If multiple such optimizations need to be made, then the new variables are called *_hoist1, …, _hoistN*. |
| **Dependency Analysis** | Sometimes adds special comment statements (called dummy statements in the compiler). These are used to express complex dependencies. For example: <br /><br /> `If(priorType) {` <br /> `x_F = …` <br /> `}` <br /> `If(!priorType) {` <br /> `x_F = ..` <br /> `}` <br /> `// x_F is now defined in all cases    <-- This is the dummy statement` <br /> `g(x_F)` |
