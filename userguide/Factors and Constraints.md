---
layout: default 
--- 
[Infer.NET user guide](index.md)

## Factors and Constraints

Infer.NET has a number of inbuilt factors and constraints which you can use to build your models. These are normally exposed as static methods on the `Variable` class, such as `Variable.Discrete()`. You can also [add your own factors](How to add a new factor and message operators.md) or [constraints](How to add a new constraint.md) and include them in your model by using `Variable<T>.Factor` or `Variable.Constrain`.

Factors are functions that define a new variable in terms of other existing variables. A factor can be stochastic (even deterministic inputs produce an uncertain output) or deterministic (deterministic inputs will produce a deterministic output). Constraints act on existing variables and do not define a new variable. Below we mainly categorize factors in terms of the type of their **output** variable; constraints and experimental factors relating to undirected models are documented separately in their own sections.

*   [Boolean factors](Bool factors.md)
*   [Integer and enum factors](Int factors.md)
*   [Double factors](Double factors.md)
*   [String and char factors](String and char factors.md)
*   [Array and list factors](Array and list factors.md)
*   [Vector and matrix factors](Vector and matrix factors.md)
*   [Generic factors](Generic factors.md)
*   [Constraints](Constraints.md)
*   [Undirected factors](Undirected factors.md)

**Note:** some factors are less mature than others. For example, their implementation may only support a limited class of models or algorithms, and they may not be as thoroughly tested as other factors. The [list of factors and constraints](list of factors and constraints.md) shows an assessment of the maturity of each factor using the mechanism of [quality bands](Quality bands.md). If you have an issue or question concerning an experimental factor, please raise it in the forums.
