---
layout: default 
--- 
[Infer.NET development](index.md)

## Usability issues

When people have difficulty understanding part of Infer.NET or get stuck implementing something we should note it here and try and come up with a solution.

#### Modelling API

*   Breaking symmetry is very clumsy at the moment and InitialiseTo can be confusing. <br />
    Solution: introduce BreakSymmetry attribute? or InitialiseToSamplesFrom(distribution)?

*   Handling DistributionArrays and doing inference on arrays is non intuitive <br />
    Solution: hide away DistributionArrays even more and work with .NET arrays where possible

*   Using from F# can be awkward <br />
    Solution: examine F# code and create convenience methods for using from F# e.g. operator overloads

*   Support explicit modelling language, like BUGS? <br />
    Solution: perhaps introduce Csoft using Oslo technology or offer direct support for BUGS models

*   Confusion over SetTo() vs. assignment <br />
    Solution: provide better documentation of this issue

*   Confusion when indexing by a random variable and the need to surround this by a switch <br />
    Solution: can we automatically handle this case by creating a surrounding switch as needed? - we have discussed this in the past.

*   Accidentally calling the wrong constructors for Vector, Discrete etc. <br />
    Solution: use named factory methods e.g. Discrete.FromProbs(), Discrete.Uniform() and deprecate the constructors

#### Inference & monitoring

*   It is not clear when inference is happening and what is being inferred. <br />
    Solution: implement new inference proposal

*   Factor graph display does not show gates <br />
    Solution: gates are partially shown via edges named "selector".

