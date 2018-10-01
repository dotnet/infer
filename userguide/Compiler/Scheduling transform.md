---
layout: default
---
[Infer.NET user guide](../index.md) : [Infer.NET development](../Infer.NET development.md) : [Infer.NET compiler design](../Infer.NET compiler design.md)

## Scheduling transform

The scheduling transform re-orders statements within while loops to respect dependency constraints. It may duplicate statements within the loop to respect trigger dependencies. It may also copy statements from inside the loop to outside the loop in order to provide initializations to meet requirement dependencies. Duplicating statements is assumed to be valid since each statement has no side-effects. Unnecessary duplications are removed by the following transform (dead code). If the code does not contain any while loops, this transform does nothing. The transform assumes that statements have been annotated with DependencyInformation attributes (see [Dependency analysis transform](Dependency analysis transform.md)). 
 
Each `while` loop is processed independently. For each `while` loop, the scheduler does the following steps:

1. Sequence the statements inside the `while` loop (this is called the iteration schedule).
2. Collect the requirements for the loop to get started.
3. Construct an initialization sequence to initialize those requirements.

The transform works by constructing a dependency graph where each node is a statement inside the loop and edges correspond to the links in DependencyInformation. Statements outside of the loop are not included in the dependency graph. Each edge is annotated with the type of dependency (Declaration, Trigger, etc) inferred by the [Dependency analysis transform](Dependency analysis transform.md). Note that all information regarding the message passing algorithm is encoded in these annotations; the transform has no inherent knowledge of inference algorithms.

Intuitively, you can think of each statement as a message update (either from factor to variable or variable to factor), and the dependencies are the other messages that this message is computed from. This intuition isn't quite right since the code may have additional variables that do not correspond to messages, but to a first approximation you can only think about messages. 

If an edge from A to B is annotated with **Required**, we say that "B requires A". If the edge is annotated with **Trigger**, we say that "A triggers B" or equivalently "A invalidates B". If an edge from A to B is annotated with **Fresh**, we say that "B needs a fresh A". The meaning of these annotations is as follows:

| Annotation | Scheduling constraint |
|------------------------------------| 
| B requires A | A must be executed at least once before executing B for the first time | 
| A invalidates B | B must always be consistent with the latest value of A | 
| B needs a fresh A | A must be up-to-date before executing B |

Trigger and fresh edges automatically apply transitively, i.e. if A triggers B and B triggers C, then A triggers C. Directed cycles of **Required** edges are prohibited (they don't make sense). Same for directed cycles of **Trigger** edges and directed cycles of **Fresh** edges.
In the following discussion, we ignore the influence of user-provided initializations. These are discussed at the end.

#### Constructing the iteration schedule

When constructing the iteration schedule, we assume that the initialization schedule will handle the "B requires A" constraints. Therefore we assume that every statement has already been executed at least once. We only need to consider invalidation constraints. At any point of the schedule, we can label each message as valid or invalid. A message is invalid if it is the (transitive) target of a **Trigger** and it has not been updated since the source of the trigger was updated. Otherwise it is valid. At the beginning of the schedule, we assume all messages are valid, and at the end of the schedule we ensure that they are again valid. According to the constraint above, an invalid message cannot be used to compute another message. Thus in a sense, invalid messages block the schedule. 

The scheduler works sequentially, building up the schedule one node at a time. The scheduler maintains a list of nodes waiting to be scheduled, and marks them as valid or invalid, and fresh or stale. The initial list corresponds to all nodes which can reach an output node, and they are all marked valid and stale. Each waiting node is given a heuristic score, and the highest scoring node is scheduled next. The score considers various features such as the number of source nodes which are already scheduled, and the number of target nodes which would be made invalid. Nodes which violate the above rule are given the worst possible score. When a node is scheduled, several things happen:

1. If the node needs Fresh sources and these are stale, then the Fresh sources are scheduled first. These may themselves have stale sources, so we continue recursively.
2. The node is marked valid and fresh.
3. All targets are marked stale.
4. Every node that it triggers is marked invalid and added to the waiting list if not already there.

The above algorithm, because it is conservative, may schedule a node more times than it is needed. Therefore the next transform performs a dead code elimination that scans the schedule and prunes updates whose value is never used before the next update in cyclic order. For example, suppose A invalidates B, and the optimal schedule begins with B and ends with A. Since the scheduler conservatively tries to leave all nodes valid, it appends a redundant update of B at the end. The dead-code-elimination step detects and removes this redundant update.

#### Constructing the initialization schedule

The initialization schedule is needed only to satisfy **Required** constraints that are not already satisfied by the iteration schedule. By inspecting the iteration schedule, we can determine the set S of nodes which need to be executed prior to entering the loop. Then we run a scheduling algorithm similar to above, where the set of output nodes is S. We also have to worry about invalidations. From the iteration schedule we can determine the set of nodes whose value is used before it is updated. These nodes must be valid (but not necessarily initialized) at the end of the initialization. 

For example, suppose A invalidates B, C depends on (but does not require) B, and the iteration schedule begins with C. If the initialization schedule does not update A, then it does not have to worry about initializing B. But if the initialization schedule does update A, then it must subsequently update B.

The scheduling algorithm we use is similar to the one for the iteration schedule, except now each node is also marked with whether it has been executed. A node is given the worst possible score if it requires an un-executed node. In theory, this greedy algorithm could get stuck by a pathological combination of **Required** constraints mixed with **Trigger** constraints, because it doesn't use enough lookahead. But in practice, since the set S of output nodes is relatively small at this stage, the number of constraints is too small to cause problems.

#### User-provided initializations

In many cases, user intervention is needed for inference to converge to a sensible result. This may be to break symmetries or to avoid local minima. It is not enough to initialize the messages a particular way and then run the scheduling algorithm above. We want the schedule to actually make use of our initialization and not clobber it by recomputing the message that we carefully initialized. In other words, the choice of variables to initialize suggests what schedule to use.

We incorporate this heuristic by assigning a penalty to these events:

*  re-computing a variable which was initialized by the user and is still valid
*  invalidating a variable which was initialized by the user

Note that if the user provides too many initializations, then some of these may need to be ignored. This is currently detected by analyzing the generated schedule and seeing which initialized messages are re-computed before they are used. If so, a warning is printed to the console. Note that even though these initializations will not be used, the schedule is affected by having provided them, because of the penalties above. That is, the schedule is not recomputed with these initializations removed.

#### Where do the annotations come from?

This section describes why inference algorithms might want to use Required, Trigger, or Fresh constraints.

**Required** constraints. These arise from SkipIfUniform or Proper annotations on the arguments to message operators. These annotations are used in three situations:

*  The output message would be uniform if the input is uniform. For example in EP, if the output of a directed factor is not observed, then the upward message is uniform. This case is used for optimization, mainly by the Pruning transform.
*  The output message is not defined if the input is uniform. For example, if the precision of a Gaussian is uniform. This case is used for catching errors. While we could return a uniform message in this case, it could lead to undetected inference failures. Therefore the message operator throws an exception. Obviously we don't want the schedule to start by throwing an exception. The Proper annotation is used in this case, to distinguish from the other cases.
*  The output message is not uniform if the input is uniform, but it would lead to poor initialization. For example in EP, the IsBetween factor has three inputs (x, lowerBound, upperBound). If one of these inputs is uniform, then a non-uniform message can still be sent to the other inputs. Similarly in VMP, if the message to a BernoulliFromBeta factor from the child is uniform, then the message to the parent is non-uniform. However, we don't want to initialize the schedule this way, because it would likely push the inference into a local minimum (not to mention wasting computation). In this case, the annotation is not meant as a hard constraint but more like a penalty.

**Trigger** constraints. These arise from Trigger annotations on the arguments to message operators. The argument with the annotation is the trigger and the output message is the triggee. Intuitively, these annotations mean that the triggee is a deterministic function of the trigger, in the sense that whenever the triggee is used, we want it to be re-computed from the trigger. This situation does not arise in EP, but it does arise in VMP and Gibbs sampling. Some examples:

*  In VMP and Power EP, the message from a variable to a factor must always reflect the latest message that was sent from the factor to the variable. If not, the iterations may oscillate and never converge. 
*  Derived variables in VMP. Derived variables are marginalized out and therefore do not have their own q functions. Whenever a derived variable is used, we need to re-compute its distribution from its parent variables. 
*  In Gibbs sampling, when constructing the conditional distribution for a variable, we want to use the latest value sampled for the other variables. Otherwise the sampler is invalid (it samples from the wrong distribution).

**Fresh** constraints. These arise from Fresh annotations on the arguments to message operators. These annotations are used for sharing computations among several operators of the same factor. For example, suppose a factor has two edges A and B, and the message to A needs as part of its work to compute essentially the same quantity as the message to B. In this case, the message to A can take the message to B as an argument, and avoid repeating the work. However, the message to B must be fresh, otherwise the answer won't be the same as if A had repeated the work.