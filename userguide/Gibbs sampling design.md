---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Infer.NET development](Infer.NET development.md)

## Gibbs sampling design

This page documents our ongoing design decisions for supporting Gibbs sampling in Infer.NET.

#### Overview of the algorithm

Gibbs sampling is a special case of the Metropolis-Hastings algorithm. From a given state of the variables it randomly moves to another state, preserving the desired equilibrium distribution. The simplest version is single-site Gibbs sampling, where each variable is visited in turn and sampled from its conditional distribution given the current state of its neighbors. Note that we may visit the variables in any order and may even sample some variables more than others, as long as each variable gets sampled sufficiently often. 
 
A more efficient version, especially in the case of deterministic constraints, is the block Gibbs sampler. Here we group the variables into blocks, visit each block in turn and sample the entire block from its conditional distribution given the state of its neighbors. Note that this procedure is valid even if blocks overlap; variables in multiple blocks will simply be sampled more often. 
 
To sample the entire block from its conditional distribution, we require that the block is acyclic, i.e. there is no path from a variable to itself in the factor graph induced by variables in the block. To sample, we designate a variable in the block as the root variable, compute the conditional distribution of this root variable (with all other variables in the block marginalized out), sample the root variable, then sample neighbors of the root (with all other variables except the root marginalized out), then their neighbors, and so on until all variables are sampled conditionally on the state of previously sampled variables.
 
#### Message reuse for overlapping blocks
 
Some kinds of overlapping blocks allow messages to be reused, while others do not. For example, suppose we have blocks (A,C,D) and (B,C,D) where A and B are roots. When sampling (A,C,D) we will send a distribution from D to C to A. When sampling (B,C,D) we will send a distribution from D to C to B. Assuming no other blocks were updated in between, the message from D to C is the same in both cases. Therefore we can reuse it.
 
However, suppose we have blocks (A,C,D) and (B,C) where A and B are roots. When sampling (A,C,D) we will send a distribution from D to C to A. When sampling (B,C) we will send a distribution from C to B. Since C is connected to D in a different block, C uses the current sample of D. So here we must send two distinct messages from D to C, depending on which block we are sampling.
 
In the general case, suppose we have two non-nested blocks B1 and B2 which share a subset of variables S. Then we can share messages iff (1) S has exactly one neighbor in B1 and in B2 and (2) S does not contain the root of B1 or B2.
In the first example above, S=(C,D) and both conditions are satisfied. In the second example, S=(C) and the first condition is not satisfied (C has two neighbors in the first block).

### Inference API

The GibbsSampling:IAlgorithm object has properties:

*  BurnIn - number of samples to discard at beginning
*  Thin (or KeepNthSample) - reduction factor when constructing sample and conditional lists.

The NumberOfIterations property on the InferenceEngine object determines the total number of samples computed. So if NumberOfIterations = 100, 
BurnIn = 50, Thin = 5, then the user receives a sample list of size 10.

To get an accumulated marginal distribution: `Infer<Bernoulli>(firstCoin)`
To get a list of samples: `Infer<SampleList<bool>>(firstCoin, QueryTypes.Samples)`
To get a list of conditionals: `Infer<ConditionalList<Bernoulli>>(firstCoin, QueryTypes.Conditionals)`
- in principle, you could infer any type which has an associated boolean accumulator

To specify the set of query types that will be made on a variable, the user can attach `QueryType` attributes to it. If no such attributes are attached, then a default set of query types are assumed (defined by `IAlgorithm.SetDefaultQueryTypes`).

#### Implementation

When `Infer<T>(v)` is called:
*  loop all variables w reachable from v:
  *  if(w has no inference attributes) alg.SetDefaultAttributes(w);
*  `v.AddAttribute(new WillBeInferred(typeof(T)));`

To determine if the model needs to be recompiled, the ModelBuilder should store, for each variable, the set of types produced by the current CompiledAlgorithm. If none of these types is assignable to T then the model needs to be recompiled.
 
It is necessary to distinguish between the types produced by the CompiledAlgorithm and the type returned to the user. For example, the user may request `object` while the CompiledAlgorithm produces `Gaussian`. If the CompiledAlgorithm produces multiple types, the inference engine needs to determine which of these to return to the user. One proposal is that the types produced by the CompiledAlgorithm are ordered by priority, e.g. with the MarginalPrototype first, and the first type in this list which is assignable to T is returned to the user.

### Blocking annotations

The user creates variable blocks using inferenceEngine.Group(v1,v2,v3). Each call creates one group. Groups can overlap, and multiple calls create new groups. inferenceEngine.Group(v1) does nothing. The arguments are variables and must form a connected subgraph. The Group method returns a VariableGroup which is an ICollection. In the generated MSL, the variables are attributed with a GroupMemberAttribute, holding an index and a pointer to a common GroupAttribute object (or separate GroupAttributes sharing the same integer id, if we want to use value equality for attributes).
 
To designate a block root, use group.AddAttributes(v, new RootAttribute()). Otherwise a root is chosen automatically, for example, the first variable in the list.
 
Group attributes need to be maintained as the model is transformed. Currently, when a variable is cloned its attributes are copied. This could lead to multiple clones holding the same group index.
 
To decide whether a variable sends a distribution or sample to a factor, we need to know which of two variables is closer to the root of the block, breaking ties consistently. This can be decided quickly by precomputing the position of each variable in a depth-first traversal from the root of the block. This has the nice side-effect of also detecting loops in the block.

#### Collapsing

Besides blocking, we may want to collapse variables as well. For example, suppose x = a*b. We can't sample (x,a,b) jointly since the factor isn't conjugate. We could block (x,a) but then b is frozen once (x,a) are sampled. The correct way to handle this factor is to marginalize x when sampling a (conditional on b) and sampling b (conditional on a). This is known as collapsing a variable. A collapsed variable acts just like an EP variable; it sends a distribution in all directions.
 
Collapsing can be specified via overlapping groups. For example, to collapse x = a*b the user would say Group(a,x); Group(b,x). If a and b are the roots of their groups, then x will be marginalized with respect to both. However, we want x to be sampled only once. The scheduler should take care of this for us.
 
A related constraint is that if a derived variable belongs to a group, then at least one of its parents must also be in the group. Otherwise, the derived variable would be frozen when sampling the group.
 
Internally, the most useful place for these annotations is on the method call for the factor instance, specifying two named arguments of the factor. SendDistribution(from="mean", to="x"), SendDistribution(from="x", to=any).
 
By default, an unobserved derived variable should send a distribution to all parents. If observed, then one of parents must send a distribution to the other parents. To implement this in general for each factor, we need a mechanism to detect if the outgoing message will be a point mass or if it is non-conjugate. These act as constraints on the annotations. In particular, non-conjugate messages are not allowed, and if a non-observed variable receives a pointmass message from a factor, then it must not send a pointmass to that factor (for any recipient). Furthermore, a variable should receive at most one pointmass message.
 
To determine if a set of SendDistribution annotations is valid, we need to recover the underlying blocking scheme. This can be done via the following rules (let x->y be shorthand for "x sends a distribution to y in some factor") (note that blocks can overlap):

1. If x->y then x and y share a block. 
2. If x sends a sample to y and y sends a sample to x in some factor, then x and y must not share any block.
3. If x->y and y->z via separate factors, then x and z share a block. (Connection by directed path)
4. If x->z and y->z then x and y share a block. (Connection by converging path)

The requirement that blocks are acyclic gives us the following rules:

1. x cannot be connected to itself by a directed or converging path of distribution links.
2. If x sends a sample to y and y sends a sample to x in some factor, then there must not be a directed or converging path of distribution links between x and y.

### Message passing

Message passing is complicated by the fact that a factor may connect multiple variables in different blocks. For example, a factor may connect variables (a1,a2) in block 1 with variables (b1,b2) in block 2. When sending a message to a1, the factor should receive a distribution from a2 and samples from (b1,b2). When sending a message to b1, the factor should receive a distribution from b2 and samples from (a1,a2). Thus the type of message a factor receives can depend on the target variable.

Another complication is deterministic factors. When a deterministic factor receives a sample from all of its parents, the output is a single value, not a distribution. Thus the type of message a factor sends to its child need not be the marginal prototype.

One way to support this is to send a single message to the factor which holds both the sample and distribution of the variable. The MessagePassing transform needs to change in two ways:

1. Must support different types for messages from var->factor versus factor->var. This is implemented by a method IAlgorithm.GetMessagePrototype(channelInfo, direction, marginalPrototypeExpression)->message prototype Expression. This is called in ConvertVarDeclExpr. Requires changing MakeDeclExpression, ConvertArrayCreate, DoConvertArray, 
2. GetMessagePrototype itself relies on MessagePath attributes attached to channels by GroupTransform.

#### Example

A boolean root variable coin with marginal prototype Bernoulli gets the following associated messages in the MessageTransform:

```csharp
GibbsMsg<Bernoulli,bool> coin_Marginal;
Bernoulli coin_F; // if parent factor is stochastic
bool coin_F; // if parent factor is deterministic
GibbsMsg<Bernoulli,bool> coin_B;
GibbsMsg<Bernoulli,bool>[] coin_uses_F;
Bernoulli[] coin_uses_B;
with the following definitions:
coin_Marginal.Distribution = MultiplyAll(coin_F, coin_uses_B);
coin_Marginal.Sample = coin_Marginal.Distribution.Sample();
coin_B.Sample = coin_Marginal.Sample;
coin_B.Distribution = MultiplyAll(coin_uses_B);
coin_uses_F[i].Sample = coin_Marginal.Sample;
coin_uses_F[i].Distribution = MultiplyAllExcept(coin_F, coin_uses_B, i);
These definitions are implemented with a UsesEqualDef operator acting on GibbsMsg structs:
coin_Marginal = UsesEqualDef.MarginalConditional(coin_F, coin_uses_B);
coin_B = UsesEqualDef.DefConditional(coin_F, coin_uses_B, coin_Marginal);
coin_uses_F[i] = UsesEqualDef.UsesConditional(coin_F, coin_uses_B, coin_Marginal, i);
```

Notice that the UsesEqualDef methods need to refer to coin_Marginal. This is explained in the Local State section below.
When calling a message operator that needs a sample, we pass coin_uses_F[i].Sample. Otherwise we pass coin_uses_F[i].Distribution.

#### Array example

A boolean array variable coins gets the following associated messages in the MessageTransform:

```csharp
GibbsMsg<DistributionArray<Bernoulli>,bool[]> coin_Marginal;
DistributionArray<Bernoulli> coin_F; // if parent factor is stochastic
bool[] coin_F; // if parent factor is deterministic
GibbsMsg<DistributionArray<Bernoulli>,bool[]> coin_B;
GibbsMsg<DistributionArray<Bernoulli>,bool[]>[] coin_uses_F;
DistributionArray<Bernoulli>[] coin_uses_B;
```

The definitions of these messages are the same as for coin, but using result buffers for efficiency.

### More efficient design

A more efficient design would allow coin_Sample to be used in message operator calls. Then we don't need a GibbsMsg struct. Instead of returning a field name into this struct, the method IAlgorithm.GetMessageName returns the name of an argument to the UsesEqualDef operator (i.e. "Uses", "Def", "Sample", or "Marginal"). The corresponding message from UsesEqualDef is sent to the factor. This has the added benefit of allowing VMP to use Marginal for its outgoing messages. 

#### Example

A boolean root variable coin with marginal prototype Bernoulli gets the following associated messages in the MessageTransform:

```csharp
Bernoulli coin_Conditional;
bool coin_Sample;
Bernoulli coin_F; // if parent factor is stochastic
bool coin_F; // if parent factor is deterministic
Bernoulli coin_B;
Bernoulli[] coin_uses_F;
Bernoulli[] coin_uses_B;
with the following definitions:
coin_Conditional = MultiplyAll(coin_F, coin_uses_B);
coin_Sample = coin_Conditional.Sample();
coin_B = MultiplyAll(coin_uses_B);
coin_uses_F[i] = MultiplyAllExcept(coin_F, coin_uses_B, i);
```

When calling a message operator that needs a sample, we pass coin_Sample. Otherwise we pass coin_uses_F[i].

### Plates

Plates require variables to be replicated. For example, we may have a factor `x[i] = Gaussian(mean)` which gets transformed into the two lines:

```
 mean_rep[i] = Replicate(mean);
 x[i] = Gaussian(mean_rep[i]);
```

In this case, we require that the distribution/sample exchange between x[i] and mean is the same for all i. Therefore when we pass mean_F to ReplicateOp, we can extract the relevant field of GibbsMsg and we only need to replicate this field, not the entire struct. 

Note that mean_rep is given a WillOnlyHaveOneUse attribute so it will not get a UsesEqualDef factor. The two messages sent through mean_rep[i] will not be of type GibbsMsg so no field should be extracted from them. GroupTransform should not generate MessageName attributes for this channel.

### Gates

When a variable enters a gate, it gets cloned by Gate.Enter. We require that the distribution/sample exchange is the same for all cases of the gate, so that we can give the relevant field of GibbsMsg to Gate.Enter. These clones receive a sample from their parent and send no evidence message, no matter how many times they are used (note this is different from EP). This is because their child factors already send the correct evidence. Gate.Enter receives a sample from the gate condition.

When a variable exits a gate, there is a clone of the variable in each case, to be merged by Gate.Exit. There are two different behaviors depending on whether the variable is stochastic or derived:

*   If the variable is stochastic, then each clone receives a distribution from its parent, and sends a sample to its parent and children. Each clone is the root of a group containing the exit variable, i.e. each clone receives a distribution from its children. The total evidence contribution for each case should be the integral of all distributions received. Since the clone is sending samples out, it must cancel the evidence contribution from these samples. Thus each clone must send an evidence message which is the desired total evidence minus the unwanted contributions from its parent and child factors. (GibbsGateExitConstraintTest2 has a good example of this.)

*   If the variable is derived, then it is not the root of its group. It receives a sample from its parent and a distribution from its children. It does not need to send any evidence message because its surrounding factors (including Gate.Exit) will already do so.
A gate is not allowed to have two stochastic variables connected by a path (this includes co-parents of the same factor).

### Local state

The implement Gibbs sampling, the variables must have local state to hold the current sample. This can be implemented by allowing UsesEqualDef to refer to its outgoing messages. Then UsesEqualDef can send out a sample and later read back this sample. There is no requirement that the sample is up-to-date with respect to the incoming distributions.

### Scheduling

When a variable is resampled, all distribution messages that depend on that sample must be invalidated. One way to implement this is to put trigger annotations on all messages in the graph other than those that produce the samples. Note this automatically detects loops of distribution messages.

### Accumulation

Accumulation of samples into a distribution is done by a special internal factor Gibbs. Accumulate, whose deterministic implementation takes a value (item) and a constant estimator object and returns the value (distribution). The operator for this factor will send a message to 'distribution' by (1) updating the estimator with the current item and (2) calling GetDistribution on the estimator. The operator sends a message to 'item' by forwarding the message from 'distribution'.

These special factors are added to the model by the ModelBuilder, which needs to output an expression to construct the estimator. This constructor should eventually go into the Initialize method of the IterativeProcess, so that the estimator is reset whenever the inference is.