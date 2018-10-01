---
layout: default
---
[Infer.NET user guide](../index.md) : [Infer.NET development](../Infer.NET development.md) : [Infer.NET compiler design](../Infer.NET compiler design.md)

## Group transform

The group transform creates ChannelPathAttributes for use by the [Message transform](Message transform.md). A ChannelPathAttribute is attached to a channel and specifies the message type for a particular direction on that channel. A ChannelPath can be 'default' or 'non-default'. The same message can have up to two conflicting ChannelPaths: a default one and a non-default one. The non-default is always preferred. It is an error to have conflicting default or conflicting non-default ChannelPaths for the same message. 
 
The group transform creates groups for deterministic factors, where the parent variables are roots and the child variable is not. 
 
ChannelPathAttributes are collected directly from the MessagePathAttributes that are attached to MethodInvokeExpressions.
 
ConvertExpressionStatement finds all variables in a statement and uses their GroupMember attributes to build the nodesOfGroup dictionary. PostProcessDependencies then does the rest of the work.
 
PostProcessDependencies creates or extends groups to satisfy the following conditions:

1. Derived variables. For each derived variable, check that each group it belongs to includes at least one parent. If not, add the first parent to the group. Note that the parent may be derived itself. If a parent variable ends up being added to multiple groups, then these groups must be merged, otherwise we will have an invalid root placement. GateEnter must be treated specially because its 'cases' argument has no effect on the output so it is not really a parent.
2. Deterministic factors. The child variable must be paired with each of its parents.
3. Variable factors. The def and uses channels must belong to the same set of groups.
4. GateExitRandom factors. The child variable must be paired with the clone array, and the child must be the root.
Because these constraints all interact, they are imposed as nodes are finished by DFS.
 
CreateMessagePathAttr loops all messages from a variable to a factor and determines whether it should be a distribution or sample, depending on the distance to the group root. Messages from a stochastic factor to the child variable are always distributions. Undetermined cases are left without an attribute.
 
algorithm.ModifyFactorAttributes is called on all factor expressions and modifies the previously attached MessagePathAttributes as well as attaching default MessagePathAttributes.
