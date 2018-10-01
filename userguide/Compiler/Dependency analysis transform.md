---
layout: default
---
[Infer.NET user guide](../index.md) : [Infer.NET development](../Infer.NET development.md) : [Infer.NET compiler design](../Infer.NET compiler design.md)

## Dependency analysis transform

This transform annotates each statement with dependency information, indicating which other statements it depends on. This transform does not change existing code, though it may add new statements in the form of comments. Statements are assumed to be of three types:

*  Declarations and allocations
*  Initializers
*  Updates

For each variable, the transform collects a list of all statements that declare or modify it. Statements are then classified according to the following rules:

*  The last statement to modify a variable is an update.
*  A statement with an OperatorStatement attribute is an update. 
*  Otherwise, if the statement has an Initializer attribute, it is an initializer.
*  Otherwise, the statement is a declaration/allocation. 

The basic assumption behind these rules is that after initialization, a variable is updated by a single unconditional statement or by a set of disjoint conditional statements. In each condition, an array can be updated by either a single statement (looping over the whole array) or a set of assignments to distinct elements. The dependencies of a statement are constructed by searching for all variable references within that statement. Note that the dependencies output by this transform are from statement to statement, not variable to statement. Dependencies are classified into 7 types:

*  Declaration dependency
*  (Ordinary) dependency
*  Requirement
*  Trigger
*  Fresh dependency
*  Initializer
*  Parameter dependency (these are on parameters not statements)

If the statement read or writes to a variable, then it has a Declaration dependency on all declarations/allocations of that variable. If the statement reads or writes to a method parameter then it has a Parameter dependency on that parameter. If an update statement writes to a variable then it has an Initializer dependency on all initializers of that variable.
 
If the statement is an initializer, then it can only depend on declarations/allocations or other initializers. If the statement reads and writes a variable, and it is an initializer, then the statement has an ordinary dependency on the previous initializer of that variable. 
 
If the statement is an update and reads a variable, then it has an ordinary dependency on all updates of the variable. If the statement also writes to that variable then it has a dependency on itself. If the statement contains a method invocation, a variable is provided as a method argument, and the method argument is annotated with SkipIfUniform/Trigger/Fresh, then the statement has a requirement/trigger/fresh dependency on updates of the variable.
 
The difference between normal dependency, requirements and triggers is explained in [the scheduling transform](Scheduling transform.md). The actual annotation takes the form of a DependencyInformation attribute attached to the statement.

The dependency information also indicates if the statement is an output i.e. it computes a value that has been requested by the user. Infer() statements are currently the only statements marked as being outputs.

All statements inside of a loop have a Declaration dependency on declarations/allocations of the size expression of the loop. All statements inside of a conditional have a Declaration dependency on declarations/allocations of the condition expression. An array indexer expression (on the lhs or rhs) depends on the index expression.

The dependency information also indicates if the statement creates a uniform distribution or empty array. The statement is assumed to create a uniform distribution if it invokes a method with the Skip annotation or if it invokes ArrayHelper.MakeUniform.

#### Comment statements

The input statements to the transform may have deterministic conditions. In this case, the same variable may be updated by multiple statements having disjoint conditions. When this variable is used outside of a condition, we need to ensure that the dependency refers to all of the conditional statements. This is done by creating a dummy statement (in the form of a comment) that depends on all of the conditional updates. Any statement that uses the variable is made to depend on the dummy statement. The algorithm for creating these dummy statements is subtle because we have to deal with cases where different parts of an array are updated in disjoint conditions. For example, suppose we have the following 3 statements:

1. if(a) x = (...);
2. if(!a) x[0] = (...);
3. if(!a) x[1] = (...);

Then we need to create two dummy statements, one which represents "x[0]" across both conditions and another which represents "x[1]" across both conditions:

1. Comment "x[0]" depends on (1,2)
2. Comment "x[1]" depends on (1,3)

A statement that uses "x[0]" will depend on the first dummy statement, while a statement that uses "x" will depend on both dummy statments. A statement with an AnyItem dependency on "x" will get an Any dependency on both dummy statements. Note this is not the same as an Any dependency on the original 3 statements, which would not be correct.

To determine what dummy statements need to be created, the transform first creates a graph where each node is an update, and updates are linked if they overlap (this implies they are in disjoint conditions). A clique in this graph corresponds to a set of updates that all overlap (which implies that their conditions are all disjoint). A dummy statement is created for each maximal clique in the graph bigger than 1. A clique of size 1 is represented by the original statement. A statement that uses a variable depends on all cliques where it is affected by all nodes in the clique.