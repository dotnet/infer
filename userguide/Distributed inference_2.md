---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Infer.NET development](Infer.NET development.md)

## Distributed inference

by Tom Minka

This document describes general guidelines for implementing distributed inference algorithms, using Matchbox/EP as a running example. Given a model specification in a form of a factor graph, there are 3 steps in designing a distributed algorithm:

1. Partitioning the factors into blocks
2. Deciding which blocks should execute in parallel (the parallel schedule)
3. Deciding how the blocks should communicate (the communication schedule)

#### Partitioning

For EP, the factor graph is partitioned by assigning each factor to a block. If a variable is adjacent to two factors in different blocks, then that variable is shared between those blocks. To minimize communication overhead, the graph should be partitioned so that number of shared variables in a block is much smaller than the number of factors in that block. Often it is natural to define blocks by subdividing the largest Ranges in a model. It is not essential to subdivide the ranges equally, but it is a good idea to give each thread roughly the same amount of work to do. 

For example, in Matchbox, the largest Ranges are users, items, and observations. By subdividing the users and items, we implicitly subdivide the observations, i.e. the data matrix is subdivided into tiles. Define U = the number of unique users in a tile and V = the number of unique items in a tile. These tiles are ideal for distributed inference because the number of shared variables is proportional to U+V while the number of factors in the tile is proportional to U*V. This formula suggests that we should make the tiles as square as possible, since optimal ratio of factors to shared variables is achieved when U=V. To ensure the tiles are roughly balanced in the number of factors, we can shuffle the user/item ranges before subdividing them. 

When deciding on the size of blocks, it is not essential to make the blocks as large as possible. In fact, you don't want large blocks since these reduce the available parallelism. As long as the computation within a block is larger than the amount of I/O, you can use pipelining to hide the I/O cost, as shown later.

#### Parallelism

The message passing algorithm will have a double-loop structure. An outer iteration consists of visiting each block and update its messages to its shared variables. When processing a block, an inner iteration consists of visiting each factor and updating its outgoing messages.

In principle, any subset of blocks can be updated in parallel, without changing the fixed points of EP. However, different choices lead to different convergence rates. A poor schedule with slow convergence rate can erase the speedups that you hoped to gain from parallelism.

A good rule of thumb is that blocks that share a variable should not run in parallel. If you imagine a "block graph" in which the nodes correspond to blocks and edges indicate shared variables, then this rule corresponds to coloring the nodes so that adjacent nodes have different colors. For example, suppose we have 4 blocks arranged in a chain, i.e. blocks (1,2) share variable X, blocks (2,3) share variable Y, and blocks (3,4) share variable Z. If we have 2 threads, the rule tells us to update blocks (1,3) in parallel, followed by (2,4) in parallel. Note that this provides the same result as if we had used one thread and updated the blocks in the order 1,3,2,4. In other words, by following this exclusion principle, we obtain a parallel schedule that produces equivalent results to a sequential schedule. This generally leads to faster convergence, compared to a schedule such as (1,2) followed by (3,4). To see why, note that information can flow from node 1 to node 4 in two iterations with exclusion, while it takes three iterations without exclusion. This principle applies equally well to scheduling cores on a shared-memory machine as scheduling machines in a cluster.

Note that the exclusion principle is at odds with parallelism. In the example with 4 blocks, we can only execute two blocks in parallel with exclusion. If we want to use more than 2 threads, we have to give up on some exclusion. In a model with lots of sharing, it may not be possible to achieve any parallelism with full exclusion, e.g. if one variable is shared by all blocks. In these cases, we have to strike a compromise between exclusion and parallelism.

In the Matchbox example, each block corresponds to a tile of the data matrix. Tiles in the same row share the same user variables, while tiles in the same column share the same item variables. The exclusion principle tells us not to schedule two tiles in the same row or same column together. One such schedule is to update all blocks on the diagonal in parallel, then shift one column to the right and update the next diagonal in parallel, and so on. In this schedule, the maximum amount of parallelism is bounded by the number of tiles in a row or column. Experiments confirm that the convergence rate is improved by 2-4x by using this schedule compared to overlapped schedules (see Matchbox/TileTest). A similar approach was used in the paper "Large-Scale Matrix Factorization with Distributed Stochastic Gradient Descent" (Gemulla et al, KDD 2011).

Besides the per-user and per-item variables, Matchbox can have global parameters, such as the prior means, or weights on metadata features. These global parameters are shared by all blocks, and thus there is no parallel schedule that can satisfy the exclusion principle. In this case, a reasonable compromise is to follow the diagonal schedule above, achieving exclusion on the per-user and per-item variables, but not on the global variables.

In the double-loop algorithm, we have a choice of how many outer iterations and how many inner iterations we want to perform. Increasing the number of inner iterations will decrease the required number of outer iterations, up to a point. However, this decrease is generally not one to one, i.e. doubling the number of inner iterations does not quite halve the number of outer iterations. As a result, using more than 1 inner iteration is wasteful, in the sense that it increases the total amount of work performed to reach convergence. The only reason to use more than 1 inner iteration is if the blocks would otherwise not have enough work to overcome the communication overhead of loading the next block.

#### Communication

In order to process a block, three things have to be loaded into memory:

1. The data and graph structure for the block
2. The current marginals for the shared variables of the block
3. The latest messages from the block to its shared variables

When the computation is finished, (2) and (3) are updated. To minimize communication, a good strategy is to assign blocks to machines, so that the set of blocks updated by a given machine is the same in every iteration. This allows (1) and (3) to be stored on the local disk of the machine. The remaining question is how to communicate (2) between machines. 

The communication pattern for the marginal of a shared variable depends on what type of parallel schedule you are using. We can divide the shared variables into two types: exclusive and non-exclusive. A shared variable is "exclusive" if only one thread uses the variable at a given time. In the Matchbox example, the per-user and per-item variables are exclusive, while the global variables are not. 

For an exclusive variable, we always know the next machine in the schedule that will need its marginal. Thus whenever a block completes, we can send the marginals for its exclusive shared variables to the next machine that will use them. The receiving machine knows when and from whom it will receive a marginal message, so it can spawn a thread at the appropriate time to wait for it. The receiver thread can write the marginals to disk in anticipation of when they will be used. Ideally, if we have scheduled the blocks carefully, the marginals will be received exactly when they are needed, and they can be left in memory to be consumed immediately.

For a non-exclusive variable, there are two tasks to perform:

1. When parallel threads update a variable, the updates need to be fused into a single marginal.
2. The fused marginal needs to be distributed to all machines that need it.

The optimal communication structure for this collect/distribute pattern is a balanced binary tree, and we can keep the marginals up-to-date via parallel belief propagation. In this algorithm, each machine knows its neighbors in the tree and stores the latest received message from each neighbor. When a machine updates its marginal for a variable (either by processing a block or receiving a message from a neighbor), it sends a separate message to each neighbor. The message to a neighbor is the current marginal divided by the latest received message from that neighbor. When a machine receives a new message from a neighbor, it divides by the last received message to get an increment, and multiplies this increment into the current marginal. The time required for machine A to receive an update originated by machine B is proportional to their distance in the tree, which for a balanced binary tree is logarithmic in the number of machines. Rather than stall the block processing, it would make sense to perform this message passing in parallel with processing blocks. This means that some marginals will be stale but this doesn't affect fixed points and they would only be stale by a small amount due to the short path lengths.

Note that if you only store the messages from a block to its shared variables, and none of the internal messages inside the block, then these internal messages will need to be recomputed every time the block is visited. Recomputing these messages may require multiple iterations of inference. As argued earlier, we want to minimize the number of iterations we spend within a block, so that information spreads quickly across blocks. One way to balance these demands is to initially perform one inner iteration per outer iteration, then two inner iterations per outer, and so on, gradually increasing the number of inner iterations until convergence is reached.

#### Pipelining

Because a machine will process multiple blocks per outer iteration, we can increase efficiency by pipelining the read/compute/write steps, so that I/O is always interleaved with computation. To illustrate this, consider the Matchbox example where we have tiled the data matrix into 2 rows of 3 blocks each, and we have 2 machines. Machine A will process the tiles in the first row, and machine B will process the tiles in the second row. Let the blocks in the first row be numbered (1,2,3) while the second row is (4,5,6). To achieve exclusion, the machines must process a different column at a time. This allows two possible schedules: (A=1,B=5)(A=2,B=6)(A=3,B=4) or (A=1,B=6)(A=2,B=4)(A=3,B=5). The second schedule provides more opportunities for pipelining. The pipelined schedule is shown below. Each line is a chunk of time, and we assume each machine can perform 1 I/O step and 1 compute step in parallel. The first three rows initialize the pipeline; the remaining rows loop around and define the steady state.

| A I/O | A compute | B I/O | B compute |
|---------------------------------------|
| read 1 |          |       |           |
|        |process 1 | read 6|           |
| read 2 |process 1 |       | process 6 |
| write 1|process 2 |read 4 | process 6 |
| read 3 |process 2 |write 6| process 4 |
| write 2|process 3 |read 5 | process 4 |
| read 1 |process 3 |write 4| process 5 |
| write 3|process 1 |read 6 | process 5 |
| read 2 |process 1 |write 5| process 6 |

Notice that, in the steady state, whenever one machine writes an update, the other machine is reading it and consumes it immediately. Thus the marginals never need to be saved to disk.

#### Incremental updates

f the factor graph or data changes, rather than re-partition and restart the algorithm from scratch, it may be possible to make local changes to the blocks and resume the algorithm from its previous state, allowing for faster convergence and less communication. For example, if the interior of a block changes but its set of shared variables remains the same, then we can simply update the block from its previous state. This works because when we load the block, we will divide out its messages to the shared variables, replacing these messages when the block is finished. This also works to some extent if the set of shared variables changes. In this approach, we use the same schedule as before, just with fewer iterations. 

A further speedup is possible by prioritizing the updates so that new data is processed first. Recall that, within a block, we need to cycle through all factors and send messages from them. Instead of cycling through in an arbitrary order, we can process the changed factors first. If very few factors have changed, then we could process only a subset of the factors every time we visit the block, i.e. perform less than a complete iteration through the block. This allows the new information to propagate quickly among the blocks and improve the convergence rate. Similarly, if very few blocks have changed, we could modify the schedule to process the changed blocks first. With these optimizations, it is possible to reach convergence on the new graph without performing a full iteration of inference.

For example, if we receive new observations in Matchbox, these will get distributed to their respective blocks and all we need to do is restart inference from its previous state. If there is a new user or item, it can be added to an existing row or column group. When any block in that row/column group is loaded, there will be no stored message to that user/item, which we can treat as uniform. When the block is updated, we save the new messages, including the new user/item. The main thing to avoid is re-partitioning the user/item Ranges. Moving a user from one group to another, for example, requires updating all blocks in both affected rows. 

When new items appear in Matchbox, it is tempting to put these together in a new item group. However, new items will tend to be popular and have a lot of new data associated with them. If these are in the same group, then you will end up with many blocks in the same column having lots of new data. Since blocks in the same column are not processed in parallel, this new data will all be processed serially, leading to slow inference. A better approach is to leave room in the existing item groups so that new items can be distributed evenly among different groups, creating more opportunities for parallelism. When the item groups get full, you will have to re-partition and restart the algorithm from scratch (similar to resizing a hash table).