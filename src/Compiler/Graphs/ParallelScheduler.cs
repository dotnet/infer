// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Constructs a schedule for multiprocessor execution given a task graph.
    /// </summary>
    public class ParallelScheduler
    {
        IndexedGraph g = new IndexedGraph();
        int[] heights;
        HeightHistogram histogram;
        int[] unscheduledSourceCounts;
        PriorityQueue<QueueEntry> readyQueue;
        int[] queuePositions;
        IReadOnlyList<int[]> variablesUsedByNode;
        /// <summary>
        /// Must be non-negative.  Larger values lead to fewer stages.
        /// </summary>
        public int PruningImbalanceThreshold = 0;
        /// <summary>
        /// Lower values speed up scheduling but create more stages.
        /// </summary>
        public int BuildTreeImbalanceThreshold = 10000;
        /// <summary>
        /// If true, each stage is pruned until the remaining nodes have no idle slots when sorted by height.
        /// </summary>
        public bool FillIdleSlots = false;
        public Action<string> LoggingAction;

        public void CreateGraph(IReadOnlyList<int[]> variablesUsedByNode)
        {
            int maxIndex = variablesUsedByNode.DefaultIfEmpty(new int[0]).Max(variableIndices => variableIndices.DefaultIfEmpty(-1).Max());
            const int missing = -1;
            var lastNodeOfVariable = Util.ArrayInit(maxIndex + 1, i => missing);

            this.variablesUsedByNode = variablesUsedByNode;
            foreach (var variableIndices in variablesUsedByNode)
            {
                int node = g.AddNode();
                foreach (int variableIndex in variableIndices)
                {
                    int lastNode = lastNodeOfVariable[variableIndex];
                    if (lastNode != missing && lastNode != node && !g.ContainsEdge(lastNode, node))
                    {
                        g.AddEdge(lastNode, node);
                    }
                    lastNodeOfVariable[variableIndex] = node;
                }
            }
        }

        public int NewNode()
        {
            return g.AddNode();
        }

        public int AddEdge(int source, int target)
        {
            return g.AddEdge(source, target);
        }

        public bool ContainsEdge(int source, int target)
        {
            return g.ContainsEdge(source, target);
        }

        /// <summary>
        /// Get an array where array[node] is the height of a node in the dependency graph.
        /// </summary>
        /// <returns></returns>
        public int[] GetHeights()
        {
            if (heights == null)
                heights = ComputeHeights();
            return heights;
        }

        private int[] ComputeHeights()
        {
            g.NodeCountIsConstant = true;
            g.IsReadOnly = true;
            int[] heights = new int[g.Nodes.Count];
            var dfs = new DepthFirstSearch<int>(g);
            dfs.FinishNode += delegate (int node)
            {
                int height = -1;
                foreach (int target in g.TargetsOf(node))
                {
                    height = System.Math.Max(height, heights[target]);
                }
                heights[node] = height + 1;
            };
            dfs.SearchFrom(g.Nodes);
            return heights;
        }

        private int[] GetSourceCounts()
        {
            int[] parentCounts = new int[g.Nodes.Count];
            foreach (var node in g.Nodes)
            {
                parentCounts[node] = g.SourceCount(node);
            }
            return parentCounts;
        }

        private class Block
        {
            /// <summary>
            /// The predecessors in the task graph.
            /// </summary>
            public List<Block> parentBlocks = new List<Block>();
            public List<int> nodes = new List<int>();
            /// <summary>
            /// The successors in the task graph.
            /// </summary>
            private Block childBlock;
            private int size;
            public int Size { get { return size; } }

            public Block GetRoot()
            {
                if (childBlock == null)
                    return this;
                Block root = childBlock.GetRoot();
                // path compression
                childBlock = root;
                return root;
            }

            public void AddParentBlock(Block block)
            {
                parentBlocks.Add(block);
                size += block.Size;
                block.childBlock = this;
            }

            public void AddParentBlocks(IEnumerable<Block> blocks)
            {
                foreach (var block in blocks)
                {
                    AddParentBlock(block);
                }
            }

            public void Add(int node)
            {
                nodes.Add(node);
                size++;
            }

            public void RemoveLast()
            {
                nodes.RemoveAt(nodes.Count - 1);
                size--;
            }

            public void ForEachNode(Action<int> action)
            {
                ForEachBlock(block =>
                {
                    foreach (int node in block.nodes)
                    {
                        action(node);
                    }
                });
            }

            /// <summary>
            /// Invoke action on this block and every ancestor block, parent blocks first.
            /// </summary>
            /// <param name="action"></param>
            public void ForEachBlock(Action<Block> action)
            {
                if (parentBlocks.Count > 0)
                {
                    Stack<Block> searchStack = new Stack<Block>();
                    searchStack.Push(this);
                    Stack<Block> actionStack = new Stack<Block>();
                    while (searchStack.Count > 0)
                    {
                        Block block = searchStack.Pop();
                        actionStack.Push(block);
                        foreach (var parentBlock in block.parentBlocks)
                        {
                            searchStack.Push(parentBlock);
                        }
                    }
                    while (actionStack.Count > 0)
                    {
                        var block = actionStack.Pop();
                        action(block);
                    }
                }
                else
                {
                    action(this);
                }
            }

            public int GetSizeDebug()
            {
                int count = 0;
                ForEachBlock(block => count += block.nodes.Count);
                return count;
            }

            public override string ToString()
            {
                return "Block(" + nodes.Count + ")";
            }
        }

        private class QueueEntry : IComparable<QueueEntry>
        {
            public readonly int Node;
            public readonly int Height;

            public QueueEntry(int node, int height)
            {
                this.Node = node;
                this.Height = height;
            }

            public int CompareTo(QueueEntry other)
            {
                // order by decreasing height
                return other.Height.CompareTo(this.Height);
            }
        }

        private class VariableLocation
        {
            public readonly int Stage;
            public readonly int Thread;

            public VariableLocation(int stage, int thread)
            {
                this.Stage = stage;
                this.Thread = thread;
            }
        }

        /// <summary>
        /// Convert [stage][block][item] into [thread][stage][item]
        /// </summary>
        /// <param name="schedule"></param>
        /// <param name="threadCount"></param>
        /// <returns></returns>
        public int[][][] ConvertToSchedulePerThread(IReadOnlyList<IReadOnlyList<IReadOnlyList<int>>> schedule, int threadCount)
        {
            Dictionary<int, VariableLocation> lastThreadOfVariable = new Dictionary<int, VariableLocation>();
            int numStages = schedule.Count;
            int[][][] perThread = Util.ArrayInit(threadCount, thread =>
                Util.ArrayInit(numStages, stageIndex => default(int[])));
            Dictionary<int, int>[] localIndices = Util.ArrayInit(threadCount, thread => new Dictionary<int, int>());
            for (int stageIndex = 0; stageIndex < numStages; stageIndex++)
            {
                var stage = schedule[stageIndex];
                // build a cost vector for each block
                int numBlocks = stage.Count;
                if (numBlocks > threadCount)
                    throw new ArgumentException("stage.Count > threadCount");
                // [block][thread]
                int[][] commCostsOfBlock = Util.ArrayInit(numBlocks, blockIndex =>
                {
                    int[] commCosts = new int[threadCount];
                    var block = stage[blockIndex];
                    foreach (var node in block)
                    {
                        foreach (var variableIndex in this.variablesUsedByNode[node])
                        {
                            VariableLocation location;
                            if (lastThreadOfVariable.TryGetValue(variableIndex, out location))
                            {
                                commCosts[location.Thread]++;
                            }
                        }
                    }
                    //Debug.WriteLine($"stage {stageIndex} block {blockIndex} has costs {StringUtil.CollectionToString(commCosts, " ")}");
                    return commCosts;
                });
                // assign threads to blocks
                ICollection<int> threadsToAllocate = FindOptimalAssignment(commCostsOfBlock, (blockIndex, thread) =>
                {
                    var block = stage[blockIndex];
                    perThread[thread][stageIndex] = block.ToArray();
                    //Debug.WriteLine($"assigned block {blockIndex} to thread {thread}");
                    foreach (var node in block)
                    {
                        foreach (var variableIndex in this.variablesUsedByNode[node])
                        {
                            lastThreadOfVariable[variableIndex] = new VariableLocation(stageIndex, thread);
                        }
                    }
                });
                // remaining threads have no work to do
                foreach (var thread in threadsToAllocate)
                {
                    perThread[thread][stageIndex] = new int[0];
                }
            }
            CheckSchedulePerThread(perThread);
            return perThread;
        }

        /// <summary>
        /// Find the best thread for each block
        /// </summary>
        /// <param name="commCostsOfBlock">Score of each [block][thread] combination</param>
        /// <param name="action">Invoked on every assignment of block to thread</param>
        /// <returns>Unassigned threads</returns>
        private ICollection<int> FindOptimalAssignment(int[][] commCostsOfBlock, Action<int, int> action)
        {
            int numBlocks = commCostsOfBlock.Length;
            int threadCount = commCostsOfBlock[0].Length;
            int[] threadOfBlock = new int[numBlocks];
            // file the cost vectors into heaps
            PriorityQueue<QueueEntry>[] queueForThread = Util.ArrayInit(threadCount, t => new PriorityQueue<QueueEntry>());
            HashSet<int> threadsToAllocate = new HashSet<int>();
            for (int i = 0; i < threadCount; i++)
            {
                threadsToAllocate.Add(i);
            }
            for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++)
            {
                int[] commCosts = commCostsOfBlock[blockIndex];
                // gap = difference between highest cost and 2nd highest cost
                int gap;
                int threadWithMaxCost = GetThreadWithMaxCost(threadsToAllocate, commCosts, out gap);
                queueForThread[threadWithMaxCost].Add(new QueueEntry(blockIndex, gap));
            }
            while (threadsToAllocate.Count > 0)
            {
                // find the queue with largest height
                int height = -1;
                int threadWithMaxHeight = -1;
                for (int thread = 0; thread < threadCount; thread++)
                {
                    var queue = queueForThread[thread];
                    if (queue.Count == 0)
                        continue;
                    var entry = queue[0];
                    if (entry.Height > height)
                    {
                        threadWithMaxHeight = thread;
                        height = entry.Height;
                    }
                }
                if (threadWithMaxHeight < 0)
                    break;
                // assign to that thread
                var bestQueue = queueForThread[threadWithMaxHeight];
                var bestEntry = bestQueue.ExtractMinimum();
                var blockIndex = bestEntry.Node;
                action(blockIndex, threadWithMaxHeight);
                // remove the thread from consideration and re-allocate its heap
                threadsToAllocate.Remove(threadWithMaxHeight);
                foreach (var entry in bestQueue.Items)
                {
                    blockIndex = entry.Node;
                    int[] commCosts = commCostsOfBlock[blockIndex];
                    int gap;
                    int threadWithMaxCost = GetThreadWithMaxCost(threadsToAllocate, commCosts, out gap);
                    queueForThread[threadWithMaxCost].Add(new QueueEntry(blockIndex, gap));
                }
                bestQueue.Clear();
            }
            return threadsToAllocate;
        }

        private int GetThreadWithMaxCost(IEnumerable<int> threads, int[] commCosts, out int gap)
        {
            // find the highest cost thread and gap
            gap = 0;
            int maxCost = -1;
            int threadWithMaxCost = 0;
            foreach (int thread in threads)
            {
                int cost = commCosts[thread];
                if (cost > maxCost)
                {
                    gap = cost - maxCost;
                    maxCost = cost;
                    threadWithMaxCost = thread;
                }
            }
            return threadWithMaxCost;
        }

        /// <summary>
        /// Convert [stage][block][item] into [thread][block][item]
        /// </summary>
        /// <param name="schedule"></param>
        /// <param name="threadCount"></param>
        /// <returns></returns>
        public int[][][] ConvertToSchedulePerThreadFast(IReadOnlyList<IReadOnlyList<IReadOnlyList<int>>> schedule, int threadCount)
        {
            List<List<IReadOnlyList<int>>> perThread = new List<List<IReadOnlyList<int>>>();
            foreach (var stage in schedule)
            {
                for (int blockIndex = 0; blockIndex < stage.Count; blockIndex++)
                {
                    if (perThread.Count <= blockIndex)
                    {
                        perThread.Add(new List<IReadOnlyList<int>>());
                    }
                    perThread[blockIndex].Add(stage[blockIndex]);
                }
                if (stage.Count > threadCount)
                    throw new ArgumentException("stage.Count > threadCount");
                for (int blockIndex = stage.Count; blockIndex < threadCount; blockIndex++)
                {
                    if (perThread.Count <= blockIndex)
                    {
                        perThread.Add(new List<IReadOnlyList<int>>());
                    }
                    perThread[blockIndex].Add(new List<int>());
                }
            }
            return Util.ArrayInit(perThread.Count, i =>
                    Util.ArrayInit(perThread[i].Count, j =>
                        Util.ArrayInit(perThread[i][j].Count, k => perThread[i][j][k])));
        }

        /// <summary>
        /// Renumbers the nodes in a schedule to be process-local.
        /// </summary>
        /// <param name="schedulePerThread"></param>
        /// <returns></returns>
        public static int[][][] ConvertToSchedulePerProcess(int[][][] schedulePerThread)
        {
            int numThreads = schedulePerThread.Length;
            var scheduleForProcess = Util.ArrayInit(numThreads, process => new int[0][]);
            for (int process = 0; process < numThreads; process++)
            {
                int processLocalNodeIndex = 0;
                List<int[]> processLocalStages = new List<int[]>();
                for (int stageIndex = 0; stageIndex < schedulePerThread[process].Length; stageIndex++)
                {
                    List<int> processLocalBlock = new List<int>();
                    for (int indexInBlock = 0; indexInBlock < schedulePerThread[process][stageIndex].Length; indexInBlock++)
                    {
                        int nodeIndex = schedulePerThread[process][stageIndex][indexInBlock];
                        processLocalBlock.Add(processLocalNodeIndex++);
                    }
                    processLocalStages.Add(processLocalBlock.ToArray());
                }
                scheduleForProcess[process] = processLocalStages.ToArray();
            }
            return scheduleForProcess;
        }

        public class NodeLocation
        {
            public readonly int Process;
            public readonly int LocalIndex;

            public NodeLocation(int process, int localIndex)
            {
                this.Process = process;
                this.LocalIndex = localIndex;
            }
        }

        // gameIndicesToSend[sender][recipient]
        // gameIndicesToReceive[recipient][sender]
        public static void ConvertToSchedulePerProcess(int[][][] schedulePerThread, NodeLocation[] locations, 
            out List<int>[][] nodeIndicesToSend, out List<int>[][] nodeIndicesToReceive)
        {
            int processCount = schedulePerThread.Length;
            nodeIndicesToSend = Util.ArrayInit(processCount, sender => Util.ArrayInit(processCount, recipient => new List<int>()));
            nodeIndicesToReceive = Util.ArrayInit(processCount, recipient => Util.ArrayInit(processCount, sender => new List<int>()));
            for (int process = 0; process < processCount; process++)
            {
                int processLocalNodeIndex = 0;
                for (int stageIndex = 0; stageIndex < schedulePerThread[process].Length; stageIndex++)
                {
                    for (int indexInBlock = 0; indexInBlock < schedulePerThread[process][stageIndex].Length; indexInBlock++)
                    {
                        int nodeIndex = schedulePerThread[process][stageIndex][indexInBlock];
                        var location = locations[nodeIndex];
                        nodeIndicesToSend[location.Process][process].Add(location.LocalIndex);
                        nodeIndicesToReceive[process][location.Process].Add(processLocalNodeIndex);
                        schedulePerThread[process][stageIndex][indexInBlock] = processLocalNodeIndex;
                        processLocalNodeIndex++;
                    }
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="schedulePerThread">[process][stageIndex][indexInBlock]</param>
        /// <returns></returns>
        public static double GetMedianMinBlockSize(int[][][] schedulePerThread)
        {
            int numStages = schedulePerThread[0].Length;
            var minBlockSizes = Util.ArrayInit(numStages, stageIndex => schedulePerThread.Min(s => (double)s[stageIndex].Length));
            return MMath.Median(minBlockSizes);
        }

        public static double GetMedianMinBlockSize(int[][][][] schedulePerThreadInProcess, out double medianThreadStages)
        {
            int distributedStageCount = schedulePerThreadInProcess.Length;
            var minBlockSizes = Util.ArrayInit(distributedStageCount, distributedStageIndex =>
            {
                var schedulePerThread = schedulePerThreadInProcess[distributedStageIndex];
                return GetMedianMinBlockSize(schedulePerThread);
            });
            var threadStageCounts = Util.ArrayInit(distributedStageCount, distributedStageIndex =>
            {
                var schedulePerThread = schedulePerThreadInProcess[distributedStageIndex];
                return (double)schedulePerThread[0].Length;
            });
            medianThreadStages = MMath.Median(threadStageCounts);
            return MMath.Median(minBlockSizes);
        }

        public static int[][][][] GetSchedulePerThreadInProcess(int[][] scheduleForProcess, IReadOnlyList<int[]> variablesUsedByNodeInProcess, int numThreads, Action<string> loggingAction = null)
        {
            int distributedStageCount = scheduleForProcess.Length;
            int[][][][] schedulePerThreadPerStage = new int[distributedStageCount][][][];
            for (int distributedStageIndex = 0; distributedStageIndex < distributedStageCount; distributedStageIndex++)
            {
                var nodesInStage = scheduleForProcess[distributedStageIndex];
                var variablesUsedByNodeInStage = GetItems(variablesUsedByNodeInProcess, nodesInStage);
                var ps = new ParallelScheduler();
                ps.LoggingAction = loggingAction;
                ps.CreateGraph(variablesUsedByNodeInStage);
                var scheduleForStage = ps.GetScheduleWithBarriers(numThreads);
                var schedulePerThread = ps.ConvertToSchedulePerThread(scheduleForStage, numThreads);
                // remap stage-local indices to process-local indices
                foreach (var thread in schedulePerThread)
                {
                    foreach (var threadStage in thread)
                    {
                        for (int i = 0; i < threadStage.Length; i++)
                        {
                            threadStage[i] = nodesInStage[threadStage[i]];
                        }
                    }
                }
                schedulePerThreadPerStage[distributedStageIndex] = schedulePerThread;
            }
            return schedulePerThreadPerStage;
        }

        private static T[] GetItems<T>(IReadOnlyList<T> array, IList<int> indices)
        {
            T[] result = new T[indices.Count];
            for (int i = 0; i < indices.Count; i++)
            {
                result[i] = array[indices[i]];
            }
            return result;
        }

        // arrayIndicesToSend[process][stage][recipient] is the set of array elements to send prior to the stage.
        // arrayIndicesToReceive[process][stage][sender] is the set of array elements to receive prior to the stage.
        private static void GetArrayIndicesToSend(int[][][] schedulePerProcess, IReadOnlyList<IReadOnlyList<int>> variablesUsedByNode,
            out int[][][][] arrayIndicesToSend, out int[][][][] arrayIndicesToReceive)
        {
            Dictionary<int, int> lastProcessOfVariable = new Dictionary<int, int>();
            int numProcesses = schedulePerProcess.Length;
            List<int>[][][] arrayIndicesToSendList = Util.ArrayInit(numProcesses, process =>
                Util.ArrayInit(schedulePerProcess[process].Length, stageIndex =>
                Util.ArrayInit(numProcesses, recipient =>
                new List<int>())));
            List<int>[][][] arrayIndicesToReceiveList = Util.ArrayInit(numProcesses, process =>
                Util.ArrayInit(schedulePerProcess[process].Length, stageIndex =>
                Util.ArrayInit(numProcesses, sender =>
                new List<int>())));
            int numStages = schedulePerProcess[0].Length;
            for (int stageIndex = 0; stageIndex < numStages; stageIndex++)
            {
                for (int process = 0; process < numProcesses; process++)
                {
                    for (int indexInBlock = 0; indexInBlock < schedulePerProcess[process][stageIndex].Length; indexInBlock++)
                    {
                        int nodeIndex = schedulePerProcess[process][stageIndex][indexInBlock];
                        foreach (var arrayIndex in variablesUsedByNode[nodeIndex])
                        {
                            int lastProcess;
                            if (lastProcessOfVariable.TryGetValue(arrayIndex, out lastProcess) && (lastProcess != process))
                            {
                                // lastProcess must send prior to this stage
                                arrayIndicesToSendList[lastProcess][stageIndex][process].Add(arrayIndex);
                                // process must receive prior to this stage
                                arrayIndicesToReceiveList[process][stageIndex][lastProcess].Add(arrayIndex);
                            }
                            lastProcessOfVariable[arrayIndex] = process;
                        }
                    }
                }
            }
            arrayIndicesToSend = arrayIndicesToSendList.Select(a => a.Select(b => b.Select(c => c.ToArray()).ToArray()).ToArray()).ToArray();
            arrayIndicesToReceive = arrayIndicesToReceiveList.Select(a => a.Select(b => b.Select(c => c.ToArray()).ToArray()).ToArray()).ToArray();
        }

        public DistributedCommunicationInfo[] GetDistributedCommunicationInfos(int[][][] schedulePerProcess)
        {
            Dictionary<int, int>[] processLocalIndexOfArrayIndexForProcess;
            return GetDistributedCommunicationInfos(schedulePerProcess, this.variablesUsedByNode, out processLocalIndexOfArrayIndexForProcess);
        }

        public static DistributedCommunicationInfo[] GetDistributedCommunicationInfos(int[][][] schedulePerProcess,
            IReadOnlyList<int[]> variablesUsedByNode)
        {
            Dictionary<int, int>[] processLocalIndexOfArrayIndexForProcess;
            return GetDistributedCommunicationInfos(schedulePerProcess, variablesUsedByNode, out processLocalIndexOfArrayIndexForProcess);
        }

        public static DistributedCommunicationInfo[] GetDistributedCommunicationInfos(int[][][] schedulePerProcess,
            IReadOnlyList<int[]> variablesUsedByNode,
            out Dictionary<int, int>[] processLocalIndexOfArrayIndexForProcess)
        {
            int processCount = schedulePerProcess.Length;
            int[][][][] arrayIndicesToSend, arrayIndicesToReceive;
            GetArrayIndicesToSend(schedulePerProcess, variablesUsedByNode, out arrayIndicesToSend, out arrayIndicesToReceive);
            processLocalIndexOfArrayIndexForProcess = new Dictionary<int, int>[processCount];
            var distributedCommunicationInfos = new DistributedCommunicationInfo[processCount];
            for (int process = 0; process < processCount; process++)
            {
                List<int[]> indicesThisProcess = new List<int[]>();
                Dictionary<int, int> processLocalIndexOfArrayIndex = new Dictionary<int, int>();
                for (int stageIndex = 0; stageIndex < schedulePerProcess[process].Length; stageIndex++)
                {
                    List<int> processLocalBlock = new List<int>();
                    for (int indexInBlock = 0; indexInBlock < schedulePerProcess[process][stageIndex].Length; indexInBlock++)
                    {
                        int nodeIndex = schedulePerProcess[process][stageIndex][indexInBlock];
                        List<int> processLocalIndicesThisNode = new List<int>(); //[variablesUsedByNode[nodeIndex].Length];
                        foreach (var arrayIndex in variablesUsedByNode[nodeIndex])
                        {
                            int processLocalArrayIndex;
                            if (!processLocalIndexOfArrayIndex.TryGetValue(arrayIndex, out processLocalArrayIndex))
                            {
                                // allocate a new process local index
                                processLocalArrayIndex = processLocalIndexOfArrayIndex.Count;
                                processLocalIndexOfArrayIndex[arrayIndex] = processLocalArrayIndex;
                                //Debug.WriteLine($"process {process} local index {processLocalArrayIndex} = global index {arrayIndex} for node {nodeIndex}");
                            }
                            processLocalIndicesThisNode.Add(processLocalArrayIndex);
                        }
                        indicesThisProcess.Add(processLocalIndicesThisNode.ToArray());
                    }
                    // translate the send/receive arrays
                    foreach (var arrayIndicesFromOther in arrayIndicesToReceive[process][stageIndex])
                    {
                        for (int i = 0; i < arrayIndicesFromOther.Length; i++)
                        {
                            //Debug.WriteLine($"process {process} stage {stageIndex} receives global index {arrayIndicesFromOther[i]}");
                            arrayIndicesFromOther[i] = processLocalIndexOfArrayIndex[arrayIndicesFromOther[i]];
                        }
                    }
                    foreach (var arrayIndicesToOther in arrayIndicesToSend[process][stageIndex])
                    {
                        for (int i = 0; i < arrayIndicesToOther.Length; i++)
                        {
                            //Debug.WriteLine($"process {process} stage {stageIndex} sends global index {arrayIndicesToOther[i]}");
                            arrayIndicesToOther[i] = processLocalIndexOfArrayIndex[arrayIndicesToOther[i]];
                        }
                    }
                }
                processLocalIndexOfArrayIndexForProcess[process] = processLocalIndexOfArrayIndex;
                DistributedCommunicationInfo distributedCommunicationInfo = new DistributedCommunicationInfo();
                distributedCommunicationInfo.indices = indicesThisProcess.ToArray();
                distributedCommunicationInfo.arrayLength = processLocalIndexOfArrayIndex.Count;
                distributedCommunicationInfo.arrayIndicesToSend = arrayIndicesToSend[process];
                distributedCommunicationInfo.arrayIndicesToReceive = arrayIndicesToReceive[process];
                distributedCommunicationInfos[process] = distributedCommunicationInfo;
            }
            return distributedCommunicationInfos;
        }

        /// <summary>
        /// Get a schedule where all blocks have size 1
        /// </summary>
        /// <param name="degreeOfParallelism"></param>
        /// <returns>[stage][block][item]</returns>
        public List<List<List<int>>> GetScheduleWithBarriers1(int degreeOfParallelism)
        {
            int[] heights = GetHeights();
            int[] unscheduledSourceCounts = GetSourceCounts();
            // [stage][block][item]
            List<List<List<int>>> schedule = new List<List<List<int>>>();
            PriorityQueue<QueueEntry> readyQueue = new PriorityQueue<QueueEntry>();
            // add all nodes with no parents to the ready queue
            foreach (var node in g.Nodes)
            {
                if (unscheduledSourceCounts[node] == 0)
                {
                    readyQueue.Add(new QueueEntry(node, heights[node]));
                }
            }
            List<int> newNodes = new List<int>();
            while (true)
            {
                //Trace.WriteLine("new stage");
                // find ready children of the newly scheduled nodes
                foreach (var node in newNodes)
                {
                    foreach (int target in g.TargetsOf(node))
                    {
                        if (unscheduledSourceCounts[target] <= 0)
                            throw new Exception();
                        unscheduledSourceCounts[target]--;
                        bool allParentsScheduled = (unscheduledSourceCounts[target] == 0);
                        if (allParentsScheduled)
                        {
                            readyQueue.Add(new QueueEntry(target, heights[target]));
                        }
                    }
                }
                newNodes.Clear();
                if (readyQueue.Count == 0)
                    break;

                // build a stage from the nodes on the queue
                List<List<int>> scheduleForStage = new List<List<int>>();
                for (int i = 0; i < degreeOfParallelism; i++)
                {
                    if (readyQueue.Count == 0)
                        break;
                    // extract the ready node with largest height
                    var entry = readyQueue.ExtractMinimum();
                    var node = entry.Node;
                    newNodes.Add(node);
                    List<int> block = new List<int>() { node };
                    scheduleForStage.Add(block);
                }
                //Trace.Write(string.Format("stage {0}: ", schedule.Count));
                //WriteStage(scheduleForStage, true);
                //Trace.WriteLine("");
                schedule.Add(scheduleForStage);
            }
            return schedule;
        }

        public List<List<List<int>>> GetScheduleWithBarriers(int degreeOfParallelism)
        {
            if (degreeOfParallelism < 1)
                throw new ArgumentException($"degreeOfParallelism ({degreeOfParallelism}) < 1");
            // [stage][block][game]
            List<List<List<int>>> schedule = new List<List<List<int>>>();
            if (degreeOfParallelism == 1)
            {
                // preserve the order of nodes so that results are consistent with a sequential schedule.
                List<int> block = new List<int>();
                foreach (var node in g.Nodes)
                {
                    block.Add(node);
                }
                schedule.Add(new List<List<int>>() { block });
                return schedule;
            }
            int[] heights = GetHeights();
            histogram = new HeightHistogram(heights, degreeOfParallelism);
            //Trace.WriteLine(string.Format("{0} idle slots", histogram.CountIdleAtOrAboveHeight(0)));
            unscheduledSourceCounts = GetSourceCounts();
            queuePositions = Util.ArrayInit(g.Nodes.Count, i => -1);
            readyQueue = new PriorityQueue<QueueEntry>();
            readyQueue.Moved += (entry, position) => queuePositions[entry.Node] = position;
            // add all nodes with no parents to the ready queue
            foreach (var node in g.Nodes)
            {
                if (unscheduledSourceCounts[node] == 0)
                {
                    readyQueue.Add(new QueueEntry(node, heights[node]));
                }
            }
            bool[] inPreviousStage = new bool[g.Nodes.Count];
            bool debug = false;
            int prohibitedHeight = -1;
            while (true)
            {
                if (schedule.Count > 0 && schedule.Count % 100 == 0)
                    LoggingAction?.Invoke($"scheduling stage {schedule.Count} " +
                        $"GC.GetTotalMemory = {GC.GetTotalMemory(false) / 1024 / 1024 / 1024} GB " + 
                        $"GC.CollectionCount(2) = {GC.CollectionCount(2)} ");
                HashSet<Block> roots = new HashSet<Block>();
                //Trace.WriteLine("building in-tree");
                int minHeightOfReadyNode;
                BuildForest(roots, degreeOfParallelism, inPreviousStage, prohibitedHeight, out minHeightOfReadyNode);
                if (debug)
                {
                    CheckForest(roots);
                    int totalSize = roots.Sum(block => block.Size);
                    Trace.WriteLine(string.Format("tree has {0} nodes among {1} roots", totalSize, roots.Count));
                }
                if (roots.Count == 0)
                    break;
                if (roots.Count >= degreeOfParallelism)
                {
                    //Trace.WriteLine("pruning");
                    PruneForest(roots, degreeOfParallelism);
                    CheckForest(roots);
                    int totalSize = roots.Sum(block => block.Size);
                    if (debug)
                        Trace.WriteLine(string.Format("stage has {0} nodes", totalSize));
                    if (totalSize == 0)
                    {
                        prohibitedHeight = minHeightOfReadyNode;
                        continue;
                    }
                    CopyParentBlocksIntoBase(roots);
                }
                if (roots.Count < degreeOfParallelism)
                {
                    // do nothing
                }
                prohibitedHeight = -1;

                List<List<int>> scheduleForStage = new List<List<int>>();
                foreach (var block in roots)
                {
                    if (block.parentBlocks.Count > 0)
                        throw new Exception("block.parentBlocks.Count > 0");
                    scheduleForStage.Add(block.nodes);
                    foreach (var node in block.nodes)
                    {
                        inPreviousStage[node] = true;
                    }
                }
                if (debug)
                {
                    Trace.Write(string.Format("stage {0}: ", schedule.Count));
                    WriteStage(scheduleForStage, true);
                    Trace.WriteLine("");
                }
                schedule.Add(scheduleForStage);
            }
            CheckSchedule(schedule);
            return schedule;
        }

        /// <summary>
        /// Build a set of in-trees of blocks
        /// </summary>
        /// <param name="roots">On exit, holds the roots of the in-trees</param>
        /// <param name="degreeOfParallelism"></param>
        /// <param name="inPreviousStage"></param>
        /// <param name="prohibitedHeight"></param>
        /// <param name="minHeight">Minimum height of a ready node.</param>
        private void BuildForest(HashSet<Block> roots, int degreeOfParallelism, bool[] inPreviousStage, int prohibitedHeight, out int minHeight)
        {
            // no block should ever exceed this size
            int sizeThreshold = g.Nodes.Count / degreeOfParallelism;
            Block[] blockOfNode = new Block[g.Nodes.Count];
            List<int> newNodes = new List<int>();
            List<int> skippedNodes = new List<int>();
            List<int> readyNodes = new List<int>();
            Action<Block, int> addToBlock = delegate (Block block, int node)
            {
                block.Add(node);
                blockOfNode[node] = block;
                newNodes.Add(node);
                histogram.RemoveNodeAtHeight(heights[node]);
            };
            bool refreshQueue = false; // for debugging
            if (refreshQueue)
            {
                readyQueue.Clear();
                Array.Clear(unscheduledSourceCounts, 0, unscheduledSourceCounts.Length);
                // add all nodes with no parents to the ready queue
                foreach (var node in g.Nodes)
                {
                    if (inPreviousStage[node])
                        continue;
                    foreach (int source in g.SourcesOf(node))
                    {
                        if (inPreviousStage[source])
                            continue;
                        unscheduledSourceCounts[node]++;
                    }
                    if (unscheduledSourceCounts[node] == 0)
                    {
                        readyQueue.Add(new QueueEntry(node, heights[node]));
                    }
                }
            }
            // all ready nodes (above the prohibited height) become blocks
            minHeight = int.MaxValue;
            while (readyQueue.Count > 0)
            {
                if (readyQueue.Items[0].Height <= prohibitedHeight)
                    break;
                // extract the node with largest height
                var entry = readyQueue.ExtractMinimum();
                int node = entry.Node;
                var block = new Block();
                roots.Add(block);
                addToBlock(block, node);
                minHeight = System.Math.Min(minHeight, entry.Height);
            }
            if (roots.Count == 0)
                return;
            HashSet<Block> parentBlocksGlobal = new HashSet<Block>();
            Func<int, HashSet<Block>> getParentBlocks = delegate (int target)
            {
                parentBlocksGlobal.Clear();
                foreach (int source in g.SourcesOf(target))
                {
                    if (inPreviousStage[source])
                        continue;
                    // source must be in current stage
                    var parentBlock = blockOfNode[source].GetRoot();
                    blockOfNode[source] = parentBlock;
                    parentBlocksGlobal.Add(parentBlock);
                }
                return parentBlocksGlobal;
            };
            int count = 0;
            // construct an in-tree
            while (true)
            {
                // find new candidates from the newly scheduled nodes
                foreach (var node in newNodes)
                {
                    foreach (int target in g.TargetsOf(node))
                    {
                        unscheduledSourceCounts[target]--;
                        bool allParentsScheduled = (unscheduledSourceCounts[target] == 0);
                        if (allParentsScheduled)
                        {
                            readyNodes.Add(target);
                        }
                    }
                }
                newNodes.Clear();
                readyQueue.AddRange(readyNodes.Select(node => new QueueEntry(node, heights[node])));
                readyNodes.Clear();
                if (readyQueue.Count == 0)
                    break;
                if (roots.Count < degreeOfParallelism)
                {
                    // TODO
                    break;
                }

                if (++count % 100 == 0)
                {
                    Block largestBlock = roots.First();
                    int totalSize = 0;
                    foreach (var block in roots)
                    {
                        if (block.Size > largestBlock.Size)
                            largestBlock = block;
                        totalSize += block.Size;
                    }
                    int otherSize = totalSize - largestBlock.Size;
                    if (largestBlock.Size - otherSize > BuildTreeImbalanceThreshold && degreeOfParallelism > 1)
                        break;
                    bool traceQueue = false;
                    if (traceQueue)
                    {
                        HashSet<Block> readyBlocks = new HashSet<Block>();
                        HashSet<Block> mergingWithLargest = new HashSet<Block>();
                        HashSet<Block> mergingWithOther = new HashSet<Block>();
                        foreach (var entry in readyQueue.Items)
                        {
                            var node = entry.Node;
                            var parentBlocks = getParentBlocks(node);
                            if (parentBlocks.Count == 1)
                                readyBlocks.AddRange(parentBlocks);
                            int newCount = roots.Count - parentBlocks.Count + 1;
                            if (newCount < degreeOfParallelism)
                                continue;
                            if (parentBlocks.Contains(largestBlock))
                                mergingWithLargest.AddRange(parentBlocks);
                            else
                                mergingWithOther.AddRange(parentBlocks);
                        }
                        mergingWithLargest.RemoveWhere(block => readyBlocks.Contains(block) || mergingWithOther.Contains(block));
                        mergingWithOther.RemoveWhere(block => readyBlocks.Contains(block));
                        // this check slows things down because of the size of the readyQueue (1000s of nodes)
                        //Trace.WriteLine(string.Format("readyQueue.Count = {0}", readyQueue.Count));
                        //Trace.WriteLine(string.Format("block sizes = {0}", StringUtil.CollectionToString(blocks.Select(m => m.Size), " ")));
                        //Trace.WriteLine(string.Format("{0} ready, {1} merging with largest, {2} merging with other", readyBlocks.Count, mergingWithLargest.Count, mergingWithOther.Count));
                        if (mergingWithOther.Count + readyBlocks.Count < degreeOfParallelism)
                        {
                            int notReadyCount = 0;
                            foreach (var block in roots)
                            {
                                if (readyBlocks.Contains(block) || mergingWithOther.Contains(block))
                                    continue;
                                notReadyCount += block.Size;
                            }
                            sizeThreshold = System.Math.Min(sizeThreshold, notReadyCount);
                        }
                    }
                }

                while (readyQueue.Count > 0)
                {
                    if (readyQueue.Items[0].Height <= prohibitedHeight)
                        break;
                    // extract the node with largest height
                    var entry = readyQueue.ExtractMinimum();
                    int mergeNode = entry.Node;
                    bool traceActive = false;
                    if (traceActive)
                    {
                        int activeBlockCount = 0;
                        int pendingBlockCount = 0;
                        int mergeHeight = heights[mergeNode];
                        foreach (var block in roots)
                        {
                            int node = block.nodes[block.nodes.Count - 1];
                            int height = heights[node];
                            if (height < mergeHeight)
                                pendingBlockCount++;
                            else
                                activeBlockCount++;
                        }
                        Trace.WriteLine(string.Format("{0} active, {1} pending", activeBlockCount, pendingBlockCount));
                        if (activeBlockCount == 0)
                            throw new Exception();
                    }
                    var parentBlocks = getParentBlocks(mergeNode);
                    if (parentBlocks.Count == 1)
                    {
                        var parentBlock = parentBlocks.First();
                        if (parentBlock.Size >= sizeThreshold)
                        {
                            skippedNodes.Add(mergeNode);
                        }
                        else
                        {
                            addToBlock(parentBlock, mergeNode);
                            break;
                        }
                    }
                    else
                    {
                        int newSize = 1 + parentBlocks.Sum(parentBlock => parentBlock.Size);
                        int newCount = roots.Count - parentBlocks.Count + 1;
                        if (newCount < degreeOfParallelism || newSize > sizeThreshold)
                        {
                            skippedNodes.Add(mergeNode);
                            continue;
                        }
                        // merge into a new block
                        Block block = new Block();
                        addToBlock(block, mergeNode);
                        block.AddParentBlocks(parentBlocks);
                        foreach (var parentBlock in parentBlocks)
                        {
                            roots.Remove(parentBlock);
                        }
                        roots.Add(block);
                        break;
                    }
                }
            }
            // put skipped nodes back onto the queue
            readyQueue.AddRange(skippedNodes.Select(node => new QueueEntry(node, heights[node])));
        }

        private void PruneForest(HashSet<Block> roots, int degreeOfParallelism)
        {
            List<MergedBlock<Block>> originalBlocks = new List<MergedBlock<Block>>();
            foreach (var root in roots)
            {
                originalBlocks.Add(new MergedBlock<Block>(root, root.Size));
            }
            List<int> prunedNodes = new List<int>();
            Func<MergedBlock<Block>, Block, int, MergedBlock<Block>> removeFromBlock = delegate (MergedBlock<Block> mergedBlock, Block block, int count)
            {
                if (count > block.nodes.Count)
                    throw new Exception("count > block.nodes.Count");
                if (count == block.nodes.Count)
                {
                    foreach (int node in block.nodes)
                    {
                        histogram.AddNodeAtHeight(heights[node]);
                        prunedNodes.Add(node);
                    }
                    // this must be a base block
                    if (!roots.Contains(block))
                        throw new Exception();
                    // break up the block
                    roots.Remove(block);
                    mergedBlock.Size -= block.Size;
                    MergedBlock<Block> mergedBlockOfBlock;
                    var newMergedBlock = mergedBlock.Remove(block, out mergedBlockOfBlock);
                    originalBlocks.Remove(mergedBlockOfBlock);
                    // add non-empty parent blocks
                    ForEachNonEmptyBlock(block.parentBlocks, b =>
                    {
                        roots.Add(b);
                        var mergedBlock2 = new MergedBlock<Block>(b, b.Size);
                        originalBlocks.Add(mergedBlock2);
                        if (newMergedBlock == null)
                            newMergedBlock = mergedBlock2;
                        else
                            newMergedBlock.Add(mergedBlock2);
                    });
                    return newMergedBlock;
                }
                else
                {
                    mergedBlock.Size -= count;
                    while (count > 0)
                    {
                        int node = block.nodes[block.nodes.Count - 1];
                        histogram.AddNodeAtHeight(heights[node]);
                        prunedNodes.Add(node);
                        block.RemoveLast();
                        count--;
                    }
                    return mergedBlock;
                }
            };
            bool firstIteration = true;
            while (true)
            {
                if (roots.Count < degreeOfParallelism) // TODO
                    break;
                // erase merging information
                if (originalBlocks.Count != roots.Count)
                    throw new Exception("originalBlocks.Count != roots.Count");
                foreach (var mergedBlock in originalBlocks)
                {
                    mergedBlock.Size = mergedBlock.OriginalBlock.Size;
                    mergedBlock.NextBlock = null;
                }
                // merge blocks 
                List<MergedBlock<Block>> mergedBlocks = PackBlocks(originalBlocks, degreeOfParallelism);
                // sort the blocks by decreasing size
                mergedBlocks.Sort();
                mergedBlocks.Reverse();
                // are the merged blocks sufficiently balanced?
                MergedBlock<Block> largestBlock = mergedBlocks[0];
                MergedBlock<Block> smallestBlock = mergedBlocks[mergedBlocks.Count - 1];

                if (!firstIteration || !FillIdleSlots)
                {
                    if (largestBlock.Size - smallestBlock.Size <= PruningImbalanceThreshold)
                    {
                        // create final blocks
                        foreach (MergedBlock<Block> mergedBlock in mergedBlocks)
                        {
                            Block targetBlock = mergedBlock.OriginalBlock;
                            foreach (Block block in mergedBlock.GetBlocks().Skip(1))
                            {
                                targetBlock.AddParentBlock(block);
                                roots.Remove(block);
                            }
                        }
                        break;
                    }

                    // prune nodes from the largest block until it is smaller than the 2nd largest block, or it splits into multiple blocks
                    //Trace.WriteLine(string.Format("block sizes = {0}", StringUtil.CollectionToString(mergedBlocks.Select(m => m.Size), " ")));
                    int desiredSize = mergedBlocks[1].Size;
                    if (desiredSize > smallestBlock.Size)
                    {
                        // prune until the largest block is smaller than the 2nd largest
                        desiredSize--;
                    }
                    int originalBlockCount = largestBlock.Count;
                    while (largestBlock.Size > desiredSize && largestBlock.Count == originalBlockCount)
                    {
                        if (largestBlock.NextBlock == null)
                        {
                            // remove many nodes at once
                            Block blockOfLowestNode = largestBlock.OriginalBlock;
                            int removeCount = System.Math.Min(largestBlock.Size - desiredSize, blockOfLowestNode.nodes.Count);
                            largestBlock = removeFromBlock(largestBlock, blockOfLowestNode, removeCount);
                        }
                        else
                        {
                            Block blockOfLowestNode;
                            int lowestNode = FindLowestNode(largestBlock.GetBlocks(), out blockOfLowestNode);
                            largestBlock = removeFromBlock(largestBlock, blockOfLowestNode, 1);
                        }
                    }
                }
                firstIteration = false;

                if (FillIdleSlots)
                {
                    // prune nodes to fill idle slots
                    int heightWithZeroIdle = int.MaxValue;
                    foreach (MergedBlock<Block> mergedBlock in mergedBlocks)
                    {
                        var mergedBlock2 = mergedBlock;
                        // keep pruning from this block 
                        while (mergedBlock2 != null)
                        {
                            Block blockOfLowestNode;
                            int lowestNode = FindLowestNode(mergedBlock2.GetBlocks(), out blockOfLowestNode);
                            if (lowestNode == -1)
                                break;  // block is empty
                            int height = heights[lowestNode];
                            if (height >= heightWithZeroIdle)
                                break;
                            int idleCount = histogram.CountIdleAtOrAboveHeight(height);
                            if (idleCount > 0)
                            {
                                mergedBlock2 = removeFromBlock(mergedBlock2, blockOfLowestNode, 1);
                                idleCount--;
                            }
                            if (idleCount == 0)
                            {
                                heightWithZeroIdle = height;
                                break;
                            }
                        }
                    }
                }
            }
            //Trace.WriteLine(string.Format("pruned {0} nodes", prunedNodes.Count));
            foreach (int node in prunedNodes)
            {
                foreach (int target in g.TargetsOf(node))
                {
                    unscheduledSourceCounts[target]++;
                    if (queuePositions[target] >= 0)
                    {
                        readyQueue.RemoveAt(queuePositions[target]);
                    }
                }
            }
            List<int> readyNodes = new List<int>();
            foreach (int node in prunedNodes)
            {
                if (unscheduledSourceCounts[node] == 0)
                {
                    readyNodes.Add(node);
                }
            }
            readyQueue.AddRange(readyNodes.Select(node => new QueueEntry(node, heights[node])));
        }

        private static void ForEachNonEmptyBlock(IEnumerable<Block> blocks, Action<Block> action)
        {
            foreach (var block in blocks)
            {
                if (block.nodes.Count > 0)
                    action(block);
                else
                    ForEachNonEmptyBlock(block.parentBlocks, action);
            }
        }

        public static bool ScheduleHasConflict(int[][][] schedulePerThread, IReadOnlyList<int[]> variablesUsedByNode)
        {
            HashSet<int> inCurrentStage = new HashSet<int>();
            HashSet<int> inCurrentBlock = new HashSet<int>();
            int threadCount = schedulePerThread.Length;
            int stageCount = schedulePerThread[0].Length;
            for (int stageIndex = 0; stageIndex < stageCount; stageIndex++)
            {
                inCurrentStage.Clear();
                for (int thread = 0; thread < threadCount; thread++)
                {
                    inCurrentBlock.Clear();
                    foreach (int node in schedulePerThread[thread][stageIndex])
                    {
                        inCurrentBlock.AddRange(variablesUsedByNode[node]);
                    }
                    if (inCurrentStage.ContainsAny(inCurrentBlock))
                        return true;
                    inCurrentStage.AddRange(inCurrentBlock);
                }
            }
            return false;
        }

        // [thread][stage][item]
        private void CheckSchedulePerThread(int[][][] schedule)
        {
            HashSet<int> allNodes = new HashSet<int>();
            int stageCount = schedule[0].Length;
            for (int stageIndex = 0; stageIndex < stageCount; stageIndex++)
            {
                for (int thread = 0; thread < schedule.Length; thread++)
                {
                    foreach(var node in schedule[thread][stageIndex])
                    {
                        if (allNodes.Contains(node))
                            throw new Exception("duplicate node");
                        allNodes.Add(node);
                    }
                }
            }
            if (allNodes.Count != g.Nodes.Count)
                throw new Exception("schedule does not contain all nodes");
        }

        private void CheckSchedule(List<List<List<int>>> schedule)
        {
            HashSet<int> allNodes = new HashSet<int>();
            bool[] inPreviousStage = new bool[g.Nodes.Count];
            HashSet<int> inCurrentStage = new HashSet<int>();
            HashSet<int> inCurrentBlock = new HashSet<int>();
            foreach (var stage in schedule)
            {
                inCurrentStage.Clear();
                foreach (var block in stage)
                {
                    inCurrentBlock.Clear();
                    foreach (var node in block)
                    {
                        if (allNodes.Contains(node))
                            throw new Exception("duplicate node");
                        allNodes.Add(node);
                        inCurrentStage.Add(node);
                        foreach (var source in g.SourcesOf(node))
                        {
                            if (!inPreviousStage[source] && !inCurrentBlock.Contains(source))
                                throw new Exception("cross-block dependency");
                        }
                        inCurrentBlock.Add(node);
                    }
                }
                foreach (var node in inCurrentStage)
                {
                    inPreviousStage[node] = true;
                }
            }
            if (allNodes.Count != g.Nodes.Count)
                throw new Exception("schedule does not contain all nodes");
        }

        private void CheckForest(IEnumerable<Block> roots)
        {
            HashSet<Block> set = new HashSet<Block>();
            Queue<Block> queue = new Queue<Block>();
            foreach (var block in roots)
            {
                if (set.Contains(block))
                    throw new Exception("blocks have a cycle");
                set.Add(block);
                queue.Enqueue(block);
            }
            while (queue.Count > 0)
            {
                var block = queue.Dequeue();
                // check nodes are sorted by height
                for (int i = 1; i < block.nodes.Count; i++)
                {
                    if (heights[block.nodes[i]] > heights[block.nodes[i - 1]])
                        throw new Exception("heights are not sorted");
                }
                foreach (var parentBlock in block.parentBlocks)
                {
                    queue.Enqueue(parentBlock);
                    if (set.Contains(parentBlock))
                        throw new Exception("blocks have a cycle");
                    set.Add(parentBlock);
                }
                if (block.Size != block.GetSizeDebug())
                    throw new Exception("block reports incorrect size");
            }
        }

        private static void CopyParentBlocksIntoBase(IEnumerable<Block> blocks)
        {
            List<int> temp = new List<int>();
            foreach (var block in blocks)
            {
                if (block.parentBlocks.Count == 0)
                    continue;
                temp.Clear();
                temp.AddRange(block.nodes);
                block.nodes.Clear();
                foreach (var parentBlock in block.parentBlocks)
                {
                    parentBlock.ForEachNode(block.nodes.Add);
                }
                // add the original nodes at the end to keep them in correct order
                block.nodes.AddRange(temp);
                block.parentBlocks.Clear();
            }
        }

        private int FindLowestNode(IEnumerable<Block> blocks, out Block blockOfLowestNode)
        {
            blockOfLowestNode = null;
            int lowestNode = -1;
            int minHeight = int.MaxValue;
            foreach (var block in blocks)
            {
                // block.parentBlocks are not examined since they should have larger heights
                // last node must have lowest height
                if (block.nodes.Count == 0)
                    throw new Exception("block is empty");
                int node = block.nodes[block.nodes.Count - 1];
                int height = heights[node];
                if (height < minHeight)
                {
                    lowestNode = node;
                    blockOfLowestNode = block;
                    minHeight = height;
                }
            }
            return lowestNode;
        }

        private class HeightHistogram
        {
            /// <summary>
            /// should not contain any zeros
            /// </summary>
            List<int> countAtHeight = new List<int>();
            /// <summary>
            /// Can be negative, to indicate idle processors
            /// </summary>
            List<int> surplusAtHeight = new List<int>();
            int degreeOfParallelism;
            bool countsChanged;

            public HeightHistogram(int[] heights, int degreeOfParallelism)
            {
                this.degreeOfParallelism = degreeOfParallelism;
                foreach (int height in heights)
                {
                    AddNodeAtHeight(height);
                }
            }

            private void UpdateSurplus()
            {
                if (!countsChanged)
                    return;
                int maxSurplusCount = this.countAtHeight.Count;
                while (surplusAtHeight.Count < maxSurplusCount)
                {
                    surplusAtHeight.Add(0);
                }
                if (surplusAtHeight.Count > maxSurplusCount)
                {
                    surplusAtHeight.RemoveRange(maxSurplusCount, surplusAtHeight.Count - maxSurplusCount);
                }
                for (int height = 0; height < countAtHeight.Count; height++)
                {
                    int available = countAtHeight[height];
                    if (available == 0)
                        throw new Exception("countAtHeight is 0");
                    if (height > 0)
                    {
                        int surplus = surplusAtHeight[height - 1];
                        if (surplus > 0)
                            available += surplus;
                    }
                    surplusAtHeight[height] = available - degreeOfParallelism;
                }
                countsChanged = false;
            }

            public int CountIdleAtOrAboveHeight(int height)
            {
                UpdateSurplus();
                int idleCount = 0;
                for (int i = height; i < surplusAtHeight.Count; i++)
                {
                    if (surplusAtHeight[i] < 0)
                        idleCount -= surplusAtHeight[i];
                }
                if (surplusAtHeight.Count > 0)
                {
                    int finalSurplus = surplusAtHeight[surplusAtHeight.Count - 1];
                    // if finalSurplus=7 with degreeOfParallelism=4, then there is 1 idle slot
                    if (finalSurplus > 0 && (finalSurplus % this.degreeOfParallelism > 0))
                    {
                        idleCount += this.degreeOfParallelism - (finalSurplus % this.degreeOfParallelism);
                    }
                }
                return idleCount;
            }

            public void AddNodeAtHeight(int height)
            {
                while (countAtHeight.Count <= height)
                {
                    countAtHeight.Add(0);
                }
                countAtHeight[height]++;
                countsChanged = true;
            }

            public void RemoveNodeAtHeight(int height)
            {
                countAtHeight[height]--;
                if (countAtHeight[height] == 0)
                {
                    if (countAtHeight.Count != height + 1)
                        throw new Exception("count is 0 at the non-maximum height");
                    countAtHeight.RemoveAt(height);
                }
                countsChanged = true;
            }
        }

        private class MergedBlock<T> : IComparable<MergedBlock<T>>
        {
            public T OriginalBlock;
            public int Size;
            public MergedBlock<T> NextBlock;

            public MergedBlock(T originalBlock, int size)
            {
                this.OriginalBlock = originalBlock;
                this.Size = size;
            }

            public void Add(MergedBlock<T> that)
            {
                if (that.NextBlock != null)
                    throw new Exception("that.NextBlock != null");
                that.NextBlock = this.NextBlock;
                this.NextBlock = that;
                this.Size += that.Size;
            }

            /// <summary>
            /// Does not update Size
            /// </summary>
            /// <param name="value"></param>
            /// <param name="blockOfValue"></param>
            /// <returns></returns>
            public MergedBlock<T> Remove(T value, out MergedBlock<T> blockOfValue)
            {
                if (OriginalBlock.Equals(value))
                {
                    blockOfValue = this;
                    MergedBlock<T> next = this.NextBlock;
                    if (next != null)
                    {
                        next.Size = this.Size;
                    }
                    return next;
                }
                for (MergedBlock<T> current = this; current.NextBlock != null; current = current.NextBlock)
                {
                    MergedBlock<T> next = current.NextBlock;
                    if (next.OriginalBlock.Equals(value))
                    {
                        blockOfValue = next;
                        // skip over this block
                        current.NextBlock = next.NextBlock;
                        next.NextBlock = null;
                        return this;
                    }
                }
                throw new Exception("not found");
            }

            public IEnumerable<T> GetBlocks()
            {
                for (MergedBlock<T> current = this; current != null; current = current.NextBlock)
                {
                    yield return current.OriginalBlock;
                }
            }

            public int Count { get { return GetBlocks().Count(); } }

            public int CompareTo(MergedBlock<T> other)
            {
                return this.Size.CompareTo(other.Size);
            }

            public override string ToString()
            {
                return "MergedBlock(" + Size + ")";
            }
        }

        /// <summary>
        /// Merge blocks until there are a fixed number remaining
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="originalBlocks">Must be in decreasing order of size</param>
        /// <param name="desiredNumberOfBlocks"></param>
        /// <returns></returns>
        private List<MergedBlock<T>> PackBlocks<T>(List<MergedBlock<T>> originalBlocks, int desiredNumberOfBlocks)
        {
            if (desiredNumberOfBlocks >= originalBlocks.Count)
                return originalBlocks;
            // longest processing time algorithm: sort blocks decreasing in size, assign each to the bin with smallest size so far
            PriorityQueue<MergedBlock<T>> bins = new PriorityQueue<MergedBlock<T>>();
            originalBlocks.Sort();  // puts into increasing order of size
            originalBlocks.Reverse();
            foreach (var block in originalBlocks)
            {
                if (bins.Count < desiredNumberOfBlocks)
                {
                    bins.Add(block);
                }
                else
                {
                    var smallestBin = bins[0];
                    smallestBin.Add(block);
                    bins.Changed(0);
                }
            }
            return bins.Items;
        }

        public static void WriteStage(IEnumerable<IReadOnlyCollection<int>> stage, bool blockCounts = false)
        {
            foreach (var block in stage)
            {
                if (blockCounts)
                    Trace.Write(string.Format("{0} ", block.Count));
                else
                {
                    Trace.Write("(");
                    bool firstTime = true;
                    foreach (var node in block)
                    {
                        if (firstTime)
                            firstTime = false;
                        else
                            Trace.Write(",");
                        Trace.Write(string.Format("{0}", node));
                    }
                    Trace.Write(") ");
                }
            }
        }

        public static void WriteSchedule(IEnumerable<IEnumerable<IReadOnlyCollection<int>>> schedule, bool blockCounts = false)
        {
            int stageIndex = 0;
            foreach (var stage in schedule)
            {
                Trace.Write(string.Format("stage {0}: ", stageIndex++));
                WriteStage(stage, blockCounts);
                Trace.WriteLine("");
            }
        }

        public static void WriteThreadSchedule(int[][][] blocksOfThread)
        {
            for (int stageIndex = 0; ; stageIndex++)
            {
                if (blocksOfThread[0].Length <= stageIndex)
                    break;
                Trace.Write(string.Format("stage {0}: ", stageIndex));
                WriteStage(Util.ArrayInit(blocksOfThread.Length, thread => blocksOfThread[thread][stageIndex]));
                Trace.WriteLine("");
            }
        }
    }

    [Serializable]
    public class DistributedCommunicationInfo
    {
        public int[][][] arrayIndicesToSend;
        public int[][][] arrayIndicesToReceive;
        public int arrayLength;
        // same as variablesUsedByNode but variables are renumbered to be process-local
        public int[][] indices;

        public int[] indicesCount
        {
            get
            {
                return Util.ArrayInit(indices.Length, i => indices[i].Length);
            }
        }

        public int[][] arrayIndicesToReceiveCount
        {
            get
            {
                return Util.ArrayInit(arrayIndicesToReceive.Length, stageIndex =>
                        Util.ArrayInit(arrayIndicesToReceive[stageIndex].Length, otherProcess =>
                            arrayIndicesToReceive[stageIndex][otherProcess].Length));
            }
        }

        public int[][] arrayIndicesToSendCount
        {
            get
            {
                return Util.ArrayInit(arrayIndicesToSend.Length, stageIndex =>
                        Util.ArrayInit(arrayIndicesToSend[stageIndex].Length, otherProcess =>
                            arrayIndicesToSend[stageIndex][otherProcess].Length));
            }
        }
    }
}
