// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Microsoft.ML.Probabilistic.Distributions;
    using Math;
    using Xunit;
    using Models;
    using Utilities;
    using Assert = Xunit.Assert;

    using ParallelScheduler = Microsoft.ML.Probabilistic.Compiler.Graphs.ParallelScheduler;
    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Models.Attributes;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    public class ParallelSchedulerTests
    {
        internal void ParallelSchedulerTest()
        {
            ParallelScheduler ps = new ParallelScheduler();
            int[] nodes = Util.ArrayInit(5, i => ps.NewNode());
            ps.AddEdge(nodes[0], nodes[2]);
            ps.AddEdge(nodes[1], nodes[2]);
            ps.AddEdge(nodes[2], nodes[4]);
            ps.AddEdge(nodes[3], nodes[4]);
            var schedule = ps.GetScheduleWithBarriers(2);
            ParallelScheduler.WriteSchedule(schedule);
            // TODO: add assertion
        }

        [Fact]
        public void ParallelSchedulerTest2()
        {
            ParallelScheduler ps = new ParallelScheduler();
            ps.FillIdleSlots = true;
            int depth = 10;
            int[][] variablesUsedByNode = Util.ArrayInit(2 * depth + 1, i =>
              {
                  if (i < depth)
                  {
                      return new int[] { 0 };
                  }
                  else if (i < 2 * depth)
                  {
                      return new int[] { i - depth };
                  }
                  else
                  {
                      return Util.ArrayInit(depth, j => j);
                  }
              });
            ps.CreateGraph(variablesUsedByNode);
            var schedule = ps.GetScheduleWithBarriers(2);
            ParallelScheduler.WriteSchedule(schedule);
            var perThread = ps.ConvertToSchedulePerThread(schedule, 2);
            Assert.True(perThread[1][0].Length == 0);
            Assert.True(perThread[0][2].Length == 0);
        }

        [Fact]
        public void ParallelSchedulerTest3()
        {
            var schedule = ParallelSchedulerTest(20000, 2);
            ParallelScheduler.WriteSchedule(schedule, true);
            Assert.True(schedule.Count == 10);
            Assert.True(schedule[3].Count == 2);
            //Assert.True(schedule[3][0].Count == 21);
            Assert.True(schedule[9].Count == 2);
            //Assert.True(schedule[9][0].Count == 3758);
        }

        [Fact]
        public void ParallelSchedulerEmptySequenceTest()
        {
            ParallelScheduler ps = new ParallelScheduler();
            var variablesUsedByNode = new[] { new int[] { }  };
            ps.CreateGraph(variablesUsedByNode);
        }

        public static List<List<List<int>>> ParallelSchedulerTest(int nodeCount, int parentCount)
        {
            Rand.Restart(0);
            ParallelScheduler ps = new ParallelScheduler();
            int[] nodes = Util.ArrayInit(nodeCount, i => ps.NewNode());
            for (int i = 0; i < nodeCount; i++)
            {
                if (i > parentCount)
                {
                    for (int j = 0; j < parentCount; j++)
                    {
                        int parent = Rand.Int(i);
                        if (!ps.ContainsEdge(nodes[parent], nodes[i]))
                            ps.AddEdge(nodes[parent], nodes[i]);
                    }
                }
            }
            var schedule = ps.GetScheduleWithBarriers(2);
            return schedule;
        }

        /// <summary>
        /// Test parallel inference in a toy model where the array is a time series 
        /// and each factor constrains the next element to be 1+ the previous element. 
        /// The previous element is chosen randomly.
        /// </summary>
        [Fact]
        public void ParallelScheduleTest()
        {
            IReadOnlyList<Gaussian> xMarginal;
            Gaussian shiftMarginal;
            IReadOnlyList<int[]> variablesUsedByNode;
            parallelScheduleTest(2, 20, out xMarginal, out shiftMarginal, out variablesUsedByNode);
        }

        private int[][][] parallelScheduleTest(int numThreads, int nodeCount, out IReadOnlyList<Gaussian> xMarginal, out Gaussian shiftMarginal, out IReadOnlyList<int[]> variablesUsedByNode)
        {
            int maxParentCount = 1;
            variablesUsedByNode = GenerateVariablesUsedByNode(nodeCount, maxParentCount);
            ParallelScheduler ps = new ParallelScheduler();
            ps.CreateGraph(variablesUsedByNode);
            var schedule = ps.GetScheduleWithBarriers(numThreads);
            ParallelScheduler.WriteSchedule(schedule, true);
            var schedulePerThread = ps.ConvertToSchedulePerThread(schedule, numThreads);

            var nodeCountVar = Variable.Observed(nodeCount).Named("nodeCount");
            Range node = new Range(nodeCountVar).Named("node");
            node.AddAttribute(new Sequential() { BackwardPass = true });
            var x = Variable.Array<double>(node).Named("x");
            x[node] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(node);
            var parentCount = Variable.Observed(variablesUsedByNode.Select(a => a.Length).ToArray(), node).Named("parentCount");
            Range parent = new Range(parentCount[node]).Named("parent");
            var indices = Variable.Observed(variablesUsedByNode.ToArray(), node, parent).Named("indices");
            var shift = Variable.GaussianFromMeanAndPrecision(0, 1).Named("shift");
            shift.AddAttribute(new PointEstimate());
            shift.InitialiseTo(Gaussian.PointMass(0));
            using (Variable.ForEach(node))
            {
                var subArray = Variable.Subarray(x, indices[node]).Named("subArray");
                using (Variable.If(parentCount[node] == 1))
                {
                    Variable.ConstrainEqualRandom(subArray[0], Gaussian.FromMeanAndVariance(0,1));
                }
                using (Variable.If(parentCount[node] == 2))
                {
                    Variable.ConstrainEqual(subArray[0], subArray[1] + 1);
                }
                Variable.ConstrainEqualRandom(shift, new Gaussian(1, 2));
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            engine.NumberOfIterations = 2;
            engine.OptimiseForVariables = new IVariable[] { x, shift };
            var xExpected = engine.Infer(x);
            //Console.WriteLine(xExpected);
            var shiftExpected = engine.Infer(shift);

            var threadCount = Variable.Observed(0).Named("threadCount");
            Range thread = new Range(threadCount).Named("thread");
            var blockCount = Variable.Observed(0).Named("blockCount");
            Range gameBlock = new Range(blockCount).Named("block");
            var gameCountInBlock = Variable.Observed(default(int[][]), thread, gameBlock).Named("GameCountInBlock");
            Range gameInBlock = new Range(gameCountInBlock[thread][gameBlock]).Named("gameInBlock");
            var gamesInBlock = Variable.Observed(default(int[][][]), thread, gameBlock, gameInBlock).Named("GamesInBlock");
            node.AddAttribute(new ParallelSchedule(gamesInBlock));

            threadCount.ObservedValue = schedulePerThread.Length;
            blockCount.ObservedValue = (schedulePerThread.Length == 0) ? 0 : schedulePerThread[0].Length;
            gameCountInBlock.ObservedValue = Util.ArrayInit(schedulePerThread.Length, t =>
                Util.ArrayInit(schedulePerThread[t].Length, b => schedulePerThread[t][b].Length));
            gamesInBlock.ObservedValue = schedulePerThread;

            var xActual = engine.Infer<IReadOnlyList<Gaussian>>(x);
            //Debug.WriteLine(xActual);
            Assert.True(xExpected.Equals(xActual));
            var shiftActual = engine.Infer<Gaussian>(shift);
            Assert.True(shiftExpected.Equals(shiftActual));

            xMarginal = xActual;
            shiftMarginal = shiftActual;
            return schedulePerThread;
        }

        /// <summary>
        /// Test parallel inference in a toy model where the array is a time series 
        /// and each factor constrains the next element to be 1+ the previous element. 
        /// The previous element is chosen randomly.
        /// </summary>
        [Fact]
        public void DistributedScheduleTest()
        {
            DistributedSchedule(2, 1);
            DistributedSchedule(2, 2);
        }

        private void DistributedSchedule(int processCount, int threadCount)
        {
            int nodeCount = 100;
            IReadOnlyList<Gaussian> xExpected;
            Gaussian shiftExpected;
            IReadOnlyList<int[]> variablesUsedByNode;
            var schedulePerThread = parallelScheduleTest(processCount, nodeCount, out xExpected, out shiftExpected, out variablesUsedByNode);

            int[][][] schedulePerProcess = ParallelScheduler.ConvertToSchedulePerProcess(schedulePerThread);
            Dictionary<int, int>[] processLocalIndexOfArrayIndexPerProcess;
            IReadOnlyList<Compiler.Graphs.DistributedCommunicationInfo> distributedCommunicationInfos = ParallelScheduler.GetDistributedCommunicationInfos(schedulePerThread, variablesUsedByNode, out processLocalIndexOfArrayIndexPerProcess);
            Console.WriteLine("communication cost = {0}", TotalCommunicationCost(distributedCommunicationInfos));
            int[][][][][] schedulePerThreadPerProcess;
            if (threadCount > 1)
            {
                schedulePerThreadPerProcess = Util.ArrayInit(processCount, process =>
                    ParallelScheduler.GetSchedulePerThreadInProcess(schedulePerProcess[process], distributedCommunicationInfos[process].indices, threadCount)
                );
            }
            else
            {
                schedulePerThreadPerProcess = Util.ArrayInit(processCount, process => default(int[][][][]));
            }

            IList<Gaussian>[] xResultPerProcess = new IList<Gaussian>[processCount];
            Gaussian[] shiftResultPerProcess = new Gaussian[processCount];
            bool debug = false;
            if (debug)
            {
                Assert.Equal(1, processCount);
                SequentialCommunicator comm = new SequentialCommunicator();
                DistributedScheduleTestProcess(comm, xResultPerProcess, shiftResultPerProcess, distributedCommunicationInfos[comm.Rank], schedulePerProcess[comm.Rank], schedulePerThreadPerProcess[comm.Rank]);
            }
            else
            {
                ThreadCommunicator.Run(processCount, comm =>
                    DistributedScheduleTestProcess(comm, xResultPerProcess, shiftResultPerProcess, distributedCommunicationInfos[comm.Rank], schedulePerProcess[comm.Rank], schedulePerThreadPerProcess[comm.Rank])
                );
            }
            Gaussian[] arrayMarginal = new Gaussian[nodeCount];
            for (int process = 0; process < processCount; process++)
            {
                ////Console.WriteLine($"shift {shiftExpected} {shiftResultPerProcess[process]}");
                Assert.Equal(shiftExpected, shiftResultPerProcess[process]);
                foreach (var entry in processLocalIndexOfArrayIndexPerProcess[process])
                {
                    int arrayIndex = entry.Key;
                    int localIndex = entry.Value;
                    arrayMarginal[arrayIndex] = xResultPerProcess[process][localIndex];
                }
            }
            var arrayActual = new DistributionStructArray<Gaussian, double>(arrayMarginal);
            ////Console.WriteLine(arrayActual);
            double error = arrayActual.MaxDiff(xExpected);
            Assert.True(error < 1e-10);
        }

        private static void DistributedScheduleTestProcess(
            ICommunicator comm,
            IList<Gaussian>[] xResults,
            Gaussian[] shiftResults,
            Compiler.Graphs.DistributedCommunicationInfo distributedCommunicationInfo,
            int[][] scheduleForProcess,
            // [distributedStage][thread][block][i]
            int[][][][] schedulePerThreadForProcess)
        {
            var nodeCountVar = Variable.Observed(0).Named("nodeCount");
            Range node = new Range(nodeCountVar).Named("node");
            node.AddAttribute(new Sequential() { BackwardPass = true });
            var itemCountVar = Variable.Observed(0).Named("itemCount");
            Range item = new Range(itemCountVar).Named("item");
            var x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(item);
            var parentCount = Variable.Observed(default(int[]), node).Named("parentCount");
            Range parent = new Range(parentCount[node]).Named("parent");
            var indices = Variable.Observed(default(int[][]), node, parent).Named("indices");
            indices.SetValueRange(item);
            var shift = Variable.GaussianFromMeanAndPrecision(0, 1).Named("shift");
            shift.AddAttribute(new PointEstimate());
            shift.InitialiseTo(Gaussian.PointMass(0));
            using (Variable.ForEach(node))
            {
                var subArray = Variable.Subarray(x, indices[node]).Named("subArray");
                using (Variable.If(parentCount[node] == 1))
                {
                    Variable.ConstrainEqualRandom(subArray[0], Gaussian.FromMeanAndVariance(0, 1));
                }
                using (Variable.If(parentCount[node] == 2))
                {
                    Variable.ConstrainEqual(subArray[0], subArray[1] + 1);
                }
                Variable.ConstrainEqualRandom(shift, new Gaussian(1, 2));
            }

            // this dummy part of the model causes depth cloning to occur
            Range item2 = new Range(0);
            var indices2 = Variable.Observed(new int[0], item2).Named("indices2");
            var subArray2 = Variable.Subarray(x, indices2);
            Variable.ConstrainEqual(subArray2[item2], 0.0);

            var distributedStageCount = Variable.Observed(0).Named("distributedStageCount");
            Range distributedStage = new Range(distributedStageCount).Named("stage");
            var commVar = Variable.Observed(default(ICommunicator)).Named("comm");

            if (schedulePerThreadForProcess != null)
            {
                var threadCount = Variable.Observed(0).Named("threadCount");
                Range thread = new Range(threadCount).Named("thread");
                var blockCountOfDistributedStage = Variable.Observed(default(int[]), distributedStage).Named("blockCount");
                Range gameBlock = new Range(blockCountOfDistributedStage[distributedStage]).Named("block");
                var gameCountInBlockOfDistributedStage = Variable.Observed(default(int[][][]), distributedStage, thread, gameBlock).Named("GameCountInBlock");
                Range gameInBlock = new Range(gameCountInBlockOfDistributedStage[distributedStage][thread][gameBlock]).Named("gameInBlock");
                var gamesInBlockOfDistributedStage = Variable.Array(Variable.Array(Variable.Array(Variable.Array<int>(gameInBlock), gameBlock), thread), distributedStage).Named("GamesInBlock");
                gamesInBlockOfDistributedStage.ObservedValue = default(int[][][][]);
                node.AddAttribute(new DistributedSchedule(commVar, gamesInBlockOfDistributedStage));

                threadCount.ObservedValue = schedulePerThreadForProcess[0].Length;
                blockCountOfDistributedStage.ObservedValue = Util.ArrayInit(schedulePerThreadForProcess.Length, stageIndex => schedulePerThreadForProcess[stageIndex][0].Length);
                gameCountInBlockOfDistributedStage.ObservedValue = Util.ArrayInit(schedulePerThreadForProcess.Length, stageIndex =>
                    Util.ArrayInit(schedulePerThreadForProcess[stageIndex].Length, t =>
                    Util.ArrayInit(schedulePerThreadForProcess[stageIndex][t].Length, b =>
                    schedulePerThreadForProcess[stageIndex][t][b].Length)));
                gamesInBlockOfDistributedStage.ObservedValue = schedulePerThreadForProcess;
            }
            else
            {
                var gameCountInLocalBlock = Variable.Observed(new int[0], distributedStage).Named("gameCountInLocalBlock");
                Range gameInLocalBlock = new Range(gameCountInLocalBlock[distributedStage]).Named("gameInLocalBlock");
                var nodesInLocalBlock = Variable.Observed(new int[0][], distributedStage, gameInLocalBlock).Named("nodesInLocalBlock");
                node.AddAttribute(new DistributedSchedule(commVar, nodesInLocalBlock));

                gameCountInLocalBlock.ObservedValue = Util.ArrayInit(scheduleForProcess.Length, stageIndex =>
                    scheduleForProcess[stageIndex].Length);
                nodesInLocalBlock.ObservedValue = scheduleForProcess;
            }

            var processCount = Variable.Observed(0).Named("processCount");
            Range sender = new Range(processCount);
            var arrayIndicesToSendCount = Variable.Observed(default(int[][]), distributedStage, sender).Named("arrayIndicesToSendCount");
            Range arrayIndexToSend = new Range(arrayIndicesToSendCount[distributedStage][sender]);
            var arrayIndicesToSendVar = Variable.Observed(default(int[][][]), distributedStage, sender, arrayIndexToSend).Named("arrayIndicesToSend");
            var arrayIndicesToReceiveCount = Variable.Observed(default(int[][]), distributedStage, sender).Named("arrayIndicesToReceiveCount");
            Range arrayIndexToReceive = new Range(arrayIndicesToReceiveCount[distributedStage][sender]);
            var arrayIndicesToReceiveVar = Variable.Observed(default(int[][][]), distributedStage, sender, arrayIndexToReceive).Named("arrayIndexToReceive");
            indices.AddAttribute(new DistributedCommunication(arrayIndicesToSendVar, arrayIndicesToReceiveVar));

            distributedStageCount.ObservedValue = scheduleForProcess.Length;

            commVar.ObservedValue = comm;
            processCount.ObservedValue = comm.Size;
            nodeCountVar.ObservedValue = distributedCommunicationInfo.indices.Length;
            itemCountVar.ObservedValue = distributedCommunicationInfo.arrayLength;
            parentCount.ObservedValue = distributedCommunicationInfo.indicesCount;
            indices.ObservedValue = distributedCommunicationInfo.indices;
            arrayIndicesToSendCount.ObservedValue = distributedCommunicationInfo.arrayIndicesToSendCount;
            arrayIndicesToSendVar.ObservedValue = distributedCommunicationInfo.arrayIndicesToSend;
            arrayIndicesToReceiveCount.ObservedValue = distributedCommunicationInfo.arrayIndicesToReceiveCount;
            arrayIndicesToReceiveVar.ObservedValue = distributedCommunicationInfo.arrayIndicesToReceive;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            engine.ModelName = "DistributedScheduleTest" + comm.Rank;
            engine.ShowProgress = false;
            engine.NumberOfIterations = 2;
            engine.OptimiseForVariables = new IVariable[] { x, shift };
            var xActual = engine.Infer<IList<Gaussian>>(x);
            xResults[comm.Rank] = xActual;
            var shiftActual = engine.Infer<Gaussian>(shift);
            shiftResults[comm.Rank] = shiftActual;
        }

        private static List<int[]> GenerateVariablesUsedByNode(int nodeCount, int parentCount)
        {
            Rand.Restart(0);
            List<int[]> variablesUsedByNode = new List<int[]>();
            for (int i = 0; i < nodeCount; i++)
            {
                List<int> variablesUsedThisNode = new List<int>();
                variablesUsedThisNode.Add(i);
                if (i > parentCount)
                {
                    HashSet<int> parents = new HashSet<int>();
                    while (parents.Count < parentCount)
                    {
                        int parent = Rand.Int(i);
                        if (!parents.Contains(parent))
                        {
                            parents.Add(parent);
                            variablesUsedThisNode.Add(parent);
                        }
                    }
                }
                //Debug.WriteLine($"node {i} uses variables {StringUtil.CollectionToString(variablesUsedThisNode, " ")}");
                variablesUsedByNode.Add(variablesUsedThisNode.ToArray());
            }
            return variablesUsedByNode;
        }

        public static int TotalCommunicationCost(IReadOnlyList<Compiler.Graphs.DistributedCommunicationInfo> distributedCommunicationInfos)
        {
            int cost = 0;
            foreach (var info in distributedCommunicationInfos)
            {
                foreach (var stage in info.arrayIndicesToReceiveCount)
                {
                    foreach (var count in stage)
                    {
                        cost += count;
                    }
                }
            }
            return cost;
        }

        public class SequentialCommunicator : CommunicatorBase
        {
            public override int Rank
            {
                get
                {
                    return 0;
                }
            }

            public override int Size
            {
                get
                {
                    return 1;
                }
            }

            public override void Barrier()
            {
                // do nothing
            }

            public class Request : ICommunicatorRequest
            {
                public bool Test()
                {
                    return true;
                }

                public void Wait()
                {
                    // do nothing
                }
            }

            public override ICommunicatorRequest ImmediateSend<T>(T value, int dest, int tag)
            {
                return new Request();
            }

            public override void Receive<T>(int source, int tag, out T value)
            {
                value = default(T);
            }

            public override ICommunicatorRequest ImmediateReceive<T>(int source, int tag, Action<T> action)
            {
                throw new NotImplementedException();
            }

            public override ICommunicatorRequest ImmediateReceive<T>(int source, int tag, T[] values)
            {
                throw new NotImplementedException();
            }

            public override double PercentTimeSpentWaiting => 0;
        }
    }
}