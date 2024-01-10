// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Compiler.Visualizers;
using Microsoft.ML.Probabilistic.Models.Attributes;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Reorders statements within while loops.  Some statements may be copied inside the loop to meet trigger annotations.  
    /// Some statements may be copied outside the loop to meet requirement (SkipIfUniform) annotations.
    /// Attaches InitializerSet attribute to while loops.
    /// </summary>
    internal class SchedulingTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "SchedulingTransform";
            }
        }

        /// <summary>
        /// Enables debug messages to be generated
        /// </summary>
        internal static bool debug;

        /// <summary>
        /// set this to true to work around certain scheduling bugs
        /// </summary>
        internal static bool deleteAllOffsetEdges = false;

        internal static int LastScheduleLength = 0;
        private readonly ModelCompiler compiler;
        private int whileDepth;
        private readonly List<IList<IStatement>> initStmtsOfWhile = new List<IList<IStatement>>();
        private readonly Set<IStatement> topLevelWhileStmts = new Set<IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
        private readonly Dictionary<IWhileStatement, IWhileStatement> convertedWhiles = new Dictionary<IWhileStatement, IWhileStatement>(ReferenceEqualityComparer<IWhileStatement>.Instance);
        private readonly List<IStatement> statementsToFinalize = new List<IStatement>();
        private Dictionary<NodeIndex, NodeIndex> groupOf;
        private Dictionary<NodeIndex, SerialLoopInfo> loopInfoOfGroup;
        private int nextGroupIndex, nextNodeIndex;
        private int whileCount;
        private LoopMergingInfo loopMergingInfo;
        /// <summary>
        /// If true, user-provided initializations will affect the iteration schedule.  This can sometimes improve the convergence rate.
        /// </summary>
        private bool ForceInitializedNodes;
        private LoopAnalysisTransform analysis;

        internal SchedulingTransform(ModelCompiler compiler)
        {
            this.compiler = compiler;
            this.ForceInitializedNodes = compiler.InitialisationAffectsSchedule;
        }

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            if (this.compiler.UseExperimentalSerialSchedules)
            {
                analysis = new LoopAnalysisTransform();
                analysis.Context.InputAttributes = context.InputAttributes;
                analysis.Transform(itd);
                context.Results = analysis.Context.Results;
                if (!context.Results.IsSuccess)
                {
                    Error("analysis failed");
                    return itd;
                }
            }
            var td = base.Transform(itd);
            return td;
        }

        protected override IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            if (!context.InputAttributes.Has<OperatorMethod>(imd))
                return imd;
            loopMergingInfo = context.InputAttributes.Get<LoopMergingInfo>(imd);
            return base.DoConvertMethod(md, imd);
        }

        /// <summary>
        /// Except for while loops, add the original statement to the output, without transforming.  All attributes of the statement are preserved.
        /// </summary>
        /// <param name="ist"></param>
        /// <returns></returns>
        protected override IStatement DoConvertStatement(IStatement ist)
        {
            if (ist is IWhileStatement)
                return ConvertWhile((IWhileStatement)ist);
            else
                return ist;
        }

        /// <summary>
        /// Call Schedule() to reorder the body.
        /// </summary>
        /// <param name="iws"></param>
        protected override IStatement ConvertWhile(IWhileStatement iws)
        {
            if (convertedWhiles.ContainsKey(iws))
                return convertedWhiles[iws];
            IWhileStatement ws = Builder.WhileStmt(iws);
            context.SetPrimaryOutput(ws);
            ws.Condition = ConvertExpression(iws.Condition);
            ws.Body = Builder.BlockStmt();
            Context.OpenStatement(iws.Body);
            context.SetPrimaryOutput(ws.Body);
            IList<IStatement> outputInit;
            if (whileDepth == initStmtsOfWhile.Count)
            {
                outputInit = new List<IStatement>();
                initStmtsOfWhile.Add(outputInit);
            }
            else
            {
                outputInit = initStmtsOfWhile[whileDepth];
                outputInit.Clear();
            }
            if (whileDepth == 0)
                this.statementsToFinalize.Clear();
            whileDepth++;
            Set<IVariableDeclaration> offsetVarsToDelete = new Set<IVariableDeclaration>();
            var ssinfo = context.InputAttributes.Get<SerialSchedulingInfo>(iws);
            if (ssinfo != null)
            {
                foreach (var info in ssinfo.loopInfos)
                {
                    // offset dependencies could form a cycle if there is both a forward and backward loop.
                    // otherwise, you will get pseudo-cycles that disrupt the schedule.
                    // to eliminate pseudo-cycles, we delete the offset edges.
                    if (info.forwardEdges.Count == 0 || info.backwardEdges.Count == 0)
                        offsetVarsToDelete.Add(info.loopVar);
                }
            }
            bool wasForceInitializedNodes = this.ForceInitializedNodes;
            if (whileDepth == 1 && compiler.UseSpecialFirstIteration)
                this.ForceInitializedNodes = true;
            var backEdgeSources = new List<IStatement>();
            bool isCyclic = Schedule(outputInit, ws.Body.Statements, (IReadOnlyList<IStatement>)iws.Body.Statements, backEdgeSources, context.InputAttributes.Has<DoNotSchedule>(iws), offsetVarsToDelete);
            this.ForceInitializedNodes = wasForceInitializedNodes;
            whileDepth--;
            Context.CloseStatement(iws.Body);
            if (ws.Body.Statements.Count == 0)
                ws = null;
            else
            {
                context.InputAttributes.CopyObjectAttributesTo(iws, context.OutputAttributes, ws);
                context.OutputAttributes.Set(ws, new InitializerSet(backEdgeSources));
                // we still surround the body with a while(true) even if it is not cyclic, in order to ensure that the statements do not get separated later
                if (backEdgeSources.Count == 0 && compiler.UseSerialSchedules)
                    context.OutputAttributes.Set(ws, new HasOffsetIndices());
            }
            if (whileDepth == 0)
            {
                context.AddStatementsBeforeAncestorIndex(context.InputStack.Count - 1, outputInit);
                topLevelWhileStmts.Clear();
                context.AddStatementsAfterCurrent(this.statementsToFinalize);
            }
            else if (!this.ForceInitializedNodes)
            {
                convertedWhiles[iws] = ws;
            }
            return ws;
        }

        internal static void ForEachStatement(IEnumerable<IStatement> stmts, Action<IStatement> action)
        {
            foreach (IStatement ist in stmts)
            {
                action(ist);
                if (ist is IWhileStatement)
                {
                    IWhileStatement iws = (IWhileStatement)ist;
                    ForEachStatement(iws.Body.Statements, action);
                }
            }
        }

        private void ForEachLeafStatement(IEnumerable<IStatement> stmts, Action<IStatement> action)
        {
            foreach (IStatement ist in stmts)
            {
                if (ist is IWhileStatement)
                {
                    IWhileStatement iws = (IWhileStatement)ist;
                    ForEachLeafStatement(iws.Body.Statements, action);
                }
                else if (!context.InputAttributes.Has<FirstIterationPostProcessingBlock>(ist))
                {
                    action(ist);
                }
            }
        }

        /// <summary>
        /// Reorder statements within a while loop.
        /// </summary>
        /// <param name="outputInit">Statements to place before the loop.</param>
        /// <param name="outputLoop">Statements to place inside the loop.</param>
        /// <param name="inputStmts">The input statements in the loop.  If doNotSchedule=false, they must have a strongly connected dependency graph.</param>
        /// <param name="backEdgeSources">Modified to have statements whose output is used before being assigned in the loop.</param>
        /// <param name="doNotSchedule">If true, the statements are added to outputLoop in their original order</param>
        /// <param name="offsetVarsToDelete">Offset edges on these variables will be deleted prior to scheduling</param>
        /// <returns>True if the schedule is cyclic, i.e. it needs to be iterated</returns>
        protected bool Schedule(ICollection<IStatement> outputInit, ICollection<IStatement> outputLoop, IReadOnlyList<IStatement> inputStmts, ICollection<IStatement> backEdgeSources, bool doNotSchedule, ICollection<IVariableDeclaration> offsetVarsToDelete)
        {
            if (ReferenceEquals(outputInit, outputLoop))
                throw new ArgumentException("outputInit is the same object as outputLoop");

            DependencyGraph.debug = debug;
            IStatement firstIterPostBlock = ForwardBackwardTransform.ExtractFirstIterationPostProcessingBlock(context, ref inputStmts);
            DependencyGraph g = null;
            bool makeSpecialFirst = (compiler.UseSpecialFirstIteration && whileDepth == 1);
            bool experimental = compiler.UseExperimentalSerialSchedules;
            if (!experimental || doNotSchedule)
            {
                List<IStatement> flatStmts = new List<IStatement>();
                ForEachLeafStatement(inputStmts, flatStmts.Add);
                nextGroupIndex = flatStmts.Count;
                nextNodeIndex = 0;
                groupOf = new Dictionary<NodeIndex, NodeIndex>();
                loopInfoOfGroup = new Dictionary<EdgeIndex, SerialLoopInfo>();
                // this will fill in groupOf and loopInfoOfGroup
                BuildGroups(inputStmts, -1);
                g = new DependencyGraph(context, flatStmts, ignoreMissingNodes: true, ignoreRequirements: false, deleteCancels: true);
                // If replaceTargetIndex is true then TrueSkillChainTest3 fails when OptimiseInferenceCode=False
                // A required statement is not added to the init schedule by repair because it is believed to be fresh
                // due to another statement that updates the same target.  For repair to work correctly, required and fresh
                // constraints must be kept consistent.
                bool replaceTargetIndex = false;
                if (replaceTargetIndex)
                {
                    g.getTargetIndex = delegate (NodeIndex node)
                    {
                        return new DependencyGraph.TargetIndex(loopMergingInfo.GetIndexOf(flatStmts[node]));
                    };
                }
                if (compiler.UseSerialSchedules && !compiler.UseExperimentalSerialSchedules)
                {
                    bool anyDeleted;
                    if (deleteAllOffsetEdges)
                    {
                        // delete all offset edges because the init scheduling code can't handle them yet (GridTest3 will fail)
                        anyDeleted = g.DeleteAllOffsetIndexEdges();
                    }
                    else
                    {
                        // Delete negative offset index edges in forward loops, and positive offset index edges in backward loops.
                        // The source of these offset edges will always be available, so we don't need to consider them when scheduling.
                        // Needed for SerialTests.TrueSkillChainTest
                        anyDeleted = DeleteOffsetEdges(g, doNotSchedule);
                    }
                    bool showOffsetEdges = false;
                    if (debug && g.dependencyGraph.Nodes.Count < 200 && anyDeleted && showOffsetEdges)
                        DrawOffsetEdges(g, groupOf);
                }
                if (doNotSchedule)
                {
                    AddConvertedStatements(outputLoop, inputStmts, new Range(0, inputStmts.Count), firstIterPostBlock);
                    ForEachBackEdgeSource(g, g.dependencyGraph.Nodes, usedNode => backEdgeSources.Add(flatStmts[usedNode]));
                    return true;
                }
                inputStmts = flatStmts;
            }
            NodeIndex[] groupOf2;
            if (experimental) groupOf2 = null;
            else
            {
                // Convert groupOf into an array.
                groupOf2 = new NodeIndex[nextGroupIndex];
                for (int i = 0; i < groupOf2.Length; i++)
                {
                    if (groupOf.ContainsKey(i))
                        groupOf2[i] = groupOf[i];
                    else
                        groupOf2[i] = -1;
                }
            }
            if (!compiler.AllowSerialInitialisers)
            {
                List<IWhileStatement> ancestors = context.FindAncestors<IWhileStatement>(context.InputStack.Count - 2);
                bool isNestedWhile = ancestors.Exists(ancestor => !context.InputAttributes.Has<DoNotSchedule>(ancestor));
                foreach (NodeIndex node in g.dependencyGraph.Nodes)
                {
                    if (inputStmts[node] is IWhileStatement || isNestedWhile ||
                        (groupOf2 != null && groupOf2[node] != -1))
                    {
                        g.mustNotInit.Add(node);
                    }
                }
            }
            List<NodeIndex> schedule = Schedule(g, inputStmts, offsetVarsToDelete, out List<int> initSchedule, out bool isCyclic, groupOf2);
            if (schedule == null)
            {
                schedule = new List<NodeIndex>(new Range(0, inputStmts.Count));
                RecordSchedule("Schedule", schedule, inputStmts);
                AddConvertedStatements(outputLoop, inputStmts, schedule, firstIterPostBlock);
                return true;
            }
            List<NodeIndex> schedule1 = null; // special first iteration
            if (makeSpecialFirst)
            {
                // make a second schedule to follow the special first iteration
                schedule = SecondSchedule(g, initSchedule, schedule, groupOf2, out schedule1);
            }

            if (context.trackTransform)
            {
                RecordSchedule("Init", initSchedule, inputStmts);
                whileCount++;
                // note that if the schedule contains while loops (due to newMethod=false) then their bodies will not be scheduled yet.
                if (schedule1 != null)
                {
                    RecordSchedule("Special", schedule1, inputStmts);
                }
                RecordSchedule("Schedule", schedule, inputStmts);
            }

            // output sources that are inside the convergence loop need to be finalized at the end
            AddFinalizers(g, inputStmts, schedule);

            if (initSchedule.Count > 0)
            {
                AddConvertedStatements(outputInit, inputStmts, initSchedule, null, compiler.AllowSerialInitialisers, compiler.AllowSerialInitialisers, groupOf2);
            }
            if (schedule1 != null)
            {
                // wrap the special first iteration in a fused block, to keep the statements together until IPT
                IWhileStatement ws = Builder.FusedBlockStatement(Builder.LiteralExpr(true));
                AddConvertedStatements(ws.Body.Statements, inputStmts, schedule1, null, compiler.AllowSerialInitialisers, compiler.AllowSerialInitialisers, groupOf2);
                if (firstIterPostBlock != null)
                {
                    // put firstIterPostBlock at the end of the special first iteration
                    ws.Body.Statements.AddRange(((IConditionStatement)firstIterPostBlock).Then.Statements);
                    firstIterPostBlock = null;
                }
                outputInit.Add(ws);
                context.OutputAttributes.Set(ws, new HasOffsetIndices());
                initSchedule.AddRange(schedule1);
            }
            bool wasForceInitializedNodes = this.ForceInitializedNodes;
            this.ForceInitializedNodes = false;
            AddConvertedStatements(outputLoop, inputStmts, schedule, firstIterPostBlock, true, compiler.AllowSerialInitialisers, groupOf2);
            this.ForceInitializedNodes = wasForceInitializedNodes;

            // CheckSchedule will change the deleted edges.
            CheckSchedule(g, experimental, initSchedule, schedule);
            // This must run after CheckSchedule has changed the deleted edges.
            ForEachBackEdgeSource(g, schedule, usedNode => backEdgeSources.Add(inputStmts[usedNode]));

            LastScheduleLength = schedule.Count;
            return isCyclic;
        }

        private void ForEachBackEdgeSource(DependencyGraph g, IEnumerable<NodeIndex> schedule, Action<NodeIndex> action)
        {
            var usedNodes = DeadCodeTransform.CollectUses(g.dependencyGraph, schedule, e => g.isDeleted[e]);
            foreach (var usedNode in usedNodes)
            {
                action(usedNode);
            }
        }

        private List<int> Schedule(DependencyGraph g, IReadOnlyList<IStatement> inputStmts, ICollection<IVariableDeclaration> offsetVarsToDelete, out List<int> initSchedule, out bool isCyclic, int[] groupOf2)
        {
            if (true)
            {
                // This code handles the case where the dependency graph is acyclic after deleting offset edges.
                // It starts by constructing a dependency graph where inner while loops are single nodes.
                // Then it recursively tests for cycles, constructing a schedule in the process.
                // If there are no cycles, then the constructed schedule is returned.
                Dictionary<IStatement, IStatement> replacements = new Dictionary<IStatement, IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
                // if a DependencyInformation refers to an inner statement of a block, make it refer to the block instead
                foreach (IStatement stmt in inputStmts)
                {
                    if (stmt is IWhileStatement iws)
                    {
                        ForEachStatement(iws.Body.Statements, ist => replacements[ist] = iws);
                    }
                }
                DependencyGraph g2 = new DependencyGraph(context, inputStmts, replacements, ignoreMissingNodes: true, ignoreRequirements: false, deleteCancels: true);
#if false
                if (g2.dependencyGraph.Nodes.Count == 1 && !g2.dependencyGraph.ContainsEdge(0, 0) && inputStmts[0] is IWhileStatement)
                {
                    IWhileStatement iws = (IWhileStatement)inputStmts[0];
                    IWhileStatement ws = Builder.WhileStmt(iws);
                    //context.SetPrimaryOutput(ws);
                    ws.Condition = ConvertExpression(iws.Condition);
                    ws.Body = Builder.BlockStmt();
                    Context.OpenStatement(iws.Body);
                    context.SetPrimaryOutput(ws.Body);
                    bool isCyclic = Schedule(outputInit, ws.Body.Statements, (IReadOnlyList<IStatement>)iws.Body.Statements, backEdgeSources, context.InputAttributes.Has<DoNotSchedule>(iws), offsetVarsToDelete);
                    Context.CloseStatement(iws.Body);
                    if (!isCyclic)
                    {
                        outputLoop.Add(ws);
                        context.OutputAttributes.Set(ws, new HasOffsetIndices());
                        return isCyclic;
                    }
                    // fall through
                }
                List<int> schedule = DeleteOffsetEdges(g2, inputStmts, offsetVarsToDelete);
                if (schedule != null)
                    return schedule;
#endif
                // The dependency graph is cyclic even after deleting offset edges.  Fall through and schedule everything as a unit.
            }
            isCyclic = true;
            var scheduler = new Scheduler();
            scheduler.debug = debug;
            scheduler.doRepair = compiler.EnforceTriggers;
            scheduler.RecordText = this.RecordText;
            scheduler.useExperimentalSerialSchedules = compiler.UseExperimentalSerialSchedules;
            if (compiler.UseExperimentalSerialSchedules)
            {
                IndexedProperty<NodeIndex, HashSet<IVariableDeclaration>> loopVarsOfNode;
                loopVarsOfNode = g.dependencyGraph.CreateNodeData<HashSet<IVariableDeclaration>>();
                foreach (var node in g.dependencyGraph.Nodes)
                {
                    loopVarsOfNode[node] = analysis.loopVarsOfStatement[inputStmts[node]];
                }
                scheduler.loopVarsOfNode = loopVarsOfNode;
                var ssinfo = GetSerialSchedulingInfo(g);
                // sort infos by loop priority
                SortByLoopPriority(context, ssinfo.loopInfos);
                IWhileStatement parent = context.FindAncestor<IWhileStatement>();
                context.OutputAttributes.Set(parent, ssinfo);
                //st2.loopVarsWithOffset = ssinfo.loopInfos.Select(loopInfo => loopInfo.loopVar).ToList();
            }
            try
            {
                List<int> schedule = scheduler.IterationSchedule(g, groupOf: groupOf2, forceInitializedNodes: this.ForceInitializedNodes);
                initSchedule = scheduler.InitSchedule(schedule, false);
                bool showScheduleInfo = debug;
                if (showScheduleInfo)
                {
                    int maxBack = scheduler.MaxBackEdgeCount(schedule, out double maxBackOverLength);
                    Trace.WriteLine($"schedule.Count = {schedule.Count}, max back = {maxBack}, product = {schedule.Count * maxBack}, max back/length = {maxBackOverLength}, product = {schedule.Count * maxBackOverLength}");
                }
                return schedule;
            }
            catch (Exception ex) when (!debug)
            {
                Error("Scheduling failed", ex);
                initSchedule = null;
                return null;
            }
        }

        private void AddFinalizers(DependencyGraph g, IReadOnlyList<IStatement> inputStmts, List<int> schedule)
        {
            Set<NodeIndex> outputSources = new Set<NodeIndex>();
            foreach (NodeIndex node in g.dependencyGraph.Nodes)
            {
                IStatement stmt = inputStmts[node];
                if (context.InputAttributes.Has<OutputSource>(stmt))
                    outputSources.Add(node);
            }
            if (outputSources.Count > 0)
            {
                // collect the set of invalid/stale nodes at the end of the iter schedule
                Set<NodeIndex> invalid = new Set<EdgeIndex>();
                Set<DependencyGraph.TargetIndex> stale = new Set<DependencyGraph.TargetIndex>();
                g.RepairSchedule(schedule, invalid, stale);
                List<NodeIndex> tail = new List<EdgeIndex>();
                while (outputSources.Count > 0)
                {
                    NodeIndex source = outputSources.First();
                    if (stale.Contains(g.getTargetIndex(source)) || invalid.Contains(source))
                    {
                        tail.Clear();
                        tail.Add(source);
                        tail = g.RepairSchedule(tail, invalid, stale);
                        foreach (NodeIndex node in tail)
                        {
                            this.statementsToFinalize.Add(inputStmts[node]);
                            outputSources.Remove(node);
                        }
                    }
                    else
                        outputSources.Remove(source);
                }
            }
        }

        private List<int> SecondSchedule(DependencyGraph g, List<int> initSchedule, List<int> schedule, int[] groupOf2, out List<int> schedule1)
        {
            // schedule is now the special first iteration.
            schedule1 = schedule;
            List<NodeIndex> combinedSchedule = new List<NodeIndex>();
            combinedSchedule.AddRange(initSchedule);
            combinedSchedule.AddRange(schedule);
            Scheduler scheduler = new Scheduler();
            scheduler.debug = debug;
            scheduler.doRepair = compiler.EnforceTriggers;
            scheduler.RecordText = this.RecordText;
            scheduler.useExperimentalSerialSchedules = compiler.UseExperimentalSerialSchedules;
            // recover the set of reversed edges in the schedule and make them soft constraints
            Set<EdgeIndex> reversedEdges = new Set<EdgeIndex>();
            foreach (NodeIndex node in combinedSchedule)
            {
                reversedEdges.Remove(g.dependencyGraph.EdgesInto(node));
                reversedEdges.AddRange(g.dependencyGraph.EdgesOutOf(node));
            }
            // construct the new set of initializedNodes (nodes which follow all of their parents in the first schedule, i.e. fresh nodes)
            Set<NodeIndex> initializedNodes = g.initializedNodes;
            g.initializedNodes = new Set<NodeIndex>();
            Set<EdgeIndex> initializedEdges = g.initializedEdges;
            g.initializedEdges = new Set<EdgeIndex>();
            schedule = scheduler.IterationSchedule(g, groupOf: groupOf2, forceInitializedNodes: false, edgesToReverse: reversedEdges);
            g.initializedNodes = initializedNodes;
            g.initializedEdges = initializedEdges;
            if (compiler.EnforceTriggers)
            {
                combinedSchedule = scheduler.RepairCombinedSchedule(schedule, combinedSchedule);
                // extract the second part of the combined schedule after repair
                schedule1 = combinedSchedule.Skip(initSchedule.Count).ToList();
            }

            return schedule;
        }

        private void CheckSchedule(DependencyGraph g, bool experimental, List<int> initSchedule, List<int> schedule)
        {
            Set<NodeIndex> available = Set<NodeIndex>.FromEnumerable(g.hasNonUniformInitializer);
            available.AddRange(g.initializedNodes);
            List<NodeIndex> fullSchedule = new List<EdgeIndex>();
            fullSchedule.AddRange(initSchedule);
            fullSchedule.AddRange(schedule);
            fullSchedule.AddRange(schedule);
            try
            {
                g.CheckSchedule(fullSchedule, available);
            }
            catch (Exception ex)
            {
                Error("Internal error: Invalid schedule", ex);
            }
            // restore offset edges
            g.isDeleted.Clear();
            if (!experimental && compiler.UseSerialSchedules)
                DeleteOffsetEdges(g);
            available = Set<NodeIndex>.FromEnumerable(g.hasNonUniformInitializer);
            available.AddRange(g.initializedNodes);
            // check that all requirements are satisfied
            List<string> shouldBeEmpty = g.CollectRequirements(initSchedule, available, true);
            foreach (string message in shouldBeEmpty)
            {
                Error(message);
            }
            shouldBeEmpty = g.CollectRequirements(schedule, available, false);
            foreach (string message in shouldBeEmpty)
            {
                Error(message);
            }
        }

        internal static SerialSchedulingInfo GetSerialSchedulingInfo(DependencyGraph g)
        {
            // collect the set of forward and backward offset edges
            Dictionary<IVariableDeclaration, SerialLoopInfo> loopInfoOfVariable = new Dictionary<IVariableDeclaration, SerialLoopInfo>();
            List<SerialLoopInfo> infos = new List<SerialLoopInfo>();
            foreach (KeyValuePair<EdgeIndex, IOffsetInfo> entry in g.OffsetIndices)
            {
                EdgeIndex edge = entry.Key;
                IOffsetInfo offsetIndices = entry.Value;
                foreach (var entry2 in offsetIndices)
                {
                    IVariableDeclaration loopVar = entry2.loopVar;
                    int offset = entry2.offset;
                    SerialLoopInfo info;
                    if (!loopInfoOfVariable.TryGetValue(loopVar, out info))
                    {
                        info = new SerialLoopInfo(loopVar);
                        infos.Add(info);
                        loopInfoOfVariable[loopVar] = info;
                    }
                    if (offset > 0)
                    {
                        info.backwardEdges.Add(edge);
                    }
                    else if (offset < 0)
                    {
                        info.forwardEdges.Add(edge);
                    }
                }
            }
            // attach group information for SchedulingTransform
            SerialSchedulingInfo ssinfo = new SerialSchedulingInfo();
            ssinfo.loopInfos.AddRange(infos);
            return ssinfo;
        }

        internal static void SortByLoopPriority(BasicTransformContext context, List<SerialLoopInfo> infos)
        {
            // sort by decreasing priority
            infos.Sort((info1, info2) =>
            {
                var priority1 = context.InputAttributes.Get<LoopPriority>(info1.loopVar).Priority;
                var priority2 = context.InputAttributes.Get<LoopPriority>(info2.loopVar).Priority;
                return Comparer<int>.Default.Compare(priority2, priority1);
            });
        }

        private List<NodeIndex> DeleteOffsetEdges(DependencyGraph g, IReadOnlyList<IStatement> inputStmts, ICollection<IVariableDeclaration> offsetVarsToDelete, out List<NodeIndex> initSchedule)
        {
            // note we could use SerialSchedulingInfo to do the deletion here
            bool anyBlockIsCyclicLocal;
            if (compiler.UseSerialSchedules && !compiler.UseExperimentalSerialSchedules &&
                g.DeleteAllOffsetIndexEdges(offsetVarsToDelete, out anyBlockIsCyclicLocal))
            {
                if (debug)
                {
                    Debug.WriteLine($"offsetVarsToDelete: {offsetVarsToDelete}");
                    if (g.dependencyGraph.Nodes.Count < 100 && false)
                        DrawOffsetEdges(g);
                }
                // if edges were deleted, recompute SCCs
                List<NodeIndex> schedule = ProcessStrongComponents(g, inputStmts, offsetVarsToDelete, out initSchedule);
                if (schedule != null)
                {
                    return schedule;
                }
                // fall through
            }
            initSchedule = null;
            return null;
        }

        private List<NodeIndex> ProcessStrongComponents(DependencyGraph g, IReadOnlyList<IStatement> inputStmts, ICollection<IVariableDeclaration> offsetVarsToDelete, out List<NodeIndex> initSchedule)
        {
            bool isStronglyConnected = false;
            DirectedGraphFilter<NodeIndex, EdgeIndex> graph2 = new DirectedGraphFilter<NodeIndex, EdgeIndex>(g.dependencyGraph, edge => !g.isDeleted[edge]);
            StrongComponents2<NodeIndex> scc = new StrongComponents2<NodeIndex>(graph2.SourcesOf, graph2);
            List<NodeIndex> block = new List<NodeIndex>();
            // BeginComponent is not needed since block is cleared by EndComponent
            //scc.BeginComponent += delegate() { block.Clear(); };
            scc.AddNode += block.Add;
            List<NodeIndex> pendingOutput = new List<NodeIndex>();
            List<NodeIndex> pendingReverse = new List<NodeIndex>();
            List<NodeIndex> schedule = new List<NodeIndex>();
            var initScheduleLocal = new List<NodeIndex>();
            scc.EndComponent += delegate ()
            {
                bool isSingleStmt = (block.Count == 1 && !graph2.ContainsEdge(block[0], block[0]) && !(inputStmts[block[0]] is IWhileStatement));
                if (isSingleStmt)
                {
                    NodeIndex node1 = block[0];
                    if (this.ForceInitializedNodes && g.initializedNodes.Contains(node1))
                    {
                        // updates to initialized nodes must be put at the end, in reverse order
                        pendingReverse.AddRange(block);
                    }
                    else
                    {
                        RecordSchedule("Single", block, inputStmts);
                        pendingOutput.AddRange(block);
                    }
                }
                else if (block.Count == inputStmts.Count)
                {
                    isStronglyConnected = true;
                }
                else
                {
                    List<NodeIndex> schedule2 = Schedule(g, inputStmts, offsetVarsToDelete, out List<NodeIndex> initSchedule2, out bool isCyclic2, null);
                    if (initSchedule2.Count > 0)
                    {
                        initScheduleLocal.AddRange(pendingOutput);
                        initScheduleLocal.AddRange(initSchedule2);
                        schedule.AddRange(pendingOutput);
                        pendingOutput.Clear();
                    }
                    if (this.ForceInitializedNodes)
                    {
                        // updates to initialized nodes must be put at the end, in reverse order
                        foreach (NodeIndex node in schedule2)
                        {
                            if (HasUserInitializedInitializer(context, inputStmts[node]))
                                pendingReverse.Add(node);
                            else
                                pendingOutput.Add(node);
                        }
                    }
                    else
                    {
                        pendingOutput.AddRange(schedule2);
                    }
                }
                block.Clear();
            };
            scc.SearchFrom(g.dependencyGraph.Nodes);
            if (!isStronglyConnected)
            {
                // add pendingReverse statements in reverse order
                pendingReverse.Reverse();
                pendingOutput.AddRange(pendingReverse);
                schedule.AddRange(pendingOutput);
                initSchedule = initScheduleLocal;
                return schedule;
            }
            else
            {
                initSchedule = null;
                return null;
            }
        }

        internal static bool IsUserInitialized(BasicTransformContext context, IStatement ist)
        {
            Initializer attr = context.InputAttributes.Get<Initializer>(ist);
            bool userInitialized = (attr != null) && attr.UserInitialized;
            if (userInitialized || context.InputAttributes.Has<InitialiseBackward>(ist))
            {
                return true;
            }
            return false;
        }

        internal static bool HasUserInitializedInitializer(BasicTransformContext context, IStatement ist)
        {
            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
            if (di == null)
                return false;
            foreach (IStatement source in di.Overwrites)
            {
                if (IsUserInitialized(context, source))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Delete offset edges within a group matching the loop variable being offset
        /// </summary>
        /// <param name="g"></param>
        /// <param name="keepUnavailable"></param>
        /// <returns></returns>
        private bool DeleteOffsetEdges(DependencyGraph g, bool keepUnavailable = false)
        {
            bool anyDeleted = false;
            foreach (KeyValuePair<EdgeIndex, IOffsetInfo> entry in g.OffsetIndices)
            {
                EdgeIndex edge = entry.Key;
                NodeIndex source = g.dependencyGraph.SourceOf(edge);
                NodeIndex target = g.dependencyGraph.TargetOf(edge);
                Set<NodeIndex> sourceGroups = new Set<NodeIndex>();
                ForEachGroupOf(source, group => sourceGroups.Add(group));
                Set<IVariableDeclaration> commonLoopVars = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
                ForEachGroupOf(target, group =>
                {
                    if (sourceGroups.Contains(group))
                    {
                        var info = loopInfoOfGroup[group];
                        if (info != null)
                        {
                            commonLoopVars.Add(info.loopVar);
                        }
                    }
                });
                IOffsetInfo offsetIndices = entry.Value;
                bool isBackward = false;
                bool isForward = false;
                bool isAvailable = true;
                foreach (var entry2 in offsetIndices)
                {
                    var loopVar = entry2.loopVar;
                    if (!commonLoopVars.Contains(loopVar))
                        continue;
                    int offset = entry2.offset;
                    if (offset > 0)
                    {
                        isBackward = true;
                    }
                    else if (offset < 0)
                    {
                        isForward = true;
                    }
                    isAvailable &= entry2.isAvailable;
                }
                if ((isForward || isBackward) && (!keepUnavailable || isAvailable))
                {
                    if (debug)
                        Debug.WriteLine("deleting offset edge (" + source + "," + target + ")");
                    g.isDeleted[edge] = true;
                    anyDeleted = true;
                }
            }
            return anyDeleted;
        }

        private void BuildGroups(IEnumerable<IStatement> stmts, NodeIndex group)
        {
            foreach (IStatement ist in stmts)
            {
                if (ist is IWhileStatement iws)
                {
                    NodeIndex newGroup = nextGroupIndex;
                    nextGroupIndex++;
                    if (group >= 0)
                        groupOf[newGroup] = group;
                    BuildGroups(iws.Body.Statements, newGroup);
                    SerialLoopInfo info = context.InputAttributes.Get<SerialLoopInfo>(iws);
                    loopInfoOfGroup[newGroup] = info;
                }
                else
                {
                    if (group >= 0)
                        groupOf[nextNodeIndex] = group;
                    nextNodeIndex++;
                }
            }
        }

        private void ForEachGroupOf(NodeIndex node, Action<NodeIndex> action)
        {
            NodeIndex group = node;
            while (true)
            {
                if (!groupOf.TryGetValue(group, out group))
                    break;
                action(group);
            }
        }

        internal void RecordSchedule(string prefix, IEnumerable<NodeIndex> schedule, IReadOnlyList<IStatement> inputStmts)
        {
            // record the schedule for transform browser
            var itdOut = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
            string suffix = (whileCount == 1) ? "" : whileCount.ToString(CultureInfo.InvariantCulture);
            IBlockStatement block = Builder.BlockStmt();
            foreach (NodeIndex node in schedule)
            {
                var stmt = inputStmts[node];
                if (context.OutputAttributes.Has<BackEdgeAttribute>(stmt))
                {
                    foreach (var attr in context.OutputAttributes.GetAll<BackEdgeAttribute>(stmt))
                    {
                        block.Statements.Add(Builder.CommentStmt("Uses previous value of " + attr.Message));
                    }
                }
                block.Statements.Add(stmt);
            }
            context.OutputAttributes.Add(itdOut, new DebugInfo()
            {
                Transform = this,
                Name = prefix + suffix,
                Value = block
            });
        }

        /// <summary>
        /// Record debugging information for the TransformBrowser
        /// </summary>
        /// <param name="name">Name of the Tab in the browser</param>
        /// <param name="text">Text to put in the Tab</param>
        internal void RecordText(string name, IEnumerable<string> text)
        {
            var itdOut = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
            IBlockStatement block = Builder.BlockStmt();
            foreach (string line in text)
            {
                block.Statements.Add(Builder.CommentStmt(line));
            }
            context.OutputAttributes.Add(itdOut, new DebugInfo()
            {
                Transform = this,
                Name = name,
                Value = block
            });
        }

        internal static void DrawOffsetEdges(DependencyGraph g, Dictionary<NodeIndex, NodeIndex> groupOf = null, bool inThread = true)
        {
            if (Models.InferenceEngine.Visualizer?.DependencyGraphVisualizer != null)
            {
                Debug.WriteLine("DrawOffsetEdges:");
                foreach (NodeIndex node in g.dependencyGraph.Nodes)
                {
                    if (groupOf == null || !groupOf.ContainsKey(node))
                        Debug.WriteLine("{0} {1}", node, g.NodeToShortString(node));
                    else
                        Debug.WriteLine("{0} [{1}] {2}", node, groupOf[node], g.NodeToShortString(node));
                }
                if (groupOf != null)
                {
                    foreach (NodeIndex node in groupOf.Keys)
                    {
                        if (node < g.dependencyGraph.Nodes.Count)
                            continue;
                        Debug.WriteLine("{0} [{1}] (group)", node, groupOf[node]);
                    }
                }
                // show dependency graph with offset edges colored
                bool isNegative(int edge)
                {
                    IOffsetInfo info;
                    return g.OffsetIndices.TryGetValue(edge, out info) && (info.Count(offset => offset.offset < 0) > 0);
                }
                bool isPositive(int edge)
                {
                    IOffsetInfo info;
                    return g.OffsetIndices.TryGetValue(edge, out info) && (info.Count(offset => offset.offset > 0) > 0);
                }
                bool isNoInit(int edge) => g.noInit[edge];
                var edgeStyles = new EdgeStylePredicate[] {
                    new EdgeStylePredicate("Positive", isPositive, EdgeStyle.Back),
                    new EdgeStylePredicate("Negative", isNegative, EdgeStyle.Blue),
                    new EdgeStylePredicate("NoInit", isNoInit, EdgeStyle.Dashed)
                };
                string title = "offset edges";
                if (inThread)
                {
                    var viewThread = new System.Threading.Thread(delegate ()
                    {
                        Models.InferenceEngine.Visualizer.DependencyGraphVisualizer.VisualizeDependencyGraph(g.dependencyGraph, edgeStyles, null, null, title);
                    });
                    viewThread.Start();
                }
                else
                {
                    Models.InferenceEngine.Visualizer.DependencyGraphVisualizer.VisualizeDependencyGraph(g.dependencyGraph, edgeStyles, null, null, title);
                }
            }
        }

        /// <summary>
        /// Wraps a statement with FusedBlockStatements according to its hierarchical group memberships.
        /// </summary>
        private class GroupWrapper
        {
            readonly int[] groupOf;
            readonly Stack<ICollection<IStatement>> containerStack = new Stack<ICollection<IStatement>>();
            readonly Stack<NodeIndex> groupStack = new Stack<int>();
            readonly Set<NodeIndex> currentGroups = new Set<int>();
            readonly List<NodeIndex> newGroups = new List<int>();
            readonly BasicTransformContext context;
            readonly Dictionary<NodeIndex, SerialLoopInfo> loopInfoOfGroup;

            public GroupWrapper(BasicTransformContext context, ICollection<IStatement> output, int[] groupOf, Dictionary<NodeIndex, SerialLoopInfo> loopInfoOfGroup)
            {
                this.context = context;
                this.groupOf = groupOf;
                this.loopInfoOfGroup = loopInfoOfGroup;
                groupStack.Push(-1);
                currentGroups.Add(-1);
                containerStack.Push(output);
            }

            public void Add(NodeIndex node, IStatement ist)
            {
                if (groupOf != null)
                {
                    // update the set of containers
                    newGroups.Clear();
                    NodeIndex group = groupOf[node];
                    while (true)
                    {
                        if (currentGroups.Contains(group))
                        {
                            // pop until we reach group
                            NodeIndex topGroup = groupStack.Peek();
                            while (topGroup != group)
                            {
                                currentGroups.Remove(topGroup);
                                groupStack.Pop();
                                containerStack.Pop();
                                topGroup = groupStack.Peek();
                            }
                            break;
                        }
                        else
                        {
                            newGroups.Add(group);
                            group = groupOf[group];
                        }
                    }
                    // order from largest to smallest group
                    newGroups.Reverse();
                    foreach (NodeIndex newGroup in newGroups)
                    {
                        groupStack.Push(newGroup);
                        currentGroups.Add(newGroup);
                        SerialLoopInfo info = loopInfoOfGroup[newGroup];
                        IExpression condition;
                        if (info == null) condition = Builder.LiteralExpr(true);
                        else condition = Builder.VarRefExpr(info.loopVar);
                        IWhileStatement ws = Builder.FusedBlockStatement(condition);
                        context.OutputAttributes.Set(ws, new HasOffsetIndices());
                        containerStack.Peek().Add(ws);
                        containerStack.Push(ws.Body.Statements);
                    }
                }
                containerStack.Peek().Add(ist);
            }
        }

        private void AddConvertedStatements(ICollection<IStatement> output, IReadOnlyList<IStatement> inputStmts, IEnumerable<NodeIndex> nodes, IStatement firstIterPostBlock, bool allowNestedWhile = true,
                                            bool allowNestedInit = true, int[] groupOf = null)
        {
            List<IStatement> pendingOutput = new List<IStatement>();
            var groupWrapper = new GroupWrapper(context, pendingOutput, groupOf, loopInfoOfGroup);
            foreach (NodeIndex node in nodes)
            {
                IStatement ist = inputStmts[node];
                IStatement st = ConvertStatement(ist);
                if (st == null)
                    continue;
                IList<IStatement> init = GetInitOfConvertedStatement();
                if (init != null && init.Count > 0)
                {
                    if (!allowNestedInit)
                        throw new Exception("Internal: Serial initializer not allowed");
                    IList<IStatement> myInit = GetInitOfCurrentStatement();
                    bool duplicatePendingOutput = !ReferenceEquals(myInit, output);
                    myInit.AddRange(pendingOutput);
                    myInit.AddRange(init);
                    if (duplicatePendingOutput)
                        output.AddRange(pendingOutput);
                    pendingOutput.Clear();
                    init.Clear();
                }
                // flatten nested while loops
                if (st is IWhileStatement)
                {
                    IWhileStatement iws = (IWhileStatement)st;
                    if (!allowNestedWhile)
                        throw new Exception("Internal: Serial initializer not allowed");
                    pendingOutput.AddRange(iws.Body.Statements);
                }
                else
                {
                    groupWrapper.Add(node, st);
                    //pendingOutput.Add(st);
                }
            }
            output.AddRange(pendingOutput);
            if (firstIterPostBlock != null)
                output.Add(firstIterPostBlock);
        }

        private IList<IStatement> GetInitOfConvertedStatement()
        {
            if (initStmtsOfWhile.Count == whileDepth)
                return null;
            return initStmtsOfWhile[whileDepth];
        }

        private IList<IStatement> GetInitOfCurrentStatement()
        {
            return initStmtsOfWhile[whileDepth - 1];
        }
    }

    internal class BackEdgeAttribute : ICompilerAttribute
    {
        public readonly string Message;

        public BackEdgeAttribute(string message)
        {
            this.Message = message;
        }

        public override string ToString()
        {
            return $"BackEdgeAttribute({Message})";
        }
    }

    internal class LoopAnalysisTransform : ShallowCopyTransform
    {
        HashSet<IVariableDeclaration> loopVars;
        public Dictionary<IStatement, HashSet<IVariableDeclaration>> loopVarsOfStatement = new Dictionary<IStatement, HashSet<IVariableDeclaration>>(ReferenceEqualityComparer<IStatement>.Instance);

        protected override IStatement ConvertWhile(IWhileStatement iws)
        {
            context.SetPrimaryOutput(iws);
            foreach (var ist in iws.Body.Statements)
            {
                loopVars = new HashSet<IVariableDeclaration>();
                ConvertStatement(ist);
                loopVarsOfStatement.Add(ist, loopVars);
                loopVars = null;
            }
            return iws;
        }

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            if (loopVars != null)
            {
                IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
                loopVars.Add(loopVar);
            }
            return base.ConvertFor(ifs);
        }
    }
}