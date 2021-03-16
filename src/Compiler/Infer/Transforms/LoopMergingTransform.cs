// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using System.Linq;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Fuses consecutive 'for' loops over the same range, when it is safe to do so.
    /// </summary>
    internal class LoopMergingTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "LoopMergingTransform";
            }
        }

        public static bool debug;
        private bool hasDebugInfo;
        private bool isTopLevel;
        private LoopMergingInfo loopMergingInfo;
        private List<IStatement> openContainers = new List<IStatement>();
        private Dictionary<IStatement, Set<int>> stmtsOfContainer = new Dictionary<IStatement, Set<int>>(ReferenceEqualityComparer<IStatement>.Instance);

        protected override IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            loopMergingInfo = context.InputAttributes.Get<LoopMergingInfo>(imd);
            if (debug && loopMergingInfo != null && !hasDebugInfo)
            {
                var itdOut = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
                context.OutputAttributes.Add(itdOut, loopMergingInfo.GetDebugInfo(this));
                hasDebugInfo = true;
            }
            isTopLevel = true;
            return base.DoConvertMethod(md, imd);
        }

        /// <summary>
        /// Converts each statement, analyses their containers, then merges them
        /// </summary>
        /// <param name="outputs"></param>
        /// <param name="inputs"></param>
        protected override void ConvertStatements(IList<IStatement> outputs, IEnumerable<IStatement> inputs)
        {
            if (loopMergingInfo == null || !isTopLevel)
            {
                base.ConvertStatements(outputs, inputs);
                return;
            }
            OpenOutputBlock(outputs);
            List<IStatement> converted = new List<IStatement>();
            foreach (IStatement ist in inputs)
            {
                bool isConvergenceLoop = context.InputAttributes.Has<ConvergenceLoop>(ist);
                bool isFirstIterPost = context.InputAttributes.Has<FirstIterationPostProcessingBlock>(ist);
                if (!(isConvergenceLoop || isFirstIterPost))
                    isTopLevel = false;
                IStatement st = ConvertStatement(ist);
                if (st != null)
                    converted.Add(st);
                FinishConvertStatement();
                isTopLevel = true;
            }
            // a container is represented by the first statement that had the container
            openContainers.Clear();
            List<IStatement> newContainers = new List<IStatement>();
            List<int> newPriority = new List<int>();
            // determine the size of each container, ignoring nesting constraints
            Func<IStatement, int> getStmtIndex = delegate (IStatement ist)
             {
                 bool cannotMerge = context.InputAttributes.Has<ConvergenceLoop>(ist) ||
                    context.InputAttributes.Has<FirstIterationPostProcessingBlock>(ist) ||
                    (ist is ICommentStatement);
                 int stmtIndex = cannotMerge ? -1 : loopMergingInfo.GetIndexOf(ist);
                 return stmtIndex;
             };
            foreach (IStatement ist in converted)
            {
                int stmtIndex = getStmtIndex(ist);
                IList<IStatement> core = UnwrapStatement(ist, newContainers, newPriority, stmtIndex, keepNested: false);
                openContainers.AddRange(newContainers);
                foreach (IStatement container in openContainers)
                {
                    Set<int> stmts;
                    if (!stmtsOfContainer.TryGetValue(container, out stmts))
                    {
                        stmts = new Set<int>();
                        stmtsOfContainer[container] = stmts;
                    }
                    stmts.Add(stmtIndex);
                }
            }
            openContainers.Clear();
            stmtsOfContainer.Clear();
            List<IStatement> outputContainers = new List<IStatement>();
            // merge containers, obeying nesting constraints
            foreach (IStatement ist in converted)
            {
                int stmtIndex = getStmtIndex(ist);
                IList<IStatement> core = UnwrapStatement(ist, newContainers, newPriority, stmtIndex, keepNested: true);
                RemoveLast(outputContainers, openContainers.Count);
                openContainers.AddRange(newContainers);
                // create output statements for the new containers
                foreach (IStatement container in newContainers)
                {
                    IStatement outputContainer = Containers.CreateContainer(container);
                    context.InputAttributes.CopyObjectAttributesTo(container, context.OutputAttributes, outputContainer);
                    AddToLastContainer(outputs, outputContainers, outputContainer);
                    outputContainers.Add(outputContainer);
                }
                foreach (IStatement st in core)
                {
                    AddToLastContainer(outputs, outputContainers, st);
                }
                int i = 0;
                foreach (IStatement container in openContainers)
                {
                    Set<int> stmts;
                    if (!stmtsOfContainer.TryGetValue(container, out stmts))
                    {
                        stmts = new Set<int>();
                        stmtsOfContainer[container] = stmts;
                    }
                    if (!context.InputAttributes.Has<HasOffsetIndices>(outputContainers[i]) &&
                        HasOffsetToContainer(container, stmtIndex))
                        context.OutputAttributes.Set(outputContainers[i], new HasOffsetIndices()); // for ParallelForTransform
                    stmts.Add(stmtIndex);
                    i++;
                }
            }
            CloseOutputBlock();
        }

        private void AddToLastContainer(ICollection<IStatement> outputs, List<IStatement> containers, IStatement ist)
        {
            if (containers.Count == 0)
            {
                outputs.Add(ist);
            }
            else
            {
                IStatement container = containers[containers.Count - 1];
                Containers.AddToContainer(container, ist);
            }
        }

        private int GetPriority(IStatement container)
        {
            if (container is IForStatement)
            {
                IForStatement ifs = (IForStatement)container;
                IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
                return GetPriority(context, loopVar);
            }
            return 0;
        }

        internal static int GetPriority(BasicTransformContext context, IVariableDeclaration loopVar)
        {
            var lp = context.InputAttributes.Get<LoopPriority>(loopVar);
            if (lp != null)
                return lp.Priority;
            bool isSequential = context.InputAttributes.Has<Sequential>(loopVar);
            if (isSequential)
                return 1;
            bool isPartitioned = context.InputAttributes.Has<Partitioned>(loopVar);
            if (isPartitioned)
                return 1;
            return 0;
        }

        private int[] GetPriorities(IList<IStatement> containers)
        {
            int[] priority = new int[containers.Count];
            // loop containers starting at innermost
            for (int i = containers.Count - 1; i >= 0; i--)
            {
                int newPriority = GetPriority(containers[i]);
                if (priority[i] < newPriority)
                {
                    priority[i] = newPriority;
                    // find all parent containers and give them at least the same priority
                    IForStatement ifs = (IForStatement)containers[i];
                    // affectingVariables is the set of variables whose value affects the bounds of ifs.
                    // initialize affectingVariables to the variables in the loop bound expressions.
                    Set<IVariableDeclaration> affectingVariables = Set<IVariableDeclaration>.FromEnumerable(
                        Recognizer.GetVariables(Recognizer.LoopStartExpression(ifs)).Concat(
                        Recognizer.GetVariables(ifs.Condition)
                        ));
                    if (affectingVariables.Count > 0)
                    {
                        for (int j = i - 1; j >= 0; j--)
                        {
                            // if a container has a higher priority, then its parents must also have a higher priority, so we can skip without modifying affectingVariables.
                            if (priority[j] >= newPriority)
                                continue;
                            IStatement container = containers[j];
                            if (ContainerAffectsVariables(container, affectingVariables))
                                priority[j] = newPriority;
                        }
                    }
                }
            }
            // Set priorities to prevent re-ordering across broken loops.
            // loop containers starting at innermost
            int maxInnerPriority = 0;
            int offset = 0;
            for (int i = containers.Count - 1; i >= 0; i--)
            {
                priority[i] += offset;
                if(containers[i] is IBrokenForStatement)
                {
                    // priority must be higher than all inner loops.
                    priority[i] = System.Math.Max(priority[i], maxInnerPriority);
                    // outer loops must have higher priority.
                    offset = priority[i];
                }
                maxInnerPriority = System.Math.Max(maxInnerPriority, priority[i]);
            }
            return priority;
        }

        // Modifies affectingVariables.
        internal static bool ContainerAffectsVariables(IStatement container, Set<IVariableDeclaration> affectingVariables)
        {
            if (container is IForStatement)
            {
                IForStatement ifs2 = (IForStatement)container;
                if (affectingVariables.Contains(Recognizer.LoopVariable(ifs2)))
                {
                    // all variables in the loop bounds become affecting variables.
                    affectingVariables.AddRange(
                        Recognizer.GetVariables(Recognizer.LoopStartExpression(ifs2)).Concat(
                        Recognizer.GetVariables(ifs2.Condition)
                        ));
                    return true;
                }
                return false;
            }
            else if (container is IConditionStatement)
            {
                IConditionStatement ics = (IConditionStatement)container;
                // if the condition refers to an affecting variable, then it is a parent.
                bool isParent = Recognizer.GetVariables(ics.Condition).Any(affectingVariables.Contains);
                if (isParent)
                {
                    // all variables in the condition become affecting variables, 
                    // since they change the outcome of the condition and therefore the bounds of ifs.
                    affectingVariables.AddRange(Recognizer.GetVariables(ics.Condition));
                    return true;
                }
                return false;
            }
            else
                throw new NotImplementedException($"Unrecognized container type: {StringUtil.TypeToString(container.GetType())}");
        }

        private IList<IStatement> UnwrapStatement(IStatement ist, IList<IStatement> newContainers, IList<int> newPriority, int stmtIndex, bool keepNested)
        {
            List<IStatement> stmtContainers = new List<IStatement>();
            IList<IStatement> core = UnwrapStatement(ist, stmtContainers);
            int[] priority = GetPriorities(stmtContainers);
            int[] openPriority = Util.ArrayInit(openContainers.Count, i => -1);
            int highestNewPriority = 0;
            // determine the set of open containers which can be kept
            for (int j = 0; j < stmtContainers.Count; j++)
            {
                IStatement container = stmtContainers[j];
                if (container is IBrokenForStatement)
                    continue;
                bool found = false;
                for (int i = 0; i < openContainers.Count; i++)
                {
                    IStatement openContainer = openContainers[i];
                    if (Containers.ContainersAreEqual(openContainer, container))
                    {
                        found = true;
                        if (CanAddToContainer(openContainer, stmtIndex))
                            openPriority[i] = priority[j];
                        break;
                    }
                }
                if (!found && priority[j] > highestNewPriority)
                    highestNewPriority = priority[j];
            }
            // priority of containers:
            // 1. a sequential container that is already open, and its parent containers (whether they are sequential or not).
            // 2. a new sequential container and its parent containers.
            // 3. a container that is already open.
            if (keepNested)
            {
                // close all open containers whose priority is lower than highestNewPriority
                for (int i = openContainers.Count - 1; i >= 0; i--)
                {
                    if (openPriority[i] < highestNewPriority)
                        openPriority[i] = -1;
                    else
                        break;
                }
                // close all containers beyond the first closed one
                for (int i = 0; i < openContainers.Count; i++)
                {
                    if (openPriority[i] == -1)
                    {
                        RemoveLast(openContainers, i);
                        break;
                    }
                }
            }
            else
            {
                List<IStatement> keptContainers = new List<IStatement>();
                for (int i = 0; i < openContainers.Count; i++)
                {
                    if (openPriority[i] >= 0)
                        keptContainers.Add(openContainers[i]);
                }
                openContainers.Clear();
                openContainers.AddRange(keptContainers);
            }
            List<KeyValuePair<int, IStatement>> newContainersWithPriority = new List<KeyValuePair<int, IStatement>>();
            for (int j = 0; j < stmtContainers.Count; j++)
            {
                IStatement container = stmtContainers[j];
                bool found = false;
                for (int i = 0; i < openContainers.Count; i++)
                {
                    IStatement openContainer = openContainers[i];
                    if (Containers.ContainersAreEqual(openContainer, container))
                    {
                        if (openPriority[i] >= 0)
                            found = true;
                        break;
                    }
                }
                if (!found)
                {
                    newContainersWithPriority.Add(new KeyValuePair<int, IStatement>(priority[j], container));
                }
            }
            newContainers.Clear();
            newPriority.Clear();
            foreach (var kvp in newContainersWithPriority.OrderByDescending(kvp => kvp.Key))
            {
                newPriority.Add(kvp.Key);
                newContainers.Add(kvp.Value);
            }
            return core;
        }

        private bool CanAddToContainer(IStatement container, int stmtIndex)
        {
            if (container is IBrokenForStatement)
                return false;
            // convergence loops cannot be merged because they raise separate ProgressChanged events
            if (context.InputAttributes.Has<ConvergenceLoop>(container))
                return false;
            if (loopMergingInfo != null && container is IForStatement && stmtIndex >= 0)
            {
                IForStatement ifs = (IForStatement)container;
                bool isForwardLoop = Recognizer.IsForwardLoop(ifs);
                Set<int> stmts = stmtsOfContainer[container];
                IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
                int conflict = loopMergingInfo.GetConflictingStmt(stmts, stmtIndex, loopVar, isForwardLoop);
                return (conflict == -1);
            }
            return true;
        }

        private bool HasOffsetToContainer(IStatement container, int stmtIndex)
        {
            if (loopMergingInfo != null && container is IForStatement && stmtIndex >= 0)
            {
                IForStatement ifs = (IForStatement)container;
                Set<int> stmts = stmtsOfContainer[container];
                IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
                foreach (int edge in loopMergingInfo.graph.EdgesInto(stmtIndex))
                {
                    int source = loopMergingInfo.graph.SourceOf(edge);
                    IOffsetInfo offsetInfo = loopMergingInfo.offsetInfos[edge];
                    if ((source == stmtIndex || stmts.Contains(source)) && offsetInfo != null && offsetInfo.ContainsKey(loopVar))
                        return true;
                }
                foreach (int edge in loopMergingInfo.graph.EdgesOutOf(stmtIndex))
                {
                    int target = loopMergingInfo.graph.TargetOf(edge);
                    IOffsetInfo offsetInfo = loopMergingInfo.offsetInfos[edge];
                    if ((target == stmtIndex || stmts.Contains(target)) && offsetInfo != null && offsetInfo.ContainsKey(loopVar))
                        return true;
                }
            }
            return false;
        }

        internal static IList<IStatement> UnwrapStatement(IStatement ist, List<IStatement> containers)
        {
            if (ist is IForStatement)
            {
                containers.Add(ist);
                IForStatement ifs = (IForStatement)ist;
                if (ifs.Body.Statements.Count == 1)
                {
                    return UnwrapStatement(ifs.Body.Statements[0], containers);
                }
                else
                {
                    return ifs.Body.Statements;
                }
            }
            else if (ist is IConditionStatement)
            {
                containers.Add(ist);
                IConditionStatement ics = (IConditionStatement)ist;
                if (ics.Then.Statements.Count == 1)
                {
                    return UnwrapStatement(ics.Then.Statements[0], containers);
                }
                else
                {
                    return ics.Then.Statements;
                }
            }
            else
            {
                return new List<IStatement>() { ist };
            }
        }

        private void RemoveLast<T>(List<T> list, int newSize)
        {
            if (newSize > list.Count)
                throw new ArgumentException("newSize > list.Count");
            list.RemoveRange(newSize, list.Count - newSize);
        }
    }

    internal class LoopPriority : ICompilerAttribute
    {
        public int Priority;

        public override string ToString()
        {
            return "LoopPriority(" + Priority + ")";
        }
    }


#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}