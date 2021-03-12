// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Makes copies of duplicate statements, ensuring that all statements are unique objects (and therefore can have unique attributes).
    /// </summary>
    internal class UniquenessTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "UniquenessTransform";
            }
        }

        bool isInnerStatement;
        private LoopMergingInfo loopMergingInfo;
        Set<IStatement> previousStatements = new Set<IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
        Dictionary<IStatement, IStatement> originalStatementOfClone = new Dictionary<IStatement, IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
        Dictionary<IStatement, IEnumerable<IStatement>> clonesOfStatement = new Dictionary<IStatement, IEnumerable<IStatement>>(ReferenceEqualityComparer<IStatement>.Instance);
        Dictionary<IStatement, IStatement> replacements = new Dictionary<IStatement, IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
        int whileCount;

        protected override IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            if (!context.InputAttributes.Has<OperatorMethod>(imd))
                return imd;
            loopMergingInfo = context.InputAttributes.Get<LoopMergingInfo>(imd);
            return base.DoConvertMethod(md, imd);
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            base.DoConvertMethodBody(outputs, inputs);
            PostProcessDependencies(outputs);
        }

        /// <summary>
        /// Update the dependency information of all statements, assuming that the original statements have been permuted and duplicated, but not transformed.
        /// </summary>
        /// <param name="outputs"></param>
        private void PostProcessDependencies(ICollection<IStatement> outputs)
        {
            Dictionary<IStatement, int> indexOfStatement = new Dictionary<IStatement, NodeIndex>(ReferenceEqualityComparer<IStatement>.Instance);
            if (clonesOfStatement.Count == 0) return;
            DeadCodeTransform.ForEachStatement(outputs,
                _ => { },
                _ => { },
                _ => { },
                _ => { },
                ist =>
                {
                    DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                    if (di != null)
                    {
                        // must make a clone since this statement may appear in multiple contexts
                        DependencyInformation di2 = (DependencyInformation)di.Clone();
                        di2.AddClones(clonesOfStatement);
                        IStatement originalStatement;
                        originalStatementOfClone.TryGetValue(ist, out originalStatement);
                        // originalStatement is non-null iff ist is a clone
                        IEnumerable<IStatement> clones;
                        if (clonesOfStatement.TryGetValue(originalStatement ?? ist, out clones))
                        {
                            // clones become Overwrites
                            foreach (var clone in clones)
                            {
                                di2.Add(DependencyType.Overwrite, clone);
                            }
                            if (originalStatement != null)
                                di2.Add(DependencyType.Overwrite, originalStatement);
                        }
                        // keep only the most recent overwrite that is not an allocation, and all overwrites that are allocations.
                        IStatement mostRecentWriter = null;
                        int mostRecentWriterIndex = 0;
                        List<IStatement> allocations = new List<IStatement>();
                        foreach (var writer in di2.Overwrites)
                        {
                            int index;
                            indexOfStatement.TryGetValue(writer, out index);
                            if (index > mostRecentWriterIndex)
                            {
                                mostRecentWriterIndex = index;
                                mostRecentWriter = writer;
                            }
                            if (di2.HasDependency(DependencyType.Declaration, writer))
                                allocations.Add(writer);
                        }
                        di2.Remove(DependencyType.Overwrite);
                        // all allocations must remain as Overwrites
                        foreach (var dep in allocations)
                            di2.Add(DependencyType.Overwrite, dep);
                        if (mostRecentWriter != null)
                            di2.Add(DependencyType.Overwrite, mostRecentWriter);
                        context.OutputAttributes.Remove<DependencyInformation>(ist);
                        context.OutputAttributes.Set(ist, di2);
                    }
                    indexOfStatement[ist] = indexOfStatement.Count + 1;
                });
        }

        protected override IStatement DoConvertStatement(IStatement ist)
        {
            if (ist is IWhileStatement)
            {
                IWhileStatement iws = (IWhileStatement)ist;
                if (whileCount == 0)
                {
                    replacements.Clear();
                }
                whileCount++;
                IStatement convertedSt = ConvertWhile(iws);
                whileCount--;
                if (replacements.Count > 0)
                {
                    InitializerSet initializerSet = context.InputAttributes.Get<InitializerSet>(iws);
                    if (initializerSet != null)
                    {
                        initializerSet.Replace(replacements);
                    }
                }
                return convertedSt;
            }
            else if (isInnerStatement) return ist;
            else if (context.InputAttributes.Has<FirstIterationPostProcessingBlock>(ist))
            {
                return base.DoConvertStatement(ist);
            }
            else if (!previousStatements.Contains(ist))
            {
                previousStatements.Add(ist);
                return ist;
            }
            else
            {
                this.ShallowCopy = true;
                isInnerStatement = true;
                IStatement convertedSt = base.DoConvertStatement(ist);
                isInnerStatement = false;
                this.ShallowCopy = false;
                AddClone(ist, convertedSt);
                loopMergingInfo.AddEquivalentStatement(convertedSt, loopMergingInfo.GetIndexOf(ist));
                return convertedSt;
            }
        }

        private void AddClone(IStatement ist, IStatement clone)
        {
            IEnumerable<IStatement> clones;
            if (!clonesOfStatement.TryGetValue(ist, out clones))
            {
                clones = new List<IStatement>();
                clonesOfStatement.Add(ist, clones);
            }
            ((ICollection<IStatement>)clones).Add(clone);
            replacements[ist] = clone;
            originalStatementOfClone[clone] = ist;
        }
    }
}
