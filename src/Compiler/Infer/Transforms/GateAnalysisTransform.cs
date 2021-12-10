// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Collections;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Attaches GateBlock attributes to ConditionStatements.  Does not make any changes to the code.
    /// A GateBlock attribute describes all external variables used or defined inside the block.  
    /// For arrays, we need to group uses of an array into disjoint sets of indices, so that each use is covered by one of the sets.  
    /// Each use is described by an indexing pattern and a set of bindings (assignments to condition variables).
    /// </summary>
    internal class GateAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "GateAnalysisTransform"; }
        }

        /// <summary>
        /// The list of bindings made by all conditional statements in the input stack
        /// </summary>
        private readonly List<ConditionBinding> conditionContext = new List<ConditionBinding>();

        /// <summary>
        /// The list of open gate blocks.  gateBlockContext.Count &lt;= conditionContext.Count.
        /// </summary>
        private readonly List<GateBlock> gateBlockContext = new List<GateBlock>();

        /// <summary>
        /// A dictionary mapping condition left-hand sides to GateBlocks.  Used to associate multiple conditional statements with the same GateBlock.
        /// </summary>
        private readonly Dictionary<Set<ConditionBinding>, Dictionary<IExpression, GateBlock>> gateBlocks =
            new Dictionary<Set<ConditionBinding>, Dictionary<IExpression, GateBlock>>();

        /// <summary>
        /// A dictionary storing the set of conditions in which a variable is declared.  Used for checking that a variable is defined in all conditions.
        /// </summary>
        private readonly Dictionary<IVariableDeclaration, ICollection<ConditionBinding>> declarationBindings =
            new Dictionary<IVariableDeclaration, ICollection<ConditionBinding>>();

        /// <summary>
        /// A dictionary storing the set of conditions in which a variable is defined.  Used for checking that a variable is defined in all conditions.
        /// </summary>
        private readonly Dictionary<IVariableDeclaration, Set<ICollection<ConditionBinding>>> definitionBindings =
            new Dictionary<IVariableDeclaration, Set<ICollection<ConditionBinding>>>();

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            context.SetPrimaryOutput(ics);
            ConvertExpression(ics.Condition);
            ConditionBinding binding = GateTransform.GetConditionBinding(ics.Condition, context, out IForStatement loop);
            IExpression caseValue = binding.rhs;
            if (!GateTransform.IsLiteralOrLoopVar(context, caseValue, out loop))
            {
                Error("If statement condition must compare to a literal or loop counter, was: " + ics.Condition);
                return ics;
            }
            bool isStochastic = CodeRecognizer.IsStochastic(context, binding.lhs);
            IExpression gateBlockKey;
            if (isStochastic)
                gateBlockKey = binding.lhs;
            else
                // definitions must not be unified across deterministic gate conditions
                gateBlockKey = binding.GetExpression();
            GateBlock gateBlock = null;
            Set<ConditionBinding> bindings = ConditionBinding.Copy(conditionContext);
            Dictionary<IExpression, GateBlock> blockMap;
            if (!gateBlocks.TryGetValue(bindings, out blockMap))
            {
                // first time seeing these bindings
                blockMap = new Dictionary<IExpression, GateBlock>();
                gateBlocks[bindings] = blockMap;
            }
            if (!blockMap.TryGetValue(gateBlockKey, out gateBlock))
            {
                // first time seeing this lhs
                gateBlock = new GateBlock();
                blockMap[gateBlockKey] = gateBlock;
            }
            if (gateBlock.hasLoopCaseValue && loop == null)
                Error("Cannot compare " + binding.lhs + " to a literal, since it was previously compared to a loop counter.  Put this test inside the loop.");
            if (!gateBlock.hasLoopCaseValue && gateBlock.caseValues.Count > 0 && loop != null)
                Error("Cannot compare " + binding.lhs + " to a loop counter, since it was previously compared to a literal.  Put the literal case inside the loop.");
            gateBlock.caseValues.Add(caseValue);
            if (loop != null)
                gateBlock.hasLoopCaseValue = true;
            gateBlockContext.Add(gateBlock);
            context.OutputAttributes.Set(ics, gateBlock);
            int startIndex = conditionContext.Count;
            conditionContext.Add(binding);
            ConvertBlock(ics.Then);
            if (ics.Else != null)
            {
                conditionContext.RemoveRange(startIndex, conditionContext.Count - startIndex);
                binding = binding.FlipCondition();
                conditionContext.Add(binding);
                ConvertBlock(ics.Else);
            }
            conditionContext.RemoveRange(startIndex, conditionContext.Count - startIndex);
            gateBlockContext.RemoveAt(gateBlockContext.Count - 1);
            // remove any uses that match a def
            //RemoveUsesOfDefs(gateBlock);
            if (gateBlockContext.Count > 0)
            {
                GateBlock currentBlock = gateBlockContext[gateBlockContext.Count - 1];
                // all variables defined/used in the inner block must be processed by the outer block
                foreach (ExpressionWithBindings eb in gateBlock.variablesDefined.Values)
                {
                    if (eb.Bindings.Count > 0)
                    {
                        foreach (List<ConditionBinding> binding2 in eb.Bindings)
                        {
                            ProcessUse(eb.Expression, true, Union(conditionContext, binding2));
                        }
                    }
                    else
                    {
                        ProcessUse(eb.Expression, true, conditionContext);
                    }
                }
                foreach (List<ExpressionWithBindings> ebs in gateBlock.variablesUsed.Values)
                {
                    foreach (ExpressionWithBindings eb in ebs)
                    {
                        if (eb.Bindings.Count > 0)
                        {
                            foreach (ICollection<ConditionBinding> binding2 in eb.Bindings)
                            {
                                ProcessUse(eb.Expression, false, Union(conditionContext, binding2));
                            }
                        }
                        else
                        {
                            ProcessUse(eb.Expression, false, conditionContext);
                        }
                    }
                }
            }
            return ics;
        }

        private static List<T> Union<T>(ICollection<T> list, ICollection<T> itemsToAdd)
        {
            List<T> result = new List<T>(list);
            foreach (T item in itemsToAdd)
            {
                if (!result.Contains(item))
                {
                    result.Add(item);
                }
            }
            return result;
        }

        protected override IVariableDeclaration ConvertVariableDecl(IVariableDeclaration ivd)
        {
            context.InputAttributes.Remove<GateBlock>(ivd);
            if (gateBlockContext.Count > 0)
            {
                GateBlock currentBlock = gateBlockContext[gateBlockContext.Count - 1];
                context.InputAttributes.Set(ivd, currentBlock);
            }
            RegisterDeclaration(ivd);
            return ivd;
        }

        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            ProcessExpression(ivre);
            return ivre;
        }

        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            // register the whole expression only
            // indices are not registered as separate expressions
            ProcessExpression(iaie);
            return iaie;
        }

        protected void ProcessExpression(IExpression expr)
        {
            bool isDef = Recognizer.IsBeingMutated(context, expr);
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
            if (ivd == null) return;
            if (!CodeRecognizer.IsStochastic(context, ivd)) return;
            if (isDef && !Recognizer.IsBeingAllocated(context, expr))
            {
                RegisterDefinition(ivd);
            }
            ProcessUse(expr, isDef, conditionContext);
        }

        /// <summary>
        /// Add expr to currentBlock.variablesDefined or currentBlock.variablesUsed
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="isDef"></param>
        /// <param name="conditionContext"></param>
        protected void ProcessUse(IExpression expr, bool isDef, List<ConditionBinding> conditionContext)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
            if (ivd == null) return;
            if (!CodeRecognizer.IsStochastic(context, ivd)) return;
            if (gateBlockContext.Count == 0) return;
            GateBlock currentBlock = gateBlockContext[gateBlockContext.Count - 1];
            GateBlock gateBlockOfVar = context.InputAttributes.Get<GateBlock>(ivd);
            if (gateBlockOfVar == currentBlock) return; // local variable of the gateBlock

            ExpressionWithBindings eb = new ExpressionWithBindings();
            eb.Expression = ReplaceLocalIndices(currentBlock, expr);
            List<ConditionBinding> bindings = FilterConditionContext(currentBlock, conditionContext);
            if (bindings.Count > 0) eb.Bindings.Add(bindings);
            //eb.Containers = Containers.InsideOf(context, GetAncestorIndexOfGateBlock(currentBlock));
            if (isDef)
            {
                ExpressionWithBindings eb2;
                if (!currentBlock.variablesDefined.TryGetValue(ivd, out eb2))
                {
                    currentBlock.variablesDefined[ivd] = eb;
                }
                else
                {
                    // all definitions of the same variable must have a common parent
                    currentBlock.variablesDefined[ivd] = GetCommonParent(eb, eb2);
                }
            }
            else
            {
                List<ExpressionWithBindings> ebs;
                if (!currentBlock.variablesUsed.TryGetValue(ivd, out ebs))
                {
                    ebs = new List<ExpressionWithBindings>();
                    ebs.Add(eb);
                    currentBlock.variablesUsed[ivd] = ebs;
                }
                else
                {
                    // collect all uses that overlap with eb, and replace with their common parent
                    List<ExpressionWithBindings> notOverlapping = new List<ExpressionWithBindings>();
                    while (true)
                    {
                        foreach (ExpressionWithBindings eb2 in ebs)
                        {
                            ExpressionWithBindings parent = GetCommonParent(eb, eb2);
                            if (CouldOverlap(eb, eb2)) eb = parent;
                            else notOverlapping.Add(eb2);
                        }
                        if (notOverlapping.Count == ebs.Count) break; // nothing overlaps
                        // eb must have changed, so try again using the new eb
                        ebs.Clear();
                        ebs.AddRange(notOverlapping);
                        notOverlapping.Clear();
                    }
                    ebs.Add(eb);
                    currentBlock.variablesUsed[ivd] = ebs;
                }
            }
        }

        protected void RegisterDeclaration(IVariableDeclaration ivd)
        {
            declarationBindings[ivd] = ConditionBinding.Copy(conditionContext);
        }

        protected void RegisterDefinition(IVariableDeclaration ivd)
        {
            Set<ICollection<ConditionBinding>> defBindings;
            if (!definitionBindings.TryGetValue(ivd, out defBindings))
            {
                defBindings = new Set<ICollection<ConditionBinding>>();
                definitionBindings[ivd] = defBindings;
            }
            else if (defBindings.Count == 0) return;
            if (conditionContext.Count > 0)
            {
                Set<ConditionBinding> bindings = ConditionBinding.Copy(conditionContext);
                bindings.Remove(declarationBindings[ivd]);
                if (bindings.Count > 0) defBindings.Add(bindings);
            }
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            base.DoConvertMethodBody(outputs, inputs);
            // check for complete definitions
            foreach (var entry in definitionBindings)
            {
                BindingSet reducedBindings = BindingSet.FromBindings(entry.Value).Simplify();
                if (reducedBindings.set.Count > 0)
                {
                    Error("'" + entry.Key.Name + "' is not defined in all cases.  It is only defined for " + reducedBindings);
                }
            }
        }

        /// <summary>
        /// Stores a logical formula in disjunctive normal form.
        /// </summary>
        private class BindingSet
        {
            public Set<Dictionary<IExpression, IExpression>> set = new Set<Dictionary<IExpression, IExpression>>(new DictionaryComparer<IExpression, IExpression>());

            internal static BindingSet FromBindings(IEnumerable<ICollection<ConditionBinding>> bindingSet)
            {
                BindingSet result = new BindingSet();
                foreach (ICollection<ConditionBinding> bindings in bindingSet)
                {
                    // only boolean bindings are returned since we know there must be two cases.
                    // integer bindings could have only one case.
                    var dict = ConditionBinding.ToDictionary(bindings, true);
                    if (dict.Count > 0) result.set.Add(dict);
                }
                return result;
            }

            /// <summary>
            /// Simplify a logical formula.
            /// </summary>
            /// <returns>An empty formula if the target is a tautology.</returns>
            public BindingSet Simplify()
            {
                // The algorithm to simplify a BindingSet works as follows:
                // Collect all condition variables and put them in some order.
                // Starting with the first condition variable (b), partition all bindings into 3 groups:
                // 1. bindings which contain (b)
                // 2. bindings which contain (!b)
                // 3. bindings which do not mention b
                // Recursively simplify the bindings in each group.
                // If groups 1 and 2 are equal except for b, remove b from the groups and merge them.
                List<IExpression> allLhs = new List<IExpression>();
                foreach (Dictionary<IExpression, IExpression> bindings in set)
                {
                    foreach (IExpression lhs in bindings.Keys)
                    {
                        if (!allLhs.Contains(lhs)) allLhs.Add(lhs);
                    }
                }
                return Simplify(allLhs, 0);
            }

            private BindingSet Simplify(IList<IExpression> allLhs, int lhsIndex)
            {
                if (set.Count == 0) return this;
                if (lhsIndex >= allLhs.Count) return this;
                IExpression lhs = allLhs[lhsIndex];
                var groups = GroupBy(lhs);
                bool allMatch = true;
                BindingSet bindingSet1 = null;
                BindingSet newBindingSet = new BindingSet();
                BindingSet missing = null;
                int count = 0;
                foreach (var entry in groups)
                {
                    var bindingSet2 = entry.Value.Simplify(allLhs, lhsIndex + 1);
                    if (IsMissingExpr(entry.Key))
                    {
                        if (bindingSet2.set.Count > 0) missing = bindingSet2;
                    }
                    else
                    {
                        count++;
                        if (bindingSet1 == null) bindingSet1 = bindingSet2;
                        else if (!BindingSet.AreEqual(bindingSet1, bindingSet2, allLhs, lhsIndex + 1)) allMatch = false;
                    }
                    newBindingSet.set.AddRange(bindingSet2.set);
                }
                if (count == 2 && allMatch)
                {
                    newBindingSet.set.Clear();
                    foreach (Dictionary<IExpression, IExpression> dict in bindingSet1.set)
                    {
                        var dict2 = Remove(dict, lhs);
                        if (dict2.Count > 0) newBindingSet.set.Add(dict2);
                    }
                    if (missing != null) newBindingSet.set.AddRange(missing.set);
                }
                return newBindingSet;
            }

            public BindingSet Clone()
            {
                BindingSet result = new BindingSet();
                foreach (Dictionary<IExpression, IExpression> bindings in set)
                {
                    result.set.Add(Clone(bindings));
                }
                return result;
            }

            public static Dictionary<TKey, TValue> Clone<TKey, TValue>(Dictionary<TKey, TValue> dict)
            {
                Dictionary<TKey, TValue> result = new Dictionary<TKey, TValue>();
                result.AddRange(dict);
                return result;
            }

            public static Dictionary<TKey, TValue> Remove<TKey, TValue>(Dictionary<TKey, TValue> dict, TKey key)
            {
                Dictionary<TKey, TValue> result = Clone(dict);
                result.Remove(key);
                return result;
            }

            public IExpression MissingExpr()
            {
                IAddressOutExpression iaoe = Builder.AddrOutExpr();
                iaoe.Expression = Builder.LiteralExpr(0);
                return iaoe;
            }

            public bool IsMissingExpr(IExpression expr)
            {
                return (expr is IAddressOutExpression);
            }

            public Dictionary<IExpression, BindingSet> GroupBy(IExpression lhs)
            {
                Dictionary<IExpression, BindingSet> result = new Dictionary<IExpression, BindingSet>();
                foreach (Dictionary<IExpression, IExpression> bindings in set)
                {
                    IExpression rhs;
                    if (!bindings.TryGetValue(lhs, out rhs)) rhs = MissingExpr();
                    BindingSet group;
                    if (!result.TryGetValue(rhs, out group))
                    {
                        group = new BindingSet();
                        result[rhs] = group;
                    }
                    group.set.Add(bindings);
                }
                return result;
            }

            public static bool AreEqual(BindingSet bindingSet1, BindingSet bindingSet2, IList<IExpression> allLhs, int lhsIndex)
            {
                if (bindingSet1.set.Count != bindingSet2.set.Count) return false;
                if (bindingSet1.set.Count > 1) return false;
                foreach (Dictionary<IExpression, IExpression> dict1 in bindingSet1.set)
                {
                    foreach (Dictionary<IExpression, IExpression> dict2 in bindingSet2.set)
                    {
                        return AreEqual(dict1, dict2, allLhs, lhsIndex);
                    }
                }
                return true;
            }

            public static bool AreEqual(Dictionary<IExpression, IExpression> dict1, Dictionary<IExpression, IExpression> dict2, IList<IExpression> allLhs, int lhsIndex)
            {
                for (int i = lhsIndex; i < allLhs.Count; i++)
                {
                    IExpression lhs = allLhs[i];
                    IExpression rhs1, rhs2;
                    bool found1 = dict1.TryGetValue(lhs, out rhs1);
                    bool found2 = dict2.TryGetValue(lhs, out rhs2);
                    if (found1 != found2) return false;
                    else if (found1 && !rhs1.Equals(rhs2)) return false;
                }
                return true;
            }

            public override string ToString()
            {
                return ToString(set);
            }

            public static string ToString<TKey, TValue>(IEnumerable<Dictionary<TKey, TValue>> items)
            {
                StringBuilder sb = new StringBuilder();
                foreach (Dictionary<TKey, TValue> item in items)
                {
                    sb.Append("(");
                    AppendString(sb, item);
                    sb.Append(")");
                }
                return sb.ToString();
            }

            public static void AppendString<TKey, TValue>(StringBuilder sb, Dictionary<TKey, TValue> dict)
            {
                bool first = true;
                foreach (var entry in dict)
                {
                    if (!first) sb.Append(',');
                    sb.Append(entry.Key);
                    sb.Append('=');
                    sb.Append(entry.Value);
                    first = false;
                }
            }
        }

        private int GetAncestorIndexOfGateBlock(GateBlock gateBlock)
        {
            for (int i = 0; i < context.InputStack.Count; i++)
            {
                if (context.InputStack[i].inputElement is IStatement st && 
                    context.InputAttributes.Get<GateBlock>(st) == gateBlock) return i;
            }
            return -1;
        }

        private List<ConditionBinding> FilterConditionContext(GateBlock gateBlock, List<ConditionBinding> conditionContext)
        {
            return conditionContext
                .Where(binding => !CodeRecognizer.IsStochastic(context, binding.lhs) &&
                                  !ContainsLocalVars(gateBlock, binding.lhs) && 
                                  !ContainsLocalVars(gateBlock, binding.rhs))
                .ToList();
        }

        /// <summary>
        /// Use wildcards to replace any indices in expr which involve local variables of the gateBlock
        /// </summary>
        /// <param name="gateBlock"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
        private IExpression ReplaceLocalIndices(GateBlock gateBlock, IExpression expr)
        {
            if (expr is IArrayIndexerExpression iaie)
            {
                IExpression target = ReplaceLocalIndices(gateBlock, iaie.Target);
                List<IExpression> indices = new List<IExpression>();
                bool replaced = !ReferenceEquals(target, iaie.Target);
                foreach (IExpression index in iaie.Indices)
                {
                    if (ContainsLocalVars(gateBlock, index))
                    {
                        indices.Add(Builder.StaticMethod(new Func<int>(GateAnalysisTransform.AnyIndex)));
                        replaced = true;
                    }
                    else indices.Add(index);
                }
                if (replaced) expr = Builder.ArrayIndex(target, indices);
            }
            return expr;
        }

        private bool ContainsLocalVars(GateBlock gateBlock, IExpression expr)
        {
            return Recognizer.GetVariables(expr).Any(ivd => 
                (context.InputAttributes.Get<GateBlock>(ivd) == gateBlock)
            );
        }

        private void RemoveUsesOfDefs(GateBlock gateBlock)
        {
            foreach (KeyValuePair<IVariableDeclaration, ExpressionWithBindings> entry in gateBlock.variablesDefined)
            {
                IVariableDeclaration ivd = entry.Key;
                ExpressionWithBindings eb = entry.Value;
                if (gateBlock.variablesUsed.TryGetValue(ivd, out List<ExpressionWithBindings> ebs))
                {
                    List<ExpressionWithBindings> keep = new List<ExpressionWithBindings>();
                    foreach (ExpressionWithBindings eb2 in ebs)
                    {
                        if (!CouldOverlap(eb, eb2)) keep.Add(eb2);
                    }
                    gateBlock.variablesUsed[ivd] = keep;
                }
            }
        }

        /// <summary>
        /// Apply all bindings to expr, in order.
        /// </summary>
        /// <param name="bindings"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
        internal static IExpression ReplaceExpression(IEnumerable<ConditionBinding> bindings, IExpression expr)
        {
            foreach (ConditionBinding binding in bindings)
            {
                int replaceCount = 0;
                expr = Builder.ReplaceExpression(expr, binding.lhs, binding.rhs, ref replaceCount);
            }
            return expr;
        }

        /// <summary>
        /// Used only in ArrayIndexerExpressions, to represent a wildcard.
        /// </summary>
        /// <returns></returns>
        public static int AnyIndex()
        {
            throw new Exception("This function should never be called");
        }

        internal static bool CouldOverlap(ExpressionWithBindings eb1, ExpressionWithBindings eb2)
        {
            var emptyBindings = (IEnumerable<IReadOnlyCollection<ConditionBinding>>)new[] { new ConditionBinding[0] };
            IEnumerable<IReadOnlyCollection<ConditionBinding>> bindings1 = (eb1.Bindings.Count > 0) ?
                eb1.Bindings : emptyBindings;
            IEnumerable<IReadOnlyCollection<ConditionBinding>> bindings2 = (eb2.Bindings.Count > 0) ?
                eb2.Bindings : emptyBindings;
            foreach (IReadOnlyCollection<ConditionBinding> binding1 in bindings1)
            {
                foreach (IReadOnlyCollection<ConditionBinding> binding2 in bindings2)
                {
                    // TODO: investigate whether we need to provide local var predicate.
                    if (Recognizer.MutatingFirstAffectsSecond(eb1.Expression, eb2.Expression, binding1, binding2, ivd => false))
                        return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Return the common parent of the two expressions, and indicate if the expressions overlap
        /// </summary>
        /// <param name="eb1"></param>
        /// <param name="eb2"></param>
        /// <returns></returns>
        /// <remarks>
        /// When finding the common parent, we want to preserve any array slicing.  Indexing brackets are not removed if present in both expressions.
        /// Mismatching indices are replaced by AnyIndex.
        /// Examples:
        /// i = loop index inside the gate block
        /// j = loop index outside the gate block, or a method parameter
        /// (a,a[i]) => (a,true)
        /// (a,a[j]) => (a,true)
        /// (a[i],a[0]) => (a[*],true)
        /// (a[i],a[j]) => (a[*],true)
        /// (a[j],a[0]) => (a[*],assumeMatch)
        /// (a[0],a[1]) => (a[*],false)
        /// (a[0][0],a[1][0]) => (a[*][0],false)
        /// (a[0][0],a[1][1]) => (a[*][*],false)
        /// </remarks>
        private static ExpressionWithBindings GetCommonParent(ExpressionWithBindings eb1, ExpressionWithBindings eb2)
        {
            List<IExpression> prefixes1 = Recognizer.GetAllPrefixes(eb1.Expression);
            List<IExpression> prefixes2 = Recognizer.GetAllPrefixes(eb2.Expression);
            if (!prefixes1[0].Equals(prefixes2[0])) throw new Exception("Expressions have no overlap: " + eb1 + "," + eb2);
            IExpression parent = prefixes1[0];
            int count = System.Math.Min(prefixes1.Count, prefixes2.Count);
            for (int i = 1; i < count; i++)
            {
                IExpression prefix1 = prefixes1[i];
                IExpression prefix2 = prefixes2[i];
                if (prefix1 is IArrayIndexerExpression iaie1)
                {
                    if (prefix2 is IArrayIndexerExpression iaie2)
                    {
                        if (iaie1.Indices.Count != iaie2.Indices.Count) throw new Exception("Array rank mismatch: " + eb1 + "," + eb2);
                        IList<IExpression> indices = Builder.ExprCollection();
                        for (int ind = 0; ind < iaie1.Indices.Count; ind++)
                        {
                            IExpression index1 = iaie1.Indices[ind];
                            IExpression index2 = iaie2.Indices[ind];
                            IExpression index = Unify(index1, eb1.Bindings, index2, eb2.Bindings);
                            indices.Add(index);
                        }
                        parent = Builder.ArrayIndex(parent, indices);
                    }
                    else
                    {
                        break;
                    }
                }
                else throw new Exception("Unhandled expression type: " + prefix1);
            }
            ExpressionWithBindings result = new ExpressionWithBindings();
            result.Expression = parent;
            if (eb1.Bindings.Count > 0 && eb2.Bindings.Count > 0)
            {
                result.Bindings.AddRange(eb1.Bindings);
                result.Bindings.AddRange(eb2.Bindings);
            }
            return result;

            // Returns an expression equal to expr1 and expr2 under their respective bindings, or null if the expressions are not equal.  
            IExpression Unify(
                IExpression expr1,
                IEnumerable<IReadOnlyCollection<ConditionBinding>> bindings1,
                IExpression expr2,
                IEnumerable<IReadOnlyCollection<ConditionBinding>> bindings2)
            {
                if (expr1.Equals(expr2))
                {
                    return expr1;
                }
                foreach (IReadOnlyCollection<ConditionBinding> binding2 in bindings2)
                {
                    IExpression expr1b = ReplaceExpression(binding2, expr1);
                    if (expr1b.Equals(expr2))
                    {
                        return expr1;
                    }
                }
                foreach (IReadOnlyCollection<ConditionBinding> binding1 in bindings1)
                {
                    IExpression expr2b = ReplaceExpression(binding1, expr2);
                    if (expr2b.Equals(expr1))
                    {
                        return expr2;
                    }
                }
                bool lift = false;
                if (lift)
                {
                    IExpression lifted1 = GetLiftedExpression(expr1, bindings1);
                    IExpression lifted2 = GetLiftedExpression(expr2, bindings2);
                    if (lifted1 != null && lifted1.Equals(lifted2)) return lifted1;
                }
                return Builder.StaticMethod(new Func<int>(GateAnalysisTransform.AnyIndex));

                IExpression GetLiftedExpression(IExpression expr, IEnumerable<IReadOnlyCollection<ConditionBinding>> bindings)
                {
                    IExpression lifted = null;
                    foreach (IReadOnlyCollection<ConditionBinding> binding in bindings)
                    {
                        IExpression lhs = null;
                        foreach (ConditionBinding b in binding)
                        {
                            if (b.rhs.Equals(expr)) lhs = b.lhs;
                        }
                        if (lifted == null)
                        {
                            lifted = lhs;
                        }
                        else if (lhs == null || !lifted.Equals(lhs))
                        {
                            return null;
                        }
                    }
                    return lifted;
                }
            }
        }
    }

    /// <summary>
    /// Describes all external variables used or defined inside the block.  
    /// </summary>
    internal class GateBlock : ICompilerAttribute
    {
        /// <summary>
        /// Variables defined inside the block, along with their (unique) indexing pattern.
        /// </summary>
        internal Dictionary<IVariableDeclaration, ExpressionWithBindings> variablesDefined = new Dictionary<IVariableDeclaration, ExpressionWithBindings>();

        /// <summary>
        /// Variables used inside the block, paired with a set of disjoint indexing patterns.  Does not keep track of which branch the variable was used in.
        /// </summary>
        internal Dictionary<IVariableDeclaration, List<ExpressionWithBindings>> variablesUsed = new Dictionary<IVariableDeclaration, List<ExpressionWithBindings>>();

        /// <summary>
        /// Current condition context inside the block (deterministic conditions only)
        /// </summary>
        internal List<ConditionBinding> conditionContext = new List<ConditionBinding>();

        /// <summary>
        /// The set of all case values encountered
        /// </summary>
        internal Set<IExpression> caseValues = new Set<IExpression>();

        /// <summary>
        /// True if one of the case values is a loop counter
        /// </summary>
        internal bool hasLoopCaseValue;

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder("GateBlock");
            sb.AppendLine(StringUtil.JoinColumns("  defs = ", StringUtil.ToString(variablesDefined)));
            sb.AppendLine(StringUtil.JoinColumns("  uses = ", StringUtil.ToString(variablesUsed)));
            return sb.ToString();
        }
    }

    /// <summary>
    /// Stores a unified expression along with the set of conditions where the expression appears.
    /// </summary>
    internal class ExpressionWithBindings
    {
        public IExpression Expression;

        /// <summary>
        /// A set of condition contexts.  An empty set means the expression has an empty condition context.
        /// </summary>
        public Set<IReadOnlyCollection<ConditionBinding>> Bindings = new Set<IReadOnlyCollection<ConditionBinding>>(new EnumerableComparer<ConditionBinding>());

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(Expression);
            foreach (IReadOnlyCollection<ConditionBinding> binding in Bindings)
            {
                if (binding.Count > 0)
                {
                    sb.Append(" (");
                    sb.Append(StringUtil.CollectionToString(binding, ","));
                    sb.Append(")");
                }
            }
            return sb.ToString();
        }
    }
}