// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Graphs;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Attaches DependencyInformation attributes to all statements in the input program.
    /// </summary>
    /// <remarks>
    /// Dependency analysis operates in two passes:
    /// 
    /// 1. Each statement is analysed to determine if it declares or mutates a variable or array element.  All
    /// such declaring/mutating statements are cached against the variable or array declaration element. The mutation
    /// can be quite complex for jagged arrays. Any previous statement which mutates the same variable as the current
    /// statement is considered as and marked as an 'allocation' (for example there may be many allocation statements
    /// defining a jagged array). These previous statements are added as dependencies of the current statement.
    /// 
    /// Dependencies are marked using a DependencyInformation attribute stored against each statement.  Dependencies
    /// are represented as either statements or as reference expressions.
    /// 
    /// 2. Using the statement cache prepared above, any expression dependencies are converted into statement
    /// dependencies by finding all statements which mutate the expression and are not marked as 'allocations'.
    /// Declaration dependencies are added on statements marked as allocations.
    /// After this process, all DependencyInformation attributes contain only dependencies on statements.
    /// </remarks>
    /// <summary>
    /// Analyses expressions to determine their dependencies.
    /// </summary>
    internal class DependencyAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "DependencyAnalysisTransform";
            }
        }

        /// <summary>
        /// Special offset to indicate a sequential increment dependency
        /// </summary>
        public const int sequentialOffset = int.MinValue;

        /// <summary>
        /// The dependency information for the statement being transformed.
        /// </summary>
        private DependencyInformation dependencyInformation;

        /// <summary>
        /// The dependency type for the expression being transformed.
        /// </summary>
        private DependencyType dependencyType;

        private Set<IExpression> mutatedExpressions = new Set<IExpression>();

        /// <summary>
        /// The set of parameter declarations for the top-level method being transformed.
        /// </summary>
        private Set<IParameterDeclaration> topLevelParameters = new Set<IParameterDeclaration>();

        private LoopMergingInfo loopMergingInfo;

        /// <summary>
        /// Records all times that a variable is declared or assigned to
        /// </summary>
        private Dictionary<IVariableDeclaration, MutationInformation> mutInfos =
            new Dictionary<IVariableDeclaration, MutationInformation>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);

        private bool convertingLoopInitializer;

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            mutInfos.Clear();
            OpenOutputBlock(outputs);
            foreach (IStatement ist in inputs)
            {
                // Create a dependency information instance for this top level statement
                dependencyInformation = context.InputAttributes.GetOrCreate<DependencyInformation>(ist, () => new DependencyInformation());
                // this is the default dependency type when the statement has no MethodInvoke, e.g. y = x + 1
                dependencyType = DependencyType.Dependency | DependencyType.SkipIfUniform;
                // ConvertStatement will populate dependencyInformation and mutatedExpressions
                mutatedExpressions.Clear();
                ConvertStatement(ist);
                outputs.Add(ist);
                Dictionary<IVariableDeclaration, CodeRecognizer.Bounds> bounds = null;
                if (mutatedExpressions.Count > 0)
                {
                    bounds = new Dictionary<IVariableDeclaration, CodeRecognizer.Bounds>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
                    Recognizer.AddLoopBounds(bounds, ist);
                }
                foreach (IExpression mutated in mutatedExpressions)
                {
                    RegisterMutation(ist, mutated, ist, bounds);
                }
            }
            CloseOutputBlock();
            if (context.Results.IsErrors())
                return;
            // Convert variable reference dependencies into statement dependencies.
            PostProcessDependencies(outputs);
        }

        /// <summary>
        /// Find expressions that have overlapping mutations, and replace these mutations with a dummy statement that depends on all of them.
        /// </summary>
        /// <param name="newStatement">Invoked on each new statement</param>
        /// <remarks>
        /// Statements with non-overlapping conditionals are not considered overlapping, 
        /// so these two statements will not create a dummy statement:
        /// <code>
        ///   if(b) x = 0;
        ///   if(!b) x = 1;
        /// </code>
        /// Examples of overlapping mutations:
        /// ArrayConstraintsTest
        /// ConstrainBetweenTest
        /// </remarks>
        private void CreateDummyStatements(Action<IStatement> newStatement)
        {
            foreach (MutationInformation mutInfo in mutInfos.Values)
            {
                // find overlapping mutations
                Dictionary<int, List<int>> overlappingMutations = new Dictionary<int, List<int>>();
                HashSet<int> candidates = new HashSet<int>();
                for (int i = 0; i < mutInfo.mutations.Count; i++)
                {
                    candidates.Add(i);
                    List<int> overlap = new List<int>();
                    MutationInformation.Mutation mut = mutInfo.mutations[i];
                    bool isIncrement = context.InputAttributes.Has<IncrementStatement>(mut.stmt);
                    IExpression expr = mut.expr;
                    for (int j = 0; j < mutInfo.mutations.Count; j++)
                    {
                        if (j == i)
                            continue;
                        MutationInformation.Mutation mut2 = mutInfo.mutations[j];
                        if (mut2.isAllocation != mut.isAllocation)
                            continue;
                        if (mut2.isAllocation)
                            continue;
                        // only statements with the same increment status are considered overlapping
                        if (isIncrement != context.InputAttributes.Has<IncrementStatement>(mut2.stmt))
                            continue;
                        if (isIncrement)
                            continue;
                        IExpression expr2 = mut2.expr;
                        if (Recognizer.MutatingFirstAffectsSecond(context, expr, expr2, false, mut.stmt, mut2.stmt))
                        {
                            overlap.Add(j);
                        }
                    }
                    overlappingMutations[i] = overlap;
                }
                List<MutationInformation.Mutation> newMutations = new List<MutationInformation.Mutation>();
                var cliqueFinder = new CliqueFinder<int>(node => overlappingMutations[node]);
                cliqueFinder.ForEachClique(candidates, c =>
                    CreateDummyStatement(GetMutationInfos(mutInfo, c), newStatement, newMutations.Add));
                mutInfo.mutations = newMutations;
            }
        }

        /// <summary>
        /// Create a new statement that represents all of the given mutations
        /// </summary>
        /// <param name="mutations"></param>
        /// <param name="newStatement"></param>
        /// <param name="newMutation"></param>
        private void CreateDummyStatement(ICollection<MutationInformation.Mutation> mutations, Action<IStatement> newStatement, Action<MutationInformation.Mutation> newMutation)
        {
            if (mutations.Count == 1)
            {
                foreach (var mut2 in mutations)
                {
                    newMutation(mut2);
                }
                return;
            }
            bool allAllocations = true;
            bool anyAllocations = false;
            foreach (var mut2 in mutations)
            {
                if (mut2.isAllocation)
                    anyAllocations = true;
                else
                    allAllocations = false;
            }
            if (anyAllocations && !allAllocations)
                throw new Exception("Invalid clique");
            MutationInformation.Mutation mut = new MutationInformation.Mutation();
            mut.isAllocation = anyAllocations;
            mut.expr = GetCommonParent(mutations);
            StringBuilder sb = new StringBuilder();
            sb.Append(mut.expr);
            sb.Append(" is now updated in all contexts");
            Containers containers = null;
            // TODO: these dependencies should include offsets and extraIndices
            DependencyInformation di = new DependencyInformation();
            di.IsFresh = true;
            AnyStatement skipSt = new AnyStatement();
            foreach (var mut2 in mutations)
            {
                IStatement required = mut2.stmt;
                if (required is AnyStatement)
                {
                    AnyStatement requiredAny = (AnyStatement)required;
                    foreach (IStatement required2 in requiredAny.Statements)
                    {
                        di.Add(DependencyType.Dependency | DependencyType.Fresh | DependencyType.Trigger | DependencyType.Overwrite, required2);
                        if (!(required2 is AnyStatement))
                        {
                            Containers c = Containers.GetContainers(required2);
                            if (containers == null)
                                containers = c;
                            else
                                containers = Containers.Intersect(containers, c);
                        }
                    }
                    di.Add(DependencyType.Requirement, required);
                }
                else
                {
                    di.Add(DependencyType.Dependency | DependencyType.Requirement | DependencyType.Fresh | DependencyType.Trigger | DependencyType.Overwrite, required);
                    Containers c = Containers.GetContainers(required);
                    if (containers == null)
                        containers = c;
                    else
                        containers = Containers.Intersect(containers, c);
                }
                skipSt.Statements.Add(required);
            }
            di.Add(DependencyType.SkipIfUniform, skipSt);
            IStatement dummySt = Builder.CommentStmt(sb.ToString());
            // wrap with containers for ForwardBackwardTransform and LoopMerging
            dummySt = Containers.WrapWithContainers(dummySt, containers.inputs);
            context.OutputAttributes.Set(dummySt, di);
            newStatement(dummySt);
            mut.stmt = dummySt;
            newMutation(mut);
        }


        private ICollection<MutationInformation.Mutation> GetMutationInfos(MutationInformation mutInfo, IEnumerable<int> indices)
        {
            var list = new List<MutationInformation.Mutation>();
            foreach (int i in indices)
            {
                list.Add(mutInfo.mutations[i]);
            }
            return list;
        }

        /// <summary>
        /// Get an expression whose assignment would change the value of all the given expressions
        /// </summary>
        /// <param name="mutations"></param>
        /// <returns></returns>
        private IExpression GetCommonParent(ICollection<MutationInformation.Mutation> mutations)
        {
            bool useAny = true;
            if (useAny)
            {
                List<IExpression> exprs = new List<IExpression>();
                foreach (MutationInformation.Mutation m in mutations)
                {
                    if (Recognizer.IsStaticMethod(m.expr, new Func<object[], object>(FactorManager.Any)))
                    {
                        IMethodInvokeExpression imie = (IMethodInvokeExpression)m.expr;
                        exprs.AddRange(imie.Arguments);
                    }
                    else
                    {
                        exprs.Add(m.expr);
                    }
                }
                IExpression anyExpr = Builder.StaticMethod(new Func<object[], object>(FactorManager.Any), exprs.ToArray());
                return anyExpr;
            }

            int minCount = int.MaxValue;
            var allPrefixes = new List<List<IExpression>>();
            var allBindings = new List<ICollection<ConditionBinding>>();
            foreach (var mut in mutations)
            {
                List<IExpression> prefixes = Recognizer.GetAllPrefixes(mut.expr);
                allPrefixes.Add(prefixes);
                if (prefixes.Count < minCount)
                    minCount = prefixes.Count;
                allBindings.Add(Recognizer.GetBindings(mut.stmt));
            }
            IExpression parent = allPrefixes[0][0];
            for (int i = 1; i < minCount; i++)
            {
                int bracketSize = -1;
                for (int j = 0; j < allPrefixes.Count; j++)
                {
                    var prefix = allPrefixes[j][i];
                    if (!(prefix is IArrayIndexerExpression))
                        throw new Exception("Unhandled expression type: " + prefix.GetType());
                    IArrayIndexerExpression iaie = (IArrayIndexerExpression)prefix;
                    if (bracketSize == -1)
                        bracketSize = iaie.Indices.Count;
                    else if (bracketSize != iaie.Indices.Count)
                        throw new Exception("inconsistent bracket sizes: " + prefix);
                }
                IExpression[] indices = new IExpression[bracketSize];
                for (int k = 0; k < bracketSize; k++)
                {
                    List<IExpression> exprsToUnify = new List<IExpression>();
                    for (int j = 0; j < allPrefixes.Count; j++)
                    {
                        var prefix = allPrefixes[j][i];
                        IArrayIndexerExpression iaie = (IArrayIndexerExpression)prefix;
                        exprsToUnify.Add(iaie.Indices[k]);
                    }
                    IExpression index = Unify(exprsToUnify, allBindings);
                    indices[k] = index;
                }
                parent = Builder.ArrayIndex(parent, indices);
            }
            return parent;
        }

        private IExpression Unify(IList<IExpression> exprs, IList<ICollection<ConditionBinding>> bindingsOfExpr)
        {
            // for each expr, check if it could be the unifier
            // if not, use AnyIndex
            for (int i = 0; i < exprs.Count; i++)
            {
                IExpression parent = exprs[i];
                bool allMatch = true;
                for (int j = 0; j < exprs.Count; j++)
                {
                    if (j == i)
                        continue;
                    IExpression replacedParent = GateAnalysisTransform.ReplaceExpression(bindingsOfExpr[j], parent);
                    if (!exprs[j].Equals(replacedParent))
                    {
                        allMatch = false;
                        break;
                    }
                }
                if (allMatch)
                    return parent;
            }
            return Builder.StaticMethod(new Func<int>(GateAnalysisTransform.AnyIndex));
        }

        /// <summary>
        /// Post-process dependencies replacing message expressions with the operator
        /// blocks that compute them.
        /// </summary>
        internal void PostProcessDependencies(IList<IStatement> stmts)
        {
            List<IStatement> dummyStatements = new List<IStatement>();
            CreateDummyStatements(dummyStatements.Add);
            // loopMergingInfo stores dependency information used later by LoopMergingTransform
            List<IStatement> stmts2 = new List<IStatement>(stmts);
            stmts2.AddRange(dummyStatements);
            loopMergingInfo = new LoopMergingInfo(stmts2);
            IMethodDeclaration imd = context.FindAncestor<IMethodDeclaration>();
            context.OutputAttributes.Set(imd, loopMergingInfo);
            List<Tuple<IStatement, IStatement>> checkForCancels = new List<Tuple<IStatement, IStatement>>();
            foreach (IStatement ist in stmts)
            {
                // note: this assumes that converted statements are ReferenceEqual to the input statements.
                DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                if (di == null)
                {
                    // print some debugging information
                    //context.InputAttributes.WriteObjectAttributesTo(ist, System.Console.Out);
                    foreach (var kvp in context.InputAttributes)
                    {
                        if (kvp.Key.ToString() == ist.ToString())
                            Console.WriteLine(kvp.Key);
                    }
                    Error("Cannot find dependency information for statement: " + ist);
                    continue;
                }
                var bounds = new Dictionary<IVariableDeclaration, CodeRecognizer.Bounds>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
                Recognizer.AddLoopBounds(bounds, ist);
                bool debug = false;
                if (debug)
                {
                    // for debugging
                    IStatement innerStmt = ist;
                    while (Containers.IsContainer(innerStmt))
                    {
                        if (innerStmt is IForStatement ifs)
                            innerStmt = ifs.Body.Statements[0];
                        else if (innerStmt is IConditionStatement ics)
                            innerStmt = ics.Then.Statements[0];
                        else
                            throw new Exception();
                    }
                    if (innerStmt is IExpressionStatement ies)
                    {
                        if (ies.Expression is IAssignExpression iae)
                        {
                            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
                            if (ivd != null && ivd.Name == "NoisyNodes_index0__Q" && iae.Expression.ToString().Contains("nodes_uses_F"))
                                Console.WriteLine(ivd);
                        }
                    }
                }
                bool isInitializer = context.InputAttributes.Has<Initializer>(ist);
                bool isOperatorStatement = context.InputAttributes.Has<OperatorStatement>(ist);
                DependencyInformation di2 = (DependencyInformation)di.Clone();
                di2.dependencyTypeOf.Clear();
                // For each mutating statement, stores the loop indices in mutated that must be iterated to cover affected (i.e. loops that cannot be merged).
                Dictionary<IStatement, Set<IVariableDeclaration>> extraIndicesOfStmt = new Dictionary<IStatement, Set<IVariableDeclaration>>(ReferenceEqualityComparer<IStatement>.Instance);
                foreach (IStatement exprStmt in di.DeclDependencies)
                {
                    var mutatingStmts = GetStatementsThatMutate(ist, exprStmt, true, false, ist, bounds, di2.offsetIndexOf, extraIndicesOfStmt);
                    foreach (IStatement mutatingStmt in mutatingStmts)
                    {
                        di2.Add(DependencyType.Declaration, mutatingStmt);
                    }
                }
                List<KeyValuePair<IStatement, OffsetInfo>> extraOffsetInfos = new List<KeyValuePair<IStatement, OffsetInfo>>();
                bool allocationsOnly = isInitializer;
                foreach (KeyValuePair<IStatement, DependencyType> entry in di.dependencyTypeOf)
                {
                    IStatement exprStmt = entry.Key;
                    DependencyType type = entry.Value;
                    type &= ~DependencyType.Declaration; // declarations are handled above
                    if ((type & DependencyType.SkipIfUniform) > 0)
                    {
                        // Process SkipIfUniform dependencies specially
                        // Stmts with the same offsetInfo are put into Any
                        var mutatingStmtsSkip = GetStatementsThatMutate(ist, exprStmt, allocationsOnly, true, ist, bounds, di2.offsetIndexOf, extraIndicesOfStmt);
                        foreach (IStatement mutatingStmt in mutatingStmtsSkip)
                        {
                            di2.Add(DependencyType.SkipIfUniform, mutatingStmt);
                        }
                        type &= ~DependencyType.SkipIfUniform;
                        // Fall through
                    }
                    if (type == 0)
                        continue;
                    if ((type & DependencyType.Container) > 0)
                        type |= DependencyType.Dependency; // containers also become deps
                    var mutatingStmts = GetStatementsThatMutate(ist, exprStmt, allocationsOnly, false, ist, bounds, di2.offsetIndexOf, extraIndicesOfStmt);
                    foreach (IStatement mutatingStmt in mutatingStmts)
                    {
                        DependencyType type2 = type;
                        // an ordinary dependency on a fresh statement becomes a fresh dependency
                        DependencyInformation di3 = context.InputAttributes.Get<DependencyInformation>(mutatingStmt);
                        if (di3 != null && di3.IsFresh && (type2 & DependencyType.Dependency) > 0)
                            type2 |= DependencyType.Fresh;
                        if (ReferenceEquals(mutatingStmt, ist))
                            type2 &= ~(DependencyType.Trigger | DependencyType.Fresh); // do not trigger on self
                        IncrementStatement attr = context.InputAttributes.Get<IncrementStatement>(mutatingStmt);
                        if (attr != null)
                        {
                            if (attr.loopVar != null)
                            {
                                // treat an increment as having an offset
                                OffsetInfo offsets = new OffsetInfo();
                                offsets.Add(attr.loopVar, sequentialOffset, false);
                                if (attr.Bidirectional)
                                    offsets.Add(attr.loopVar, 1, true);
                                // do not add to di2.offsetIndexOf yet because we don't want to prevent loop merging.
                                extraOffsetInfos.Add(new KeyValuePair<IStatement, OffsetInfo>(mutatingStmt, offsets));
                            }
                            else
                            {
                                checkForCancels.Add(Tuple.Create(ist, mutatingStmt));
                            }
                            type2 |= DependencyType.NoInit;
                        }
                        di2.Add(type2, mutatingStmt);
                    }
                }
                foreach (KeyValuePair<IStatement, IOffsetInfo> entry in di2.offsetIndexOf)
                {
                    IStatement mutatingStmt = entry.Key;
                    loopMergingInfo.SetOffsetInfo(mutatingStmt, ist, entry.Value);
                }
                foreach (var entry in extraOffsetInfos)
                {
                    di2.AddOffsetIndices(entry.Value, entry.Key);
                }
                bool isSumOp = ist.ToString().Contains("SumOp_SHG09"); // TEMPORARY to support Recommender
                foreach (KeyValuePair<IStatement, Set<IVariableDeclaration>> entry in extraIndicesOfStmt)
                {
                    IStatement mutatingStmt = entry.Key;
                    Set<IVariableDeclaration> indices = entry.Value;
                    bool isIncrement = context.InputAttributes.Has<IncrementStatement>(mutatingStmt);
                    if (isSumOp && context.InputAttributes.Has<OperatorStatement>(mutatingStmt))
                    {
                        // add an offset and do not prevent loop merging
                        // products_B = SumOp_SHG09.ArrayAverageConditional(... products_F)
                        // will get an offset dependency on products_F
                        OffsetInfo offsets = new OffsetInfo();
                        foreach (IVariableDeclaration loopVar in entry.Value)
                        {
                            offsets.Add(loopVar, sequentialOffset, true);
                        }
                        di2.offsetIndexOf[mutatingStmt] = offsets;
                        continue;
                    }
                    if (isIncrement)
                    {
                        // if ist depends on an array and mutatingStmt is an increment to array[i] (where i is the loopVar), then these two statements can be in the same loop over i
                        indices = new Set<IVariableDeclaration>();
                        indices.AddRange(entry.Value);
                        foreach (IncrementStatement attr in context.InputAttributes.GetAll<IncrementStatement>(mutatingStmt))
                        {
                            indices.Remove(attr.loopVar);
                        }
                        if (indices.Count == 0)
                            continue;
                    }
                    // if ist depends on an array and its dependency assigns to array[i], then these two statements must not be in the same loop over i
                    loopMergingInfo.PreventLoopMerging(mutatingStmt, ist, indices);
                }
                context.InputAttributes.Remove<DependencyInformation>(ist);
                context.OutputAttributes.Set(ist, di2);
            }
            // Ignore a dependency on an increment statement with a reverse Cancels.
            // For example:
            // 1. from_x = x_marginal/to_x
            // 2. x_marginal = to_x*from_x
            // The dependency 2->1 should be ignored, since updating 2 does not affect the value of 1.
            foreach (var tuple in checkForCancels)
            {
                IStatement affectedStmt = tuple.Item1;
                // mutatingStmt is an increment
                IStatement mutatingStmt = tuple.Item2;
                DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(affectedStmt);
                DependencyInformation di2 = context.InputAttributes.Get<DependencyInformation>(mutatingStmt);
                if (di2.HasDependency(DependencyType.Cancels, affectedStmt))
                {
                    var replacements = new Dictionary<IStatement, IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
                    replacements[mutatingStmt] = null;
                    di.Replace(replacements);
                    //DependencyType type = di.dependencyTypeOf[mutatingStmt];
                    //type |= DependencyType.Cancels;
                    //di.dependencyTypeOf.Remove(mutatingStmt);
                    //di.dependencyTypeOf[mutatingStmt] = type;
                    //Console.WriteLine($"ignoring dependency of {affectedStmt} on increment {mutatingStmt}");
                }
            }
            stmts.AddRange(dummyStatements);
        }

        class OffsetInfoComparer : IEqualityComparer<OffsetInfo>
        {
            public bool Equals(OffsetInfo x, OffsetInfo y)
            {
                if (x == null) return (y == null);
                else if (y == null) return false;
                else return x.Any(offset => y.ContainsKey(offset.loopVar));
            }

            public int GetHashCode(OffsetInfo obj)
            {
                return 0;
            }
        }

        protected IExpression ConvertExpressionWithDependencyType(IExpression expr, DependencyType type)
        {
            DependencyType oldtype = dependencyType;
            try
            {
                dependencyType = type;
                IExpression newExpr = ConvertExpression(expr);
                return newExpr;
            }
            finally
            {
                dependencyType = oldtype;
            }
        }

        /// <summary>
        /// Shallow copy of 'for' statement
        /// </summary>
        protected override IStatement ConvertFor(IForStatement ifs)
        {
            // We partially convert the loop initializer and do not convert the increment.  
            // We only convert the loop size which is the right hand size of the loop condition.
            convertingLoopInitializer = true;
            ConvertStatement(ifs.Initializer);
            convertingLoopInitializer = false;
            IBinaryExpression ibe = ifs.Condition as IBinaryExpression;
            if (ibe == null)
            {
                Error("For loop conditions must be binary expressions, was :" + ifs.Condition);
            }
            else
            {
                IVariableDeclaration loopVar = CodeRecognizer.Instance.GetVariableDeclaration(ibe.Left);
                if (loopVar == null)
                {
                    Error("For loop conditions have loop counter reference on LHS, was :" + ibe.Left);
                }
                else
                {
                    if (!context.InputAttributes.Has<LoopCounterAttribute>(loopVar))
                    {
                        context.InputAttributes.Set(loopVar, new LoopCounterAttribute());
                    }
                }
                if ((ibe.Right is ILiteralExpression) && 0.Equals(((ILiteralExpression)ibe.Right).Value))
                {
                    // loop of zero length
                    return null;
                }
                ConvertExpressionWithDependencyType(ibe.Right, DependencyType.Container);
            }
            //ConvertStatement(Builder.BlockStmt(), ifs.Increment, null);
            context.SetPrimaryOutput(ifs);
            ConvertBlock(ifs.Body);
            return ifs;
        }

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            ConvertExpressionWithDependencyType(ics.Condition, DependencyType.Container);
            context.SetPrimaryOutput(ics);
            ConvertBlock(ics.Then);
            if (ics.Else != null)
                ConvertBlock(ics.Else);
            return ics;
        }

        /// <summary>
        /// Register the mutation represented by expression expr - this expression might be a
        /// target of an assignment or of a non-static method, or it might be a variable declaration
        /// expression. The expression and its top level statement are added to the registry of
        /// mutating expressions for the associated variable declaration.
        /// In addition, previous mutations are marked as 'allocations'
        /// </summary>
        /// <param name="topLevelStatement"></param>
        /// <param name="expr">An lvalue or assignment (in the case of initialization).</param>
        /// <param name="bindings">Bindings of loop variables in expr</param>
        /// <param name="bounds">Bounds on loop variables in expr</param>
        internal void RegisterMutation(
            IStatement topLevelStatement,
            IExpression expr,
            IStatement bindings,
            IReadOnlyDictionary<IVariableDeclaration, CodeRecognizer.Bounds> bounds)
        {
            // Extract the variable declaration from a, possibly indexed, reference or declaration expression.
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);

            // This case arises for targets of static methods where the target is a type reference expression
            // rather than a variable reference expression. In this case, no mutation occurs, so just return
            if (ivd == null)
                return;
            // Ignore mutations of loop variables
            if (context.InputAttributes.Has<LoopCounterAttribute>(ivd))
                return;

            // Get the MutationInformation instance that stores the list of mutating expressions for this variable declaration.
            // Create if this is the first time we have seen this variable
            MutationInformation mutInfo;
            if (!mutInfos.TryGetValue(ivd, out mutInfo))
            {
                mutInfo = new MutationInformation();
                mutInfos[ivd] = mutInfo;
            }

            // Register the current expression 
            bool isAllocation = IsAllocationStatement(context, topLevelStatement);
            mutInfo.RegisterMutationStatement(expr, topLevelStatement, isAllocation);

            // Find earlier allocations
            bool isIncrement = context.InputAttributes.Has<IncrementStatement>(topLevelStatement);
            bool allocationsOnly = !isIncrement;
            try
            {
                IList<IStatement> allocations = mutInfo.GetMutations(context, topLevelStatement, expr, allocationsOnly, false, bindings, bounds).ListSelect(m => m.stmt);

                foreach (IStatement alloc in allocations)
                {
                    dependencyInformation.Add(DependencyType.Overwrite, alloc);
                    dependencyInformation.Add(DependencyType.Declaration, alloc);
                }
            }
            catch (Exception ex)
            {
                Error(ex.Message, ex);
            }
        }

        private static bool IsAllocationStatement(BasicTransformContext context, IStatement stmt)
        {
            if (stmt is IConditionStatement)
                return IsAllocationStatement(context, ((IConditionStatement)stmt).Then);
            else if (stmt is IForStatement)
                return IsAllocationStatement(context, ((IForStatement)stmt).Body);
            else if (stmt is IBlockStatement)
                return IsAllocationStatement(context, ((IBlockStatement)stmt).Statements[0]);
            else // TODO: should use an attribute for AllocationStatement
                return !context.InputAttributes.Has<OperatorStatement>(stmt);
        }

        private bool IsBeingMutated(IExpression expr)
        {
            foreach (IExpression mutated in mutatedExpressions)
            {
                if (Recognizer.MutatingFirstAffectsSecond(context, mutated, expr, false))
                    return true;
            }
            return false;
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            if (convertingLoopInitializer)
            {
                ConvertExpression(iae.Target);
                return iae;
            }

            if (iae.Target is IVariableDeclarationExpression)
            {
                ConvertExpression(iae.Target);
                dependencyInformation.IsUniform = false;
            }
            else
                mutatedExpressions.Add(iae.Target);

            // Add dependency on RHS expression
            ConvertExpression(iae.Expression);

            // Add container dependencies on indices of LHS
            ConvertIndices(iae.Target, DependencyType.Container);

            return iae;
        }

        protected override IExpression ConvertAddressOut(IAddressOutExpression iaoe)
        {
            mutatedExpressions.Add(iaoe.Expression);
            return base.ConvertAddressOut(iaoe);
        }

        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            if (!convertingLoopInitializer)
            {
                if ((dependencyType & DependencyType.SkipIfUniform) > 0)
                    dependencyInformation.IsUniform = true;
                IExpression varRef = Builder.VarRefExpr(ivde.Variable);
                mutatedExpressions.Add(varRef);
            }
            // Record the containers that a variable is created in, for use by CodeRecognizer.MutatingFirstAffectsSecond
            context.InputAttributes.Remove<Containers>(ivde.Variable);
            context.InputAttributes.Set(ivde.Variable, new Containers(context));
            return ivde;
        }

        protected override IExpression ConvertArrayCreate(IArrayCreateExpression iace)
        {
            if ((dependencyType & DependencyType.SkipIfUniform) > 0)
            {
                bool isUniform = false;
                foreach (IExpression dim in iace.Dimensions)
                {
                    if ((dim is ILiteralExpression) && 0.Equals(((ILiteralExpression)dim).Value))
                    {
                        isUniform = true;
                    }
                }
                if (iace.Initializer == null)
                    isUniform = true;
                if (isUniform)
                    dependencyInformation.IsUniform = true;
            }
            return base.ConvertArrayCreate(iace);
        }

        protected override IExpression ConvertDefaultExpr(IDefaultExpression ide)
        {
            if ((dependencyType & DependencyType.SkipIfUniform) > 0)
                dependencyInformation.IsUniform = true;
            return base.ConvertDefaultExpr(ide);
        }

        protected override IExpression ConvertBinary(IBinaryExpression ibe)
        {
            // assume that binary operators do not have SkipIfUniform attributes
            if ((dependencyType & DependencyType.SkipIfUniform) > 0)
                return ConvertExpressionWithDependencyType(ibe, dependencyType & ~DependencyType.SkipIfUniform);
            else
                return base.ConvertBinary(ibe);
        }

        protected override IExpression ConvertObjectCreate(IObjectCreateExpression ioce)
        {
            if (ioce.Constructor == null)
            {
                Warning("null constructor - cannot compute dependencies");
                return ioce;
            }
            MethodBase method = Builder.ToMethod(ioce.Constructor);
            if (method == null)
            {
                if (ioce.Arguments.Count > 0)
                    Warning("constructor has no MethodBase - cannot compute dependencies");
                return ioce;
            }
            if ((dependencyType & DependencyType.SkipIfUniform) > 0)
            {
                if (typeof(ConvertibleToArray).IsAssignableFrom(method.DeclaringType))
                {
                    // if the constructor has a parameter named "length" that is zero, then mark the statement as uniform
                    ParameterInfo[] parameters = method.GetParameters();
                    int argIndex = 0;
                    foreach (ParameterInfo parameter in parameters)
                    {
                        if (parameter.Name == "length")
                        {
                            IExpression arg = ioce.Arguments[argIndex];
                            if (arg is ILiteralExpression)
                            {
                                object argValue = ((ILiteralExpression)arg).Value;
                                if (argValue is int && (int)argValue == 0)
                                    dependencyInformation.IsUniform = true;
                            }
                            break;
                        }
                        argIndex++;
                    }
                }
            }
            return ConvertFactor(ioce, method, ioce.Arguments);
        }

        // Shallow copy and visit children
        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if (CodeRecognizer.IsInfer(imie))
            {
                dependencyInformation.IsOutput = true;
                // make sure that IsInferred attributes are correct for IterativeProcessTransform
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(imie.Arguments[0]);
                if (ivd != null && !context.InputAttributes.Has<IsInferred>(ivd))
                    context.OutputAttributes.Set(ivd, new IsInferred());
            }
            if (context.InputAttributes.Has<DeterministicConstraint>(imie))
                dependencyInformation.IsOutput = true;

            if (!(imie.Method is IMethodReferenceExpression imre) ||
                !(Builder.ToMethod(imre.Method) is MethodInfo method))
            {
                Error("Could not find method for : " + imie.Method);
                return imie;
            }

            if (Recognizer.IsStaticMethod(imie, new Func<object[], object>(FactorManager.All)))
            {
                AllStatement allSt = new AllStatement();
                DependencyInformation parentDepInfo = dependencyInformation;
                dependencyInformation = new DependencyInformation();
                foreach (IExpression arg in imie.Arguments)
                {
                    dependencyInformation.IsUniform = false;
                    dependencyInformation.dependencyTypeOf.Clear();
                    ConvertExpression(arg);
                    if ((dependencyType & DependencyType.SkipIfUniform) > 0)
                    {
                        if (dependencyInformation.IsUniform)
                        {
                            // nothing to add
                            continue;
                        }
                    }
                    foreach (KeyValuePair<IStatement, DependencyType> entry in dependencyInformation.dependencyTypeOf)
                    {
                        DependencyType type = entry.Value & dependencyType;
                        if (type > 0)
                            allSt.Statements.Add(entry.Key);
                    }
                }
                dependencyInformation = parentDepInfo;
                if (allSt.Statements.Count > 0)
                    AddDependencyOn(allSt);
                return imie;
            }
            if (Recognizer.IsStaticMethod(imie, new Func<object[], object>(FactorManager.Any)))
            {
                AnyStatement anySt = new AnyStatement();
                DependencyInformation parentDepInfo = dependencyInformation;
                dependencyInformation = new DependencyInformation();
                List<KeyValuePair<IStatement, DependencyType>> deps = new List<KeyValuePair<IStatement, DependencyType>>();
                foreach (IExpression arg in imie.Arguments)
                {
                    dependencyInformation.IsUniform = false;
                    dependencyInformation.dependencyTypeOf.Clear();
                    ConvertExpression(arg);
                    if ((dependencyType & DependencyType.SkipIfUniform) > 0)
                    {
                        if (dependencyInformation.IsUniform)
                        {
                            // nothing to add
                            continue;
                        }
                    }
                    deps.Clear();
                    foreach (KeyValuePair<IStatement, DependencyType> entry in dependencyInformation.dependencyTypeOf)
                    {
                        DependencyType type = entry.Value & dependencyType;
                        if (type > 0)
                            deps.Add(new KeyValuePair<IStatement, DependencyType>(entry.Key, type));
                    }
                    if (deps.Count == 0)
                    {
                        // this argument has no dependencies, therefore there is no Any dependency
                        anySt = null;
                        break;
                    }
                    else if (deps.Count == 1)
                    {
                        anySt.Statements.Add(deps[0].Key);
                    }
                    else
                    {
                        // deps.Count > 1
                        if (imie.Arguments.Count == 1)
                        {
                            foreach (KeyValuePair<IStatement, DependencyType> entry in deps)
                            {
                                parentDepInfo.Add(entry.Value, entry.Key);
                            }
                        }
                        else
                        {
                            // must make a dummy statement
                            Error("Unhandled Any dependency");
                        }
                        anySt = null;
                    }
                }
                dependencyInformation = parentDepInfo;
                if (anySt != null)
                {
                    if (anySt.Statements.Count == 0)
                        dependencyInformation.IsUniform = true;
                    else
                        AddDependencyOn(anySt);
                }
                return imie;
            }
            if (Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, PlaceHolder, PlaceHolder>(FactorManager.AllExcept))
                || Recognizer.IsStaticMethod(imie, new Func<object, object>(FactorManager.InducedSource))
                || Recognizer.IsStaticMethod(imie, new Func<object, object>(FactorManager.InducedTarget)))
            {
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(imie.Arguments[0]);
                if (ivd != null)
                {
                    AddDependencyOn(imie);
                }
                return imie;
            }
            if (Recognizer.IsStaticMethod(imie, new Func<object, object>(FactorManager.NoTrigger)))
            {
                IExpression arg = imie.Arguments[0];
                ConvertExpressionWithDependencyType(arg, dependencyType & ~DependencyType.Trigger);
                return imie;
            }
            ConvertFactor(imie, method, imie.Arguments);
            // convert the target after the arguments
            if (imre.Target != null)
            {
                // We assume that the target is only used to select a method, so value of the target doesn't matter.
                ConvertExpressionWithDependencyType(imre.Target, DependencyType.Declaration);
                if (!method.Name.StartsWith("Get") && !method.Name.StartsWith("get") && !method.Name.StartsWith("Sample"))
                {
                    mutatedExpressions.Add(imre.Target);
                }
            }
            return imie;
        }

        private IExpression ConvertFactor(IExpression imie, MethodBase method, IList<IExpression> args)
        {
            MessageFcnInfo fcnInfo = context.InputAttributes.Get<MessageFcnInfo>(method);
            // Note di cannot contain ParameterDeps, ContainerDeps
            // di must not be modified!
            DependencyInformation di = (fcnInfo == null) ? FactorManager.GetDependencyInfo(method) : fcnInfo.DependencyInfo;
            if ((dependencyType & DependencyType.SkipIfUniform) > 0)
            {
                bool methodReturnsUniform = false;
                if (di.IsUniform)
                    methodReturnsUniform = true; // the method returns uniform regardless of arguments
                if (Recognizer.IsStaticMethod(imie, typeof(Collections.ArrayHelper), "Fill"))
                {
                    if (args.Count > 0)
                    {
                        IExpression arg = args[0];
                        if (arg is ILiteralExpression ile)
                        {
                            object value = ile.Value;
                            if (0.Equals(value))
                            {
                                // result is an empty array
                                methodReturnsUniform = true;
                            }
                        }
                    }
                }
                if (methodReturnsUniform)
                    dependencyInformation.IsUniform = true;
            }

            // Build a mapping between default parameter expressions and the corresponding model expressions
            int i = 0;
            Dictionary<IExpression, IExpression> parameterToExpressionMap = new Dictionary<IExpression, IExpression>();
            foreach (ParameterInfo pi in Util.GetParameters(method))
            {
                IArgumentReferenceExpression paramRef = Builder.ParamRef(Builder.Param(pi.Name, pi.ParameterType));
                if (i >= args.Count)
                    Error("method call has too few arguments");
                else
                    parameterToExpressionMap[paramRef] = args[i++];
            }
            // Now apply this map to each item in the dependency list
            bool treatTriggersAsFresh = false;
            bool treatFreshAsTriggers = false;
            bool allTriggers = context.InputAttributes.Has<AllTriggersAttribute>(imie);
            if (treatFreshAsTriggers && di.IsFresh)
            {
                allTriggers = true;
            }
            foreach (KeyValuePair<IStatement, DependencyType> entry in di.dependencyTypeOf)
            {
                DependencyType type = entry.Value;
                // If 'SkipIfUniform' is clear, then ignore all SkipIfUniform dependencies
                if ((dependencyType & DependencyType.SkipIfUniform) == 0)
                    type &= ~DependencyType.SkipIfUniform;
                // If 'all trigger' is set, all dependencies are also triggers
                if (!treatTriggersAsFresh && allTriggers && (type & DependencyType.Dependency) > 0)
                    type |= DependencyType.Trigger;
                IExpression arg = MapDependency(entry.Key, parameterToExpressionMap);
                IVariableDeclaration argVar = Recognizer.GetVariableDeclaration(arg);
                if (argVar != null && context.InputAttributes.Has<DoesNotHaveInitializer>(argVar) && !IsBeingMutated(arg))
                {
                    // If the variable does not have an initializer, then it must be updated first
                    type |= DependencyType.Requirement;
                }
                if (type == 0)
                    continue;
                ConvertExpressionWithDependencyType(arg, type);
            }
            // set IsFresh last to override any setting made by inner expressions
            dependencyInformation.IsFresh = di.IsFresh;
            if (treatTriggersAsFresh && allTriggers)
            {
                dependencyInformation.IsFresh = true;
            }
            if (treatFreshAsTriggers)
            {
                dependencyInformation.IsFresh = false;
            }
            return imie;
        }

        // This takes a list of dependencies, which are in the form of expression statements.
        // These dependencies are created by the factor manager which analyses message
        // function attributes and creates the expressions statements using the inbuilt parameter
        // names and types. The parameterToExpressionMap maps parameter expressions to the corresponding
        // argument expressions in the model. This method uses this map, to replace all
        // arguments in all dependencies with the correct model expressions.
        internal IExpression MapDependency(IStatement ist, Dictionary<IExpression, IExpression> parameterToExpressionMap)
        {
            return ReplaceArgs(((IExpressionStatement)ist).Expression, parameterToExpressionMap);
        }

        // Replace the parameter expressions in an expression with the corresponding model expressions 
        private IExpression ReplaceArgs(IExpression iExpression, Dictionary<IExpression, IExpression> parameterToExpressionMap)
        {
            foreach (KeyValuePair<IExpression, IExpression> kvp in parameterToExpressionMap)
            {
                int repCount = 0;
                iExpression = Builder.ReplaceExpression(iExpression, kvp.Key, kvp.Value, ref repCount);
            }
            return iExpression;
        }

#region Expression types on which a statement can have dependencies 

        // Add dependencies on this property reference
        // We cannot use base.ConvertPropertyRefExpr because we want the dependency to be on the entire expression,
        // not the Target expression by itself.
        protected override IExpression ConvertPropertyRefExpr(IPropertyReferenceExpression ipre)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(ipre);
            if (ivd != null)
            {
                AddDependencyOn(ipre);
                return ipre;
            }
            else
                return base.ConvertPropertyRefExpr(ipre);
        }

        protected override IExpression ConvertFieldRefExpr(IFieldReferenceExpression ifre)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(ifre);
            if (ivd != null)
            {
                AddDependencyOn(ifre);
                return ifre;
            }
            else
                return base.ConvertFieldRefExpr(ifre);
        }

        // Adds dependencies on this variable reference.
        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            if (Recognizer.GetLoopForVariable(context, ivre) == null)
            {
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(ivre);
                AddDependencyOn(ivre);
            }
            return ivre;
        }

        // Adds dependencies on this array element.
        // We cannot use base.ConvertArrayIndexer because we want the dependency to be on the entire expression,
        // not the Target expression by itself.
        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            iaie = (IArrayIndexerExpression)ConvertIndices(iaie, dependencyType);
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iaie.Target);
            if (ivd != null)
            {
                AddDependencyOn(iaie);
            }
            else
            {
                IParameterDeclaration ipd = Recognizer.GetParameterDeclaration(iaie.Target);
                if (ipd != null)
                    AddParameterDependency(ipd);
            }
            return iaie;
        }

#endregion

        /// <summary>
        /// Converts the indices in all brackets of expr, but not the innermost target.
        /// </summary>
        /// <param name="expr">An array indexer expression, otherwise does nothing</param>
        /// <param name="type">the DependencyType to apply</param>
        /// <returns></returns>
        protected IExpression ConvertIndices(IExpression expr, DependencyType type)
        {
            DependencyType dependencyTypeOld = dependencyType;
            try
            {
                // SkipIfUniform dependencies do not apply to array indices
                dependencyType = type & ~(DependencyType.SkipIfUniform);
                return ConvertIndices2(expr);
            }
            finally
            {
                dependencyType = dependencyTypeOld;
            }
        }

        protected IExpression ConvertIndices2(IExpression expr)
        {
            if (!(expr is IArrayIndexerExpression))
                return expr;
            IArrayIndexerExpression iaie = (IArrayIndexerExpression)expr;
            IArrayIndexerExpression aie = Builder.ArrayIndxrExpr();
            foreach (IExpression exp in iaie.Indices)
            {
                aie.Indices.Add(ConvertExpression(exp));
            }
            aie.Target = ConvertIndices2(iaie.Target);
            return aie;
        }

        // Adds expr to the current dependency list
        protected void AddDependencyOn(IExpression expr)
        {
            AddDependencyOn(expr, dependencyType);
        }

        protected void AddDependencyOn(IExpression expr, DependencyType type)
        {
            // commented out for efficiency, since this check is not needed (yet)
            //if (IsBeingMutated(expr)) return;
            AddDependencyOn(new ExpressionDependency(expr), type);
        }

        protected void AddDependencyOn(IStatement st)
        {
            AddDependencyOn(st, dependencyType);
        }

        protected void AddDependencyOn(IStatement st, DependencyType type)
        {
            dependencyInformation.Add(type, st);
        }

        internal void AddParameterDependency(IParameterDeclaration ipd)
        {
            dependencyInformation.ParameterDependencies.Add(ipd);
        }

        protected override IParameterDeclaration ConvertMethodParameter(IParameterDeclaration ipd, int index)
        {
            if (context.FindAncestor<IAnonymousMethodExpression>() == null)
                topLevelParameters.Add(ipd);
            return ipd;
        }

        protected override IExpression ConvertArgumentRef(IArgumentReferenceExpression iare)
        {
            IParameterDeclaration ipd = iare.Parameter.Resolve();
            // only add dependencies on top-level parameters, not anonymous method parameters
            if (topLevelParameters.Contains(ipd))
                AddParameterDependency(ipd);
            return base.ConvertArgumentRef(iare);
        }

        private void ForEachExpressionDependency(IEnumerable<IStatement> stmts, Action<ExpressionDependency> action)
        {
            foreach (IStatement ist in stmts)
            {
                if (ist is ExpressionDependency)
                    action((ExpressionDependency)ist);
                else if (ist is AnyStatement)
                {
                    AnyStatement anySt = (AnyStatement)ist;
                    ForEachExpressionDependency(anySt.Statements, action);
                }
                else
                    throw new Exception("unexpected dependency");
            }
        }

        static IEnumerable<IStatement> LiftNestedAll(AnyStatement anySt)
        {
            // Convert Any(x,All(y,z)) to All(Any(x,y),Any(x,z))
            // Convert Any(All(x,y),All(z,w)) to All(Any(x,z),Any(x,w),Any(y,z),Any(y,w))
            List<AnyStatement> results = new List<AnyStatement>();
            IEnumerable<AnyStatement> CopyAndAdd(IStatement newSt)
            {
                IStatement[] newSequence = new IStatement[] { newSt };
                if (results.Count == 0) return new[] { new AnyStatement(newSequence) };
                else return results.Select(a => new AnyStatement(a.Statements.Concat(newSequence).ToArray()));
            }
            foreach(var stmt in anySt.Statements)
            {
                if(stmt is AllStatement allSt)
                {
                    results = allSt.Statements.SelectMany(CopyAndAdd).ToList();
                }
                else if(results.Count > 0)
                {
                    // Add this stmt to all clauses
                    foreach(var clause in results)
                    {
                        clause.Statements.Add(stmt);
                    }
                }
                else
                {
                    results.Add(new AnyStatement(stmt));
                }
            }
            return results.Select(a => (a.Statements.Count == 1) ? a.Statements[0] : a);
        }

        // same as GetMutations but handles AnyStatements
        internal IEnumerable<IStatement> GetStatementsThatMutate(
            IStatement exclude,
            IStatement exprStmt,
            bool allocationsOnly,
            bool mustMutate,
            IStatement bindings,
            IReadOnlyDictionary<IVariableDeclaration, CodeRecognizer.Bounds> bounds,
            Dictionary<IStatement, IOffsetInfo> offsetInfos,
            Dictionary<IStatement, Set<IVariableDeclaration>> extraIndicesOfStmt)
        {
            if (mustMutate && exprStmt is ExpressionDependency) exprStmt = new AnyStatement(exprStmt);
            if (exprStmt is AllStatement allSt)
            {
                List<ExpressionDependency> exprDeps = new List<ExpressionDependency>();
                ForEachExpressionDependency(allSt.Statements, exprDeps.Add);
                List<IStatement> results = new List<IStatement>();
                foreach (ExpressionDependency ies in exprDeps)
                {
                    IExpression expr = ies.Expression;
                    List<MutationInformation.Mutation> mutations = GetMutations(exclude, expr, allocationsOnly, mustMutate, bindings, bounds, offsetInfos, extraIndicesOfStmt);
                    IExpression prefixExpr = expr;
                    while (prefixExpr is IMethodInvokeExpression)
                    {
                        IMethodInvokeExpression imie = (IMethodInvokeExpression)prefixExpr;
                        prefixExpr = imie.Arguments[0];
                    }
                    var prefixes = Recognizer.GetAllPrefixes(prefixExpr);
                    // algorithm: find the prefix of each mutation, compute a graph of all prefix overlaps, make an All statement for each clique
                    // example: x[i] has mutations x[0][0], x[1][0], x[i][1]
                    // prefixes are: x[0], x[1], x[i]
                    // cliques are: (x[0], x[i]), (x[1], x[i])
                    // dependency is: (x[0][0] and x[i][1]) or (x[1][0] and x[i][1])
                    List<KeyValuePair<MutationInformation.Mutation, IExpression>> mutationsToCheck = new List<KeyValuePair<MutationInformation.Mutation, IExpression>>();
                    foreach (MutationInformation.Mutation m in mutations)
                    {
                        IExpression expr2 = m.expr;
                        bool isIncrement = context.InputAttributes.Has<IncrementStatement>(m.stmt);
                        if (isIncrement)
                        {
                            // ignore
                        }
                        else
                        {
                            var prefixes2 = Recognizer.GetAllPrefixes(expr2);
                            var prefix2 = prefixes2[System.Math.Min(prefixes2.Count, prefixes.Count) - 1];
                            mutationsToCheck.Add(new KeyValuePair<MutationInformation.Mutation, IExpression>(m, prefix2));
                        }
                    }
                    if (mutationsToCheck.Count == 1)
                    {
                        results.Add(mutationsToCheck[0].Key.stmt);
                    }
                    else // if (mutationsToCheck.Count > 1)
                    {
                        List<IReadOnlyList<IStatement>> cliques = new List<IReadOnlyList<IStatement>>();
                        AddCliques(mutationsToCheck, cliques);
                        AnyStatement anyBlock = new AnyStatement();
                        foreach (var clique in cliques)
                        {
                            if(clique.Count == 1)
                            {
                                anyBlock.Statements.Add(clique[0]);
                            }
                            else
                            {
                                anyBlock.Statements.Add(new AllStatement(clique.ToArray()));
                            }
                        }
                        results.AddRange(LiftNestedAll(anyBlock));
                    }
                }
                foreach (var result in results)
                    yield return result;
            }
            else if (exprStmt is AnyStatement anySt)
            {
                // Any(expr1, expr2) => Any(stmt1, stmt2)
                List<ExpressionDependency> exprDeps = new List<ExpressionDependency>();
                ForEachExpressionDependency(anySt.Statements, exprDeps.Add);
                var newSt = new AnyStatement();
                // For pruning based on SkipIfUniform, we only want to prune a statement if it must be uniform in all cases.
                // This is only guaranteed when every dependency is uniform.
                foreach (ExpressionDependency ies in exprDeps)
                {
                    IExpression expr = ies.Expression;
                    AllStatement allBlock = new AllStatement();
                    AnyStatement anyBlock = new AnyStatement();
                    List<MutationInformation.Mutation> mutations = GetMutations(exclude, expr, allocationsOnly, mustMutate, bindings, bounds, offsetInfos, extraIndicesOfStmt);
                    foreach (MutationInformation.Mutation m in mutations)
                    {
                        anyBlock.Statements.Add(m.stmt);
                    }
                    if (anyBlock.Statements.Count > 0)
                        allBlock.Statements.Add(anyBlock);

                    // add groups to results
                    if (allBlock.Statements.Count == 1)
                    {
                        IStatement ist = allBlock.Statements[0];
                        if (ist is AnyStatement)
                        {
                            AnyStatement group = (AnyStatement)ist;
                            newSt.Statements.AddRange(group.Statements);
                        }
                        else
                        {
                            newSt.Statements.Add(ist);
                        }
                    }
                }
                if (newSt.Statements.Count == 1)
                    yield return newSt.Statements[0];
                else if (newSt.Statements.Count > 0)
                    yield return newSt;
            }
            else if (exprStmt is ExpressionDependency ies)
            {
                IExpression expr = ies.Expression;
                foreach (MutationInformation.Mutation m in GetMutations(exclude, expr, allocationsOnly, mustMutate, bindings, bounds, offsetInfos, extraIndicesOfStmt))
                {
                    IStatement mutatingStmt = m.stmt;
                    yield return mutatingStmt;
                }
            }
            else
            {
                yield return exprStmt;
            }
        }

        private void AddCliques(List<KeyValuePair<MutationInformation.Mutation, IExpression>> mutationsToCheck, IList<IReadOnlyList<IStatement>> groups)
        {
            // find overlapping mutations
            Dictionary<int, List<int>> overlappingMutations = new Dictionary<int, List<int>>();
            Set<int> candidates = new Set<int>();
            for (int i = 0; i < mutationsToCheck.Count; i++)
            {
                candidates.Add(i);
                List<int> overlap = new List<int>();
                MutationInformation.Mutation mut = mutationsToCheck[i].Key;
                IExpression expr1 = mutationsToCheck[i].Value;
                bool isIncrement = context.InputAttributes.Has<IncrementStatement>(mut.stmt);
                for (int j = 0; j < mutationsToCheck.Count; j++)
                {
                    if (j == i)
                        continue;
                    MutationInformation.Mutation mut2 = mutationsToCheck[j].Key;
                    IExpression expr2 = mutationsToCheck[j].Value;
                    //if (mut2.isAllocation != mut.isAllocation)
                    //    continue;
                    //if (mut2.isAllocation)
                    //    continue;
                    // only statements with the same increment status are considered overlapping
                    //if (isIncrement != context.InputAttributes.Has<IncrementStatement>(mut2.stmt))
                    //    continue;
                    //if (Recognizer.MutatingFirstAffectsSecond(context, expr1, expr2, false, mut.stmt, mut2.stmt))
                    if (Recognizer.MutatingFirstAffectsSecond(context, expr1, expr2, false))
                    {
                        overlap.Add(j);
                    }
                }
                overlappingMutations[i] = overlap;
            }
            var cliqueFinder = new CliqueFinder<int>(i => overlappingMutations[i]);
            cliqueFinder.ForEachClique(candidates, delegate (Stack<int> c)
            {
                List<IStatement> group = new List<IStatement>();
                foreach (int i in c)
                {
                    group.Add(mutationsToCheck[i].Key.stmt);
                }
                groups.Add(group);
            });
        }

        /// <summary>
        /// Get all statements that change the value of an expression
        /// </summary>
        /// <param name="exclude">A statement to exclude from the result</param>
        /// <param name="expr">The expression being changed</param>
        /// <param name="allocationsOnly">If true, only allocation statements are returned</param>
        /// <param name="mustMutate">If true, all returned statements are known to mutate.  Otherwise, returned statements are not guaranteed to mutate</param>
        /// <param name="bindingsInExpr">Bindings of loop variables in expr</param>
        /// <param name="boundsInExpr">Bounds on loop variables in expr</param>
        /// <param name="offsetInfos">Modified on exit</param>
        /// <param name="extraIndicesOfStmt">Modified on exit</param>
        /// <returns></returns>
        private List<MutationInformation.Mutation> GetMutations(
            IStatement exclude,
            IExpression expr,
            bool allocationsOnly,
            bool mustMutate,
            IStatement bindingsInExpr,
            IReadOnlyDictionary<IVariableDeclaration, CodeRecognizer.Bounds> boundsInExpr,
            Dictionary<IStatement, IOffsetInfo> offsetInfos,
            Dictionary<IStatement, Set<IVariableDeclaration>> extraIndicesOfStmt)
        {
            IExpression varExpr = expr;
            while (varExpr is IMethodInvokeExpression)
                varExpr = ((IMethodInvokeExpression)varExpr).Arguments[0];
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(varExpr);
            if (ivd == null)
            {
                Error("Expected a variable reference in expression, got " + expr);
                return new List<MutationInformation.Mutation>();
            }
            if (context.InputAttributes.Has<LoopCounterAttribute>(ivd))
                return new List<MutationInformation.Mutation>();
            MutationInformation mutinfo;
            if (!mutInfos.TryGetValue(ivd, out mutinfo))
            {
                Error("Mutation information not found for " + ivd); // + " in "+msg);
                return new List<MutationInformation.Mutation>();
            }
            if (allocationsOnly)
            {
                return mutinfo.GetMutations(context, exclude, expr, true, false, bindingsInExpr, boundsInExpr, offsetInfos, extraIndicesOfStmt);
            }
            else
            {
                // in the following call, we do not exclude the current statement, to allow cyclic dependencies
                return mutinfo.GetMutations(context, null, expr, false, mustMutate, bindingsInExpr, boundsInExpr, offsetInfos, extraIndicesOfStmt);
            }
        }

        /// <summary>
        /// Keeps track of all the times a particular variable is mutated (used for dependency tracking).
        /// </summary>
        public class MutationInformation
        {
            internal class Mutation
            {
                /// <summary>
                /// the expression being mutated
                /// </summary>
                internal IExpression expr;

                /// <summary>
                /// the containing statement
                /// </summary>
                internal IStatement stmt;

                /// <summary>
                /// true if the stmt is not an OperatorStatement
                /// </summary>
                internal bool isAllocation;

                public override string ToString()
                {
                    return (isAllocation ? "alloc " : "") + expr.ToString() + ": " + stmt.ToString();
                }
            }

            // must be ordered
            internal List<Mutation> mutations = new List<Mutation>();

            internal MutationInformation()
            {
            }

            public override string ToString()
            {
                return StringUtil.ToString(mutations);
            }

            /// <summary>
            /// Registers a statement which mutates the specified expression.
            /// The statement may be null.
            /// </summary>
            /// <param name="expr"></param>
            /// <param name="stmt"></param>
            /// <param name="isAllocation"></param>
            internal void RegisterMutationStatement(IExpression expr, IStatement stmt, bool isAllocation)
            {
                Mutation m = new Mutation();
                m.expr = expr;
                m.stmt = stmt;
                m.isAllocation = isAllocation;
                if (!mutations.Contains(m))
                    mutations.Add(m);
            }

            internal bool Contains(IStatement stmt)
            {
                foreach (Mutation m in mutations)
                {
                    if (m.stmt == stmt)
                        return true;
                }
                return false;
            }

            internal enum MutationStatementType
            {
                Any,
                AllocationsOnly,
                MutationsOnly
            };

            /// <summary>
            /// Get all statements that change the value of an expression
            /// </summary>
            /// <param name="context"></param>
            /// <param name="exclude">A statement to exclude from the result</param>
            /// <param name="expr">The expression being changed.  Can contain a call to AllExcept.</param>
            /// <param name="allocationsOnly">If true, only allocation statements are returned</param>
            /// <param name="mustMutate">If true, all returned statements are known to mutate.  Otherwise, returned statements are not guaranteed to mutate</param>
            /// <param name="bindingsInExpr">Bindings of loop variables in expr</param>
            /// <param name="boundsInExpr">Bounds on loop variables in expr</param>
            /// <param name="offsetInfos">Modified on exit</param>
            /// <param name="extraIndicesOfStmt">Modified on exit</param>
            /// <returns></returns>
            internal List<Mutation> GetMutations(BasicTransformContext context, IStatement exclude, IExpression expr, bool allocationsOnly,
                bool mustMutate = false,
                IStatement bindingsInExpr = null,
                IReadOnlyDictionary<IVariableDeclaration, CodeRecognizer.Bounds> boundsInExpr = null,
                Dictionary<IStatement, IOffsetInfo> offsetInfos = null,
                Dictionary<IStatement, Set<IVariableDeclaration>> extraIndicesOfStmt = null)
            {
                List<Mutation> result = new List<Mutation>();
                IExpression ignoreExpr = null;
                if (Recognizer.IsStaticGenericMethod(expr, new Func<PlaceHolder, PlaceHolder, PlaceHolder>(FactorManager.AllExcept)))
                {
                    IMethodInvokeExpression imie = (IMethodInvokeExpression)expr;
                    expr = imie.Arguments[0];
                    IExpression ignoreIndex = imie.Arguments[1];
                    // ignore only if the index is a literal
                    if (ignoreIndex is ILiteralExpression)
                        ignoreExpr = Builder.ArrayIndex(expr, ignoreIndex);
                }
                var offsets = new OffsetInfo();
                var extraIndices = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
                var bounds = new Dictionary<IVariableDeclaration, CodeRecognizer.Bounds>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
                var allocations = new List<Mutation>();
                var mutationsOnly = new List<Mutation>();
                foreach (Mutation m in mutations)
                {
                    if (m.stmt == exclude)
                        continue;
                    if (allocationsOnly && !m.isAllocation)
                        continue;
                    if (ignoreExpr != null && !(m.expr is IVariableDeclarationExpression) &&
                        Recognizer.MutatingFirstAffectsSecond(context, m.expr, ignoreExpr, true))
                    {
                        continue;
                    }
                    offsets.Clear();
                    extraIndices.Clear();
                    bounds.Clear();
                    Recognizer.AddLoopBounds(bounds, m.stmt);
                    if (Recognizer.MutatingFirstAffectsSecond(context, m.expr, expr, false, m.stmt, bindingsInExpr, bounds, boundsInExpr, offsets, extraIndices))
                    {
                        if (m.isAllocation)
                            allocations.Add(m);
                        else
                            mutationsOnly.Add(m);
                        if (offsets.Count > 0 && offsetInfos != null)
                        {
                            offsetInfos[m.stmt] = offsets;
                            offsets = new OffsetInfo();
                        }
                        if (extraIndices.Count > 0 && extraIndicesOfStmt != null)
                        {
                            extraIndicesOfStmt[m.stmt] = extraIndices;
                            extraIndices = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
                        }
                    }
                }
                // prune allocations that are fully overwritten by some other allocation or mutation
                Set<IStatement> overwritten = new Set<IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
                foreach (Mutation m in allocations)
                {
                    IStatement ist = m.stmt;
                    DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                    foreach (IStatement init in di.Overwrites)
                    {
                        overwritten.Add(init);
                    }
                    foreach (IStatement decl in di.DeclDependencies)
                    {
                        overwritten.Add(decl);
                    }
                }
                foreach (Mutation m in allocations)
                {
                    if (overwritten.Contains(m.stmt))
                        continue;
                    // if mustMutate = false, then an allocation is covered iff it is fully contained in some mutation
                    // if mustMutate = true, then an allocation is covered if it overlaps with any mutation
                    bool isCovered = mutationsOnly.Exists(m2 => Recognizer.MutatingFirstAffectsSecond(context, m.expr, m2.expr, !mustMutate));
                    if (isCovered)
                        continue;
                    // This allocation is not fully overwritten, so it must be kept as a dependency.
                    result.Add(m);
                }
                result.AddRange(mutationsOnly);
                return result;
            }
        }

        /// <summary>
        ///  Marks variables which are loop counters (since mutation information is not tracked for such variables)
        /// </summary>
        private class LoopCounterAttribute : ICompilerAttribute
        {
        }
    }

    internal class ExpressionDependency : Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete.XExpressionStatement
    {
        public ExpressionDependency(IExpression expr)
        {
            this.Expression = expr;
        }

        public override string ToString()
        {
            return "Expr(" + base.ToString() + ")";
        }
    }

    internal class AnyStatement : Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete.XBlockStatement
    {
        public AnyStatement()
        {
        }

        public AnyStatement(params IStatement[] stmts)
        {
            Statements.AddRange(stmts);
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder("Any {");
            sb.AppendLine();
            foreach (var ist in this.Statements)
            {
                sb.Append(ist);
                sb.AppendLine();
            }
            sb.Append("}");
            return sb.ToString();
        }
    }

    internal class AllStatement : Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete.XBlockStatement
    {
        public AllStatement()
        {
        }

        public AllStatement(params IStatement[] stmts)
        {
            Statements.AddRange(stmts);
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder("All {");
            sb.AppendLine();
            foreach (var ist in this.Statements)
            {
                sb.Append(ist);
                sb.AppendLine();
            }
            sb.Append("}");
            return sb.ToString();
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}