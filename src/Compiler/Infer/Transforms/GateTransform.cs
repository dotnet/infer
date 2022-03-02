// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Handles gates in the MSL, as represented by if statements and switch statements.
    /// </summary>
    /// <remarks>
    /// This transform affects:
    ///  - references to variables declared outside of an if/switch statement
    ///    The transform replaces the variable with different clones on each branch.
    ///    A Gate.Enter factor is added at the start relating the clones to the original.
    /// 
    ///  - variables defined in all branches of an if/switch statement
    ///    The transform replaces the variable with different clones in each branch. 
    ///    A Gate.Exit factor is added at the end relating the clones to the original.
    /// </remarks>
    internal class GateTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "GateTransform"; }
        }

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            GateAnalysisTransform analysis = new GateAnalysisTransform();
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            context.Results = analysis.Context.Results;
            return base.Transform(itd);
        }

        protected IAlgorithm algorithm;

        public GateTransform(IAlgorithm algorithm)
        {
            this.algorithm = algorithm;
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            base.DoConvertMethodBody(outputs, inputs);
            if (context.Results.IsErrors()) return;
            PostProcess();
        }

        internal void PostProcess()
        {
            foreach (Dictionary<IExpression, ConditionInformation> entry in condDict.Values)
            {
                foreach (KeyValuePair<IExpression, ConditionInformation> contextEntry in entry)
                {
                    foreach (var cloneEntry in contextEntry.Value.cloneMap)
                    {
                        if (!cloneEntry.Value.IsDefinedInAllCases())
                        {
                            Error(
                                $"Random variable '{cloneEntry.Key.Expression}' is not defined (before being used) in all cases of '{contextEntry.Value.conditionLhs}'");
                        }
                    }
                }
            }

            string warningText =
                "This model will consume excess memory due to the following mix of indexing expressions inside of a conditional: {0}";
            foreach (var entry in inefficientReplacements)
            {
                if (entry.Value != null)
                {
                    Warning(string.Format(warningText, StringUtil.CollectionToString(entry.Value, ", ")));
                }
            }
        }

        /// <summary>
        /// The list of bindings corresponding to conditional statements in the input stack
        /// </summary>
        private readonly List<ConditionBinding> bindings = new List<ConditionBinding>();

        /// <summary>
        /// The list of open conditional statements on the input stack.
        /// </summary>
        private readonly List<ConditionInformation> conditionContext = new List<ConditionInformation>();

        /// <summary>
        /// A cache of the ConditionalInformation for each conditionLhs in each condition context.
        /// </summary>
        private readonly Dictionary<Set<ConditionBinding>, Dictionary<IExpression, ConditionInformation>> condDict =
            new Dictionary<Set<ConditionBinding>, Dictionary<IExpression, ConditionInformation>>();

        /// <summary>
        /// A cache of the names used for 'cases' variables, to avoid collisions
        /// </summary>
        private readonly Set<string> casesNames = new Set<string>();

        internal static bool IsLiteralOrLoopVar(BasicTransformContext context, IExpression expr, out IForStatement loop)
        {
            loop = null;
            if (expr is ILiteralExpression) return true;
            if (!(expr is IVariableReferenceExpression loopRef)) return false;

            loop = Recognizer.GetLoopForVariable(context, loopRef);
            return loop != null;
        }

        internal static ConditionBinding GetConditionBinding(IExpression condition, BasicTransformContext context,
            out IForStatement loop)
        {
            ConditionBinding binding = new ConditionBinding(condition);
            IExpression caseValue = binding.rhs;
            if (!IsLiteralOrLoopVar(context, caseValue, out loop))
            {
                // flip the binding so that literal is on right
                binding = new ConditionBinding(caseValue, binding.lhs);
            }

            return binding;
        }

        /// <summary>
        /// Converts an if statement with a stochastic condition of the form 'b' or '!b' for boolean random
        /// variable b or 'i==value' for boolean or integer random variable i. The 'value' can be a literal or
        /// a loop index.
        /// </summary>
        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            Set<ConditionBinding> conditionCaseContext = new Set<ConditionBinding>();
            conditionCaseContext.AddRange(bindings);
            IForStatement loop = null;
            ConditionBinding binding = GetConditionBinding(ics.Condition, context, out loop);
            IExpression caseValue = binding.rhs;
            if (!IsLiteralOrLoopVar(context, caseValue, out loop))
            {
                Error($"If statement condition must compare to a literal or loop counter, was: {ics.Condition}");
                return ics;
            }

            bool isStochastic = CodeRecognizer.IsStochastic(context, binding.lhs);
            IExpression caseNumber;
            bool isBinaryCondition;
            if (caseValue.GetExpressionType().Equals(typeof(bool)))
            {
                isBinaryCondition = true;
                if (caseValue is ILiteralExpression)
                {
                    bool value = (bool) ((ILiteralExpression) caseValue).Value;
                    caseNumber = value ? Builder.LiteralExpr(0) : Builder.LiteralExpr(1);
                }
                else
                    throw new Exception($"Can\'t compute {nameof(caseNumber)} of {nameof(caseValue)}={caseValue}");
            }
            else
            {
                isBinaryCondition = false;
                caseNumber = caseValue;
            }

            IExpression conditionLhs = binding.lhs;
            IExpression transformedLhs = ConvertExpression(conditionLhs);
            IExpression gateBlockKey = isStochastic ? binding.lhs : ics.Condition;
            if (isStochastic)
            {
                IVariableDeclaration missingLoopVar = GetMissingLoopVar(conditionCaseContext, ics.Condition);
                if (missingLoopVar != null)
                {
                    string inputDescription;
                    if (isBinaryCondition)
                        inputDescription = $"\'if({ics.Condition})\'";
                    else if (loop == null)
                        inputDescription = $"\'case({ics.Condition})\'";
                    else
                        inputDescription = $"\'switch({gateBlockKey})\'";
                    Error($"{inputDescription} should be placed outside \'for({missingLoopVar.Name})\'");
                }
            }

            ConditionInformation ci = GetConditionInfo(conditionCaseContext, ics, conditionLhs, transformedLhs,
                gateBlockKey, isBinaryCondition, loop);
            if (ci == null)
                return ics;
            ci.caseValue = caseValue;
            ci.caseNumber = caseNumber;
            conditionContext.Add(ci);
            int startIndex = bindings.Count;
            bindings.Add(binding);
            ConvertConditionCase(ics.Then, ci);
            if ((ics.Else != null) && (ics.Else.Statements.Count > 0))
            {
                bindings.RemoveRange(startIndex, bindings.Count - startIndex);
                binding = binding.FlipCondition();
                bindings.Add(binding);
                if (!isBinaryCondition)
                    Error($"Else clause not allowed for integer condition: {ics.Condition}.");
                ILiteralExpression ile = caseNumber as ILiteralExpression;
                if (ile == null)
                {
                    Error($"Else clause not allowed for non-literal condition: {ics.Condition}.");
                    return ics;
                }

                ci.caseValue = Builder.LiteralExpr(((int) ile.Value) == 0);
                ci.caseNumber = Builder.LiteralExpr(1 - (int) ile.Value);
                ConvertConditionCase(ics.Else, ci);
            }

            conditionContext.Remove(ci);
            bindings.RemoveRange(startIndex, bindings.Count - startIndex);
            return null;
        }

        /// <summary>
        /// Get a loop variable in the current context that is not referenced by expr or conditionCaseContext.
        /// </summary>
        /// <param name="conditionCaseContext"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
        private IVariableDeclaration GetMissingLoopVar(Set<ConditionBinding> conditionCaseContext, IExpression expr)
        {
            ICollection<IVariableDeclaration> conditionedVars = Containers.GetConditionedLoopVariables(context);
            Containers c = Containers.GetContainersNeededForExpression(context, expr);
            foreach (ConditionBinding cb in conditionCaseContext)
            {
                Containers c2 = Containers.GetContainersNeededForExpression(context, cb.GetExpression());
                c = Containers.Append(c, c2);
            }

            foreach (IForStatement ifs in context.FindAncestors<IForStatement>())
            {
                if (!c.Contains(ifs))
                {
                    IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
                    if (!conditionedVars.Contains(loopVar))
                        return loopVar;
                }
            }

            return null;
        }

        private void ConvertConditionCase(IBlockStatement inputBlock, ConditionInformation condInfo)
        {
            IBlockStatement tempBlock = ConvertBlock(inputBlock);
            tempBlock = condInfo.WrapBlockWithConditionals(context, tempBlock);
            context.AddStatementsBeforeCurrent(tempBlock.Statements);
        }

        private ConditionInformation GetConditionInfo(Set<ConditionBinding> conditionCaseContext,
            IConditionStatement conditionStmt,
            IExpression conditionLhs, IExpression transformedLhs, IExpression gateBlockKey, bool isBinaryCondition,
            IForStatement loop)
        {
            //if (!context.OutputAttributes.Has<Stochastic>(condvd)) Error("Conditions must be stochastic, was : " + condition);
            if (!condDict.TryGetValue(conditionCaseContext, out var dict))
            {
                dict = new Dictionary<IExpression, ConditionInformation>();
                condDict[conditionCaseContext] = dict;
            }

            if (!dict.TryGetValue(gateBlockKey, out var conditionInfo))
            {
                bool isStochastic = CodeRecognizer.IsStochastic(context, conditionLhs);
                var gateBlock = context.InputAttributes.Get<GateBlock>(conditionStmt);
                IExpression numberOfCases =
                    isStochastic ? GetNumberOfCases(conditionLhs, isBinaryCondition, loop, gateBlock) : null;
                ConditionInformation parent =
                    (conditionContext.Count == 0) ? null : conditionContext[conditionContext.Count - 1];
                conditionInfo = new ConditionInformation(algorithm, conditionLhs, numberOfCases);
                conditionInfo.gateBlock = gateBlock;
                conditionInfo.parent = parent;
                conditionInfo.isStochastic = isStochastic;
                dict[gateBlockKey] = conditionInfo;
                // must set these before Build
                conditionInfo.conditionStmt = conditionStmt;
                conditionInfo.switchLoop = loop;
                IVariableDeclaration conditionVar = Recognizer.GetVariableDeclaration(transformedLhs);
                if (isStochastic)
                {
                    VariableInformation vi = VariableInformation.GetVariableInformation(context, conditionVar);
                    List<IList<IExpression>> indices = Recognizer.GetIndices(transformedLhs);
                    IList<IStatement> stmts = Builder.StmtCollection();
                    string selectorName = $"{transformedLhs}_selector";
                    selectorName = VariableInformation.GenerateName(context, selectorName);
                    IVariableDeclaration selectorDecl = vi.DeriveIndexedVariable(stmts, context, selectorName, indices);
                    if (!context.InputAttributes.Has<DerivedVariable>(selectorDecl))
                        context.OutputAttributes.Set(selectorDecl, new DerivedVariable());
                    conditionInfo.selector = Builder.VarRefExpr(selectorDecl);
                    IExpression copy = Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder>(Clone.Copy),
                        new Type[] {transformedLhs.GetExpressionType()}, transformedLhs);
                    stmts.Add(Builder.AssignStmt(conditionInfo.selector, copy));
                    conditionInfo.AddCasesStatements(context, stmts);
                    conditionInfo.casesArray = conditionInfo.BuildCasesArray(casesNames, context, this);
                }
                else
                {
                    conditionInfo.selector = transformedLhs;
                }
            }
            else
            {
                conditionInfo.conditionStmt = conditionStmt;
                conditionInfo.switchLoop = loop;
            }

            return conditionInfo;
        }

        private IExpression GetNumberOfCases(IExpression expr, bool isBinaryCondition, IForStatement loop,
            GateBlock gateBlock)
        {
            if (isBinaryCondition)
                return Builder.LiteralExpr(2);
            if (loop != null)
                return Recognizer.LoopSizeExpression(loop);
            // find the maximum integer case value
            int maxCaseValue = 0;
            foreach (IExpression caseValue in gateBlock.caseValues)
            {
                int caseInt = (int) ((ILiteralExpression) caseValue).Value;
                maxCaseValue = System.Math.Max(maxCaseValue, caseInt);
            }

            return Builder.LiteralExpr(maxCaseValue + 1);
        }

        protected override IStatement DoConvertStatement(IStatement ist)
        {
            // if a statement was generated by a ConditionInformation, transform it again under the parent ConditionInformation
            ConditionInformation condInfo = context.OutputAttributes.Get<ConditionInformation>(ist);
            List<ConditionInformation> removed = null;
            if (condInfo != null)
            {
                int start = conditionContext.IndexOf(condInfo);
                if (start >= 0)
                {
                    removed = new List<ConditionInformation>();
                    for (int i = start; i < conditionContext.Count; i++)
                    {
                        removed.Add(conditionContext[i]);
                    }

                    conditionContext.RemoveRange(start, conditionContext.Count - start);
                }
            }

            IStatement st = base.DoConvertStatement(ist);
            if (removed != null)
            {
                // restore previous context
                //conditionContext.Add(condInfo);
                conditionContext.AddRange(removed);
            }

            return st;
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            IExpression expr = base.ConvertAssign(iae);
            if (expr is IAssignExpression ae && ae.Expression is IMethodInvokeExpression imie)
            {
                // Check for DerivedVariable attributes again, as in ModelAnalysisTransform.
                // This is necessary when statements are re-transformed by outer conditionals, changing the target of an assignment.
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(ae.Target);
                if (ivd != null)
                {
                    FactorManager.FactorInfo info = CodeRecognizer.GetFactorInfo(context, imie);
                    if (info != null && info.IsDeterministicFactor &&
                        !context.InputAttributes.Has<DerivedVariable>(ivd))
                    {
                        context.InputAttributes.Set(ivd, new DerivedVariable());
                    }
                }
            }

            return expr;
        }

        protected override IVariableDeclaration ConvertVariableDecl(IVariableDeclaration ivd)
        {
            if (conditionContext.Count > 0 && !context.InputAttributes.Has<ConditionInformation>(ivd))
            {
                ConditionInformation innermostCloning = conditionContext[conditionContext.Count - 1];
                context.InputAttributes.Set(ivd, innermostCloning);
            }

            context.InputAttributes.Remove<Containers>(ivd);
            context.InputAttributes.Set(ivd, new Containers(context));
            return ivd;
        }

        /// <summary>
        /// Converts indexed variable references inside if/switch statements by cloning them and adding
        /// Enter/Exit factors as necessary.
        /// </summary>
        /// <param name="iaie"></param>
        /// <returns></returns>
        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            // must compute isDef before iaie is modified!
            bool isDef = Recognizer.IsBeingMutated(context, iaie);
            //iaie = (IArrayIndexerExpression)ConvertIndices(iaie);
            if (conditionContext.Count == 0) return base.ConvertArrayIndexer(iaie);
            return ReplaceWithClone(iaie, isDef);
        }

        /// <summary>
        /// Converts the indices in all brackets of expr, but not the innermost target.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        protected IExpression ConvertIndices(IExpression expr)
        {
            if (!(expr is IArrayIndexerExpression iaie)) return expr;
            IArrayIndexerExpression aie = Builder.ArrayIndxrExpr();
            foreach (IExpression exp in iaie.Indices)
            {
                aie.Indices.Add(exp); // do not convert indices
            }

            aie.Target = ConvertIndices(iaie.Target);
            return aie;
        }

        /// <summary>
        /// Converts variable references inside if/switch statements by cloning them and adding
        /// Enter/Exit factors as necessary.
        /// </summary>
        /// <param name="ivre"></param>
        /// <returns></returns>
        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            if (conditionContext.Count == 0) return ivre;
            bool isDef = Recognizer.IsBeingMutated(context, ivre);
            return ReplaceWithClone(ivre, isDef);
        }

        protected override IExpression ConvertArrayCreate(IArrayCreateExpression iace)
        {
            return iace;
        }

        protected IExpression ReplaceWithClone(IExpression expr, bool isDef)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
            if (ivd == null) return expr;
            if (!CodeRecognizer.IsStochastic(context, ivd)) return expr;
            ConditionInformation ciDecl = context.InputAttributes.Get<ConditionInformation>(ivd);
            int start = (ciDecl == null) ? 0 : (conditionContext.IndexOf(ciDecl) + 1);
            int conditionContextIndex = conditionContext.Count - 1;
            return ReplaceWithClone(expr, isDef, ivd, start, conditionContextIndex);
        }

        /// <summary>
        /// Replace expr with a reference to the clone array, substituting any indices that match wildcards.
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="isDef"></param>
        /// <param name="ivd"></param>
        /// <param name="start"></param>
        /// <param name="conditionContextIndex"></param>
        /// <returns></returns>
        /// <remarks>
        /// For example, if expr = a[0,j][1,k] and the gateBlock has a pattern a[*,j][*,k] then this function
        /// returns clone[0][1][caseNumber].
        /// </remarks>
        internal IExpression ReplaceWithClone(IExpression expr, bool isDef, IVariableDeclaration ivd, int start,
            int conditionContextIndex)
        {
            if (conditionContextIndex < start)
                return expr;
            ConditionInformation ci = conditionContext[conditionContextIndex];
            // match the expr against the defs/uses of the GateBlock
            bool replaced = false;
            if (ci.gateBlock == null)
            {
                Error($"Cannot find GateBlock for {ci}");
                return expr;
            }

            ExpressionWithBindings eb = new ExpressionWithBindings()
            {
                Expression = expr,
                Bindings = Set<IReadOnlyCollection<ConditionBinding>>.FromEnumerable(new[] {bindings}),
            };
            if (ci.gateBlock.variablesDefined.TryGetValue(ivd, out var definedExpression) &&
                GateAnalysisTransform.CouldOverlap(definedExpression, eb))
            {
                expr = ReplaceWithClone(expr, isDef, ivd, start, conditionContextIndex, ci, definedExpression,
                    out replaced);
                if (isDef && !replaced)
                {
                    IExpression boundDef =
                        GateAnalysisTransform.ReplaceExpression(bindings, definedExpression.Expression);
                    Error($"{expr} doesn\'t match bound GateBlock def: {boundDef}");
                    return expr;
                }
            }

            if (!replaced)
            {
                if (ci.gateBlock.variablesUsed.TryGetValue(ivd, out var expressionsUsingVariable))
                {
                    foreach (ExpressionWithBindings usedExpression in expressionsUsingVariable)
                    {
                        if (GateAnalysisTransform.CouldOverlap(usedExpression, eb))
                        {
                            expr = ReplaceWithClone(expr, isDef, ivd, start, conditionContextIndex, ci, usedExpression,
                                out replaced);
                            if (replaced)
                                break;
                        }
                    }
                }
            }

            if (!replaced)
                Error($"Could not find a match for {expr} in GateBlock: {ci.gateBlock}");
            return expr;
        }

        private IExpression ReplaceWithClone(IExpression expr, bool isDef, IVariableDeclaration ivd, int start,
            int conditionContextIndex, ConditionInformation ci,
            ExpressionWithBindings toClone, out bool replaced)
        {
            replaced = false;
            // apply the current bindings to eb and expr
            IExpression toClone2 = GateAnalysisTransform.ReplaceExpression(bindings, toClone.Expression);
            // toReplace is the prefix of expr that we need to replace.
            // if expr is a[index][0] and toClone2 is a[i], toReplace is a[index]
            // note expr has at least as many indexing brackets as toReplace
            // Example:
            // expr = a[i,j][k,l]
            // toClone2 = a[*,j][*,l]
            // new expr = clone[i][k]
            IExpression toReplace = GetMatchingPrefix(expr, toClone2, out var indices);
            if (toReplace != null)
            {
                var currentBindings = bindings.Take(conditionContextIndex + 1).ToList();
                IExpression clone = ci.GetClone(context, toClone, currentBindings, this, start, conditionContextIndex, isDef);
                clone = Builder.JaggedArrayIndex(clone, indices);
                // check if indices contains an expression that is not a top-level loop variable or loop local
                bool isSubset = ci.isStochastic && indices.Any(bracket => bracket.Any(index =>
                {
                    // Based on GateAnalysisTransform.ContainsLocalVars
                    bool containsLocalVars = Recognizer.GetVariables(index).Any(indexVar =>
                        context.InputAttributes.Get<GateBlock>(indexVar) == ci.gateBlock);
                    if (containsLocalVars)
                    {
                        // check for dependence on variables tagged with GateBlock that do not depend on a inner loop
                        return Recognizer.GetVariables(index).Any(indexVar =>
                                context.InputAttributes.Get<GateBlock>(indexVar) == ci.gateBlock &&
                                Recognizer.GetLoopForVariable(context, indexVar) == null);
                    }
                    else
                    {
                        var indexVarDecl = Recognizer.GetVariableDeclaration(index);
                        return indexVarDecl == null;
                    }
                }));
                RecordReplacement(expr, toClone, !isSubset);

                int replaceCount = 0;
                expr = Builder.ReplaceExpression(expr, toReplace, clone, ref replaceCount);
                replaced = true;
            }

            return expr;
        }

        /// <summary>
        /// A null list indicates that at least one replacement was efficient.
        /// </summary>
        readonly Dictionary<ExpressionWithBindings, List<IExpression>> inefficientReplacements =
            new Dictionary<ExpressionWithBindings, List<IExpression>>();

        private void RecordReplacement(IExpression expr, ExpressionWithBindings toClone, bool isEfficient)
        {
            if (isEfficient)
            {
                inefficientReplacements[toClone] = null;
            }
            else
            {
                if (!inefficientReplacements.TryGetValue(toClone, out var replacements))
                {
                    replacements = new List<IExpression>();
                    inefficientReplacements[toClone] = replacements;
                }

                replacements?.Add(expr);
            }
        }

        /// <summary>
        /// Returns a prefix of expr1 that matches expr2, where anything matches AnyItem.  The indices that matched AnyItem are returned.
        /// </summary>
        /// <param name="expr1"></param>
        /// <param name="expr2"></param>
        /// <param name="indices">Contains indices of <paramref name="expr1"/> that were replaced by wildcards in <paramref name="expr2"/></param>
        /// <returns></returns>
        private static IExpression GetMatchingPrefix(IExpression expr1, IExpression expr2,
            out List<IEnumerable<IExpression>> indices)
        {
            indices = new List<IEnumerable<IExpression>>();
            List<IExpression> prefixes1 = Recognizer.GetAllPrefixes(expr1);
            List<IExpression> prefixes2 = Recognizer.GetAllPrefixes(expr2);
            if (!prefixes1[0].Equals(prefixes2[0])) return null;
            IList<IExpression> bracket = Builder.ExprCollection();
            int count = System.Math.Min(prefixes1.Count, prefixes2.Count);
            for (int i = 1; i < count; i++)
            {
                IExpression prefix1 = prefixes1[i];
                IExpression prefix2 = prefixes2[i];
                if (prefix1 is IArrayIndexerExpression iaie1)
                {
                    if (prefix2 is IArrayIndexerExpression iaie2)
                    {
                        if (iaie1.Indices.Count != iaie2.Indices.Count) return prefixes1[i - 1];
                        for (int ind = 0; ind < iaie1.Indices.Count; ind++)
                        {
                            IExpression index1 = iaie1.Indices[ind];
                            IExpression index2 = iaie2.Indices[ind];
                            if (Recognizer.IsStaticMethod(index2, new Func<int>(GateAnalysisTransform.AnyIndex)))
                            {
                                bracket.Add(index1);
                            }
                            else if (!index1.Equals(index2))
                            {
                                return null; // indices don't match
                            }
                        }
                    }
                }

                if (bracket.Count > 0)
                {
                    indices.Add(bracket);
                    bracket = Builder.ExprCollection();
                }
            }

            return prefixes1[count - 1];
        }
    }

    /// <summary>
    /// For an if or switch statement, keeps tracks of variables that are cloned and other information.
    /// </summary>
    internal class ConditionInformation : ICompilerAttribute
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        internal static CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Helps recognize code patterns
        /// </summary>
        private static readonly CodeRecognizer Recognizer = CodeRecognizer.Instance;

        internal GateBlock gateBlock;
        internal readonly IAlgorithm algorithm;
        internal IConditionStatement conditionStmt;
        internal readonly IExpression conditionLhs;
        internal readonly IExpression numberOfCases;
        internal bool isStochastic;

        private bool isBinaryCondition
        {
            get { return conditionLhs.GetExpressionType().Equals(typeof (bool)); }
        }

        /// <summary>
        /// The zero-based index of this case in the gate block
        /// </summary>
        internal IExpression caseNumber;

        /// <summary>
        /// The value that the conditionLhs must equal in this case of the gate block.  Only used for ToString.
        /// </summary>
        internal IExpression caseValue;

        /// <summary>
        /// A bool[] used for collecting evidence and Gate.Exit
        /// </summary>
        internal IVariableDeclaration casesArray;

        internal IExpression selector;
        internal Dictionary<IExpression, IVariableDeclaration> casesVars = new Dictionary<IExpression, IVariableDeclaration>();

        /// <summary>
        /// Map from an input expression to its clone information.
        /// </summary>
        internal Dictionary<ExpressionWithBindings, ClonedVarInfo> cloneMap = new Dictionary<ExpressionWithBindings, ClonedVarInfo>();

        /// <summary>
        /// True if the condition rhs is a loop variable.
        /// </summary>
        internal bool isSwitch
        {
            get { return (switchLoop != null); }
        }

        /// <summary>
        /// If isSwitch=true, the loop defining the loop variable in the conditionLhs.
        /// </summary>
        internal IForStatement switchLoop;

        /// <summary>
        /// True if the number of cases is known at compile time.
        /// </summary>
        private bool isFixedSize
        {
            get { return (switchLoop == null); }
        }

        internal ConditionInformation parent = null;

        internal void AddCasesStatements(BasicTransformContext context, IList<IStatement> stmts)
        {
            int ancIndex = context.GetAncestorIndex(isSwitch ? (IStatement) switchLoop : conditionStmt);
            context.AddStatementsBeforeAncestorIndex(ancIndex, stmts, false);
        }

        internal void AddCloneStatements(BasicTransformContext context, IList<IStatement> stmts)
        {
            // add 2 to reach the statement just inside the current condition statement.
            int ancIndex = context.GetAncestorIndex(conditionStmt) + 2;
            context.AddStatementsBeforeAncestorIndex(ancIndex, stmts, false);
        }

        public override string ToString()
        {
            string s = caseValue == null ? conditionLhs.ToString() : $"{conditionLhs}={caseValue}";
            if (parent == null)
                return s;
            else
                return $"{s},{parent}";
        }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="algorithm"></param>
        /// <param name="conditionLhs"></param>
        /// <param name="numberOfCases"></param>
        internal ConditionInformation(IAlgorithm algorithm, IExpression conditionLhs, IExpression numberOfCases)
        {
            this.algorithm = algorithm;
            this.conditionLhs = conditionLhs;
            this.numberOfCases = numberOfCases;
        }

        //public bool EqualsOrIsParentOf(ConditionInformation that)
        //{
        //  if (that == null) return false;
        //  else if (Equals(that)) return true;
        //  else return EqualsOrIsParentOf(that.parent);
        //}

        //public override bool Equals(object obj)
        //{
        //  ConditionInformation ci = obj as ConditionInformation;
        //  if (ci == null)
        //    return false;
        //  return ci.conditionLhs.Equals(conditionLhs);
        //}

        //public override int GetHashCode()
        //{
        //  return conditionLhs.GetHashCode();
        //}

        internal IVariableDeclaration BuildCasesArray(Set<string> usedNames, BasicTransformContext context, IExpressionTransform expressionTransform)
        {
            IVariableDeclaration conditionVar = Recognizer.GetVariableDeclaration(selector);
            // Add cases statement
            string basename = CodeBuilder.MakeValid(selector.ToString()) + "_cases";
            string name = basename;
            int ct = 0;
            while (usedNames.Contains(name))
            {
                name = basename + ct;
                ct++;
            }
            usedNames.Add(name);
            IList<IStatement> stmts = Builder.StmtCollection();
            IVariableDeclaration casesArray = null;
            bool optimiseBinaryConditions = false;
            if (optimiseBinaryConditions && isBinaryCondition)
            {
                // This 'optimisation' is not actually saving time, memory or reducing the length of the generated code :(
                List<IExpression> args = new List<IExpression>();
                args.Add(selector);
                for (int i = 0; i < 2; i++)
                {
                    IVariableDeclaration casesVar = Builder.VarDecl(name + i, typeof (bool));
                    casesVars[Builder.LiteralExpr(i)] = casesVar;
                    VariableInformation vi = VariableInformation.GetVariableInformation(context, casesVar);
                    vi.IsStochastic = isStochastic;
                    context.OutputAttributes.Set(casesVar, new DerivedVariable());
                    context.OutputAttributes.Set(casesVar, new DoNotSendEvidence());
                    stmts.Add(Builder.ExprStatement(Builder.VarDeclExpr(casesVar)));
                    args.Add(Builder.VarRefExpr(casesVar));
                }
                IMethodInvokeExpression imie = Builder.StaticMethod(new Models.ActionOut2<bool, bool, bool>(Gate.CasesBool), args.ToArray());
                //context.OutputAttributes.Set(imie, new Stochastic()); // for IfCuttingTransform
                stmts.Add(Builder.ExprStatement(imie));
            }
            else
            {
                casesArray = Builder.VarDecl(name, typeof (bool[]));
                VariableInformation vi = VariableInformation.GetVariableInformation(context, casesArray);
                vi.SetSizesAtDepth(0, new IExpression[] {numberOfCases});
                ValueRange valueRange = context.InputAttributes.Get<ValueRange>(conditionVar);
                if(valueRange != null)
                {
                    IVariableDeclaration indexVar = valueRange.Range.GetIndexDeclaration();
                    vi.SetIndexVariablesAtDepth(0, new IVariableDeclaration[] { indexVar });
                }
                vi.IsStochastic = isStochastic;
                context.OutputAttributes.Set(casesArray, new DerivedVariable());
                context.OutputAttributes.Set(casesArray, new DoNotSendEvidence());
                context.OutputAttributes.Set(casesArray, new DivideMessages(false));
                if (conditionVar != null)
                {
                    context.InputAttributes.CopyObjectAttributesTo<GroupMember>(conditionVar, context.OutputAttributes, casesArray);
                    //context.InputAttributes.CopyObjectAttributesTo<DivideMessages>(conditionVar, context.OutputAttributes, casesArray);
                    context.InputAttributes.CopyObjectAttributesTo<TraceMessages>(conditionVar, context.OutputAttributes, casesArray);
                    //context.OutputAttributes.Set(conditionVar, attr);
                    //context.OutputAttributes.Set(transformedConditionVar, new CancelTrigger() { trigger = attr });
                }
                ChannelTransform.setAllGroupRoots(context, casesArray, false);
                IExpression ivde = expressionTransform.ConvertExpression(Builder.VarDeclExpr(casesArray));
                IStatement casesDecl = Builder.AssignStmt(ivde, Builder.ArrayCreateExpr(typeof (bool), numberOfCases));
                Delegate d;
                IMethodInvokeExpression imie = null;
                if (isBinaryCondition)
                {
                    d = new Func<bool, bool[]>(Gate.Cases);
                    imie = Builder.StaticMethod(d, selector);
                }
                else
                {
                    d = new Func<int, int, bool[]>(Gate.CasesInt);
                    imie = Builder.StaticMethod(d, selector, numberOfCases);
                }
                //context.OutputAttributes.Set(imie, new Stochastic());  // for IfCuttingTransform
                IStatement cases = Builder.AssignStmt(Builder.VarRefExpr(casesArray), imie);
                stmts.Add(casesDecl);
                stmts.Add(cases);
            }
            AddCasesStatements(context, stmts);
            return casesArray;
        }

        /// <summary>
        /// Returns the appropriate clone of a variable to use inside the current gate.  The clone
        /// will be created if necessary.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="eb"></param>
        /// <param name="currentBindings"></param>
        /// <param name="transform"></param>
        /// <param name="start"></param>
        /// <param name="conditionContextIndex">An index into the conditionContext array.</param>
        /// <param name="isDef"></param>
        /// <returns></returns>
        internal IExpression GetClone(BasicTransformContext context, ExpressionWithBindings eb, ICollection<ConditionBinding> currentBindings, GateTransform transform, int start, int conditionContextIndex, bool isDef)
        {
            IExpression expr = eb.Expression;
            // expr should not contain the conditionLhs
            if (Builder.ContainsExpression(expr, conditionLhs))
                context.Error($"Internal: expr ({expr}) contains the conditionLhs ({conditionLhs})");
            ClonedVarInfo cvi = GetClonedVarInfo(context, eb, isDef);
            if (cvi == null)
            {
                return expr;
            }
            if (cvi.arrayDecl == null)
            {
                var extraBindings = IndexingTransform.FilterBindingSet(eb.Bindings, 
                    binding => !currentBindings.Contains(binding) && !CodeRecognizer.IsStochastic(context, binding.lhs)
                );
                CreateCloneArray(cvi, context, transform, start, conditionContextIndex, extraBindings);
            }
            return cvi.GetCloneForCase(context, this, isDef, caseNumber);
        }

        private ClonedVarInfo GetClonedVarInfo(BasicTransformContext context, ExpressionWithBindings eb, bool isDef)
        {
            ClonedVarInfo cvi;
            if (cloneMap.TryGetValue(eb, out cvi)) return cvi;
            // must create
            IExpression expr = eb.Expression;
            Type exprType = expr.GetExpressionType();
            if (exprType == null)
            {
                context.Error($"Could not determine type of expression: {expr}");
                return null;
            }
            cvi = new ClonedVarInfo(expr, exprType, isDef);
            cloneMap[eb] = cvi;
            return cvi;
        }

        /// <summary>
        /// Replace wildcards in an expression with loop indices and return the corresponding loops
        /// </summary>
        /// <param name="context"></param>
        /// <param name="expr"></param>
        /// <param name="indices"></param>
        /// 
        /// <returns></returns>
        public IExpression ReplaceAnyItem(BasicTransformContext context, IExpression expr, List<IList<IExpression>> indices)
        {
            if (expr is IArrayIndexerExpression iaie)
            {
                IExpression result = ReplaceAnyItem(context, iaie.Target, indices);
                IList<IExpression> newIndices = Builder.ExprCollection();
                IList<IExpression> allIndices = Builder.ExprCollection();
                for (int i = 0; i < iaie.Indices.Count; i++)
                {
                    IExpression index = iaie.Indices[i];
                    if (Recognizer.IsStaticMethod(index, new Func<int>(GateAnalysisTransform.AnyIndex)))
                    {
                        IExpression newIndex = GetIndexVar(context, result, i);
                        newIndices.Add(newIndex);
                        allIndices.Add(newIndex);
                    }
                    else allIndices.Add(index);
                }
                if (newIndices.Count > 0) indices.Add(newIndices);
                return Builder.ArrayIndex(result, allIndices);
            }
            else return expr;
        }

        public IVariableReferenceExpression GetIndexVar(BasicTransformContext context, IExpression expr, int i)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
            if (ivd == null) throw new Exception($"Could not get variable declaration for {expr}");
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
            int d = Recognizer.GetIndexingDepth(expr);
            while (vi.indexVars.Count <= d)
            {
                vi.indexVars.Add(new IVariableDeclaration[vi.sizes[vi.indexVars.Count].Length]);
            }
            IVariableDeclaration v = vi.indexVars[d][i];
            if (v == null || Recognizer.GetLoopForVariable(context, v, conditionStmt) != null)
            {
                v = VariableInformation.GenerateLoopVar(context, "_gi");
                if (vi.indexVars[d][i] == null) vi.indexVars[d][i] = v;
            }
            return Builder.VarRefExpr(v);
        }

        /// <summary>
        /// Creates a clone of a variable which is entering or exiting a Gate.
        /// </summary>
        /// <param name="cvi">Cloned variable information</param>
        /// <param name="context"></param>
        /// <param name="transform"></param>
        /// <param name="start"></param>
        /// <param name="bindingSet"></param>
        /// <param name="conditionContextIndex">An index into the conditionContext array.</param>
        /// <returns></returns>
        private void CreateCloneArray(ClonedVarInfo cvi, BasicTransformContext context, GateTransform transform, int start, int conditionContextIndex,
                                      Set<IReadOnlyCollection<ConditionBinding>> bindingSet)
        {
            IExpression expr = cvi.expr;
            Type exprType = cvi.exprType;
            bool isExitVar = cvi.IsExitVar;
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);

            // this statement fills in cvi.indices
            IExpression indexedExpr = ReplaceAnyItem(context, expr, cvi.wildcardVars);
            IExpression transformedExpr;
            IVariableDeclaration transformedVar;
            if (conditionContextIndex > start)
            {
                transformedExpr = transform.ReplaceWithClone(indexedExpr, isExitVar, ivd, start, conditionContextIndex - 1);
                transformedVar = Recognizer.GetVariableDeclaration(transformedExpr);
            }
            else
            {
                transformedExpr = indexedExpr;
                transformedVar = ivd;
            }

            bool isIndexedByCondition = (isSwitch && Builder.ContainsExpression(expr, caseNumber));

            // construct cvi.containers from eb.Bindings
            // if cloning outside the loop, must treat the loop variable as a local variable and exclude it from conditions
            bool cloneOutsideLoop = (isSwitch && !isIndexedByCondition);
            IStatement bindingContainer = IndexingTransform.GetBindingSetContainer(IndexingTransform.FilterBindingSet(bindingSet,
                    binding => !cloneOutsideLoop || !Builder.ContainsExpression(binding.GetExpression(), caseNumber)));
            if (bindingContainer != null) cvi.containers.Add(bindingContainer);

            if (!isStochastic)
            {
                cvi.arrayDecl = transformedVar;
                cvi.arrayRef = transformedExpr; //indexedExpr;
                if (isExitVar)
                {
                    // TODO: this attribute should preferably be attached in GetCloneForCase
                    if(!context.OutputAttributes.Has<DerivedVariable>(cvi.arrayDecl))
                        context.OutputAttributes.Set(cvi.arrayDecl, new DerivedVariable());
                }
                return;
            }

            // An array of all the uses of this variable in each case
            IExpression arraySize;
            if (isExitVar) arraySize = numberOfCases; // exit variables must be defined in all cases of the condition.
            else if (isIndexedByCondition) arraySize = null; // used in exactly one case.
            else if (isFixedSize) arraySize = cvi.caseCount; // used in some cases.
            else arraySize = numberOfCases; // used in all cases.
            cvi.IsEnterOne = (arraySize == null);

            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);

            List<IList<IExpression>> exprIndices = Recognizer.GetIndices(expr);
            IList<IStatement> stBefore = Builder.StmtCollection();
            IList<IStatement> stAfter = Builder.StmtCollection();
            string name = ToString(transformedExpr);
            name = vi.Name + "_";
            foreach (IList<IExpression> bracket in exprIndices)
            {
                foreach (IExpression index in bracket)
                {
                    if (!Recognizer.IsStaticMethod(index, new Func<int>(GateAnalysisTransform.AnyIndex)))
                    {
                        name += $"{index}_";
                    }
                }
            }
            name = VariableInformation.GenerateName(context, name);
            IVariableDeclaration indexVar;
            if (isSwitch) indexVar = Recognizer.GetVariableDeclaration(caseNumber);
            else
            {
                indexVar = VariableInformation.GenerateLoopVar(context, "_gateind");
            }
            IExpression[][] arraySizes = (arraySize == null) ? null : new IExpression[][] { new[] { arraySize } };
            IVariableDeclaration[][] indexVars = new IVariableDeclaration[][] { new[] { indexVar } };
            // Cannot set useLiteralIndices=true because arraySize is initially zero and gets incremented later.
            cvi.arrayDecl = vi.DeriveArrayVariable(stBefore, context,
                                                   name, arraySizes, indexVars, exprIndices, cvi.wildcardVars, useArrays: true);
            context.InputAttributes.Remove<ConditionInformation>(cvi.arrayDecl);
            context.InputAttributes.Set(cvi.arrayDecl, this);
            if (!context.InputAttributes.Has<DerivedVariable>(cvi.arrayDecl))
                context.OutputAttributes.Set(cvi.arrayDecl, new DerivedVariable());
            var arrayInformation = VariableInformation.GetVariableInformation(context, cvi.arrayDecl);
            List<IStatement> wildcardLoops = arrayInformation.BuildWildcardLoops(cvi.wildcardVars);

            cvi.arrayRef = Builder.VarRefExpr(cvi.arrayDecl);
            IExpression arrayRef = Builder.JaggedArrayIndex(cvi.arrayRef, cvi.wildcardVars);
            if (isExitVar)
            {
                // *************** Gate Exit case ***********************
                if (isFixedSize)
                {
                    // keep track of which cases the variable is defined in.
                    IArrayCreateExpression ace = Builder.ArrayCreateExpr(typeof (int), arraySize);
                    ace.Initializer = Builder.BlockExpr();
                    cvi.caseNumbers = ace;
                }
                // the clone array will only be used by Gate.Exit
                context.OutputAttributes.Set(cvi.arrayDecl, new SuppressVariableFactor());
                // VMP requires special treatment of exit variables.  
                // For backwards compatibility, VMP supports the useExitRandom flag, which requires
                // a special treatment of exit variables as well as a new Gate.ExitRandom operator.
                // This is done by attaching the attribute GateExitingVariable to the clones array.
                // ChannelTransform later reads this attribute.
                bool useExitRandom =
                    (algorithm is VariationalMessagePassing vmp) &&
                     vmp.UseGateExitRandom;
                // if using Gate.ExitRandom, the clones should be marked as GateExiting variables
                if (useExitRandom)
                    context.OutputAttributes.Set(cvi.arrayDecl, new VariationalMessagePassing.GateExitRandomVariable());
                // want the grouping to come from the definition, not the variable being cloned
                context.OutputAttributes.Remove<GroupMember>(cvi.arrayDecl);
                // TODO: insert argument types
                Delegate d;
                IExpression exitMethod;
                if (casesArray == null)
                {
                    if (useExitRandom)
                    {
                        d = new Func<bool, bool, PlaceHolder[], PlaceHolder>(Gate.ExitRandomTwo<PlaceHolder>);
                    }
                    else
                    {
                        d = new Func<bool, bool, PlaceHolder[], PlaceHolder>(Gate.ExitTwo<PlaceHolder>);
                    }
                    exitMethod = Builder.StaticGenericMethod(d, new Type[] {exprType},
                                                             Builder.VarRefExpr(casesVars[Builder.LiteralExpr(0)]),
                                                             Builder.VarRefExpr(casesVars[Builder.LiteralExpr(1)]),
                                                             arrayRef);
                }
                else
                {
                    if (useExitRandom)
                    {
                        d = new Func<bool[], PlaceHolder[], PlaceHolder>(Gate.ExitRandom);
                    }
                    else
                    {
                        d = new Func<bool[], PlaceHolder[], PlaceHolder>(Gate.Exit);
                    }
                    // because Gate.Exit sends a message to cases, we must use casesArray not casesArrayEnter
                    IExpression casesArrayRef = Builder.VarRefExpr(casesArray);
                    exitMethod = Builder.StaticGenericMethod(d, new Type[] {exprType}, casesArrayRef, arrayRef);
                }

                IStatement exit = Builder.AssignStmt(transformedExpr, exitMethod);
                exit = Containers.WrapWithContainers(exit, wildcardLoops);
                stAfter.Add(exit);
                if (!useExitRandom && !context.InputAttributes.Has<DerivedVariable>(transformedVar))
                    context.OutputAttributes.Set(transformedVar, new DerivedVariable());
            }
            else
            {
                // *************** Gate Enter case ***********************
                // if the expression is indexed by the condition variable, we know that:
                // 1. The enclosing block is a switch statement.
                // 2. If the same expression appears in another branch of the switch, it must refer to a different variable.
                // 3. Since literal indexing is not (yet) allowed on input, the variable denoted by this expression cannot appear in any other branch.
                // 4. Therefore can use EnterOne to enter this variable.
                IExpression enterMethod;
                Type selectorType = selector.GetExpressionType();
                if (isIndexedByCondition)
                {
                    // used in exactly one case.
                    enterMethod = Builder.StaticGenericMethod(new Func<int, PlaceHolder, int, PlaceHolder>(Gate.EnterOne),
                                                              new Type[] {exprType}, selector, transformedExpr, caseNumber);
                }
                else if (isFixedSize)
                {
                    // used in some cases.
                    IArrayCreateExpression ace = Builder.ArrayCreateExpr(typeof (int), cvi.caseCount);
                    ace.Initializer = Builder.BlockExpr();
                    cvi.caseNumbers = ace;
                    if (casesArray == null)
                    {
                        IExpression case0Expr = Builder.VarRefExpr(casesVars[Builder.LiteralExpr(0)]);
                        IExpression case1Expr = Builder.VarRefExpr(casesVars[Builder.LiteralExpr(1)]);
                        enterMethod = Builder.StaticGenericMethod(
                            new Func<bool, bool, PlaceHolder, int[], PlaceHolder[]>(Gate.EnterPartialTwo<PlaceHolder>),
                            new Type[] {exprType}, case0Expr, case1Expr, transformedExpr, cvi.caseNumbers);
                        // attach an attribute to indicate that these expressions no longer need to be cloned by this ConditionInformation.
                        context.InputAttributes.Remove<ConditionInformation>(case0Expr);
                        context.InputAttributes.Set(case0Expr, this);
                        // attach an attribute to indicate that these expressions no longer need to be cloned by this ConditionInformation.
                        context.InputAttributes.Remove<ConditionInformation>(case1Expr);
                        context.InputAttributes.Set(case1Expr, this);
                    }
                    else
                    {
                        if (selectorType.Equals(typeof (int)))
                        {
                            enterMethod = Builder.StaticGenericMethod(
                                new Func<int, PlaceHolder, int[], PlaceHolder[]>(Gate.EnterPartial<PlaceHolder>),
                                new Type[] {exprType}, selector, transformedExpr, cvi.caseNumbers);
                        }
                        else if (selectorType.Equals(typeof (bool)))
                        {
                            enterMethod = Builder.StaticGenericMethod(
                                new Func<bool, PlaceHolder, int[], PlaceHolder[]>(Gate.EnterPartial<PlaceHolder>),
                                new Type[] {exprType}, selector, transformedExpr, cvi.caseNumbers);
                        }
                        else
                        {
                            throw new NotImplementedException($"Unhandled selector type: {StringUtil.TypeToString(selectorType)}");
                        }
                    }
                }
                else
                {
                    if (casesArray == null) context.Error("Unrolled form of Gate.Enter not yet implemented.");
                    // used in all cases.
                    if (selectorType.Equals(typeof (int)))
                    {
                        enterMethod = Builder.StaticGenericMethod(new Func<int, PlaceHolder, PlaceHolder[]>(Gate.Enter<PlaceHolder>),
                                                                  new Type[] {exprType}, selector, transformedExpr);
                    }
                    else if (selectorType.Equals(typeof (bool)))
                    {
                        enterMethod = Builder.StaticGenericMethod(new Func<bool, PlaceHolder, PlaceHolder[]>(Gate.Enter<PlaceHolder>),
                                                                  new Type[] {exprType}, selector, transformedExpr);
                    }
                    else
                    {
                        throw new NotImplementedException($"Unhandled selector type: {StringUtil.TypeToString(selectorType)}");
                    }
                }
                //                context.OutputAttributes.Set(enterMethod, new IsGateEnter());
                //context.OutputAttributes.Set(enterMethod, new Stochastic()); // for IfCuttingTransform
                IStatement enter = Builder.AssignStmt(arrayRef, enterMethod);
                enter = Containers.WrapWithContainers(enter, wildcardLoops);
                // mark this statement as generated by this ConditionInformation, so it doesn't get transformed again 
                context.OutputAttributes.Set(enter, this);
                stBefore.Add(enter);
            }

            int conditionStmtIndex = context.GetAncestorIndex(conditionStmt);
            // change keptContainers to be a single conditional stmt containing all bindings
            // caseVars must have the same conditional stmt
            stBefore = Containers.WrapWithContainers(stBefore, cvi.containers);
            stAfter = Containers.WrapWithContainers(stAfter, cvi.containers);
            int ancIndex = cloneOutsideLoop ? context.GetAncestorIndex(switchLoop) : conditionStmtIndex;
            if (!isExitVar && isIndexedByCondition)
            {
                // define the clone array inside the condition statement.
                // this is required when each array element can be used more than once.
                AddCloneStatements(context, stBefore);
                Assert.IsTrue(stAfter.Count == 0);
            }
            else
            {
                // this is required for exit vars and for switch enter vars that are not indexed by the switch loop.
                // otherwise, this is only an optimization.  if each array element will be used once, then we know that the evidence message will be uniform,
                // so we can place the declaration of the clone array outside the condition statement.
                context.AddStatementsBeforeAncestorIndex(ancIndex, stBefore, false);
                context.AddStatementsAfterAncestorIndex(ancIndex, stAfter, false);
            }
        }

        internal string ToString(IExpression expr)
        {
            if (expr is IVariableReferenceExpression ivre)
            {
                return ivre.Variable.Resolve().Name;
            }
            else if (expr is IArrayIndexerExpression iaie)
            {
                StringBuilder sb = new StringBuilder(ToString(iaie.Target));
                foreach (IExpression indExpr in iaie.Indices)
                    sb.Append("_" + ToString(indExpr));
                return sb.ToString();
            }
            else if (Recognizer.IsStaticMethod(expr, new Func<int>(GateAnalysisTransform.AnyIndex)))
            {
                return "any";
            }
            else return expr.ToString();
        }

        /// <summary>
        /// Get a fully substituted expression for the condition of this condInfo.
        /// </summary>
        /// <param name="context"></param>
        /// <returns></returns>
        internal IExpression GetConditionExpression(BasicTransformContext context)
        {
            if (!isStochastic) return conditionStmt.Condition;
            if (casesArray != null)
            {
                IExpression item = Builder.ArrayIndex(Builder.VarRefExpr(casesArray), caseNumber);
                if (caseNumber is ILiteralExpression)
                {
                    IVariableDeclaration casesVar;
                    if (!casesVars.TryGetValue(caseNumber, out casesVar))
                    {
                        VariableInformation avi = VariableInformation.GetVariableInformation(context, casesArray);
                        IList<IExpression> indices = Builder.ExprCollection();
                        indices.Add(caseNumber);
                        IList<IStatement> stmts = Builder.StmtCollection();
                        casesVar = avi.DeriveIndexedVariable(stmts, context, ToString(item), new List<IList<IExpression>> {indices});
                        casesVars[caseNumber] = casesVar;
                        IExpression copy = Builder.StaticGenericMethod(
                            new Func<PlaceHolder, PlaceHolder>(Clone.Copy<PlaceHolder>), new Type[] {typeof (bool)}, item);
                        IStatement assignSt = Builder.AssignStmt(Builder.VarRefExpr(casesVar), copy);
                        context.OutputAttributes.Set(copy, new CasesCopy());
                        stmts.Add(assignSt);
                        AddCasesStatements(context, stmts);
                    }
                    return Builder.VarRefExpr(casesVar);
                }
                else return item;
            }
            else return Builder.VarRefExpr(casesVars[caseNumber]);
        }

        /// <summary>
        /// Wrap each statement with multiple 'if' statements, as given by condInfo.
        /// </summary>
        internal IBlockStatement WrapBlockWithConditionals(BasicTransformContext context, IBlockStatement block)
        {
            IBlockStatement result = Builder.BlockStmt();
            if (isStochastic)
            {
                WrapStatementsWithConditionals(context, block.Statements, result.Statements.Add);
            }
            else
            {
                IConditionStatement cs = Builder.CondStmt(GetConditionExpression(context), block);
                result.Statements.Add(cs);
            }
            return result;
        }

        /// <summary>
        /// Wrap each statement with multiple 'if' statements, as given by condInfo.
        /// </summary>
        /// <param name="context">the transform context.</param>
        /// <param name="statements">the statements to wrap.</param>
        /// <param name="action">action to take for each wrapped statement.</param>
        internal void WrapStatementsWithConditionals(BasicTransformContext context, IList<IStatement> statements, Action<IStatement> action)
        {
            IConditionStatement currentConditionStatement = null;
            for (int i = 0; i < statements.Count; i++)
            {
                IStatement s = statements[i];
                if (s is IForStatement ifs)
                {
                    // Recursively wrap the body statements.
                    IForStatement fs = Builder.ForStmt();
                    fs.Condition = ifs.Condition;
                    fs.Increment = ifs.Increment;
                    fs.Initializer = ifs.Initializer;
                    fs.Body = WrapBlockWithConditionals(context, ifs.Body);
                    action(fs);
                    // End the current condition statement.
                    currentConditionStatement = null;
                    continue;
                }
                if (s is IRepeatStatement irs)
                {
                    // Recursively wrap the body statements.
                    IRepeatStatement rs = Builder.RepeatStmt();
                    rs.Count = irs.Count;
                    rs.Body = WrapBlockWithConditionals(context, irs.Body);
                    action(rs);
                    // End the current condition statement.
                    currentConditionStatement = null;
                    continue;
                }
                if (s is IConditionStatement ics && !CodeRecognizer.IsStochastic(context, ics.Condition))
                {
                    // Recursively wrap the body statements.
                    IConditionStatement cs = Builder.CondStmt();
                    cs.Condition = ics.Condition;
                    cs.Then = WrapBlockWithConditionals(context, ics.Then);
                    action(cs);
                    // End the current condition statement.
                    currentConditionStatement = null;
                    continue;
                }
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(s);
                if (ivd != null && context.InputAttributes.Has<DoNotSendEvidence>(ivd))
                {
                    action(s);
                    // End the current condition statement.
                    currentConditionStatement = null;
                    continue;
                }
                if (currentConditionStatement == null)
                {
                    // open a new condition statement
                    currentConditionStatement = Builder.CondStmt(GetConditionExpression(context), Builder.BlockStmt());
                    action(currentConditionStatement);
                }
                currentConditionStatement.Then.Statements.Add(s);
            }
        }
    }

    /// <summary>
    /// Describes a variable cloned across different branches of a condition.
    /// </summary>
    /// <remarks>
    /// Any variable declared outside a condition is cloned into an array of variables;
    /// one for each branch of the condition in which the variable is used.
    /// </remarks>
    internal class ClonedVarInfo
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        internal static CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Helps recognize code patterns
        /// </summary>
        internal static CodeRecognizer Recognizer = CodeRecognizer.Instance;

        /// <summary>
        /// The expression being cloned.
        /// </summary>
        internal readonly IExpression expr;

        /// <summary>
        /// Type of expr.
        /// </summary>
        internal Type exprType;

        /// <summary>
        /// Loop indices corresponding to the wildcards in expr.
        /// </summary>
        internal List<IList<IExpression>> wildcardVars = new List<IList<IExpression>>();

        /// <summary>
        /// True if the variable is defined inside the gate.
        /// </summary>
        internal readonly bool IsExitVar;

        internal bool IsEnterOne;

        /// <summary>
        /// Declaration of the array of clones.
        /// </summary>
        internal IVariableDeclaration arrayDecl;

        /// <summary>
        /// Cached variable reference to the array of clones.
        /// </summary>
        internal IExpression arrayRef;

        /// <summary>
        /// A map from case numbers to clone variables.
        /// </summary>
        internal Dictionary<IExpression, IVariableDeclaration> caseVars = new Dictionary<IExpression, IVariableDeclaration>();

        /// <summary>
        /// An array of integers, listing the condition branches in which the variable is used.
        /// </summary>
        /// <remarks>
        /// For example, if x is used in branches 0 and 2, the expression will be <c>new int[] { 0, 2 }</c>.
        /// indices is non-null only if Gate.EnterPartial or Gate.Exit is being used.  If enterOne is true, caseNumbers is null.
        /// </remarks>
        internal IArrayCreateExpression caseNumbers;

        /// <summary>
        /// The length of caseNumbers.
        /// </summary>
        internal ILiteralExpression caseCount;

        /// <summary>
        /// The number of distinct cases in which this variable is defined.
        /// </summary>
        internal int definitionCount;

        /// <summary>
        /// Containers in which the clone array (and elements thereof) are declared.
        /// </summary>
        internal List<IStatement> containers = new List<IStatement>();

        internal ClonedVarInfo(IExpression expr, Type exprType, bool isDef)
        {
            this.expr = expr;
            this.exprType = exprType;
            IsExitVar = isDef;
            caseCount = Builder.LiteralExpr(0); // will be mutated later in cvi.AddCase
        }

        /// <summary>
        /// Get a clone variable for the given case number.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="condInfo"></param>
        /// <param name="isDef"></param>
        /// <param name="caseNumber"></param>
        /// <returns></returns>
        internal IVariableReferenceExpression GetCloneForCase(BasicTransformContext context, ConditionInformation condInfo, bool isDef, IExpression caseNumber)
        {
            IVariableDeclaration caseVar;
            if (caseVars.TryGetValue(caseNumber, out caseVar))
                return Builder.VarRefExpr(caseVar);
            if (IsEnterOne)
            {
                caseVars[caseNumber] = arrayDecl;
                return Builder.VarRefExpr(arrayDecl);
            }
            IExpression item;
            if (condInfo.isStochastic)
            {
                // for a switch/default statement, caseNumber is the loop index, 
                // and AddCase returns -1.
                int index = AddCase(isDef, caseNumber);
                // when expr is a use of an EnterPartial variable, use the count as the index, otherwise use the case number.
                IExpression indexExpr = (index == -1 || IsExitVar) ? caseNumber : Builder.LiteralExpr(index);
                List<IList<IExpression>> indices2 = new List<IList<IExpression>>();
                indices2.AddRange(wildcardVars);
                IList<IExpression> lastIndex = Builder.ExprCollection();
                lastIndex.Add(indexExpr);
                indices2.Add(lastIndex);
                item = Builder.JaggedArrayIndex(arrayRef, indices2);
            }
            else
            {
                item = arrayRef;
            }
            IList<IStatement> stmts = Builder.StmtCollection();
            // caseVar will have the same type as the expression being cloned
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
            string name = condInfo.ToString(arrayRef);
            //if (!condInfo.isStochastic) prefix += "_cond_" + condInfo.conditionLhs + condInfo.cloneSuffix;
            name += "_" + caseNumber;
            name = VariableInformation.GenerateName(context, name);
            caseVar = vi.DeriveIndexedVariable(stmts, context, name, Recognizer.GetIndices(expr), this.wildcardVars);
            context.InputAttributes.Remove<ConditionInformation>(caseVar);
            context.InputAttributes.Set(caseVar, condInfo);
            if (context.OutputAttributes.Has<VariationalMessagePassing.GateExitRandomVariable>(arrayDecl))
                context.OutputAttributes.Set(caseVar, new VariationalMessagePassing.GateExitRandomVariable());
            caseVars[caseNumber] = caseVar;
            IExpression caseVarRef = Builder.VarRefExpr(caseVar);
            caseVarRef = Builder.JaggedArrayIndex(caseVarRef, wildcardVars);
            IStatement assignSt = null;
            if (isDef)
            {
                // expr is a local variable of a gate.
                // transform
                //   expr = rhs;
                // into
                //   exprType expr_caseNum = rhs;  
                //   expr_clones[caseNum] = Factor.Copy(expr_caseNum);
                IExpression copyExpr = Builder.StaticGenericMethod(
                    new Func<PlaceHolder, PlaceHolder>(Clone.Copy), new Type[] {exprType}, caseVarRef);
                assignSt = Builder.AssignStmt(item, copyExpr);
                // want the grouping to come from the definition, not the variable being cloned
                context.InputAttributes.Remove<GroupMember>(caseVar);
            }
            else
            {
                // expr_caseNum[i][j] = Factor.Copy(expr_clones[i][j][caseNum]);
                IExpression copyExpr = Builder.StaticGenericMethod(
                    new Func<PlaceHolder, PlaceHolder>(Clone.Copy), new Type[] {exprType}, item);
                assignSt = Builder.AssignStmt(caseVarRef, copyExpr);
            }
            context.OutputAttributes.Set(assignSt, condInfo);
            if (isDef)
            {
                // if the definition is derived, then this attribute will be re-attached by ConvertAssign
                context.InputAttributes.Remove<DerivedVariable>(caseVar);
            }
            else
            {
                if (!context.InputAttributes.Has<DerivedVariable>(caseVar))
                    context.InputAttributes.Set(caseVar, new DerivedVariable());
            }
            // in a stochastic gate, these are the same wildcardLoops computed in BuildCloneArray
            // in a deterministic gate, these have not been computed in BuildCloneArray, but they will be the same for every caseVar
            // so we might benefit from caching them
            List<IStatement> wildcardLoops = VariableInformation.GetVariableInformation(context, caseVar).BuildWildcardLoops(wildcardVars);
            assignSt = Containers.WrapWithContainers(assignSt, wildcardLoops);
            if (!isDef)
            {
                stmts.Add(assignSt);
            }
            else
            {
                assignSt = Containers.WrapWithContainers(assignSt, containers);
                int conditionStmtIndex = context.GetAncestorIndex(condInfo.conditionStmt);
                conditionStmtIndex += 2;
                context.AddStatementAfterAncestorIndex(conditionStmtIndex, assignSt);
            }
            stmts = Containers.WrapWithContainers(stmts, containers);
            condInfo.AddCloneStatements(context, stmts);
            return Builder.VarRefExpr(caseVar);
        }

        /// <summary>
        /// Given a case number, get an index into the clones array.
        /// </summary>
        /// <param name="isDef">True if the expression to be cloned is on the lhs of an assignment.</param>
        /// <param name="caseNumber"></param>
        /// <returns><c>indices.IndexOf(caseNumber)</c>, adding a new entry if necessary.</returns>
        internal int AddCase(bool isDef, IExpression caseNumber)
        {
            if (caseNumbers == null) return -1;
            int ct = caseNumbers.Initializer.Expressions.IndexOf(caseNumber);
            if (ct == -1)
            {
                if (isDef)
                {
                    definitionCount++;
                    //if((int)((ILiteralExpression)caseNumber).Value != indices.Initializer.Expressions.Count) throw new Exception("Exit variable definitions are out of order");
                }
                caseNumbers.Initializer.Expressions.Add(caseNumber);
                caseCount.Value = caseNumbers.Initializer.Expressions.Count;
                ct = caseNumbers.Initializer.Expressions.Count - 1;
            }
            return ct;
        }

        internal bool IsDefinedInAllCases()
        {
            if (caseNumbers == null) return true;
            if (definitionCount == 0) return true;
            IExpression dimension = caseNumbers.Dimensions[0];
            if (dimension is ILiteralExpression ile) return definitionCount == (int)ile.Value;
            return true;
        }

        public override string ToString()
        {
            return $"ClonedVarInfo({expr},{arrayRef})";
        }
    }

    /// <summary>
    /// When attached to a variable declaration, indicates that no variable factor should be generated.
    /// </summary>
    internal class SuppressVariableFactor : ICompilerAttribute
    {
    }

    internal class DoNotSendEvidence : ICompilerAttribute
    {
    }

    internal class CasesCopy : ICompilerAttribute
    {
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}