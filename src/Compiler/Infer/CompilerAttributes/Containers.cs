// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

//#define ignoreOutput

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Attributes
{
    internal class Containers : ICompilerAttribute
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Helps recognize code patterns
        /// </summary>
        private static readonly CodeRecognizer Recognizer = CodeRecognizer.Instance;

        /// <summary>
        /// Outermost container is first.
        /// </summary>
        internal readonly List<IStatement> inputs = new List<IStatement>();

        internal readonly List<IStatement> outputs = new List<IStatement>();

        /// <summary>
        /// The number of containers
        /// </summary>
        public int Count
        {
            get { return inputs.Count; }
        }

        /// <summary>
        /// The number of loops
        /// </summary>
        public int LoopCount
        {
            get { return inputs.OfType<IForStatement>().Count(); }
        }

        /// <summary>
        /// The number of conditionals
        /// </summary>
        public int ConditionCount
        {
            get { return inputs.OfType<IConditionStatement>().Count(); }
        }

        /// <summary>
        /// The containers of the current transform context.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="loopVarsHaveOneContainer"></param>
        internal Containers(BasicTransformContext context, bool loopVarsHaveOneContainer = true) // : this(FindContainers(context))
        {
            if (loopVarsHaveOneContainer)
            {
                int ancIndex2 = Recognizer.GetAncestorIndexOfLoopBeingInitialized(context);
                if (ancIndex2 != -1)
                {
                    // For a loop variable, the loop is the only container.
                    // This is needed to allow re-ordering nested loops.  Otherwise, the inner loop variable would always require the outer loop around it.
                    // It is also useful for ignoring conditioned loops.
                    TransformInfo ti = context.InputStack[ancIndex2];
                    IForStatement ifs = (IForStatement)ti.inputElement;
                    IStatement container = CreateContainer(ifs);
                    inputs.Add(container);
#if ignoreOutput
                    outputs.Add(container);
#else
                    outputs.Add((IStatement)ti.PrimaryOutput);
#endif
                    var loopVar = Recognizer.LoopVariable(ifs);
                    bool mustRemove = false;
                    if (!context.InputAttributes.Has<Containers>(loopVar))
                    {
                        context.InputAttributes.Set(loopVar, this);
                        mustRemove = true;
                    }
                    var initExpr = ((IExpressionStatement)ifs.Initializer).Expression;
                    this.AddContainersNeededForExpression(context, initExpr);
                    this.AddContainersNeededForExpression(context, ifs.Condition);
                    if (mustRemove)
                        context.InputAttributes.Remove<Containers>(loopVar);
                    return;
                }
            }
            // exclude the current statement.
            int ancIndex = context.FindAncestorIndex<IStatement>();
            for (int i = 0; i < ancIndex; i++)
            {
                TransformInfo ti = context.InputStack[i];
                IStatement inputElement = ti.inputElement as IStatement;
                if (IsContainer(inputElement) && !context.InputAttributes.Has<ConvergenceLoop>(inputElement))
                {
                    inputs.Add(CreateContainer(inputElement));
                    outputs.Add((IStatement)ti.PrimaryOutput);
                }
            }
        }

        internal static Containers InsideOf(BasicTransformContext context, int exclude)
        {
            Containers containers = new Containers();
            for (int i = exclude + 1; i < context.InputStack.Count; i++)
            {
                TransformInfo ti = context.InputStack[i];
                IStatement inputElement = ti.inputElement as IStatement;
                if (IsContainer(inputElement) && !context.InputAttributes.Has<ConvergenceLoop>(inputElement))
                {
                    containers.inputs.Add(CreateContainer(inputElement));
                    containers.outputs.Add((IStatement)ti.PrimaryOutput);
                }
            }
            return containers;
        }

        internal Containers(BasicTransformContext context, int count)
        {
            for (int i = 0; i < count; i++)
            {
                TransformInfo ti = context.InputStack[i];
                IStatement inputElement = ti.inputElement as IStatement;
                if (IsContainer(inputElement) && !context.InputAttributes.Has<ConvergenceLoop>(inputElement))
                {
                    inputs.Add(CreateContainer(inputElement));
                    outputs.Add((IStatement)ti.PrimaryOutput);
                }
            }
        }

        /// <summary>
        /// Returns a list of the open containers, outermost first.
        /// </summary>
        /// <param name="context"></param>
        /// <returns></returns>
        internal static List<IStatement> FindContainers(BasicTransformContext context)
        {
            return context.FindAncestors<IStatement>().FindAll(IsContainer);
        }

        /// <summary>
        /// Returns the outermost container in the context that is not in this.
        /// </summary>
        /// <param name="context"></param>
        /// <returns></returns>
        internal int GetMatchingAncestorIndex(BasicTransformContext context)
        {
            int ancIndex = context.FindAncestorIndex<IStatement>();
            foreach (IStatement container in Containers.FindContainers(context))
            {
                if (context.InputAttributes.Has<ConvergenceLoop>(container)) continue;
                if (!this.Contains(container))
                {
                    // found a container unique to the current context.
                    // statements must be added here.
                    ancIndex = context.GetAncestorIndex(container);
                    break;
                }
            }
            return ancIndex;
        }

        internal int GetMaxAncestorIndex(BasicTransformContext context)
        {
            int ancIndex = -1;
            foreach (IStatement container in inputs)
            {
                int ancIndex2 = GetAncestorIndex(context, container);
                if (ancIndex2 > ancIndex) ancIndex = ancIndex2;
            }
            return ancIndex;
        }

        /// <summary>
        /// An empty container list
        /// </summary>
        internal Containers()
        {
        }

        private Containers(Containers c)
        {
            inputs.AddRange(c.inputs);
            outputs.AddRange(c.outputs);
        }

        public override string ToString()
        {
            return "Containers(" + StringUtil.ToString(inputs) + ")";
        }

        /// <summary>
        /// True if <c>this</c> includes the given container
        /// </summary>
        /// <param name="container"></param>
        /// <param name="allowBrokenLoops"></param>
        /// <param name="ignoreLoopDirection"></param>
        /// <returns></returns>
        internal bool Contains(IStatement container, bool allowBrokenLoops = false, bool ignoreLoopDirection = false)
        {
            return ListContains(inputs, container, allowBrokenLoops, ignoreLoopDirection);
        }

        /// <summary>
        /// True if <c>this</c> includes all of the given containers
        /// </summary>
        /// <param name="containers"></param>
        /// <returns></returns>
        internal bool Contains(Containers containers)
        {
            foreach (var c in containers.inputs)
            {
                if (!Contains(c)) return false;
            }
            return true;
        }

        internal bool SetEquals(Containers c)
        {
            return Contains(c) && c.Contains(this);
        }

        /// <summary>
        /// Override equality to mean set equality.
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            return (obj is Containers c) && SetEquals(c);
        }

        /// <summary>
        /// Override of hash code to make it match overload of Equals()
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return Hash.GetHashCodeAsSet(inputs, new ContainerComparer());
        }

        internal static bool ListContains(IEnumerable<IStatement> list, IStatement container, bool allowBrokenLoops = false, bool ignoreLoopDirection = false)
        {
            foreach (IStatement st in list)
            {
                if (ContainersAreEqual(st, container, allowBrokenLoops, ignoreLoopDirection)) return true;
            }
            return false;
        }

        internal static bool IsContainer(IStatement st)
        {
            return (st is IForStatement) || (st is IConditionStatement) || (st is IRepeatStatement);
        }

        /// <summary>
        /// The first index in the input stack matching the given container, or -1 if not found.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="ancestor">A container</param>
        /// <returns></returns>
        internal static int GetAncestorIndex(BasicTransformContext context, IStatement ancestor)
        {
            for (int i = 0; i < context.InputStack.Count; i++)
            {
                IStatement anc = context.InputStack[i].inputElement as IStatement;
                if (ContainersAreEqual(anc, ancestor)) return i;
            }
            return -1;
        }

        /// <summary>
        /// True if the containers match, ignoring the statements in their bodies.
        /// </summary>
        /// <param name="st1"></param>
        /// <param name="st2"></param>
        /// <param name="allowBrokenLoops"></param>
        /// <param name="ignoreLoopDirection"></param>
        /// <returns></returns>
        internal static bool ContainersAreEqual(IStatement st1, IStatement st2, bool allowBrokenLoops = false, bool ignoreLoopDirection = false)
        {
            if (ReferenceEquals(st1, st2))
                return true;
            if (st1 is IForStatement ifs1)
            {
                if (st2 is IForStatement ifs2)
                {
                    if (ignoreLoopDirection && Recognizer.IsForwardLoop(ifs1) != Recognizer.IsForwardLoop(ifs2))
                    {
                        ifs2 = (IForStatement)CreateContainer(ifs2);
                        Recognizer.ReverseLoopDirection(ifs2);
                    }
                    return ifs1.Initializer.Equals(ifs2.Initializer) &&
                           ifs1.Condition.Equals(ifs2.Condition) &&
                           ifs1.Increment.Equals(ifs2.Increment) &&
                           (allowBrokenLoops || ((ifs1 is IBrokenForStatement) == (ifs2 is IBrokenForStatement)));
                }
                else return false;
            }
            else if (st1 is IConditionStatement ics1)
            {
                if (st2 is IConditionStatement ics2)
                {
                    return ics1.Condition.Equals(ics2.Condition);
                }
                else return false;
            }
            else if (st1 is IRepeatStatement irs1)
            {
                if (st2 is IRepeatStatement irs2)
                {
                    return irs1.Count.Equals(irs2.Count);
                }
                else return false;
            }
            else return (st1 == st2);
        }

        public static int ContainerGetHashCode(IStatement st)
        {
            if (st is IForStatement ifs)
            {
                return ifs.Initializer.GetHashCode();
            }
            else if (st is IConditionStatement ics)
            {
                return ics.Condition.GetHashCode();
            }
            else return st.GetHashCode();
        }

        public class ContainerComparer : EqualityComparer<IStatement>
        {
            public override bool Equals(IStatement st1, IStatement st2)
            {
                return ContainersAreEqual(st1, st2);
            }

            public override int GetHashCode(IStatement st)
            {
                return ContainerGetHashCode(st);
            }
        }

        internal static Containers GetContainers(IStatement ist)
        {
            Containers c = new Containers();
            IList<IStatement> core = LoopMergingTransform.UnwrapStatement(ist, c.inputs);
            c.outputs.AddRange(c.inputs);
            return c;
        }

        /// <summary>
        /// Collect all containers that are not in the context at the given index.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="ancestorIndex">If 0, all containers are returned.</param>
        /// <returns></returns>
        internal Containers GetContainersNotInContext(BasicTransformContext context, int ancestorIndex)
        {
            Containers result = new Containers();
            for (int i = 0; i < inputs.Count; i++)
            {
                IStatement container = inputs[i];
                int index = GetAncestorIndex(context, container);
                if (index == -1 || index >= ancestorIndex)
                {
                    result.inputs.Add(container);
                    result.outputs.Add(outputs[i]);
                }
            }
            return result;
        }

        internal Containers Remove(Containers containers)
        {
            return Remove(containers.inputs);
        }

        internal Containers Remove(List<IStatement> containers)
        {
            return Remove(ist => ListContains(containers, ist));
        }

        internal Containers Remove(Predicate<IStatement> predicate)
        {
            Containers result = new Containers();
            for (int i = 0; i < inputs.Count; i++)
            {
                IStatement container = inputs[i];
                if (!predicate(container))
                {
                    result.inputs.Add(container);
                    result.outputs.Add(outputs[i]);
                }
            }
            return result;
        }

        internal Containers RemoveOneRepeat(IRepeatStatement irs)
        {
            bool found = false;
            Containers result = new Containers();
            for (int i = 0; i < inputs.Count; i++)
            {
                IStatement container = inputs[i];
                if (!found && container is IRepeatStatement rs)
                {
                    if (rs.Count.Equals(irs.Count))
                    {
                        found = true;
                        continue;
                    }
                }
                result.inputs.Add(container);
                result.outputs.Add(outputs[i]);
            }
            return result;
        }

        /// <summary>
        /// Collect all loops in the context whose index is referenced by expr or by the size expression of another collected loop.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="expr"></param>
        /// <param name="excludeAncestorIndex">Only loops whose ancestor index is greater than excludeAncestorIndex will be collected.</param>
        /// <param name="includeConditionals">Include condition statements</param>
        /// <returns>A list of IForStatements, starting with outermost.</returns>
        internal static List<IStatement> GetLoopsNeededForExpression(
            BasicTransformContext context, IExpression expr, int excludeAncestorIndex, bool includeConditionals)
        {
            List<IStatement> loopsNeeded = new List<IStatement>();
            List<IStatement> ancestors = context.FindAncestorsBelow<IStatement>(excludeAncestorIndex);
            ancestors.Reverse();
            List<IExpression> containedExpressions = new List<IExpression>();
            AddToContainedExpressions(containedExpressions, expr, context);
            // loop ancestors starting from innermost
            foreach (IStatement ist in ancestors)
            {
                if (ist is IForStatement loop)
                {
                    IVariableDeclaration loopVar = Recognizer.LoopVariable(loop);
                    try
                    {
                        IExpression loopVarRef = Builder.VarRefExpr(loopVar);
                        foreach (IExpression containedExpression in containedExpressions)
                        {
                            if (Builder.ContainsExpression(containedExpression, loopVarRef))
                            {
                                IForStatement replacedLoop = Builder.ForStmt(Recognizer.LoopVariable(loop), Recognizer.LoopSizeExpression(loop));
                                loopsNeeded.Add(replacedLoop);
                                AddToContainedExpressions(containedExpressions, Recognizer.LoopSizeExpression(loop), context);
                                break;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        context.Error("GetLoopsNeededForExpression", ex);
                    }
                }
                else if (includeConditionals && (ist is IConditionStatement ics))
                {
                    bool found = false;
                    var conditionVariables = Recognizer.GetVariables(ics.Condition).Select(Builder.VarRefExpr);
                    foreach (IExpression conditionVariable in conditionVariables)
                    {
                        foreach (IExpression containedExpression in containedExpressions)
                        {
                            if (Builder.ContainsExpression(containedExpression, conditionVariable))
                            {
                                loopsNeeded.Add(ics);
                                AddToContainedExpressions(containedExpressions, ics.Condition, context);
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }
                }
            }
            loopsNeeded.Reverse();
            return loopsNeeded;
        }

        /// <summary>
        /// Adds the expression and any dependent expressions to the supplied list of contained expressions.
        /// </summary>
        /// <param name="containedExpressions"></param>
        /// <param name="expr"></param>
        /// <param name="context"></param>
        private static void AddToContainedExpressions(List<IExpression> containedExpressions, IExpression expr, BasicTransformContext context)
        {
            containedExpressions.Add(expr);
            IVariableDeclaration baseVar = Recognizer.GetVariableDeclaration(expr);
            if (expr is IArrayIndexerExpression iaie)
            {
                foreach (var ind in iaie.Indices) AddToContainedExpressions(containedExpressions, ind, context);
            }
            if (baseVar == null) return;
            Containers containers = context.InputAttributes.Get<Containers>(baseVar);
            if (containers == null) throw new Exception("Containers not found for: " + baseVar);
            foreach (IStatement container in containers.inputs)
            {
                if (container is IForStatement ifs)
                {
                    containedExpressions.Add(Builder.VarRefExpr(Recognizer.LoopVariable(ifs)));
                }
            }
        }

        /// <summary>
        /// Reorder containers to follow their order in the input stack.  Also removes duplicates.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="start"></param>
        public void OrderByContext(BasicTransformContext context, int start = 0)
        {
            if (inputs.Count == 0) return;
            Set<IStatement> inputSet = new Set<IStatement>(new ContainerComparer());
            inputSet.AddRange(inputs);
            inputs.Clear();
            outputs.Clear();
            for (int i = 0; i < context.InputStack.Count; i++)
            {
                TransformInfo ti = context.InputStack[i];
                IStatement inputElement = ti.inputElement as IStatement;
                if (inputElement == null || !IsContainer(inputElement)) continue;
                inputElement = CreateContainer(inputElement);
                if (inputSet.Contains(inputElement))
                {
                    if (i >= start)
                    {
                        inputs.Add(inputElement);
#if ignoreOutput
                        outputs.Add(inputElement);
#else
                        outputs.Add((IStatement)ti.PrimaryOutput);
#endif
                    }
                    inputSet.Remove(inputElement);
                }
            }
            // there may be containers that are not in the current context at all.  put these on the inside.
            inputs.AddRange(inputSet);
            outputs.AddRange(inputSet);
        }

        /// <summary>
        /// Get the minimal set of containers needed for all variables in an expression to be declared.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
        internal static Containers GetContainersNeededForExpression(BasicTransformContext context, IExpression expr)
        {
            Containers containers = new Containers();
            containers.AddContainersNeededForExpression(context, expr);
            return containers;
        }

        internal void AddContainersNeededForExpression(BasicTransformContext context, IExpression expr)
        {
            foreach (var ivd in Recognizer.GetVariables(expr))
            {
                Containers c = context.InputAttributes.Get<Containers>(ivd);
                if (c == null)
                {
                    context.Error("Containers not found for '" + ivd.Name + "'.");
                    return;
                }
                Add(c);
            }
            this.OrderByContext(context);
        }

        internal static Containers GetContainersNeededForExpression(BasicTransformContext context, IExpression expr, int excludeAncestorIndex)
        {
            Containers containers = GetContainersNeededForExpression(context, expr);
            containers.OrderByContext(context, excludeAncestorIndex + 1);
            return containers;
        }

        /// <summary>
        /// Get the minimal set of containers needed to evaluate an expression, along with the first index in the context stack at which the expression's variables are all declared.
        /// </summary>
        /// <param name="context">The transform context</param>
        /// <param name="expr">Any expression</param>
        /// <param name="ancIndex">On exit, the first index in the context stack at which the expression's variables are all declared.</param>
        /// <returns>The minimal set of containers needed to evaluate expr</returns>
        internal static Containers GetContainersNeededForExpression(BasicTransformContext context, IExpression expr, out int ancIndex)
        {
            Containers containers = GetContainersNeededForExpression(context, expr);
            if (containers == null)
            {
                ancIndex = -1;
                return null;
            }
            ancIndex = containers.GetMatchingAncestorIndex(context);
            // append any conditionals with constant conditions
            //List<IStatement> loopsMissing = Containers.GetLoopsNeededForExpression(context, expr, ancIndex-1, true);
            //containers = Containers.Append(containers, loopsMissing);
            //containers.OrderByContext(context);
            return containers;
        }

        public static Set<IVariableDeclaration> GetConditionedLoopVariables(BasicTransformContext context)
        {
            Set<IVariableDeclaration> loopVars = new Set<IVariableDeclaration>();
            foreach (IConditionStatement ics in context.FindAncestors<IConditionStatement>())
            {
                ConditionBinding binding = new ConditionBinding(ics.Condition);
                if (binding.lhs is IVariableReferenceExpression)
                {
                    IVariableReferenceExpression ivre = (IVariableReferenceExpression)binding.lhs;
                    if (Recognizer.GetLoopForVariable(context, ivre) != null) loopVars.Add(Recognizer.GetVariableDeclaration(ivre));
                }
                if (binding.rhs is IVariableReferenceExpression)
                {
                    IVariableReferenceExpression ivre = (IVariableReferenceExpression)binding.rhs;
                    if (Recognizer.GetLoopForVariable(context, ivre) != null) loopVars.Add(Recognizer.GetVariableDeclaration(ivre));
                }
            }
            return loopVars;
        }

        /// <summary>
        /// Remove loops (and their dependent containers) that are not needed to evaluate the expression.
        /// </summary>
        /// <param name="containers"></param>
        /// <param name="context"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
        internal static Containers RemoveUnusedLoops(Containers containers, BasicTransformContext context, IExpression expr)
        {
            Containers needed = Containers.GetContainersNeededForExpression(context, expr);
            return RemoveUnusedLoops(containers, context, needed);
        }

        /// <summary>
        /// Remove loops (and their dependent containers) that are not needed to evaluate the expression.
        /// </summary>
        /// <param name="containers"></param>
        /// <param name="context"></param>
        /// <param name="needed"></param>
        /// <returns></returns>
        internal static Containers RemoveUnusedLoops(Containers containers, BasicTransformContext context, Containers needed)
        {
            Containers result = new Containers();
            Set<IVariableDeclaration> allowedVars = Containers.GetConditionedLoopVariables(context);
            allowedVars.Clear(); // for ReplicateWithConditionedIndexTest
            for (int i = 0; i < containers.inputs.Count; i++)
            {
                IStatement container = containers.inputs[i];
                if (container is IForStatement)
                {
                    IForStatement ifs = container as IForStatement;
                    IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
                    if (!allowedVars.Contains(loopVar) && !needed.Contains(container)) continue;
                }
                result.inputs.Add(container);
                result.outputs.Add(containers.outputs[i]);
            }
            // Removing unused loops may have left us with containers that refer to the removed loop index.  We must remove these also.
            // Note this routine could be merged with RemoveUnusedLoops.
            result = Containers.RemoveInvalidConditions(result, context);
            return result;
        }

        internal Containers Where(Predicate<IStatement> predicate)
        {
            Containers result = new Containers();
            for (int i = 0; i < inputs.Count; i++)
            {
                IStatement container = inputs[i];
                if (predicate(container))
                {
                    result.inputs.Add(container);
                    result.outputs.Add(outputs[i]);
                }
            }
            return result;
        }

        internal Containers Replace(Func<IStatement, IStatement> getReplacement)
        {
            Containers result = new Containers();
            for (int i = 0; i < inputs.Count; i++)
            {
                IStatement container = inputs[i];
                IStatement replacement = getReplacement(container);
                result.inputs.Add(replacement);
                if (!ContainersAreEqual(outputs[i], inputs[i]))
                    throw new Exception("outputs[i] != inputs[i]");
                result.outputs.Add(replacement);
            }
            return result;
        }

        /// <summary>
        /// Returns true if expr can be evaluated in the given containers, i.e. all local variables are declared.
        /// </summary>
        /// <param name="containers"></param>
        /// <param name="context"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
        internal static bool ContainsExpression(List<IStatement> containers, BasicTransformContext context, IExpression expr)
        {
            return Recognizer.GetVariables(expr).All(ivd =>
            {
                Containers c = context.InputAttributes.Get<Containers>(ivd);
                // does containers contain all of c?
                return c.inputs.TrueForAll(st => Containers.ListContains(containers, st, allowBrokenLoops: true, ignoreLoopDirection: true));
            });
        }

        internal static Containers RemoveInvalidConditions(Containers containers, BasicTransformContext context)
        {
            Containers result = new Containers();
            for (int i = 0; i < containers.inputs.Count; i++)
            {
                IStatement container = containers.inputs[i];
                if (container is IConditionStatement ics)
                {
                    IExpression condition = ics.Condition;
                    if (condition is IBinaryExpression ibe && ibe.Operator == BinaryOperator.BooleanAnd)
                    {
                        // split the condition into conjuncts
                        List<IExpression> conditions = new List<IExpression>();
                        IExpression newCondition = null;
                        bool changed = false;
                        ForEachConjunct(condition, delegate (IExpression expr)
                        {
                            if (ContainsExpression(containers.inputs, context, expr))
                            {
                                if (newCondition == null) newCondition = expr;
                                else newCondition = Builder.BinaryExpr(BinaryOperator.BooleanAnd, newCondition, expr);
                            }
                            else changed = true;
                        });
                        if (changed)
                        {
                            if (newCondition != null)
                            {
                                IConditionStatement cs = Builder.CondStmt(newCondition, Builder.BlockStmt());
                                result.inputs.Add(cs);
                                result.outputs.Add(cs);
                            }
                            continue;
                        }
                    }
                    else
                    {
                        if (!ContainsExpression(containers.inputs, context, condition)) continue;
                    }
                }
                else if (container is IRepeatStatement irs)
                {
                    if (!ContainsExpression(containers.inputs, context, irs.Count)) continue;
                }
                result.inputs.Add(container);
                result.outputs.Add(containers.outputs[i]);
            }
            return result;
        }

        internal static void ForEachConjunct(IExpression expr, Action<IExpression> action)
        {
            if (expr is IBinaryExpression ibe && ibe.Operator == BinaryOperator.BooleanAnd)
            {
                ForEachConjunct(ibe.Left, action);
                ForEachConjunct(ibe.Right, action);
            }
            else
            {
                action(expr);
            }
        }

        internal static Containers RemoveStochasticConditionals(Containers containers, BasicTransformContext context)
        {
            Containers result = new Containers();
            for (int i = 0; i < containers.inputs.Count; i++)
            {
                IStatement container = containers.inputs[i];
                if (container is IConditionStatement ics)
                {
                    if (CodeRecognizer.IsStochastic(context, ics.Condition)) continue;
                }
                result.inputs.Add(container);
                result.outputs.Add(containers.outputs[i]);
            }
            return result;
        }

        internal static Containers SortStochasticConditionals(Containers containers, BasicTransformContext context)
        {
            Containers result = new Containers();
            Containers conditionals = new Containers();
            for (int i = 0; i < containers.inputs.Count; i++)
            {
                IStatement container = containers.inputs[i];
                if (container is IConditionStatement ics)
                {
                    if (CodeRecognizer.IsStochastic(context, ics.Condition))
                    {
                        conditionals.inputs.Add(container);
                        conditionals.outputs.Add(containers.outputs[i]);
                        continue;
                    }
                }
                result.inputs.Add(container);
                result.outputs.Add(containers.outputs[i]);
            }
            for (int i = 0; i < conditionals.inputs.Count; i++)
            {
                result.inputs.Add(conditionals.inputs[i]);
                result.outputs.Add(conditionals.outputs[i]);
            }
            return result;
        }

        private void Add(Containers extraContainers)
        {
            for (int i = 0; i < extraContainers.inputs.Count; i++)
            {
                IStatement input = extraContainers.inputs[i];
                if (!this.Contains(input))
                {
                    inputs.Add(input);
                    outputs.Add(extraContainers.outputs[i]);
                }
            }
        }

        internal static Containers Append(Containers containers, Containers extraContainers)
        {
            Containers result = new Containers(containers);
            for (int i = 0; i < extraContainers.inputs.Count; i++)
            {
                IStatement input = extraContainers.inputs[i];
                if (!containers.Contains(input))
                {
                    result.inputs.Add(input);
                    result.outputs.Add(extraContainers.outputs[i]);
                }
            }
            return result;
        }

        internal static Containers Append(Containers containers, IEnumerable<IStatement> extraContainers)
        {
            Containers result = new Containers(containers);
            foreach (IStatement loop in extraContainers)
            {
                result.inputs.Add(loop);
                result.outputs.Add(loop);
            }
            return result;
        }

        internal static Containers Append(Containers containers, IEnumerable<IForStatement> extraLoops)
        {
            Containers result = new Containers(containers);
            foreach (IStatement loop in extraLoops)
            {
                result.inputs.Add(loop);
                result.outputs.Add(loop);
            }
            return result;
        }

        internal static IStatement WrapWithContainers(IStatement stmt, ICollection<IStatement> containers)
        {
            if (containers.Count == 0) return stmt;
            else
            {
                IList<IStatement> result = Builder.StmtCollection();
                AddStatementWithContainers(result, stmt, containers);
                return result[0];
            }
        }

        internal static IList<IStatement> WrapWithContainers(IList<IStatement> stmts, ICollection<IStatement> containers)
        {
            if (containers.Count == 0) return stmts;
            else
            {
                IList<IStatement> result = Builder.StmtCollection();
                AddStatementWithContainers(result, stmts, containers);
                return result;
            }
        }

        internal static void SetBodyTo(IStatement container, IBlockStatement block)
        {
            if (container is IForStatement ifs)
            {
                ifs.Body = block;
            }
            else if (container is IConditionStatement ics)
            {
                ics.Then = block;
            }
            else
            {
                throw new ArgumentException("unrecognized container: " + container);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="statementsDest"></param>
        /// <param name="statementsSource"></param>
        /// <param name="containers">Outermost container is first.  The bodies of the containers are ignored.</param>
        internal static void AddStatementWithContainers(IList<IStatement> statementsDest, IList<IStatement> statementsSource, ICollection<IStatement> containers)
        {
            if (statementsSource.Count == 0)
                return;
            if (containers.Count == 0)
            {
                statementsDest.AddRange(statementsSource);
                return;
            }
            IStatement outermostContainer = null;
            IStatement parent = null;
            List<IStatement> loopBreakers = new List<IStatement>();
            foreach (IStatement container in containers)
            {
                IStatement child = CreateContainer(container);
                if (parent != null)
                {
                    AddToContainer(parent, child);
                }
                else
                    outermostContainer = child;
                parent = child;
                if (child is IBrokenForStatement ifs)
                {
                    loopBreakers.Add(Recognizer.LoopBreakStatement(ifs));
                }
            }
            AddToContainer(parent, statementsSource);
            AddToContainer(parent, loopBreakers);
            statementsDest.Add(outermostContainer);
        }

        internal static void AddStatementWithContainers(IList<IStatement> statementsDest, IStatement statement, ICollection<IStatement> containers)
        {
            IList<IStatement> statementsSource = Builder.StmtCollection();
            statementsSource.Add(statement);
            AddStatementWithContainers(statementsDest, statementsSource, containers);
        }

        /// <summary>
        /// Create a clone of prototype with an empty body.
        /// </summary>
        /// <param name="prototype"></param>
        /// <returns></returns>
        internal static IStatement CreateContainer(IStatement prototype)
        {
            if (prototype is IForStatement loop)
            {
                IForStatement ifs = Builder.ForStmt(loop);
                ifs.Initializer = loop.Initializer;
                ifs.Condition = loop.Condition;
                ifs.Increment = loop.Increment;
                ifs.Body = Builder.BlockStmt();
                return ifs;
            }
            else if (prototype is IConditionStatement cond)
            {
                return Builder.CondStmt(cond.Condition, Builder.BlockStmt());
            }
            else if (prototype is IRepeatStatement irs)
            {
                return Builder.RepeatStmt(irs.Count);
            }
            else
            {
                throw new NotImplementedException("unrecognized container: " + prototype);
            }
        }

        internal static void AddToContainer(IStatement container, IStatement statement)
        {
            Statements(container).Add(statement);
        }

        internal static void AddToContainer(IStatement container, IList<IStatement> statements)
        {
            Statements(container).AddRange(statements);
        }

        internal static IList<IStatement> Statements(IStatement container)
        {
            if (container is IForStatement ifs)
            {
                return ifs.Body.Statements;
            }
            else if (container is IConditionStatement ics)
            {
                return ics.Then.Statements;
            }
            else if (container is IRepeatStatement irs)
            {
                return irs.Body.Statements;
            }
            else
            {
                throw new ArgumentException("unrecognized container: " + container);
            }
        }

        /// <summary>
        /// Returns the containers common to both, preserving the order in the first argument.
        /// </summary>
        /// <param name="containers"></param>
        /// <param name="containers2"></param>
        /// <param name="allowBrokenLoops"></param>
        /// <param name="ignoreLoopDirection"></param>
        /// <returns></returns>
        internal static Containers Intersect(Containers containers, Containers containers2, bool allowBrokenLoops = false, bool ignoreLoopDirection = false)
        {
            Containers result = new Containers();
            for (int i = 0; i < containers.inputs.Count; i++)
            {
                IStatement container = containers.inputs[i];
                for (int j = 0; j < containers2.inputs.Count; j++)
                {
                    IStatement container2 = containers2.inputs[j];
                    IStatement intersected = Intersect(container, container2, allowBrokenLoops, ignoreLoopDirection);
                    if (intersected != null)
                    {
                        result.inputs.Add(intersected);
                        var intersectedOutput = Intersect(containers.outputs[i], containers2.outputs[j], allowBrokenLoops, ignoreLoopDirection);
                        result.outputs.Add(intersectedOutput);
                    }
                }
            }
            return result;
        }

        internal static IStatement Intersect(IStatement st1, IStatement st2, bool allowBrokenLoops = false, bool ignoreLoopDirection = false)
        {
            if (ReferenceEquals(st1, st2))
                return st1;
            if (st1 is IForStatement ifs1)
            {
                bool isForward = Recognizer.IsForwardLoop(ifs1);
                bool isBroken = ifs1 is IBrokenForStatement;
                if (st2 is IForStatement ifs2)
                {
                    if (ignoreLoopDirection && isForward != Recognizer.IsForwardLoop(ifs2))
                    {
                        ifs2 = (IForStatement)CreateContainer(ifs2);
                        Recognizer.ReverseLoopDirection(ifs2);
                    }
                    bool isBroken2 = ifs2 is IBrokenForStatement;
                    if (ifs1.Initializer.Equals(ifs2.Initializer) &&
                        ifs1.Condition.Equals(ifs2.Condition) &&
                        ifs1.Increment.Equals(ifs2.Increment) &&
                        (allowBrokenLoops || (isBroken == isBroken2)))
                    {
                        return isBroken ? st1 : st2;
                    }
                    // fall through
                }
                // fall through
            }
            else if (st1 is IConditionStatement ics1)
            {
                if (st2 is IConditionStatement ics2)
                {
                    if (ics1.Condition.Equals(ics2.Condition)) return st1;
                    // fall through
                }
                // fall through
            }
            else if (st1 is IRepeatStatement irs1)
            {
                if (st2 is IRepeatStatement irs2)
                {
                    if (irs1.Count.Equals(irs2.Count)) return st1;
                    // fall through
                }
                // fall through
            }
            else if (st1 == st2)
                return st1;
            return null;
        }

        internal void Add(IStatement st)
        {
            st = CreateContainer(st);
            inputs.Add(st);
            outputs.Add(st);
        }

        public Containers Clone()
        {
            Containers result = new Containers();
            result.inputs.AddRange(this.inputs);
            result.outputs.AddRange(this.outputs);
            return result;
        }
    }
}