// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Sets VariableInformation.isStochastic, converts constants in random contexts into distributions, and attaches MarginalPrototype attributes.
    /// </summary>
    /// <remarks>
    /// DistributionAnalysis must be part of this transform in order to correctly convert constants.  
    /// Otherwise, you would have to execute multiple rounds of StocAnalysis and DistributionAnalysis.
    /// This is because converting a constant may use knowledge about the MarginalPrototype (if it exists) 
    /// and it may influence the choice of MarginalPrototype (if it is not known yet).
    /// Ideally, we would not need to convert constants, but that would require significant changes to later transforms.
    /// </remarks>
    internal class StocAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "StocAnalysisTransform"; }
        }

        private readonly Set<IVariableDeclaration> varsWithConstantDefinition = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
        private readonly Dictionary<IVariableDeclaration, int> stochasticConditionsOf = new Dictionary<IVariableDeclaration, int>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
        private int numberOfStochasticConditions;
        private readonly bool convertConstants;
        // for DistributionAnalysis
        private readonly Set<IVariableDeclaration> loopVars = new Set<IVariableDeclaration>();
        private bool inPartialLoop;

        internal StocAnalysisTransform(bool convertConstants = false)
        {
            this.convertConstants = convertConstants;
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            for (int argIndex = 0; argIndex < imie.Arguments.Count; argIndex++)
            {
                IExpression arg = imie.Arguments[argIndex];
                if (arg is IAddressOutExpression iaoe)
                {
                    IExpression target = iaoe.Expression;
                    bool targetHasLiteralIndices = inPartialLoop;
                    object targetDecl = Recognizer.GetDeclaration(target);
                    MarginalPrototype mpa = GetMarginalPrototype(target, targetDecl);
                    MarginalPrototype mpa2 = InferMarginalPrototypeOut(imie, argIndex, target, targetDecl);
                    if (mpa2 != null)
                    {
                        SetMarginalPrototype(target, targetDecl, mpa, mpa2, targetHasLiteralIndices);
                    }
                    if (targetDecl is IVariableDeclaration ivd)
                    {
                        bool isStoch = CodeRecognizer.IsStochastic(context, imie) || IsStochContext(ivd);
                        bool needsMarginalDividedByPrior = CodeRecognizer.NeedsMarginalDividedByPrior(context, imie);
                        SetStoch(target, isStoch, needsMarginalDividedByPrior);
                    }
                }
            }
            return base.ConvertMethodInvoke(imie);
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            // check for literal indices before the target is transformed
            bool targetHasLiteralIndices = inPartialLoop;
            List<IList<IExpression>> indices = Recognizer.GetIndices(iae.Target);
            foreach (IList<IExpression> bracket in indices)
            {
                foreach (IExpression index in bracket)
                {
                    if (index is ILiteralExpression) targetHasLiteralIndices = true;
                }
            }
            IParameterDeclaration ipd = null;
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
            if (ivd == null)
            {
                ipd = Recognizer.GetParameterDeclaration(iae.Target);
            }
            IAssignExpression ae = (IAssignExpression)base.ConvertAssign(iae);
            bool isLoopInitializer = (Recognizer.GetAncestorIndexOfLoopBeingInitialized(context) != -1);
            if (isLoopInitializer)
                return ae;
            InferMarginalPrototype(ae, targetHasLiteralIndices);
            if (convertConstants && !CodeRecognizer.IsStochastic(context, ae.Expression))
            {
                // assigning a constant value to a random variable, such as: x = c
                // try to rewrite into: x = Factor.Random(Bernoulli.PointMass(c))
                // we must do the below even for observed x's, in order to compute the MarginalPrototype
                IAssignExpression stocAssign = Builder.AssignExpr(ae.Target, ae.Expression);
                IExpression rhs = ae.Expression;
                while (Recognizer.IsStaticMethod(rhs, typeof(Factor), "Copy"))
                {
                    // change Factor.Copy(c) into c
                    rhs = ((IMethodInvokeExpression)rhs).Arguments[0];
                }
                Type domainType = rhs.GetExpressionType();
                if (ipd != null || ivd != null)
                {
                    if (domainType.Equals(typeof(bool)))
                    {
                        Type distType = typeof(Bernoulli);
                        MethodInfo method = FactorManager.GetPointMassMethod(distType, domainType);
                        if (method != null)
                        {
                            IExpression dist = Builder.StaticMethod(method, rhs);
                            IMethodInvokeExpression imie = Builder.StaticGenericMethod(new Func<Sampleable<PlaceHolder>, PlaceHolder>(Factor.Random<PlaceHolder>),
                                                                                       new Type[] { domainType }, dist);
                            stocAssign.Expression = imie;
                            InferMarginalPrototype(stocAssign, targetHasLiteralIndices);
                        }
                    }
                    else if (domainType.Equals(typeof(int)))
                    {
                        Type distType = typeof(Discrete);
                        MethodInfo method = (MethodInfo)Reflection.Invoker.GetBestMethod(distType, "PointMass",
                            BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod |
                            BindingFlags.FlattenHierarchy,
                            null, new Type[] { domainType, typeof(int) }, out Exception exception);
                        if (method != null)
                        {
                            IExpression cardinalityExpression = GetIntCardinalityExpression(rhs, ae.Target, targetHasLiteralIndices);
                            if (cardinalityExpression == null) context.Error("Unknown cardinality of '" + ae.Target + "'.  Perhaps you forgot SetValueRange?");
                            else
                            {
                                IExpression dist = Builder.StaticMethod(method, rhs, cardinalityExpression);
                                IMethodInvokeExpression imie = Builder.StaticGenericMethod(new Func<Sampleable<PlaceHolder>, PlaceHolder>(Factor.Random<PlaceHolder>),
                                                                                           new Type[] { domainType }, dist);
                                stocAssign.Expression = imie;
                                InferMarginalPrototype(stocAssign, targetHasLiteralIndices);
                            }
                        }
                    }
                }
                bool doReplace = (ipd == null && ivd != null && (CodeRecognizer.IsStochastic(context, ivd) || IsStochContext(ivd)));
                if (doReplace) ae = stocAssign;
                else if (IsConstraint())
                {
                    return Builder.StaticGenericMethod(new Action<PlaceHolder, PlaceHolder>(Constrain.Equal), new Type[] { domainType }, ae.Target, rhs);
                }
            }
            if (ipd == null && ivd != null)
            {
                // Array creation with no initialiser can be either for stochastic or deterministic
                // variables, so just return without marking the variable.
                bool setStoch = !IsArrayCreateWithoutInitializer(ae.Expression) && !IsConstraint();
                if (setStoch)
                {
                    // Sets the stochasticity of the LHS of an assignment, given the RHS.
                    bool isStoch = CodeRecognizer.IsStochastic(context, ae.Expression) || IsStochContext(ivd);
                    bool needsMarginalDividedByPrior = CodeRecognizer.NeedsMarginalDividedByPrior(context, ae.Expression);
                    SetStoch(ae.Target, isStoch, needsMarginalDividedByPrior);
                }
            }
            return ae;

            bool IsArrayCreateWithoutInitializer(IExpression expression)
            {
                return (expression is IArrayCreateExpression iace) && (iace.Initializer == null);
            }

            bool IsConstraint()
            {
                IStatement ist = context.FindAncestor<IStatement>();
                return context.InputAttributes.Has<Constraint>(ist);
            }
        }

        protected override IVariableDeclaration ConvertVariableDecl(IVariableDeclaration ivd)
        {
            // Record the loop context that a variable is created in
            context.InputAttributes.Remove<LoopContext>(ivd);
            context.InputAttributes.Set(ivd, new LoopContext(context));
            context.InputAttributes.Remove<Containers>(ivd);
            context.InputAttributes.Set(ivd, new Containers(context));
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
            vi.IsStochastic = false;
            stochasticConditionsOf[ivd] = numberOfStochasticConditions;
            return base.ConvertVariableDecl(ivd);
        }

        protected void SetStoch(IExpression expr, bool isStoch, bool needsMarginalDividedByPrior)
        {
            if (expr is IArgumentReferenceExpression) return;
            if (expr is IArrayIndexerExpression iaie)
            {
                SetStoch(iaie.Target, isStoch, needsMarginalDividedByPrior);
                return;
            }
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
            if (ivd == null)
            {
                Error("Could not find target variable of expression: " + expr);
                return;
            }
            if (isStoch)
            {
                VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
                vi.IsStochastic = true;
            }
            else
            {
                varsWithConstantDefinition.Add(ivd);
            }
            if (needsMarginalDividedByPrior)
            {
                VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
                vi.NeedsMarginalDividedByPrior = true;
            }
        }

        protected bool IsStochContext(IVariableDeclaration ivd)
        {
            return (numberOfStochasticConditions > stochasticConditionsOf[ivd]);
        }

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            bool isStochasticCondition = CodeRecognizer.IsStochastic(context, ics.Condition);
            if (isStochasticCondition) numberOfStochasticConditions++;
            bool prev_inPartialLoop = inPartialLoop;
            if (!isStochasticCondition)
            {
                if (Recognizer.GetVariables(ics.Condition).Any(loopVars.Contains))
                    inPartialLoop = true;
            }
            IStatement st = base.ConvertCondition(ics);
            inPartialLoop = prev_inPartialLoop;
            if (isStochasticCondition) numberOfStochasticConditions--;
            return st;
        }

        #region Factor insertion

        protected override IExpression ConvertUnary(IUnaryExpression iue)
        {
            int parentIndex = context.InputStack.Count - 2;
            if (context.GetAncestor(parentIndex) is IConditionStatement)
            {
                return iue;
            }

            if (!CodeRecognizer.IsStochastic(context, iue.Expression))
            {
                return base.ConvertUnary(iue);
            }

            var convertedExpression = ConvertExpression(iue.Expression);
            switch (iue.Operator)
            {
                case UnaryOperator.Negate:
                    var factorExpr = FindFactorMethod(iue.Operator, false, convertedExpression);
                    if (factorExpr != null) return factorExpr;
                    if (convertedExpression.GetExpressionType() == typeof(double))
                    {
                        return FindFactorMethod(BinaryOperator.Subtract, true, Builder.LiteralExpr(0.0), convertedExpression);
                    }
                    throw new InvalidOperationException("Operator (-) does not have a registered factor for argument type " + convertedExpression.GetExpressionType() + ".");
                case UnaryOperator.BitwiseNot:
                case UnaryOperator.BooleanNot:
                    return FindFactorMethod(iue.Operator, true, convertedExpression);
            }

            throw new InvalidOperationException("Could not find factor to replace operator " + iue.Operator);
        }

        protected override IExpression ConvertBinary(IBinaryExpression ibe)
        {
            int parentIndex = context.InputStack.Count - 2;
            if (context.GetAncestor(parentIndex) is IConditionStatement)
            {
                return ibe;
            }

            var leftIsStochastic = CodeRecognizer.IsStochastic(context, ibe.Left);
            var rightIsStochastic = CodeRecognizer.IsStochastic(context, ibe.Right);
            if (!leftIsStochastic && !rightIsStochastic)
            {
                return base.ConvertBinary(ibe);
            }

            var convertedLeft = ConvertExpression(ibe.Left);
            var convertedRight = ConvertExpression(ibe.Right);
            var defaultFactor = FindFactorMethod(ibe.Operator, false, convertedLeft, convertedRight);

            switch (ibe.Operator)
            {
                case BinaryOperator.Add:
                case BinaryOperator.Subtract:
                case BinaryOperator.Multiply:
                case BinaryOperator.Divide:
                case BinaryOperator.Modulus:
                case BinaryOperator.BitwiseExclusiveOr:
                case BinaryOperator.BooleanAnd:
                case BinaryOperator.BooleanOr:
                case BinaryOperator.BitwiseAnd:
                case BinaryOperator.BitwiseOr:
                    return FindFactorMethod(ibe.Operator, true, convertedLeft, convertedRight);
                case BinaryOperator.IdentityEquality:
                    return defaultFactor ?? FindFactorMethod(UnaryOperator.Negate, true, FindFactorMethod(BinaryOperator.IdentityInequality, true, convertedLeft, convertedRight));
                case BinaryOperator.ValueEquality:
                    return defaultFactor ?? FindFactorMethod(UnaryOperator.Negate, true, FindFactorMethod(BinaryOperator.ValueInequality, true, convertedLeft, convertedRight));
                case BinaryOperator.IdentityInequality:
                    return defaultFactor ?? FindFactorMethod(UnaryOperator.Negate, true, FindFactorMethod(BinaryOperator.IdentityEquality, true, convertedLeft, convertedRight));
                case BinaryOperator.ValueInequality:
                    return defaultFactor ?? FindFactorMethod(UnaryOperator.Negate, true, FindFactorMethod(BinaryOperator.ValueEquality, true, convertedLeft, convertedRight));
                case BinaryOperator.LessThan:
                    return GreaterThan(convertedRight, convertedLeft);
                case BinaryOperator.GreaterThan:
                    return GreaterThan(convertedLeft, convertedRight);
                case BinaryOperator.LessThanOrEqual:
                    return GreaterThanOrEqual(convertedRight, convertedLeft);
                case BinaryOperator.GreaterThanOrEqual:
                    return GreaterThanOrEqual(convertedLeft, convertedRight);
            }

            throw new InvalidOperationException("Could not find factor to replace operator " + ibe.Operator);
        }

        private static IExpression NotOrNull(IExpression expr)
        {
            return (expr == null) ? null : FindFactorMethod(UnaryOperator.Negate, false, expr);
        }

        private static IExpression GreaterThan(IExpression a, IExpression b)
        {
            var f = FindFactorMethod(BinaryOperator.GreaterThan, false, a, b);
            if (f == null) f = FindFactorMethod(Variable.Operator.LessThan, false, b, a);
            if (f == null) f = NotOrNull(FindFactorMethod(Variable.Operator.LessThanOrEqual, false, a, b));
            if (f == null) f = NotOrNull(FindFactorMethod(Variable.Operator.GreaterThanOrEqual, false, b, a));
            if (f != null) return f;

            if (typeof(double).IsAssignableFrom(a.GetExpressionType()))
            {
                IExpression diff;
                // TODO what does this mean in our world?
                /*if (b.IsObserved && b.IsReadOnly && b.IsBase && b.ObservedValue.Equals(0.0))
                {
                    diff = a;
                }
                else if (a.IsObserved && a.IsReadOnly && a.IsBase && a.ObservedValue.Equals(0.0))
                {
                    return !GreaterThanOrEqual(b, a);
                }
                else*/
                {
                    diff = FindFactorMethod(BinaryOperator.Subtract, false, a, b);
                }
                if (diff == null)
                {
                    throw new InvalidOperationException("None of the operators (<,>,-) has a registered factor for argument type " + a.GetExpressionType() + ".");
                }
                return IsPositive(diff);
            }
            throw new InvalidOperationException("Neither of the operators (<,>) has a registered factor for argument type " + a.GetExpressionType() + ".");
        }

        private static IExpression GreaterThanOrEqual(IExpression a, IExpression b)
        {
            var f = FindFactorMethod(Variable.Operator.GreaterThanOrEqual, false, a, b);
            if (f == null) f = FindFactorMethod(Variable.Operator.LessThanOrEqual, false, b, a);
            if (f == null) f = NotOrNull(FindFactorMethod(Variable.Operator.LessThan, false, a, b));
            if (f == null) f = NotOrNull(FindFactorMethod(Variable.Operator.GreaterThan, false, b, a));
            if (f != null) return f;

            if (typeof(double).IsAssignableFrom(a.GetExpressionType()))
            {
                IExpression diff;
                // TODO what does this mean in our world?
                /*if (b.IsObserved && b.IsReadOnly && b.IsBase && b.ObservedValue.Equals(0.0))
                {
                    diff = a;
                }
                else if (a.IsObserved && a.IsReadOnly && a.IsBase && a.ObservedValue.Equals(0.0))
                {
                    return !GreaterThan(b, a);
                }
                else*/
                {
                    diff = FindFactorMethod(Variable.Operator.Minus, false, a, b);
                }
                if (diff == null)
                {
                    throw new InvalidOperationException("None of the operators (<,>,-) has a registered factor for argument type " + a.GetExpressionType() + ".");
                }
                return IsPositive(diff); // should be IsPositiveOrZero
            }
            else
            {
                throw new InvalidOperationException("Neither of the operators (<,>) has a registered factor for argument type " + a.GetExpressionType() + ".");
            }
        }

        private static IExpression IsPositive(IExpression x)
        {
            return Builder.StaticMethod(new Func<double, bool>(Factor.IsPositive), x);
        }

        private static IExpression FindFactorMethod(UnaryOperator op, bool throws, IExpression arg)
        {
            return FindFactorMethod(ConvertUnaryOperator(op), throws, arg);
        }

        private static IExpression FindFactorMethod(BinaryOperator op, bool throws, IExpression arg1, IExpression arg2)
        {
            return FindFactorMethod(ConvertBinaryOperator(op), throws, arg1, arg2);
        }

        private static IExpression FindFactorMethod(Variable.Operator? op, bool throws, params IExpression[] args)
        {
            if (!op.HasValue) return null;

            var exprTypes = args.Select(a => a.GetExpressionType()).ToArray();
            var del = Variable.LookupOperatorFactor(op.Value, exprTypes);
            if (del == null)
            {
                if (throws)
                {
                    throw new InvalidOperationException("No operator factor registered for : " + op + " with argument type " + exprTypes.Select(t => t.ToString()).Aggregate((s1, s2) => s1 + "," + s2) + ".");
                }
                return null;
            }

            var methodInfo = del.Method;
            if (methodInfo.GetParameters().Length != args.Length)
            {
                throw new Exception("argument list lengths do not match");
            }
            IMethodInvokeExpression imie;
            if (methodInfo.IsGenericMethod && !methodInfo.ContainsGenericParameters)
            {
                imie = Builder.StaticGenericMethod(methodInfo, args);
            }
            else
            {
                imie = Builder.StaticMethod(methodInfo, args);
            }
            return imie;
        }

        private static Variable.Operator? ConvertUnaryOperator(UnaryOperator input)
        {
            // TODO should some of this logic move into callers?
            switch (input)
            {
                case UnaryOperator.BooleanNot:
                    return Variable.Operator.Not;
                case UnaryOperator.BitwiseNot:
                    return Variable.Operator.Complement;
                case UnaryOperator.Negate:
                    return Variable.Operator.Negative;
                case UnaryOperator.PostIncrement:
                    // only supported in for statement iterator so should be left as is
                    return null;
                case UnaryOperator.PreIncrement:
                case UnaryOperator.PreDecrement:
                case UnaryOperator.PostDecrement:
                    // TODO these are not supported but throwing here fails tests that expect them to fail later
                    //throw new NotSupportedException();
                    return null;
                default:
                    throw new ArgumentOutOfRangeException("Unknown operator");
            }
        }

        private static Variable.Operator? ConvertBinaryOperator(BinaryOperator input)
        {
            switch (input)
            {
                case BinaryOperator.Add:
                    return Variable.Operator.Plus;
                case BinaryOperator.Subtract:
                    return Variable.Operator.Minus;
                case BinaryOperator.Multiply:
                    return Variable.Operator.Multiply;
                case BinaryOperator.Divide:
                    return Variable.Operator.Divide;
                case BinaryOperator.Modulus:
                    return Variable.Operator.Modulus;
                case BinaryOperator.ShiftLeft:
                    return Variable.Operator.LeftShift;
                case BinaryOperator.ShiftRight:
                    return Variable.Operator.RightShift;
                case BinaryOperator.ValueEquality:
                    return Variable.Operator.Equal;
                case BinaryOperator.ValueInequality:
                    return Variable.Operator.NotEqual;
                case BinaryOperator.BitwiseOr:
                case BinaryOperator.BooleanOr:
                    return Variable.Operator.Or;
                case BinaryOperator.BitwiseAnd:
                case BinaryOperator.BooleanAnd:
                    return Variable.Operator.And;
                case BinaryOperator.BitwiseExclusiveOr:
                    return Variable.Operator.Xor;
                case BinaryOperator.LessThan:
                    return Variable.Operator.LessThan;
                case BinaryOperator.LessThanOrEqual:
                    return Variable.Operator.LessThanOrEqual;
                case BinaryOperator.GreaterThan:
                    return Variable.Operator.GreaterThan;
                case BinaryOperator.GreaterThanOrEqual:
                    return Variable.Operator.GreaterThanOrEqual;
                default:
                    throw new ArgumentOutOfRangeException("Unknown operator");
            }
        }

        #endregion Factor insertion

        // DistributionAnalysis

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
            loopVars.Add(loopVar);
            IStatement st = base.ConvertFor(ifs);
            loopVars.Remove(loopVar);
            return st;
        }

        protected void SetMarginalPrototype(IExpression target, object targetDecl, MarginalPrototype mpa, MarginalPrototype mpa2, bool targetHasLiteralIndices)
        {
            if (!targetHasLiteralIndices)
            {
                if (mpa != null)
                    return;
                if (context.OutputAttributes.Has<MarginalPrototype>(targetDecl))
                    throw new InferCompilerException($"{targetDecl} already has a MarginalPrototype");
                context.OutputAttributes.Set(targetDecl, mpa2);
            }
            else
            {
                // input is: a[0][1] = Factor.Random(prior01);
                // output is: a_mp[0][1] = prior01;
                List<IList<IExpression>> indices = Recognizer.GetIndices(target);
                IVariableDeclaration mpVar;
                Containers declContainers;
                if (targetDecl is IParameterDeclaration)
                    declContainers = new Containers();
                else
                    declContainers = context.InputAttributes.Get<Containers>((IVariableDeclaration)targetDecl);
                if (mpa != null)
                {
                    // a marginal prototype array has already been created
                    mpVar = Recognizer.GetVariableDeclaration(mpa.prototypeExpression);
                    if (mpVar == null || !context.InputAttributes.Has<MarginalPrototypeVariable>(mpVar))
                        return;
                }
                else
                {
                    // create a marginal prototype array using indexVars and sizes at lower depths
                    VariableInformation vi = VariableInformation.GetVariableInformation(context, targetDecl);
                    int depth = indices.Count;
                    vi.DefineSizesUpToDepth(context, depth);
                    vi.DefineIndexVarsUpToDepth(context, depth);
                    List<IExpression[]> sizes2 = new List<IExpression[]>();
                    List<IVariableDeclaration[]> indexVars2 = new List<IVariableDeclaration[]>();
                    for (int i = 0; i < depth; i++)
                    {
                        sizes2.Add(vi.sizes[i]);
                        indexVars2.Add(vi.indexVars[i]);
                    }

                    Type sourceType;
                    if (mpa2.prototypeExpression != null)
                    {
                        sourceType = mpa2.prototypeExpression.GetExpressionType();
                    }
                    else
                    {
                        sourceType = mpa2.prototype.GetType();
                    }

                    Type tp = CodeBuilder.MakeJaggedArrayType(sourceType, sizes2);
                    string name = (targetDecl is IParameterDeclaration ipd) ? ipd.Name : ((IVariableDeclaration)targetDecl).Name;
                    name = VariableInformation.GenerateName(context, name + "_mp");
                    mpVar = Builder.VarDecl(name, tp);
                    IList<IStatement> stmts = Builder.StmtCollection();
                    Builder.NewJaggedArray(stmts, mpVar, indexVars2, sizes2);
                    int ancIndex = declContainers.GetMatchingAncestorIndex(context);
                    Containers missing = declContainers.GetContainersNotInContext(context, ancIndex);
                    stmts = Containers.WrapWithContainers(stmts, missing.outputs);
                    context.AddStatementsBeforeAncestorIndex(ancIndex, stmts);
                    context.InputAttributes.Set(mpVar, new MarginalPrototypeVariable());
                    VariableInformation vi_mp = VariableInformation.GetVariableInformation(context, mpVar);
                    vi_mp.sizes = sizes2;
                    vi_mp.indexVars = indexVars2;
                    MarginalPrototype mpa3 = new MarginalPrototype(null);
                    mpa3.prototypeExpression = Builder.JaggedArrayIndex(Builder.VarRefExpr(mpVar), vi_mp.GetIndexExpressions(context, depth));
                    if (context.OutputAttributes.Has<MarginalPrototype>(targetDecl))
                        throw new InferCompilerException($"{targetDecl} already has a MarginalPrototype");
                    context.OutputAttributes.Set(targetDecl, mpa3);
                }

                IExpression expr = mpa2.prototypeExpression;
                if (expr == null)
                {
                    expr = Quoter.Quote(mpa2.prototype);
                }

                IStatement st = Builder.AssignStmt(Builder.JaggedArrayIndex(Builder.VarRefExpr(mpVar), indices), expr);
                // put st into all declContainers and (all containers of context except stochastic conditionals)
                Containers containers = new Containers(context);
                containers = Containers.RemoveStochasticConditionals(containers, context);
                containers = Containers.Append(declContainers, containers);
                int ancIndex2 = containers.GetMatchingAncestorIndex(context);
                Containers missing2 = containers.GetContainersNotInContext(context, ancIndex2);
                st = Containers.WrapWithContainers(st, missing2.outputs);
                context.AddStatementAfterAncestorIndex(ancIndex2, st);
            }
        }

        /// <summary>
        /// Sets the MarginalPrototype attribute of the target variable of an assign expression.
        /// </summary>
        /// <param name="iae"></param>
        /// <param name="targetHasLiteralIndices"></param>
        protected void InferMarginalPrototype(IAssignExpression iae, bool targetHasLiteralIndices)
        {
            IExpression target = iae.Target;
            object targetDecl = Recognizer.GetDeclaration(target);
            if (targetDecl == null) return;
            // this must be done first because it has side-effects
            MarginalPrototype mpa = GetMarginalPrototype(target, targetDecl);
            if (!targetHasLiteralIndices && mpa != null)
                return;
            IMethodInvokeExpression imie = iae.Expression as IMethodInvokeExpression;
            if (imie == null)
            {
                // if this assignment was produced by EqualityPropagation, check for MarginalPrototype attribute containing the original rhs
                MarginalPrototype rhsPrototype = context.InputAttributes.Get<MarginalPrototype>(iae);
                if (rhsPrototype != null)
                {
                    IExpression rhs = rhsPrototype.prototypeExpression;
                    imie = rhs as IMethodInvokeExpression;
                }
                if (imie == null)
                    return;
            }
            MarginalPrototype mpa2 = InferMarginalPrototype(imie, target, targetDecl);
            if (mpa2 != null)
            {
                SetMarginalPrototype(target, targetDecl, mpa, mpa2, targetHasLiteralIndices);
            }
        }

        /// <summary>
        /// Attached to generated variables that store marginal prototypes
        /// </summary>
        private class MarginalPrototypeVariable : ICompilerAttribute
        {
        }

        protected MarginalPrototype InferMarginalPrototypeOut(IMethodInvokeExpression imie, int argIndex, IExpression target, object targetDecl)
        {
            if (Recognizer.IsStaticGenericMethod(imie, new FuncOut<PlaceHolder[], int, PlaceHolder[], PlaceHolder[]>(Collection.Split)))
            {
                IExpression source = imie.Arguments[0];
                IExpression count = imie.Arguments[1];
                return GetSplitMarginalPrototype(source, target, targetDecl, count);
            }
            else if (Recognizer.IsStaticGenericMethod(imie, new FuncOut<PlaceHolder, PlaceHolder, PlaceHolder>(LowPriority.SequentialCopy)))
            {
                IExpression source = imie.Arguments[0];
                return GetMarginalPrototype(source, targetDecl);
            }
            return null;
        }

        protected MarginalPrototype InferMarginalPrototype(IMethodInvokeExpression imie, IExpression target, object targetDecl)
        {
            if (Recognizer.IsStaticGenericMethod(imie, new Func<Sampleable<PlaceHolder>, PlaceHolder>(Factor.Random)))
            {
                IExpression source = CodeRecognizer.RemoveCast(imie.Arguments[0]);
                IExpression mpe = ReplaceIndices(source, targetDecl);
                SparsityAttribute sparsity = context.InputAttributes.Get<SparsityAttribute>(targetDecl);
                if (sparsity != null && sparsity.Sparsity != null)
                {
                    var exprType = mpe.GetExpressionType();
                    if (exprType == typeof(Dirichlet))
                    {
                        IExpression dimensionExpression = GetDirichletLengthExpression(mpe);
                        return GetDirichletPrototypeExpression(targetDecl, dimensionExpression); // deals with sparsity
                    }
                    else if (exprType == typeof(Discrete))
                    {
                        IExpression dimensionExpression = GetDiscreteLengthExpression(mpe);
                        return GetDiscretePrototypeExpression(targetDecl, dimensionExpression); // deals with sparsity
                    }
                    else if (exprType == typeof(SparseGaussianList))
                    {
                        IExpression dimensionExpression = GetSparseListLengthExpression(mpe, new Func<int, Microsoft.ML.Probabilistic.Distributions.SparseGaussianList>(Microsoft.ML.Probabilistic.Distributions.SparseGaussianList.FromSize));
                        return GetSparseGaussianListPrototypeExpression(targetDecl, dimensionExpression);
                    }
                    else if (exprType == typeof(SparseGammaList))
                    {
                        IExpression dimensionExpression = GetSparseListLengthExpression(mpe, new Func<int, Microsoft.ML.Probabilistic.Distributions.SparseGammaList>(Microsoft.ML.Probabilistic.Distributions.SparseGammaList.FromSize));
                        return GetSparseGammaListPrototypeExpression(targetDecl, dimensionExpression);
                    }
                    else if (exprType == typeof(SparseBernoulliList))
                    {
                        IExpression dimensionExpression = GetSparseListLengthExpression(mpe, new Func<int, Microsoft.ML.Probabilistic.Distributions.SparseBernoulliList>(Microsoft.ML.Probabilistic.Distributions.SparseBernoulliList.FromSize));
                        return GetSparseBernoulliListPrototypeExpression(targetDecl, dimensionExpression);
                    }
                    else if (exprType == typeof(SparseBetaList))
                    {
                        IExpression dimensionExpression = GetSparseListLengthExpression(mpe, new Func<int, Microsoft.ML.Probabilistic.Distributions.SparseBetaList>(Microsoft.ML.Probabilistic.Distributions.SparseBetaList.FromSize));
                        return GetSparseBetaListPrototypeExpression(targetDecl, dimensionExpression);
                    }
                    else if (exprType == typeof(BernoulliIntegerSubset))
                    {
                        IExpression dimensionExpression = GetSparseListLengthExpression(mpe, new Func<int, Microsoft.ML.Probabilistic.Distributions.BernoulliIntegerSubset>(Microsoft.ML.Probabilistic.Distributions.BernoulliIntegerSubset.FromSize));
                        return GetBernoulliIntegerSubsetPrototypeExpression(targetDecl, dimensionExpression);
                    }
                }
                MarginalPrototype mpa = new MarginalPrototype(null);
                mpa.prototypeExpression = mpe;
                return mpa;
            }
            else if (Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<PlaceHolder>, int, PlaceHolder>(Collection.GetItem)))
            {
                if (imie.Arguments.Count == 2)
                {
                    IExpression source = imie.Arguments[0];
                    IExpression index = imie.Arguments[1];
                    if (typeof(Vector).IsAssignableFrom(source.GetExpressionType()))
                    {
                        MarginalPrototype mpa = GetMarginalPrototype(source, targetDecl);
                        if (mpa != null)
                        {
                            if (mpa.prototype != null)
                            {
                                if (mpa.prototype is VectorGaussian)
                                    mpa = new MarginalPrototype(new Gaussian());
                                else if (mpa.prototype is Dirichlet)
                                    mpa = new MarginalPrototype(new Beta());
                            }
                            else if (mpa.prototypeExpression != null)
                            {
                                IExpression prototypeExpression = mpa.prototypeExpression;
                                if (typeof(VectorGaussian).IsAssignableFrom(prototypeExpression.GetExpressionType()))
                                {
                                    mpa = new MarginalPrototype(new Gaussian());
                                }
                                else if (typeof(Dirichlet).IsAssignableFrom(prototypeExpression.GetExpressionType()))
                                {
                                    mpa = new MarginalPrototype(new Beta());
                                }
                            }
                            return mpa;
                        }
                    }
                    else
                    {
                        IExpression source2 = Builder.ArrayIndex(source, index);
                        CopyIndexVariables(target, source2);
                        return GetMarginalPrototype(source2, targetDecl);
                    }
                }
            }
            else if (
                Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<PlaceHolder>, IReadOnlyList<int>, PlaceHolder[]>(Collection.GetItems)) ||
                Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<PlaceHolder>, IReadOnlyList<int>, PlaceHolder[]>(Collection.Subarray)) ||
                Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<PlaceHolder>, int[][], PlaceHolder[][]>(Collection.SplitSubarray))
                )
            {
                if (imie.Arguments.Count == 2)
                {
                    IExpression source = imie.Arguments[0];
                    IExpression indices = imie.Arguments[1];
                    int targetDepth = Recognizer.GetIndexingDepth(target);
                    VariableInformation targetInfo = VariableInformation.GetVariableInformation(context, targetDecl);
                    CopyIndexVariables(target, indices);
                    bool isSplitSubarray = Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<PlaceHolder>, int[][], PlaceHolder[][]>(Collection.SplitSubarray));
                    int indexingDepth = isSplitSubarray ? 2 : 1;
                    targetInfo.DefineIndexVarsUpToDepth(context, targetDepth + indexingDepth);
                    IExpression index = indices;
                    for (int depth = 0; depth < indexingDepth; depth++)
                    {
                        IVariableDeclaration indexVar = targetInfo.indexVars[targetDepth + depth][0];
                        var indexExpr = Builder.VarRefExpr(indexVar);
                        index = Builder.ArrayIndex(index, indexExpr);
                        target = Builder.ArrayIndex(target, indexExpr);
                    }
                    IExpression source2 = Builder.ArrayIndex(source, index);
                    CopyIndexVariables(target, source2);
                    // source's marginal prototype may use source's indexVars.  we need to map these into target's indexVars.
                    // this is done by indexing source with target's indexVars and calling GetMarginalPrototype.
                    for (int d = targetDepth + indexingDepth; d < targetInfo.indexVars.Count; d++)
                    {
                        IVariableDeclaration[] bracket = targetInfo.indexVars[d];
                        IExpression[] indexExprs = new IExpression[bracket.Length];
                        for (int i = 0; i < bracket.Length; i++)
                        {
                            indexExprs[i] = Builder.VarRefExpr(bracket[i]);
                        }
                        source2 = Builder.ArrayIndex(source2, indexExprs);
                    }
                    return GetMarginalPrototype(source2, targetDecl);
                }
            }
            else if (
                Recognizer.IsStaticGenericMethod(imie,
                                                 new Func<IList<PlaceHolder>, IList<string>, Dictionary<string, int>, PlaceHolder[]>(ExperimentalFactor.GetItemsWithDictionary))
                )
            {
                if (imie.Arguments.Count == 3)
                {
                    IExpression source = imie.Arguments[0];
                    IExpression indices = imie.Arguments[1];
                    IExpression dict = imie.Arguments[2];
                    object indicesDecl = Recognizer.GetDeclaration(indices);
                    VariableInformation indicesInfo = (indicesDecl == null) ? null : VariableInformation.GetVariableInformation(context, indicesDecl);
                    int targetDepth = Recognizer.GetIndexingDepth(target);
                    VariableInformation targetInfo = VariableInformation.GetVariableInformation(context, targetDecl);
                    IVariableDeclaration indexVar;
                    if (targetInfo.indexVars.Count <= targetDepth || targetInfo.indexVars[targetDepth] == null || targetInfo.indexVars[targetDepth][0] == null)
                    {
                        if (indicesInfo != null)
                        {
                            int indicesDepth = Recognizer.GetIndexingDepth(indices);
                            indexVar = indicesInfo.indexVars[indicesDepth][0];
                        }
                        else
                        {
                            indexVar = VariableInformation.GenerateLoopVar(context, "_iv");
                        }
                        targetInfo.SetIndexVariablesAtDepth(targetDepth, new IVariableDeclaration[] { indexVar });
                    }
                    else
                    {
                        indexVar = targetInfo.indexVars[targetDepth][0];
                    }
                    object sourceDecl = Recognizer.GetDeclaration(source);
                    VariableInformation sourceInfo = VariableInformation.GetVariableInformation(context, sourceDecl);
                    IExpression index = Builder.ArrayIndex(dict, Builder.ArrayIndex(indices, Builder.VarRefExpr(indexVar)));
                    //mpa.prototypeExpression = sourceInfo.GetMarginalPrototypeExpression(prototypeExpression, index);
                    CopyIndexVariables(target, source);
                    IExpression source2 = Builder.ArrayIndex(source, index);
                    // source's marginal prototype may use source's indexVars.  we need to map these into target's indexVars.
                    // this is done by indexing source with target's indexVars and calling GetMarginalPrototype.
                    for (int d = targetDepth + 1; d < targetInfo.indexVars.Count; d++)
                    {
                        IVariableDeclaration[] bracket = targetInfo.indexVars[d];
                        IExpression[] indexExprs = new IExpression[bracket.Length];
                        for (int i = 0; i < bracket.Length; i++)
                        {
                            indexExprs[i] = Builder.VarRefExpr(bracket[i]);
                        }
                        source2 = Builder.ArrayIndex(source2, indexExprs);
                    }
                    return GetMarginalPrototype(source2, targetDecl);
                }
            }
            else if (
                Recognizer.IsStaticMethod(imie, new Func<bool[], int>(Factor.CountTrue))
                )
            {
                IExpression source = imie.Arguments[0];
                IExpression lengthExpression = GetArrayLengthExpression(source, targetDecl);
                if (lengthExpression != null)
                {
                    lengthExpression = Builder.BinaryExpr(lengthExpression, BinaryOperator.Add, Builder.LiteralExpr(1));
                    return GetDiscretePrototypeExpression(targetDecl, lengthExpression);
                }
            }
            else if (
                Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, PlaceHolder>(Clone.Copy)) ||
                Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, PlaceHolder>(Diode.Copy)) ||
                Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, PlaceHolder>(Cut.Backward)) ||
                Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, bool, PlaceHolder>(Cut.ForwardWhen)) ||
                Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, double, PlaceHolder>(PowerPlate.Enter)) ||
                Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, double, PlaceHolder>(Damp.Forward)) ||
                Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, double, PlaceHolder>(Damp.Backward)) ||
                Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, PlaceHolder>(LowPriority.Backward)) ||
                Recognizer.IsStaticGenericMethod(imie, new FuncOut<PlaceHolder, PlaceHolder, PlaceHolder>(LowPriority.SequentialCopy))
                )
            {
                if (imie.Arguments.Count > 0)
                {
                    IExpression source = imie.Arguments[0];
                    CopyIndexVariables(target, source);
                    return GetMarginalPrototype(source, targetDecl);
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<int, int>(Factor.DiscreteUniform)))
            {
                IExpression arg = imie.Arguments[0];
                object argDecl = Recognizer.GetVariableDeclaration(arg);
                if (argDecl != null && context.InputAttributes.Has<MarginalPrototype>(argDecl))
                {
                    IExpression dimensionExpression = GetIntCardinalityExpression(arg, targetDecl);
                    if (dimensionExpression != null)
                    {
                        dimensionExpression = Builder.BinaryExpr(dimensionExpression, BinaryOperator.Subtract, Builder.LiteralExpr(1));
                        return GetDiscretePrototypeExpression(targetDecl, dimensionExpression);
                    }
                }
                else
                {
                    return GetDiscretePrototypeExpression(targetDecl, arg);
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<int, double, int>(Rand.Binomial))
                )
            {
                if (imie.Arguments.Count > 0)
                {
                    IExpression trialCountExpr = imie.Arguments[0];
                    MarginalPrototype mpa = GetMarginalPrototype(trialCountExpr, targetDecl);
                    if (mpa == null)
                    {
                        IExpression dimensionExpression = Builder.BinaryExpr(trialCountExpr, BinaryOperator.Add, Builder.LiteralExpr(1));
                        return GetDiscretePrototypeExpression(targetDecl, dimensionExpression);
                    }
                    return mpa;
                }
            }
            else if (Recognizer.IsStaticMethod(imie, typeof(Factor), "Plus") ||
                     Recognizer.IsStaticMethod(imie, typeof(NoSkip), "Plus")
                     )
            {
                if (imie.Arguments.Count == 2 &&
                    typeof(int).IsAssignableFrom(imie.Arguments[0].GetExpressionType()) &&
                    typeof(int).IsAssignableFrom(imie.Arguments[1].GetExpressionType())
                    )
                {
                    bool isPoisson1 = typeof(Poisson).Equals(GetMarginalType(imie.Arguments[0]));
                    bool isPoisson2 = typeof(Poisson).Equals(GetMarginalType(imie.Arguments[1]));
                    if (isPoisson1 || isPoisson2)
                    {
                        MarginalPrototype mpa = new MarginalPrototype(null);
                        mpa.prototypeExpression = Builder.StaticMethod(new Func<Poisson>(Poisson.Uniform));
                        return mpa;
                    }
                    else
                    {
                        // int plus
                        // the following code comes from CopyMarginalPrototype
                        IExpression dimension1 = GetIntCardinalityExpression(imie.Arguments[0], targetDecl);
                        IExpression dimension2 = GetIntCardinalityExpression(imie.Arguments[1], targetDecl);
                        if (dimension1 != null && dimension2 != null)
                        {
                            IExpression dimensionExpression = CardinalityOfBinaryExpression(dimension1, BinaryOperator.Add, dimension2);
                            return GetDiscretePrototypeExpression(targetDecl, dimensionExpression);
                        }
                    }
                }
                else
                {
                    // everything else
                    return GetFirstMarginalPrototype(imie.Arguments, targetDecl);
                }
            }
            else if (
                Recognizer.IsStaticMethod(imie, typeof(System.Math), "Max") ||
                Recognizer.IsStaticMethod(imie, typeof(System.Math), "Min")
                )
            {
                var mpa = GetFirstMarginalPrototype(imie.Arguments, targetDecl);
                if (IsGamma(mpa))
                    return new MarginalPrototype(new TruncatedGamma());
                else
                    return mpa;
            }
            else if (Recognizer.IsStaticMethod(imie, typeof(Factor), "Difference"))
            {
                if (imie.Arguments.Count == 2 &&
                    typeof(double).IsAssignableFrom(imie.Arguments[0].GetExpressionType()) &&
                    typeof(double).IsAssignableFrom(imie.Arguments[1].GetExpressionType())
                    )
                {
                    Type[] types = new[]
                    {
                        typeof(Beta),
                        typeof(Gamma),
                        typeof(Gaussian),
                    };
                    bool anyArgumentMatches = imie.Arguments.Any(arg => types.Any(type => type.Equals(GetMarginalType(arg))));
                    if (anyArgumentMatches)
                    {
                        MarginalPrototype mpa = new MarginalPrototype(null);
                        mpa.prototypeExpression = Builder.StaticMethod(new Func<Microsoft.ML.Probabilistic.Distributions.Gaussian>(Microsoft.ML.Probabilistic.Distributions.Gaussian.Uniform));
                        return mpa;
                    }
                    // fall through
                }
                return GetFirstMarginalPrototype(imie.Arguments, targetDecl);
            }
            else if (Recognizer.IsStaticMethod(imie, typeof(Factor), "Ratio"))
            {
                return GetFirstMarginalPrototype(imie.Arguments, targetDecl);
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<PositiveDefiniteMatrix, double, PositiveDefiniteMatrix>(Factor.Product)))
            {
                if (imie.Arguments.Count > 0 && typeof(PositiveDefiniteMatrix).IsAssignableFrom(imie.Arguments[0].GetExpressionType()))
                {
                    // matrix product
                    IExpression rowsExpression = GetMatrixRowsExpression(imie.Arguments[0], targetDecl);
                    if (rowsExpression != null)
                    {
                        MarginalPrototype mpa = new MarginalPrototype(null);
                        mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Microsoft.ML.Probabilistic.Distributions.Wishart>(Microsoft.ML.Probabilistic.Distributions.Wishart.Uniform), rowsExpression);
                        return mpa;
                    }
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<double, PositiveDefiniteMatrix, PositiveDefiniteMatrix>(Factor.Product)))
            {
                if (imie.Arguments.Count > 0 && typeof(PositiveDefiniteMatrix).IsAssignableFrom(imie.Arguments[1].GetExpressionType()))
                {
                    // matrix product
                    IExpression rowsExpression = GetMatrixRowsExpression(imie.Arguments[1], targetDecl);
                    if (rowsExpression != null)
                    {
                        MarginalPrototype mpa = new MarginalPrototype(null);
                        mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Microsoft.ML.Probabilistic.Distributions.Wishart>(Microsoft.ML.Probabilistic.Distributions.Wishart.Uniform), rowsExpression);
                        return mpa;
                    }
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<Matrix, Vector, Vector>(Factor.Product)) ||
                     Recognizer.IsStaticMethod(imie, new Func<double[,], Vector, Vector>(Factor.Product)))
            {
                IExpression rowsExpression;
                if (typeof(Matrix).IsAssignableFrom(imie.Arguments[0].GetExpressionType()))
                {
                    // matrix product
                    rowsExpression = GetMatrixRowsExpression(imie.Arguments[0], targetDecl);
                }
                else
                {
                    rowsExpression = GetArrayLengthExpression(imie.Arguments[0], targetDecl);
                }
                if (rowsExpression != null)
                {
                    MarginalPrototype mpa = new MarginalPrototype(null);
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Microsoft.ML.Probabilistic.Distributions.VectorGaussian>(Microsoft.ML.Probabilistic.Distributions.VectorGaussian.Uniform), rowsExpression);
                    return mpa;
                }
            }
            else if (Recognizer.IsStaticMethod(imie, typeof(Factor), "Product"))
            {
                if (imie.Arguments.Count == 2 &&
                    typeof(int).IsAssignableFrom(imie.Arguments[0].GetExpressionType()) &&
                    typeof(int).IsAssignableFrom(imie.Arguments[1].GetExpressionType())
                    )
                {
                    // int product
                    // if A ranges 0,...,(dim1-1) and B ranges 0,...,(dim2-1)
                    // then A*B ranges 0,...,(dim1-1)*(dim2-1)
                    IExpression dimension1 = GetIntCardinalityExpression(imie.Arguments[0], targetDecl);
                    IExpression dimension2 = GetIntCardinalityExpression(imie.Arguments[1], targetDecl);
                    if (dimension1 != null && dimension2 != null)
                    {
                        IExpression dimensionExpression = CardinalityOfBinaryExpression(dimension1, BinaryOperator.Multiply, dimension2);
                        return GetDiscretePrototypeExpression(targetDecl, dimensionExpression);
                    }
                }
                else
                {
                    // everything else
                    return GetFirstMarginalPrototype(imie.Arguments, targetDecl);
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<double, double, double>(Factor.Gaussian))
                     || Recognizer.IsStaticMethod(imie, new Func<double, double, double>(Factor.GaussianFromMeanAndVariance))
                     || Recognizer.IsStaticMethod(imie, new Func<double, double, double>(Gaussian.Sample))
                     || Recognizer.IsStaticMethod(imie, new Func<double, double, double>(Rand.Normal))
                     || Recognizer.IsStaticMethod(imie, new Func<IList<double>, double>(Factor.Sum))
                )
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                mpa.prototypeExpression = Builder.StaticMethod(new Func<Microsoft.ML.Probabilistic.Distributions.Gaussian>(Microsoft.ML.Probabilistic.Distributions.Gaussian.Uniform));
                return mpa;
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<double, double, double, double, double>(TruncatedGaussian.Sample))
                )
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                mpa.prototypeExpression = Builder.StaticMethod(new Func<Microsoft.ML.Probabilistic.Distributions.TruncatedGaussian>(Microsoft.ML.Probabilistic.Distributions.TruncatedGaussian.Uniform));
                return mpa;
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<double, double, double, double, double>(Factor.TruncatedGammaFromShapeAndRate))
                )
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                mpa.prototypeExpression = Builder.StaticMethod(new Func<Microsoft.ML.Probabilistic.Distributions.TruncatedGamma>(Microsoft.ML.Probabilistic.Distributions.TruncatedGamma.Uniform));
                return mpa;
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<double, double, double>(Factor.BetaFromMeanAndTotalCount))
                     || Recognizer.IsStaticMethod(imie, new Func<double, double, double>(Beta.SampleFromMeanAndVariance))
                     || Recognizer.IsStaticMethod(imie, new Func<double, double, double>(Beta.Sample))
                     || Recognizer.IsStaticMethod(imie, new Func<double>(Rand.Double))
                     || Recognizer.IsStaticMethod(imie, new Func<double, double, double>(Rand.Beta))
                     || Recognizer.IsStaticMethod(imie, new Func<double, double>(MMath.Logistic))
                )
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                mpa.prototypeExpression = Builder.StaticMethod(new Func<Microsoft.ML.Probabilistic.Distributions.Beta>(Microsoft.ML.Probabilistic.Distributions.Beta.Uniform));
                return mpa;
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<double, double>(System.Math.Exp))
                     || Recognizer.IsStaticMethod(imie, new Func<double, double, double>(Factor.GammaFromShapeAndRate))
                     || Recognizer.IsStaticMethod(imie, new Func<double, double, double>(Gamma.Sample))
                     || Recognizer.IsStaticMethod(imie, new Func<double, double, double>(Gamma.SampleFromMeanAndVariance))
                     || Recognizer.IsStaticMethod(imie, new Func<double, double>(Rand.Gamma))
                )
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                mpa.prototypeExpression = Builder.StaticMethod(new Func<Microsoft.ML.Probabilistic.Distributions.Gamma>(Microsoft.ML.Probabilistic.Distributions.Gamma.Uniform));
                return mpa;
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<IList<double>, Vector>(MMath.Softmax)))
            {
                if (imie.Arguments.Count == 1)
                {
                    MarginalPrototype mpa = new MarginalPrototype(null);
                    IExpression arg = imie.Arguments[0];
                    Type argType = arg.GetExpressionType();
                    IExpression dimensionExpression = (argType.IsArray)
                                                          ? GetArrayLengthExpression(arg, targetDecl)
                                                          : GetVectorLengthExpression(arg, targetDecl);
                    return GetSoftmaxPrototypeExpression(targetDecl, dimensionExpression);
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<double, int>(Factor.Poisson))
                     || Recognizer.IsStaticMethod(imie, new Func<double, int>(Poisson.Sample))
                )
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                mpa.prototypeExpression = Builder.StaticMethod(new Func<Microsoft.ML.Probabilistic.Distributions.Poisson>(Microsoft.ML.Probabilistic.Distributions.Poisson.Uniform));
                return mpa;
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<Vector, PositiveDefiniteMatrix, Vector>(Factor.VectorGaussian))
                     || Recognizer.IsStaticMethod(imie, new Func<Vector, PositiveDefiniteMatrix, Vector>(VectorGaussian.Sample))
                     || Recognizer.IsStaticMethod(imie, new Func<Vector, PositiveDefiniteMatrix, Vector>(VectorGaussian.SampleFromMeanAndVariance))
                )
            {
                if (imie.Arguments.Count > 0)
                {
                    IExpression dimensionExpression = GetVectorLengthExpression(imie.Arguments[0], targetDecl);
                    if (dimensionExpression != null)
                    {
                        MarginalPrototype mpa = new MarginalPrototype(null);
                        mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Microsoft.ML.Probabilistic.Distributions.VectorGaussian>(Microsoft.ML.Probabilistic.Distributions.VectorGaussian.Uniform), dimensionExpression);
                        return mpa;
                    }
                    else
                    {
                        return GetMarginalPrototype(imie.Arguments[0], targetDecl);
                    }
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<Vector[], Vector>(Factor.Sum))
                     || Recognizer.IsStaticMethod(imie, new Func<IList<Vector>, Vector>(Factor.Sum)))
            {
                if (imie.Arguments.Count > 0)
                {
                    IExpression source = Builder.ArrayIndex(imie.Arguments[0], Builder.LiteralExpr(0));
                    IExpression dimensionExpression = GetVectorLengthExpression(source, targetDecl);

                    if (dimensionExpression != null)
                    {
                        MarginalPrototype mpa = new MarginalPrototype(null);
                        mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Microsoft.ML.Probabilistic.Distributions.VectorGaussian>(Microsoft.ML.Probabilistic.Distributions.VectorGaussian.Uniform), dimensionExpression);
                        return mpa;
                    }
                    else
                    {
                        return GetMarginalPrototype(source, targetDecl);
                    }
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<double, PositiveDefiniteMatrix, PositiveDefiniteMatrix, PositiveDefiniteMatrix>(Wishart.Sample))
                     || Recognizer.IsStaticMethod(imie, new Func<double, PositiveDefiniteMatrix, PositiveDefiniteMatrix>(Wishart.SampleFromShapeAndScale))
                     || Recognizer.IsStaticMethod(imie, new Func<double, PositiveDefiniteMatrix, PositiveDefiniteMatrix>(Wishart.SampleFromShapeAndRate))
                )
            {
                if (imie.Arguments.Count >= 2)
                {
                    IExpression dimensionExpression = GetMatrixRowsExpression(imie.Arguments[1], targetDecl);
                    if (dimensionExpression != null)
                    {
                        MarginalPrototype mpa = new MarginalPrototype(null);
                        mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Microsoft.ML.Probabilistic.Distributions.Wishart>(Microsoft.ML.Probabilistic.Distributions.Wishart.Uniform), dimensionExpression);
                        return mpa;
                    }
                    else
                    {
                        return GetMarginalPrototype(imie.Arguments[1], targetDecl);
                    }
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<double[], Vector>(Vector.FromArray)))
            {
                if (imie.Arguments.Count == 1)
                {
                    MarginalPrototype mpa = new MarginalPrototype(null);
                    IExpression dimensionExpression = GetArrayLengthExpression(imie.Arguments[0], targetDecl);
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Microsoft.ML.Probabilistic.Distributions.VectorGaussian>(Microsoft.ML.Probabilistic.Distributions.VectorGaussian.Uniform),
                                                                   dimensionExpression);
                    return mpa;
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<Vector, Vector, Vector>(Vector.Concat)))
            {
                if (imie.Arguments.Count == 2)
                {
                    IExpression dimension1 = GetVectorLengthExpression(imie.Arguments[0], targetDecl);
                    IExpression dimension2 = GetVectorLengthExpression(imie.Arguments[1], targetDecl);
                    if (dimension1 != null && dimension2 != null)
                    {
                        IExpression dimensionExpression = Builder.BinaryExpr(dimension1, BinaryOperator.Add, dimension2);
                        MarginalPrototype mpa = new MarginalPrototype(null);
                        mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Microsoft.ML.Probabilistic.Distributions.VectorGaussian>(Microsoft.ML.Probabilistic.Distributions.VectorGaussian.Uniform), dimensionExpression);
                        return mpa;
                    }
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<Vector, int, int, Vector>(Vector.Subvector)))
            {
                if (imie.Arguments.Count == 3)
                {
                    IExpression dimensionExpression = imie.Arguments[2];
                    MarginalPrototype mpa = new MarginalPrototype(null);
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Microsoft.ML.Probabilistic.Distributions.VectorGaussian>(Microsoft.ML.Probabilistic.Distributions.VectorGaussian.Uniform), dimensionExpression);
                    return mpa;
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<Vector, int>(Factor.Discrete))
                     || Recognizer.IsStaticMethod(imie, new Func<Vector, int>(Rand.Sample))
                     || Recognizer.IsStaticMethod(imie, new Func<Vector, int>(Discrete.Sample))
                )
            {
                if (imie.Arguments.Count > 0)
                {
                    // the following code comes from CopyMarginalPrototype
                    IExpression dimensionExpression = GetVectorLengthExpression(imie.Arguments[0], targetDecl);
                    return GetDiscretePrototypeExpression(targetDecl, dimensionExpression);
                }
            }
            else if (Recognizer.IsStaticMethod(imie, typeof(Factor), "DiscreteFromLogProbs"))
            {
                if (imie.Arguments.Count == 1)
                {
                    IExpression dimensionExpression = GetArrayLengthExpression(imie.Arguments[0], targetDecl);
                    return GetDiscretePrototypeExpression(targetDecl, dimensionExpression);
                }
            }
            else if (Recognizer.IsStaticMethod(imie, typeof(EnumSupport), "DiscreteEnum")
                     || Recognizer.IsStaticMethod(imie, typeof(DiscreteEnum<>), "Sample")
                )
            {
                if (imie.Arguments.Count > 0)
                {
                    // the following code comes from CopyMarginalPrototype
                    IExpression dimensionExpression = GetVectorLengthExpression(imie.Arguments[0], targetDecl);
                    if (dimensionExpression != null)
                    {
                        MarginalPrototype mpa = new MarginalPrototype(null);
                        Type enumType = imie.GetExpressionType();
                        Type distType = typeof(DiscreteEnum<>).MakeGenericType(enumType);
                        mpa.prototypeExpression = Builder.StaticMethod(distType.GetMethod("Uniform", BindingFlags.Public | BindingFlags.FlattenHierarchy | BindingFlags.Static));
                        return mpa;
                    }
                }
            }
            else if (Recognizer.IsStaticMethod(imie, typeof(EnumSupport), "EnumToInt"))
            {
                Type enumType = imie.Arguments[0].GetExpressionType();
                Array values = Enum.GetValues(enumType);
                int maxValue = 0;
                for (int i = 0; i < values.Length; i++)
                {
                    int value = (int)values.GetValue(i);
                    if (value > maxValue)
                        maxValue = value;
                }
                IExpression dimensionExpression = Builder.LiteralExpr(maxValue + 1);
                return GetDiscretePrototypeExpression(targetDecl, dimensionExpression);
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<Vector, Vector>(Dirichlet.SampleFromPseudoCounts))
                     || Recognizer.IsStaticMethod(imie, new Func<Vector, double, Vector>(Factor.DirichletFromMeanAndTotalCount))
                )
            {
                IExpression dimensionExpression = GetVectorLengthExpression(imie.Arguments[0], targetDecl);
                return GetDirichletPrototypeExpression(targetDecl, dimensionExpression);
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<int, double, Vector>(Factor.DirichletSymmetric)))
            {
                IExpression dimensionExpression = imie.Arguments[0];
                return GetDirichletPrototypeExpression(targetDecl, dimensionExpression);
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<ISparseList<double>, ISparseList<double>, ISparseList<double>>(SparseGaussianList.Sample)))
            {
                IExpression dimensionExpression = GetVectorLengthExpression(imie.Arguments[0], targetDecl);
                return GetSparseGaussianListPrototypeExpression(targetDecl, dimensionExpression);
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<ISparseList<double>, ISparseList<double>, ISparseList<double>>(SparseGammaList.Sample)))
            {
                IExpression dimensionExpression = GetVectorLengthExpression(imie.Arguments[0], targetDecl);
                return GetSparseGammaListPrototypeExpression(targetDecl, dimensionExpression);
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<ISparseList<double>, ISparseList<bool>>(SparseBernoulliList.Sample)))
            {
                IExpression dimensionExpression = GetVectorLengthExpression(imie.Arguments[0], targetDecl);
                return GetSparseBernoulliListPrototypeExpression(targetDecl, dimensionExpression);
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<ISparseList<double>, ISparseList<double>, ISparseList<double>>(SparseBetaList.Sample)))
            {
                IExpression dimensionExpression = GetVectorLengthExpression(imie.Arguments[0], targetDecl);
                return GetSparseBetaListPrototypeExpression(targetDecl, dimensionExpression);
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<ISparseList<double>, IList<int>>(BernoulliIntegerSubset.Sample)))
            {
                IExpression dimensionExpression = GetVectorLengthExpression(imie.Arguments[0], targetDecl);
                return GetBernoulliIntegerSubsetPrototypeExpression(targetDecl, dimensionExpression);
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<double, double, double>(System.Math.Pow)))
            {
                MarginalPrototype mp = GetMarginalPrototype(imie.Arguments[0], targetDecl);
                if (IsGamma(mp))
                {
                    MarginalPrototype mpa = new MarginalPrototype(null);
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<double, GammaPower>(GammaPower.Uniform), imie.Arguments[1]);
                    return mpa;
                }
                else if (IsGammaPower(mp))
                {
                    IExpression powerExpression;
                    if (mp.prototype is GammaPower gp)
                        powerExpression = Builder.LiteralExpr(gp.Power);
                    else
                        powerExpression = GetGammaPowerExpression(mp.prototypeExpression);
                    powerExpression = Builder.BinaryExpr(BinaryOperator.Multiply, powerExpression, imie.Arguments[1]);
                    MarginalPrototype mpa = new MarginalPrototype(null);
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<double, GammaPower>(GammaPower.Uniform), powerExpression);
                    return mpa;
                }
            }
            else if (Recognizer.IsStaticMethod(imie, new Func<IList<double>, int>(MMath.IndexOfMaximumDouble)))
            {
                IExpression dimensionExpression = GetArrayLengthExpression(imie.Arguments[0], targetDecl);
                return GetDiscretePrototypeExpression(targetDecl, dimensionExpression);
            }
            else if (Recognizer.IsStaticGenericMethod(imie, new FuncOut<PlaceHolder[], int, PlaceHolder[], PlaceHolder[]>(Collection.Split)))
            {
                IExpression source = imie.Arguments[0];
                return GetSplitMarginalPrototype(source, target, targetDecl);
            }
            //} else {
            //  // check for a stochastic attribute on the method
            //  MethodBase factorMethod = Recognizer.GetMethodReference(imie).MethodInfo;
            //  object[] attrs = factorMethod.GetCustomAttributes(typeof(Stochastic), false);
            //  if (attrs.Length > 0) {
            //    Stochastic attr = (Stochastic)attrs[0];
            //    if (attr.Type != null && attr.MethodName != null) {
            //      mpa = new MarginalPrototype(null);
            //      MethodInfo factoryMethod = new MethodReference(attr.Type, attr.MethodName).GetMethodInfo();
            //      mpa.prototypeExpression = Builder.StaticMethod(factoryMethod, imie.Arguments.ToArray<IExpression>());
            //      context.OutputAttributes.Set(targetDecl, mpa);
            //    }
            //  }
            return null;
        }

        private static bool IsGammaPower(MarginalPrototype mp)
        {
            return mp != null && (mp.prototype is GammaPower || mp.prototypeExpression?.GetExpressionType() == typeof(GammaPower));
        }

        private static bool IsGamma(MarginalPrototype mp)
        {
            return mp != null && (mp.prototype is Gamma || mp.prototypeExpression?.GetExpressionType() == typeof(Gamma));
        }

        private static IExpression CardinalityOfBinaryExpression(IExpression dimension1, BinaryOperator op, IExpression dimension2)
        {
            var one = Builder.LiteralExpr(1);
            if (op == BinaryOperator.Add)
            {
                // if A ranges 0,...,(dim1-1) and B ranges 0,...,(dim2-1)
                // then A+B ranges 0,...,(dim1+dim2-2)
                return Builder.BinaryExpr(
                    Builder.BinaryExpr(dimension1, op, dimension2),
                    BinaryOperator.Subtract, one);
            }
            else
            {
                dimension1 = Builder.BinaryExpr(dimension1, BinaryOperator.Subtract, one);
                dimension2 = Builder.BinaryExpr(dimension2, BinaryOperator.Subtract, one);
                return Builder.BinaryExpr(
                    Builder.BinaryExpr(dimension1, op, dimension2),
                    BinaryOperator.Add, one);
            }
        }

        protected MarginalPrototype GetSplitMarginalPrototype(IExpression source, IExpression target, object targetDecl, IExpression offset = null)
        {
            int targetDepth = Recognizer.GetIndexingDepth(target);
            VariableInformation targetInfo = VariableInformation.GetVariableInformation(context, targetDecl);
            IVariableDeclaration indexVar;
            if (targetInfo.indexVars.Count <= targetDepth || targetInfo.indexVars[targetDepth] == null || targetInfo.indexVars[targetDepth][0] == null)
            {
                indexVar = VariableInformation.GenerateLoopVar(context, "_iv");
                targetInfo.SetIndexVariablesAtDepth(targetDepth, new IVariableDeclaration[] { indexVar });
            }
            else
            {
                indexVar = targetInfo.indexVars[targetDepth][0];
            }
            object sourceDecl = Recognizer.GetDeclaration(source);
            VariableInformation sourceInfo = VariableInformation.GetVariableInformation(context, sourceDecl);
            CopyIndexVariables(target, source);
            IExpression index = Builder.VarRefExpr(indexVar);
            if (offset != null)
                index = Builder.BinaryExpr(BinaryOperator.Add, index, offset);
            IExpression source2 = Builder.ArrayIndex(source, index);
            // source's marginal prototype may use source's indexVars.  we need to map these into target's indexVars.
            // this is done by indexing source with target's indexVars and calling GetMarginalPrototype.
            for (int d = targetDepth + 1; d < targetInfo.indexVars.Count; d++)
            {
                IVariableDeclaration[] bracket = targetInfo.indexVars[d];
                IExpression[] indexExprs = new IExpression[bracket.Length];
                for (int i = 0; i < bracket.Length; i++)
                {
                    indexExprs[i] = Builder.VarRefExpr(bracket[i]);
                }
                source2 = Builder.ArrayIndex(source2, indexExprs);
            }
            return GetMarginalPrototype(source2, targetDecl);
        }

        protected MarginalPrototype GetFirstMarginalPrototype(IList<IExpression> sources, object targetDecl)
        {
            foreach (IExpression source in sources)
            {
                MarginalPrototype mpa = GetMarginalPrototype(source, targetDecl);
                if (mpa != null) return mpa;
            }
            return null;
        }

        protected static bool AreCompatible(MarginalPrototype mpa, MarginalPrototype mpa2)
        {
            if (ReferenceEquals(mpa2, mpa))
                return true;
            bool compatible = (mpa2.prototypeExpression == mpa.prototypeExpression) ||
                (mpa2.prototypeExpression != null && mpa2.prototypeExpression.Equals(mpa.prototypeExpression));
            if (compatible && mpa2.prototype != mpa.prototype)
            {
                if (mpa.prototype is SettableToUniform prototype && mpa2.prototype is SettableToUniform prototype2)
                {
                    prototype.SetToUniform();
                    prototype2.SetToUniform();
                    compatible = prototype.Equals(prototype2);
                }
                else
                    compatible = false;
            }
            return compatible;
        }

        /// <summary>
        /// Get the marginal prototype of an expression, with all indices substituted.
        /// </summary>
        /// <param name="source">A variable reference or array indexer expression</param>
        /// <param name="targetDecl">A variable or parameter declaration</param>
        /// <returns></returns>
        protected MarginalPrototype GetMarginalPrototype(IExpression source, object targetDecl)
        {
            object sourceDecl = Recognizer.GetDeclaration(source);
            if (sourceDecl == null) return null;
            MarginalPrototype mpa;
            var mpas = context.InputAttributes.GetAll<MarginalPrototype>(sourceDecl);
            if (mpas.Count == 0)
                mpa = null;
            else
            {
                mpa = mpas[0];
                if (mpas.Count > 1)
                {
                    bool allCompatible = mpas.TrueForAll(mpa2 => AreCompatible(mpa, mpa2));
                    if (allCompatible)
                    {
                        context.OutputAttributes.Remove<MarginalPrototype>(sourceDecl);
                        context.OutputAttributes.Set(sourceDecl, mpa);
                    }
                    else
                    {
                        Error($"{sourceDecl} has multiple inconsistent MarginalPrototype attributes");
                    }
                }
            }
            if (mpa == null)
            {
                ValueRange vr = context.InputAttributes.Get<ValueRange>(sourceDecl);
                if (vr != null && source.GetExpressionType().Equals(typeof(int)))
                {
                    IExpression dimensionExpression = vr.Range.GetSizeExpression();
                    mpa = GetDiscretePrototypeExpression(sourceDecl, dimensionExpression);
                    context.OutputAttributes.Set(sourceDecl, mpa);
                    // fall through so that indices in the ValueRange get replaced
                }
                else return null;
            }

            if (mpa.prototype != null) return mpa;
            IExpression prototypeExpression = mpa.prototypeExpression;
            // Indices of this expression
            source = ReplaceIndices(source, targetDecl);
            List<IList<IExpression>> indices = Recognizer.GetIndices(source);
            if (indices.Count != 0)
            {
                VariableInformation sourceInfo = VariableInformation.GetVariableInformation(context, sourceDecl);
                try
                {
                    prototypeExpression = sourceInfo.GetMarginalPrototypeExpression(context, prototypeExpression, indices);
                }
                catch (Exception ex)
                {
                    Error(ex.Message);
                }
            }

            if (prototypeExpression == mpa.prototypeExpression) return mpa;

            mpa = new MarginalPrototype(null);
            mpa.prototypeExpression = prototypeExpression;
            return mpa;
        }

        protected Type GetMarginalType(IExpression source)
        {
            IVariableDeclaration sourceDecl = Recognizer.GetVariableDeclaration(source);
            if (sourceDecl == null) return null;
            MarginalPrototype mpa = context.OutputAttributes.Get<MarginalPrototype>(sourceDecl);
            if (mpa == null) return null;
            if (mpa.prototype != null) return mpa.GetType();
            if (mpa.prototypeExpression != null) return mpa.prototypeExpression.GetExpressionType();
            return null;
        }

        protected bool HasExtraIndices(IExpression expr, object targetDecl)
        {
            IExpression exprWithoutIndices = ReplaceIndices(expr, targetDecl);
            return !exprWithoutIndices.Equals(expr);
        }

        /// <summary>
        /// If expr has indices which are not indexVars of the target, replace them with zeros.
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="targetDecl">The target</param>
        /// <returns></returns>
        protected IExpression ReplaceIndices(IExpression expr, object targetDecl)
        {
            Containers containers = new Containers();
            Set<IVariableDeclaration> loopVars = new Set<IVariableDeclaration>();
            // note we can't easily get the loop vars from the ancestor statements, because we may be conditioning on some of them
            if (targetDecl is IVariableDeclaration)
            {
                LoopContext lc = context.InputAttributes.Get<LoopContext>(targetDecl);
                if (lc == null)
                {
                    Error(targetDecl + " is missing LoopContext");
                }
                else
                {
                    loopVars.AddRange(lc.loopVariables);
                }
                containers = context.InputAttributes.Get<Containers>(targetDecl);
            }
            VariableInformation vi = context.InputAttributes.Get<VariableInformation>(targetDecl);
            if (vi != null)
            {
                foreach (IVariableDeclaration[] ivds in vi.indexVars)
                {
                    foreach (IVariableDeclaration ivd in ivds)
                    {
                        if (ivd != null)
                        {
                            loopVars.Add(ivd);
                            Containers c = context.InputAttributes.Get<Containers>(ivd);
                            if (c != null)
                            {
                                containers = Containers.Append(containers, c);
                            }
                        }
                    }
                }
            }
            return ReplaceIndices(containers, loopVars, expr);
        }

        /// <summary>
        /// If expr has indices which reference variables not in loopVars, replace the indices with zeros. 
        /// </summary>
        /// <param name="containers"></param>
        /// <param name="keepVars"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
        protected IExpression ReplaceIndices(Containers containers, Set<IVariableDeclaration> keepVars, IExpression expr)
        {
            if (expr is ICastExpression ice)
            {
                return Builder.CastExpr(ReplaceIndices(containers, keepVars, ice.Expression), ice.TargetType);
            }
            else if (expr is ICheckedExpression iche)
            {
                return Builder.CheckedExpr(ReplaceIndices(containers, keepVars, iche.Expression));
            }
            else if (expr is IUnaryExpression iue)
            {
                return Builder.UnaryExpr(iue.Operator, ReplaceIndices(containers, keepVars, iue.Expression));
            }
            else if (expr is IBinaryExpression ibe)
            {
                var newLeft = ReplaceIndices(containers, keepVars, ibe.Left);
                var newRight = ReplaceIndices(containers, keepVars, ibe.Right);
                return Builder.BinaryExpr(newLeft, ibe.Operator, newRight);
            }
            else if (expr is IArrayIndexerExpression iaie)
            {
                IExpression[] newIndices = new IExpression[iaie.Indices.Count];
                for (int i = 0; i < newIndices.Length; i++)
                {
                    newIndices[i] = ReplaceVarsNotContained(containers, keepVars, iaie.Indices[i]);
                }
                return Builder.ArrayIndex(ReplaceIndices(containers, keepVars, iaie.Target), newIndices);
            }
            else if (expr is IMethodInvokeExpression imie)
            {
                IMethodInvokeExpression imie2 = Builder.MethodInvkExpr();
                imie2.Method = imie.Method;
                if (imie.Method.Method.Name.Equals("get_Item"))
                {
                    foreach (IExpression arg in imie.Arguments)
                    {
                        imie2.Arguments.Add(ReplaceVarsNotContained(containers, keepVars, arg));
                    }
                }
                else
                {
                    foreach (IExpression arg in imie.Arguments)
                    {
                        imie2.Arguments.Add(ReplaceIndices(containers, keepVars, arg));
                    }
                }
                return imie2;
            }
            else if (expr is IPropertyReferenceExpression ipre)
            {
                return Builder.PropRefExpr(ReplaceIndices(containers, keepVars, ipre.Target), ipre.Property);
            }
            return expr;
        }

        /// <summary>
        /// If expr has references to variables not in containers, replace them with zeros.
        /// </summary>
        /// <param name="containers">Containers of variables to keep</param>
        /// <param name="keepVars"></param>
        /// <param name="expr">Input expression</param>
        /// <returns>expr or a new expression with variables replaced.</returns>
        /// <remarks>Only direct variable references or indices are replaced.</remarks>
        protected IExpression ReplaceVarsNotContained(Containers containers, Set<IVariableDeclaration> keepVars, IExpression expr)
        {
            if (expr is IVariableReferenceExpression)
            {
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
                bool isStoc = CodeRecognizer.IsStochastic(context, ivd);
                Containers c = context.InputAttributes.Get<Containers>(ivd);
                if (c != null)
                {
                    // stochastic conditionals will be later removed from a deterministic variable, so we don't need to consider those
                    c = Containers.RemoveStochasticConditionals(c, context);
                }
                if (isStoc || (!keepVars.Contains(ivd) && (c == null || !containers.Contains(c))))
                {
                    // didn't match
                    return Builder.LiteralExpr(0);
                }
            }
            else if (expr is IArrayIndexerExpression iaie)
            {
                IExpression target = ReplaceVarsNotContained(containers, keepVars, iaie.Target);
                if (target is ILiteralExpression ile && ile.Value.Equals(0))
                    return Builder.LiteralExpr(0);
                else
                {
                    IExpression[] newIndices = new IExpression[iaie.Indices.Count];
                    for (int i = 0; i < newIndices.Length; i++)
                    {
                        newIndices[i] = ReplaceVarsNotContained(containers, keepVars, iaie.Indices[i]);
                    }
                    IArrayIndexerExpression aie = Builder.ArrayIndex(target, newIndices);
                    return aie;
                }
            }
            return expr;
        }

        private static IExpression GetDirichletLengthExpression(IExpression prototypeExpression)
        {
            if (Recognizer.IsStaticMethod(prototypeExpression, typeof(Microsoft.ML.Probabilistic.Distributions.Dirichlet), "Uniform") ||
                Recognizer.IsStaticMethod(prototypeExpression, new Func<Vector, Dirichlet>(Dirichlet.FromMeanLog)))
            {
                IMethodInvokeExpression imie = (IMethodInvokeExpression)prototypeExpression;
                return imie.Arguments[0];
            }
            else if (prototypeExpression.GetExpressionType().Equals(typeof(Microsoft.ML.Probabilistic.Distributions.Dirichlet)))
            {
                return Builder.PropRefExpr(prototypeExpression, typeof(Microsoft.ML.Probabilistic.Distributions.Dirichlet), "Dimension", typeof(int));
            }
            else
                return null;
        }

        internal static IExpression GetDiscreteLengthExpression(IExpression prototypeExpression)
        {
            if (Recognizer.IsStaticMethod(prototypeExpression, typeof(Microsoft.ML.Probabilistic.Distributions.Discrete), "Uniform"))
            {
                IMethodInvokeExpression imie = (IMethodInvokeExpression)prototypeExpression;
                return imie.Arguments[0];
            }
            else if (prototypeExpression.GetExpressionType().Equals(typeof(Microsoft.ML.Probabilistic.Distributions.Discrete)))
            {
                return Builder.PropRefExpr(prototypeExpression, typeof(Microsoft.ML.Probabilistic.Distributions.Discrete), "Dimension", typeof(int));
            }
            else
                return null;
        }

        private static IExpression GetVectorGaussianLengthExpression(IExpression prototypeExpression)
        {
            if (Recognizer.IsStaticMethod(prototypeExpression, typeof(Microsoft.ML.Probabilistic.Distributions.VectorGaussian), "Uniform"))
            {
                IMethodInvokeExpression imie = (IMethodInvokeExpression)prototypeExpression;
                return imie.Arguments[0];
            }
            else if (prototypeExpression.GetExpressionType().Equals(typeof(Microsoft.ML.Probabilistic.Distributions.VectorGaussian)))
            {
                return Builder.PropRefExpr(prototypeExpression, typeof(Microsoft.ML.Probabilistic.Distributions.VectorGaussian), "Dimension", typeof(int));
            }
            else
                return null;
        }

        private static IExpression GetGammaPowerExpression(IExpression prototypeExpression)
        {
            if (Recognizer.IsStaticMethod(prototypeExpression, typeof(Microsoft.ML.Probabilistic.Distributions.GammaPower), "Uniform"))
            {
                IMethodInvokeExpression imie = (IMethodInvokeExpression)prototypeExpression;
                return imie.Arguments[0];
            }
            else if (typeof(GammaPower).IsAssignableFrom(prototypeExpression.GetExpressionType()))
            {
                return Builder.FieldRefExpr(prototypeExpression, typeof(GammaPower), "Power");
            }
            else
                return null;
        }

        private static IExpression GetSparseListLengthExpression<T>(IExpression prototypeExpression, Func<int, T> createFromSize)
        {
            if (Recognizer.IsStaticMethod(prototypeExpression, createFromSize))
            {
                IMethodInvokeExpression imie = (IMethodInvokeExpression)prototypeExpression;
                return imie.Arguments[0];
            }

            Type expressionType = prototypeExpression.GetExpressionType();
            if (expressionType.GetProperty("Dimension", typeof(int)) != null)
            {
                return Builder.PropRefExpr(prototypeExpression, expressionType, "Dimension", typeof(int));
            }
            else
            {
                return null;
            }
        }
        protected IExpression GetVectorLengthExpression(IExpression source, object targetDecl)
        {
            MarginalPrototype mpa = GetMarginalPrototype(source, targetDecl);
            if (mpa != null)
            {
                if (mpa.prototype is Dirichlet d)
                {
                    return Builder.LiteralExpr(d.Dimension);
                }
                else if (mpa.prototype is VectorGaussian vg)
                {
                    return Builder.LiteralExpr(vg.Dimension);
                }
                else if (mpa.prototype is SparseGaussianList sgl)
                {
                    return Builder.LiteralExpr(sgl.Dimension);
                }
                else if (mpa.prototype == null && mpa.prototypeExpression != null)
                {
                    var mpe = mpa.prototypeExpression;
                    IExpression lengthExpr;
                    lengthExpr = GetDirichletLengthExpression(mpe);
                    if (lengthExpr == null)
                        lengthExpr = GetVectorGaussianLengthExpression(mpe);
                    if (lengthExpr == null)
                        lengthExpr = GetSparseListLengthExpression(mpe, new Func<int, SparseGaussianList>(SparseGaussianList.FromSize));
                    if (lengthExpr == null)
                        lengthExpr = GetSparseListLengthExpression(mpe, new Func<int, SparseGammaList>(SparseGammaList.FromSize));
                    if (lengthExpr == null)
                        lengthExpr = GetSparseListLengthExpression(mpe, new Func<int, SparseBernoulliList>(SparseBernoulliList.FromSize));
                    if (lengthExpr == null)
                        lengthExpr = GetSparseListLengthExpression(mpe, new Func<int, SparseBetaList>(SparseBetaList.FromSize));
                    if (lengthExpr == null)
                        lengthExpr = GetSparseListLengthExpression(mpe, new Func<int, BernoulliIntegerSubset>(BernoulliIntegerSubset.FromSize));
                    if (lengthExpr != null && !HasExtraIndices(lengthExpr, targetDecl))
                    {
                        return lengthExpr;
                    }
                }
            }
            if (!CodeRecognizer.IsStochastic(context, source) && !HasExtraIndices(source, targetDecl))
            {
                // source must be a Vector or something with a Count
                return Builder.PropRefExpr(source, typeof(Vector), "Count", typeof(int));
            }
            return null;
        }

        protected IExpression GetIntCardinalityExpression(IExpression source, IExpression target, bool targetHasLiteralIndices)
        {
            if (!targetHasLiteralIndices)
            {
                IExpression expr = GetIntCardinalityExpression(target, target);
                if (expr != null)
                    return expr;
            }
            object targetDecl = Recognizer.GetDeclaration(target);
            ValueRange vr = context.InputAttributes.Get<ValueRange>(targetDecl);
            if (vr != null)
                return vr.Range.GetSizeExpression();
            return GetIntCardinalityExpression(source, targetDecl);
        }

        /// <summary>
        /// Given an integer expression, returns an expression for the maximum value plus 1.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="targetDecl"></param>
        /// <returns></returns>
        protected IExpression GetIntCardinalityExpression(IExpression source, object targetDecl)
        {
            MarginalPrototype mpa = GetMarginalPrototype(source, targetDecl);
            if (mpa != null)
            {
                if (mpa.prototype is Discrete d)
                {
                    return Builder.LiteralExpr(d.Dimension);
                }
                else if (mpa.prototype == null && mpa.prototypeExpression != null)
                {
                    IExpression discreteExpression = mpa.prototypeExpression;
                    if (Recognizer.IsStaticMethod(discreteExpression, typeof(Discrete), "Uniform"))
                    {
                        IMethodInvokeExpression imie = (IMethodInvokeExpression)discreteExpression;
                        return imie.Arguments[0];
                    }
                    if (Recognizer.IsStaticMethod(discreteExpression, typeof(Discrete), "PointMass"))
                    {
                        IMethodInvokeExpression imie = (IMethodInvokeExpression)discreteExpression;
                        return imie.Arguments[1];
                    }
                    return Builder.PropRefExpr(discreteExpression, typeof(Discrete), "Dimension", typeof(int));
                }
            }
            if (!CodeRecognizer.IsStochastic(context, source) && !ReferenceEquals(source, targetDecl))
            {
                if (!HasExtraIndices(source, targetDecl))
                    return Builder.BinaryExpr(source, BinaryOperator.Add, Builder.LiteralExpr(1));
                if (ChannelTransform.RemoveCast(source) is IBinaryExpression ibe)
                {
                    IExpression dimension1 = GetIntCardinalityExpression(ibe.Left, targetDecl);
                    IExpression dimension2 = GetIntCardinalityExpression(ibe.Right, targetDecl);
                    if (dimension1 != null && dimension2 != null)
                    {
                        return CardinalityOfBinaryExpression(dimension1, ibe.Operator, dimension2);
                    }
                }
            }
            return null;
        }

        protected IExpression GetMatrixRowsExpression(IExpression source, object targetDecl)
        {
            if (!CodeRecognizer.IsStochastic(context, source) && !HasExtraIndices(source, targetDecl))
            {
                // source must be a Matrix
                return Builder.PropRefExpr(source, typeof(Matrix), "Rows", typeof(int));
            }
            else
            {
                MarginalPrototype mpa = GetMarginalPrototype(source, targetDecl);
                if (mpa != null)
                {
                    if (mpa.prototype is Wishart w)
                    {
                        return Builder.LiteralExpr(w.Dimension);
                    }
                    else if (mpa.prototype == null && mpa.prototypeExpression != null)
                    {
                        IExpression wishartExpression = mpa.prototypeExpression;
                        if (typeof(Wishart).IsAssignableFrom(wishartExpression.GetExpressionType()))
                        {
                            if (Recognizer.IsStaticMethod(wishartExpression, typeof(Wishart), "Uniform"))
                            {
                                IMethodInvokeExpression imie = (IMethodInvokeExpression)wishartExpression;
                                return imie.Arguments[0];
                            }
                            return Builder.PropRefExpr(wishartExpression, typeof(Wishart), "Dimension", typeof(int));
                        }
                    }
                }
            }
            return null;
        }

        protected IExpression GetArrayLengthExpression(IExpression source, object targetDecl, int dim = 0)
        {
            Type arrayType;
            int depth = Recognizer.GetIndexingDepth(source);
            // GetMarginalPrototype(source, targetDecl) does not work here since it does not necessarily contain the array size.
            object sourceDecl = Recognizer.GetDeclaration(source);
            if (sourceDecl != null)
            {
                VariableInformation sourceInfo = VariableInformation.GetVariableInformation(context, sourceDecl);
                if (sourceInfo.sizes.Count > depth)
                {
                    var size = sourceInfo.sizes[depth][dim];
                    // replace indexVars with the provided indices
                    var indices = Recognizer.GetIndices(source);
                    for (int i = 0; i < indices.Count; i++)
                    {
                        for (int j = 0; j < indices[i].Count; j++)
                        {
                            if (sourceInfo.indexVars.Count > i && sourceInfo.indexVars[i].Length > j)
                            {
                                var indexVar = sourceInfo.indexVars[i][j];
                                if (indexVar != null)
                                {
                                    int count = 0;
                                    size = Builder.ReplaceVariable(size, indexVar, indices[i][j], ref count);
                                }
                            }
                        }
                    }
                    return ReplaceIndices(size, targetDecl);
                }
                // fall through
                arrayType = (sourceDecl is IVariableDeclaration ivd) ? ivd.VariableType.DotNetType : ((IParameterDeclaration)sourceDecl).ParameterType.DotNetType;
                for (int bracket = 0; bracket < depth; bracket++)
                {
                    arrayType = arrayType.GetElementType();
                }
            }
            else
            {
                var root = source;
                for (int bracket = 0; bracket < depth; bracket++)
                {
                    root = ((IArrayIndexerExpression)root).Target;
                }
                if (root is IArrayCreateExpression iace)
                {
                    arrayType = Array.CreateInstance(iace.Type.DotNetType, new int[iace.Dimensions.Count]).GetType();
                }
                else
                    throw new ArgumentException($"Can't get array length expression for source \"{source}\"", nameof(source));
            }
            if (arrayType.GetArrayRank() == 1)
                return Builder.PropRefExpr(source, arrayType, "Length");
            else
                return Builder.Method(source, arrayType.GetMethod("GetLength"), Builder.LiteralExpr(dim));
        }

        // Sets the prototype expression for a Discrete expression from a dimension expression
        private MarginalPrototype GetDiscretePrototypeExpression(object targetDecl, IExpression dimensionExpression)
        {
            SparsityAttribute sparsityAttr = context.InputAttributes.Get<SparsityAttribute>(targetDecl);
            if (dimensionExpression != null)
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                if (sparsityAttr == null || sparsityAttr.Sparsity == null)
                {
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Microsoft.ML.Probabilistic.Distributions.Discrete>(Microsoft.ML.Probabilistic.Distributions.Discrete.Uniform), dimensionExpression);
                }
                else
                {
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Sparsity, Microsoft.ML.Probabilistic.Distributions.Discrete>(Microsoft.ML.Probabilistic.Distributions.Discrete.Uniform), dimensionExpression,
                                                                   Quoter.Quote(sparsityAttr.Sparsity));
                }
                return mpa;
            }
            return null;
        }

        // Sets the prototype expression for a Dirichlet expression from a dimension expression
        private MarginalPrototype GetDirichletPrototypeExpression(object targetDecl, IExpression dimensionExpression)
        {
            SparsityAttribute sparsityAttr = context.InputAttributes.Get<SparsityAttribute>(targetDecl);
            if (dimensionExpression != null)
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                if (sparsityAttr == null || sparsityAttr.Sparsity == null)
                {
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Microsoft.ML.Probabilistic.Distributions.Dirichlet>(Microsoft.ML.Probabilistic.Distributions.Dirichlet.Uniform), dimensionExpression);
                }
                else
                {
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Sparsity, Microsoft.ML.Probabilistic.Distributions.Dirichlet>(Microsoft.ML.Probabilistic.Distributions.Dirichlet.Uniform), dimensionExpression,
                                                                   Quoter.Quote(sparsityAttr.Sparsity));
                }
                return mpa;
            }
            return null;
        }

        // Sets the prototype expression for a SparseGaussianList expression from a dimension expression
        private MarginalPrototype GetSparseGaussianListPrototypeExpression(object targetDecl, IExpression dimensionExpression)
        {
            SparsityAttribute sparsityAttr = context.InputAttributes.Get<SparsityAttribute>(targetDecl);
            if (dimensionExpression != null)
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                if (sparsityAttr == null || sparsityAttr.Sparsity == null)
                    mpa.prototypeExpression = Builder.StaticMethod(new Converter<int, Microsoft.ML.Probabilistic.Distributions.SparseGaussianList>(Microsoft.ML.Probabilistic.Distributions.SparseGaussianList.FromSize),
                                                                   dimensionExpression);
                else
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, double, Microsoft.ML.Probabilistic.Distributions.SparseGaussianList>(Microsoft.ML.Probabilistic.Distributions.SparseGaussianList.FromSize),
                                                                   dimensionExpression, Quoter.Quote(sparsityAttr.Sparsity.Tolerance));
                return mpa;
            }
            return null;
        }

        // Sets the prototype expression for a SparseGammaList expression from a dimension expression
        private MarginalPrototype GetSparseGammaListPrototypeExpression(object targetDecl, IExpression dimensionExpression)
        {
            SparsityAttribute sparsityAttr = context.InputAttributes.Get<SparsityAttribute>(targetDecl);
            if (dimensionExpression != null)
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                if (sparsityAttr == null || sparsityAttr.Sparsity == null)
                    mpa.prototypeExpression = Builder.StaticMethod(new Converter<int, Microsoft.ML.Probabilistic.Distributions.SparseGammaList>(Microsoft.ML.Probabilistic.Distributions.SparseGammaList.FromSize),
                                                                   dimensionExpression);
                else
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, double, Microsoft.ML.Probabilistic.Distributions.SparseGammaList>(Microsoft.ML.Probabilistic.Distributions.SparseGammaList.FromSize),
                                                                   dimensionExpression, Quoter.Quote(sparsityAttr.Sparsity.Tolerance));
                return mpa;
            }
            return null;
        }

        // Sets the prototype expression for a SparseBernoulliList expression from a dimension expression
        private MarginalPrototype GetSparseBernoulliListPrototypeExpression(object targetDecl, IExpression dimensionExpression)
        {
            SparsityAttribute sparsityAttr = context.InputAttributes.Get<SparsityAttribute>(targetDecl);
            if (dimensionExpression != null)
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                if (sparsityAttr == null || sparsityAttr.Sparsity == null)
                    mpa.prototypeExpression = Builder.StaticMethod(new Converter<int, Microsoft.ML.Probabilistic.Distributions.SparseBernoulliList>(Microsoft.ML.Probabilistic.Distributions.SparseBernoulliList.FromSize),
                                                                   dimensionExpression);
                else
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, double, Microsoft.ML.Probabilistic.Distributions.SparseBernoulliList>(Microsoft.ML.Probabilistic.Distributions.SparseBernoulliList.FromSize),
                                                                   dimensionExpression, Quoter.Quote(sparsityAttr.Sparsity.Tolerance));
                return mpa;
            }
            return null;
        }

        // Sets the prototype expression for a SparseBetaList expression from a dimension expression
        private MarginalPrototype GetSparseBetaListPrototypeExpression(object targetDecl, IExpression dimensionExpression)
        {
            SparsityAttribute sparsityAttr = context.InputAttributes.Get<SparsityAttribute>(targetDecl);
            if (dimensionExpression != null)
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                if (sparsityAttr == null || sparsityAttr.Sparsity == null)
                    mpa.prototypeExpression = Builder.StaticMethod(new Converter<int, Microsoft.ML.Probabilistic.Distributions.SparseBetaList>(Microsoft.ML.Probabilistic.Distributions.SparseBetaList.FromSize),
                                                                   dimensionExpression);
                else
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, double, Microsoft.ML.Probabilistic.Distributions.SparseBetaList>(Microsoft.ML.Probabilistic.Distributions.SparseBetaList.FromSize),
                                                                   dimensionExpression, Quoter.Quote(sparsityAttr.Sparsity.Tolerance));
                return mpa;
            }
            return null;
        }

        // Sets the prototype expression for a BernoulliIntegerSubset expression from a dimension expression
        private MarginalPrototype GetBernoulliIntegerSubsetPrototypeExpression(object targetDecl, IExpression dimensionExpression)
        {
            SparsityAttribute sparsityAttr = context.InputAttributes.Get<SparsityAttribute>(targetDecl);
            if (dimensionExpression != null)
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                if (sparsityAttr == null || sparsityAttr.Sparsity == null)
                    mpa.prototypeExpression = Builder.StaticMethod(new Converter<int, Microsoft.ML.Probabilistic.Distributions.BernoulliIntegerSubset>(Microsoft.ML.Probabilistic.Distributions.BernoulliIntegerSubset.FromSize),
                                                                   dimensionExpression);
                else
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, double, Microsoft.ML.Probabilistic.Distributions.BernoulliIntegerSubset>(Microsoft.ML.Probabilistic.Distributions.BernoulliIntegerSubset.FromSize),
                                                                   dimensionExpression, Quoter.Quote(sparsityAttr.Sparsity.Tolerance));
                return mpa;
            }
            return null;
        }


        // Sets the prototype expression for a Softmax expression from a dimension expression
        private MarginalPrototype GetSoftmaxPrototypeExpression(object targetDecl, IExpression dimensionExpression)
        {
            SparsityAttribute sparsityAttr = context.InputAttributes.Get<SparsityAttribute>(targetDecl);
            if (dimensionExpression != null)
            {
                MarginalPrototype mpa = new MarginalPrototype(null);
                if (sparsityAttr == null || sparsityAttr.Sparsity == null)
                {
                    mpa.prototypeExpression = Builder.StaticMethod(new Converter<int, Microsoft.ML.Probabilistic.Distributions.Dirichlet>(Microsoft.ML.Probabilistic.Distributions.Dirichlet.Uniform), dimensionExpression);
                }
                else
                {
                    mpa.prototypeExpression = Builder.StaticMethod(new Func<int, Sparsity, Microsoft.ML.Probabilistic.Distributions.Dirichlet>(Microsoft.ML.Probabilistic.Distributions.Dirichlet.Uniform), dimensionExpression,
                                                                   Quoter.Quote(sparsityAttr.Sparsity));
                }
                return mpa;
            }
            return null;
        }

        private void CopyIndexVariables(IExpression target, IExpression source)
        {
            object targetDecl = Recognizer.GetDeclaration(target);
            if (targetDecl == null) return;
            VariableInformation targetInfo = VariableInformation.GetVariableInformation(context, targetDecl);
            int targetDepth = Recognizer.GetIndexingDepth(target);
            object sourceDecl = Recognizer.GetDeclaration(source);
            if (sourceDecl == null) return;
            VariableInformation sourceInfo = VariableInformation.GetVariableInformation(context, sourceDecl);
            int sourceDepth = Recognizer.GetIndexingDepth(source);
            for (int d = sourceDepth; d < sourceInfo.indexVars.Count; d++)
            {
                int d2 = targetDepth + (d - sourceDepth);
                IVariableDeclaration[] sourceBracket = sourceInfo.indexVars[d];
                IVariableDeclaration[] targetBracket;
                if (targetInfo.indexVars.Count == d2)
                {
                    targetBracket = new IVariableDeclaration[sourceBracket.Length];
                    targetInfo.indexVars.Add(targetBracket);
                }
                else
                {
                    targetBracket = targetInfo.indexVars[d2];
                    if (targetBracket.Length != sourceBracket.Length)
                        Error("targetBracket.Length (" + targetBracket.Length + ") != sourceBracket.Length (" + sourceBracket.Length + ")");
                }
                for (int i = 0; i < sourceBracket.Length; i++)
                {
                    if (targetBracket[i] == null) targetBracket[i] = sourceBracket[i];
                }
            }
        }
    }

    internal class SparsityAttribute : ICompilerAttribute
    {
        public SparsityAttribute(Sparsity sparsity)
        {
            Sparsity = sparsity;
        }

        public Sparsity Sparsity { get; private set; }
    }
}