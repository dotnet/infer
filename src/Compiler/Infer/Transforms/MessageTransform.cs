// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Transforms a model specified in terms of channels into the set of message passing
    /// operations required to perform inference in that model.
    /// </summary>
    /// <remarks>
    /// The message passing transform does the following operations:
    ///  - converts channel declarations into pairs of variables for forwards and backwards 
    ///    messages along that channel.
    ///  - initalises message arrays using the appropriate message prototypes
    ///    - message prototypes may be different for backward and forward directions
    ///    - the algorithm determines the message prototypes in a call to GetMessagePrototypes
    ///    - default message prototype is the marginal prototype
    ///  - converts method calls into sets of operator method calls
    ///  - keeps track of all inter-operator dependencies, as these are needed by the scheduler
    /// <para>
    /// Stochastic variables must have ChannelInfo attributes.
    /// </para>
    /// </remarks>
    internal class MessageTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "MessageTransform"; }
        }

        internal static bool UseMessageAnalysis;
        internal static bool debug;
        internal bool InitializeOnSeparateLine;
        protected IAlgorithm algorithm;
        protected FactorManager factorManager;
        protected ModelCompiler compiler;

        const string resultName = "result";
        const string resultIndexName = "resultIndex";

        /// <summary>
        /// If true, operators that return their argument are replaced by copy expressions.  Note this is required for certain tests to pass.
        /// </summary>
        private readonly bool UseCopyOperators = true;

        /// <summary>
        /// The set of variables that have at least one definition with a non-unit derivative
        /// </summary>
        protected Set<IVariableDeclaration> hasNonUnitDerivative = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);

        protected Dictionary<IExpression, Type> initialiserType = new Dictionary<IExpression, Type>();

        protected Dictionary<IVariableDeclaration, IVariableDeclaration> derivOfVariable =
            new Dictionary<IVariableDeclaration, IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);

        private static readonly MessageDirection[] directions = { MessageDirection.Forwards, MessageDirection.Backwards };

        /// <summary>
        /// If true, restrictions on derived parents are ignored.
        /// </summary>
        private readonly bool AllowDerivedParents;

        /// <summary>
        /// Results of analysis
        /// </summary>
        private MessageAnalysisTransform analysis;

        /// <summary>
        /// Method declaration used for message tracing
        /// </summary>
        private IMethodDeclaration messageUpdatedMethod;

        public MessageTransform(ModelCompiler compiler, IAlgorithm algorithm, FactorManager factorManager, bool allowDerivedParents)
        {
            this.compiler = compiler;
            this.algorithm = algorithm;
            this.factorManager = factorManager;
            this.AllowDerivedParents = allowDerivedParents;
        }

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            if (UseMessageAnalysis)
            {
                analysis = new MessageAnalysisTransform(algorithm, factorManager);
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
            if (messageUpdatedMethod != null)
            {
                // Add method to fire events when a message is updated
                td.Methods.Add(messageUpdatedMethod);
            }

            return td;
        }

        public override void ConvertTypeProperties(ITypeDeclaration td, ITypeDeclaration itd)
        {
            base.ConvertTypeProperties(td, itd);
            td.Name = td.Name + "_" + algorithm.ShortName;
            if (itd.Documentation != null)
            {
                td.Documentation = itd.Documentation + " using algorithm '" + algorithm.Name + "'";
            }
        }

        protected override IMethodDeclaration ConvertMethod(IMethodDeclaration imd)
        {
            IMethodDeclaration md = base.ConvertMethod(imd);
            context.OutputAttributes.Set(md, new OperatorMethod());
            return md;
        }

        private static bool IsGibbsMarginal(Type type)
        {
            return type.IsGenericType && type.GetGenericTypeDefinition().Equals(typeof(GibbsMarginal<,>));
        }

        public static bool IsPointMass(Type type)
        {
            return type.IsGenericType && type.GetGenericTypeDefinition().Equals(typeof(PointMass<>));
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            if (UseMessageAnalysis)
            {
                // create factory variables
                foreach (KeyValuePair<IVariableDeclaration, IExpression> entry in analysis.factoryInitExprs)
                {
                    IVariableDeclaration factoryVar = entry.Key;
                    IExpression initExpr = entry.Value;
                    IStatement ist = Builder.AssignStmt(Builder.VarDeclExpr(factoryVar), initExpr);
                    outputs.Add(ist);
                }
            }
            base.DoConvertMethodBody(outputs, inputs);
        }

#if true
        /// <summary>
        /// Only converts the contained statements in a for loop, leaving the initializer,
        /// condition and increment statements unchanged.
        /// </summary>
        protected override IStatement ConvertFor(IForStatement ifs)
        {
            IForStatement fs = Builder.ForStmt();
            context.SetPrimaryOutput(fs);
            fs.Condition = ifs.Condition;
            fs.Increment = ifs.Increment;
            fs.Initializer = ifs.Initializer;
            fs.Body = ConvertBlock(ifs.Body);
            // attributes are not copied
            return fs;
        }
#endif

        /// <summary>
        /// Flatten out and remove if statements.
        /// </summary>
        /// <returns></returns>
        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            if (IsStochasticVariableReference(ics.Condition))
            {
                IBlockStatement bs = ConvertBlock(ics.Then);
                context.AddStatementsBeforeCurrent(bs.Statements);
                return null;
            }
            else
            {
                return base.ConvertCondition(ics);
            }
        }

        /// <summary>
        /// This does the work of constructing all the operator calls for the given factor call
        /// </summary>
        /// <param name="imie">The method invoke expression</param>
        /// <returns></returns>
        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if (CodeRecognizer.IsInfer(imie)) return ConvertInfer(imie);
            if (CodeRecognizer.IsIsIncreasing(imie)) return imie;
            if (context.FindAncestor<IExpressionStatement>() == null) return imie;
            IAssignExpression iae = context.FindAncestor<IAssignExpression>();
            bool isAssignment = (iae != null);
            IStatement ist = context.FindAncestor<IStatement>();
            bool resultIsObserved = context.InputAttributes.Has<Models.Constraint>(ist);
            MethodConverter methodConverter = new MethodConverter(this, imie);
            if (methodConverter.info == null)
            {
                Error("Factor information could not be found for method: " + imie);
                return imie;
            }
            if (methodConverter.info.IsVoid != !isAssignment)
            {
                if (methodConverter.info.IsVoid) Error("A void function cannot be used in an assignment");
                else if (context.InputStack[context.InputStack.Count - 2].inputElement is IExpressionStatement)
                    Error("The return value of a non-void function must be assigned to a variable");
                else Error("Nested function calls are not allowed. Create a temporary variable for each function result");
                return imie;
            }
            if (iae != null)
            {
                IVariableDeclaration targetVar = Recognizer.GetVariableDeclaration(iae.Target);
                // TODO: this is temporary until full support for UseDerivMessages
                if (targetVar != null && methodConverter.info.IsDeterministicFactor)
                {
                    if (!MethodAndAllArgumentsHaveUnitDerivative(imie, methodConverter.info))
                    {
                        hasNonUnitDerivative.Add(targetVar);
                    }
                }
            }

            return methodConverter.Convert(imie, iae, isAssignment, resultIsObserved);
        }

        private bool MethodAndAllArgumentsHaveUnitDerivative(IMethodInvokeExpression imie, FactorManager.FactorInfo info)
        {
            bool hasUnitDerivative = info.Method.IsDefined(typeof(HasUnitDerivative), true);
            if (hasUnitDerivative)
            {
                // check that all arguments have unit derivative
                foreach (IExpression arg in imie.Arguments)
                {
                    IVariableDeclaration argVar = Recognizer.GetVariableDeclaration(arg);
                    if (argVar != null && hasNonUnitDerivative.Contains(argVar))
                    {
                        hasUnitDerivative = false;
                        break;
                    }
                }
            }

            return hasUnitDerivative;
        }

        private void AddInitialiserStatement(IExpression message, IExpression factor)
        {
            IVariableDeclaration msgVar = Recognizer.GetVariableDeclaration(message);
            KeyValuePair<IVariableDeclaration, IExpression> key = new KeyValuePair<IVariableDeclaration, IExpression>(msgVar, factor);
            if (analysis.messageInitExprs.TryGetValue(key, out IExpression initExpr))
            {
                IExpressionStatement init = Builder.AssignStmt(message, initExpr);
                context.OutputAttributes.Set(init, new Initializer());
                context.AddStatementBeforeCurrent(init);
            }
        }

        private class MethodConverter
        {
            readonly IAlgorithm alg;
            public readonly FactorManager.FactorInfo info;
            readonly Dictionary<string, MessageInfo> msgInfo = new Dictionary<string, MessageInfo>();
            /// <summary>
            /// Indicates whether each argument is valid for a formal marked with the Stochastic attribute.
            /// </summary>
            readonly Dictionary<string, bool> isStochastic = new Dictionary<string, bool>();
            /// <summary>
            /// Whether this is a variable factor
            /// </summary>
            readonly bool isVariableFactor;
            readonly BasicTransformContext context;
            readonly FactorManager factorManager;
            readonly MessageTransform transform;

            public MethodConverter(MessageTransform transform, IMethodInvokeExpression imie)
            {
                this.transform = transform;
                context = transform.context;
                factorManager = transform.factorManager;
                isVariableFactor = context.InputAttributes.Has<IsVariableFactor>(imie);

                Algorithm algAttr = context.InputAttributes.Get<Algorithm>(imie);
                if (algAttr != null) alg = algAttr.algorithm;
                else alg = transform.algorithm;

                info = CodeRecognizer.GetFactorInfo(context, imie);
            }

            public IExpression Convert(IMethodInvokeExpression imie, IAssignExpression iae, bool isAssignment, bool resultIsObserved)
            {
                // Collect information about the arguments
                List<bool> isReturnOrOut = new List<bool>();
                List<IExpression> arguments = new List<IExpression>();
                if (iae != null)
                {
                    IExpression target = iae.Target;
                    if (target is IVariableDeclarationExpression)
                    {
                        target = Builder.VarRefExpr(Recognizer.GetVariableDeclaration(target));
                    }
                    isReturnOrOut.Add(!resultIsObserved);
                    arguments.Add(target);
                }
                if (!info.Method.IsStatic)
                {
                    isReturnOrOut.Add(false);
                    arguments.Add(imie.Method.Target);
                }
                foreach (IExpression arg in imie.Arguments)
                {
                    bool isOut = (arg is IAddressOutExpression);
                    isReturnOrOut.Add(isOut);
                    arguments.Add(isOut ? ((IAddressOutExpression)arg).Expression : arg);
                }


                //--------------------------------------------------------
                // Set up the message info for each target field, and record the
                // type of each field
                //--------------------------------------------------------                
                if (debug)
                {
                    context.InputAttributes.Set(imie, new MessageInfoDict()
                    {
                        msgInfo = msgInfo
                    });
                }
                var argumentTypes = new Dictionary<string, Type>();
                var resultTypes = new Dictionary<string, Type>();
                List<bool> argIsConstant = new List<bool>();
                bool resultIsConstant = info.IsDeterministicFactor && !isVariableFactor;
                for (int i = 0; i < info.ParameterNames.Count; i++)
                {
                    string parameterName = info.ParameterNames[i];
                    bool isWeaklyTyped = isVariableFactor && parameterName == "init";
                    bool isChild = isReturnOrOut[i];
                    IExpression channelRef = arguments[i];
                    bool isConstant = !CodeRecognizer.IsStochastic(context, channelRef);
                    if (!(alg is GibbsSampling))
                    {
                        IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(channelRef);
                        if (ivd != null)
                        {
                            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
                            isConstant = isConstant && !vi.NeedsMarginalDividedByPrior;
                        }
                    }
                    if (!isConstant) resultIsConstant = false;
                    argIsConstant.Add(isConstant);
                    MessageInfo mi;
                    if (isConstant)
                    {
                        mi = new MessageInfo
                        {
                            messageFromFactor = isChild ? channelRef : null,
                            messageToFactor = channelRef
                        };
                        if (!isWeaklyTyped)
                        {
                            Type inwardType = mi.messageToFactor.GetExpressionType();
                            argumentTypes[parameterName] = inwardType;
                        }
                    }
                    else
                    {
                        IExpression fwdMsg = transform.GetMessageExpression(channelRef, MessageDirection.Forwards);
                        IExpression bckMsg = transform.GetMessageExpression(channelRef, MessageDirection.Backwards);
                        //if (fwdMsg == null) { Error("forward message is null"); return imie; }
                        //if (bckMsg == null) { Error("backward message is null"); return imie; }
                        if (isChild)
                        {
                            mi = new MessageInfo(transform, channelRef)
                            {
                                messageFromFactor = fwdMsg,
                                messageToFactor = bckMsg
                            };
                        }
                        else
                        {
                            mi = new MessageInfo(transform, channelRef)
                            {
                                messageFromFactor = bckMsg,
                                messageToFactor = fwdMsg
                            };
                        }
                        if (!isWeaklyTyped)
                        {
                            if (mi.messageToFactor != null)
                            {
                                Type inwardType = mi.messageToFactor.GetExpressionType();
                                if (inwardType == null)
                                {
                                    transform.Error("Cannot determine type of " + mi.messageToFactor);
                                    return imie;
                                }
                                argumentTypes[parameterName] = inwardType;
                            }
                            if (mi.messageFromFactor != null)
                            {
                                Type outwardType = mi.messageFromFactor.GetExpressionType();
                                argumentTypes["to_" + parameterName] = outwardType;
                                resultTypes[parameterName] = outwardType;
                            }
                        }
                        isStochastic[parameterName] = !mi.hasNonUnitDerivative;
                        if (alg is VariationalMessagePassing vmp && vmp.UseDerivMessages && !isChild && mi.hasNonUnitDerivative && !isVariableFactor)
                        {
                            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(channelRef);
                            DerivMessage dm = context.InputAttributes.Get<DerivMessage>(ivd);
                            if (dm != null)
                            {
                                IExpression derivArg = Builder.ReplaceExpression(channelRef, Builder.VarRefExpr(ivd), Builder.VarRefExpr(dm.decl));
                                IExpression derivMsg = transform.GetMessageExpression(derivArg, MessageDirection.Forwards);
                                MessageInfo mi2 = new MessageInfo
                                {
                                    messageFromFactor = derivMsg,
                                    messageToFactor = derivMsg
                                };
                                string derivParameterName = parameterName + "_deriv";
                                msgInfo[derivParameterName] = mi2;
                                Type derivType = derivMsg.GetExpressionType();
                                argumentTypes[derivParameterName] = derivType;
                                resultTypes[derivParameterName] = derivType;
                            }
                        }
                    }
                    msgInfo[parameterName] = mi;
                }
                if (resultIsConstant && isAssignment && !resultIsObserved)
                {
                    // The statement is an assignment between deterministic variables.
                    // The factor will be used unchanged.  Check if it copies any of its arguments.
                    MessageFcnInfo fninfo = info.GetMessageFcnInfoFromFactor();
                    IExpression copyExpr = CheckForCopyOperators(fninfo, imie.Arguments, arguments[0]);
                    if (copyExpr != null)
                        return copyExpr;
                    return imie;
                }

                // Create operator statements for each non-deterministic argument
                List<ICompilerAttribute> factorAttributes = context.InputAttributes.GetAll<ICompilerAttribute>(imie);

                int initialPriorityListSize = this.factorManager.PriorityList.Count;
                GivePriorityFromAttributes(imie);

                // send an evidence message to the innermost stochastic condition statement.
                IConditionStatement ics = GetInnermostStochasticConditionStatement();
                if (ics != null)
                {
                    AddEvidenceMessage(ics.Condition, argumentTypes, factorAttributes);
                }
                else if (resultIsConstant)
                {
                    if (!isAssignment)
                    {
                        // Constrain.Equal(true,false) outside of a conditional statement should be left intact, and marked as an output.
                        context.OutputAttributes.Set(imie, new DeterministicConstraint());
                        return imie;
                    }
                    else if (resultIsObserved)
                    {
                        // a = Not(b)  turns into  Constrain.Equal(a,Not(b))
                        Type t = iae.Target.GetExpressionType();
                        var mie = Builder.StaticGenericMethod(new Action<PlaceHolder, PlaceHolder>(Constrain.Equal), new Type[] { t }, iae.Target, imie);
                        context.OutputAttributes.Set(mie, new DeterministicConstraint());
                        context.AddStatementBeforeCurrent(Builder.ExprStatement(mie));
                        return null;
                    }
                    else
                        return null;
                }

                // Loop over each output argument and construct the operator method
                string operatorSuffix = alg.GetOperatorMethodSuffix(factorAttributes);
                for (int i = 0; i < info.ParameterNames.Count; i++)
                {
                    string targetParameter = info.ParameterNames[i];
                    MessageInfo mi = msgInfo[targetParameter];
                    bool isChild = isReturnOrOut[i];
                    // do not generate messages to constant arguments.
                    if (argIsConstant[i] && !isChild) continue;
                    if (UseMessageAnalysis)
                    {
                        transform.AddInitialiserStatement(mi.messageFromFactor, iae ?? (IExpression)imie);
                        if (isChild) transform.AddInitialiserStatement(mi.messageToFactor, null);
                    }

                    int currentPriorityListSize = this.factorManager.PriorityList.Count;

                    Type targetType;
                    if (argIsConstant[i])
                        targetType = argumentTypes[targetParameter];
                    else
                        targetType = resultTypes[targetParameter];
                    argumentTypes[resultName] = targetType;

                    GivePriorityFromDistributionType(targetType);

                    // Give priority to operators in GivePriorityTo attributes
                    // TODO: can be moved out of the loop when we get rid of the operator lookup based on target type
                    List<GivePriorityTo> ops = context.InputAttributes.GetAll<GivePriorityTo>(imie);
                    foreach (GivePriorityTo op in ops)
                    {
                        this.factorManager.GivePriorityTo(op.Container);
                    }

                    bool removeOperatorStatement = argIsConstant[i];
                    void AddStatement(IStatement st)
                    {
                        if (removeOperatorStatement) context.OutputAttributes.Remove<OperatorStatement>(st);
                        context.AddStatementBeforeCurrent(st);
                    }
                    ForEachOperatorStatement(AddStatement, operatorSuffix, targetParameter, argumentTypes);

                    this.factorManager.PriorityList.RemoveRange(0, this.factorManager.PriorityList.Count - currentPriorityListSize);

                    if (alg is VariationalMessagePassing vmp && vmp.UseDerivMessages && isChild && mi.hasNonUnitDerivative && !isVariableFactor)
                    {
                        IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(arguments[i]);
                        DerivMessage dm = context.InputAttributes.Get<DerivMessage>(ivd);
                        if (dm != null)
                        {
                            ChannelToMessageInfo ctmi = context.InputAttributes.Get<ChannelToMessageInfo>(dm.decl);
                            IVariableDeclaration derivMsgVar = ctmi.fwd.decl;
                            IVariableDeclaration forwardVar = Recognizer.GetVariableDeclaration(mi.messageFromFactor);
                            MessageInfo mi2 = new MessageInfo
                            {
                                messageFromFactor = Builder.ReplaceExpression(mi.messageFromFactor, Builder.VarRefExpr(forwardVar), Builder.VarRefExpr(derivMsgVar))
                            };
                            string field = "";
                            msgInfo[field] = mi2;
                            string methodName = targetParameter + "Deriv";
                            ForEachOperatorStatement(context.AddStatementBeforeCurrent, methodName, field, argumentTypes);
                        }
                    }
                }
                this.factorManager.PriorityList.RemoveRange(0, this.factorManager.PriorityList.Count - initialPriorityListSize);
                return null;
            }

            private void AddEvidenceMessage(IExpression channelRef, Dictionary<string, Type> argumentTypes, List<ICompilerAttribute> factorAttributes)
            {
                if (isVariableFactor && msgInfo.ContainsKey("Def") &&
                    context.InputAttributes.Has<DoNotSendEvidence>(msgInfo["Def"].channelDecl))
                {
                    transform.Error("Sending evidence from a variable marked with DoNotSendEvidence (" + msgInfo["Def"].channelDecl + ")");
                }
                argumentTypes[resultName] = typeof(double);
                string methodName = alg.GetEvidenceMethodName(factorAttributes);
                MessageInfo mi = new MessageInfo(transform, channelRef)
                {
                    messageFromFactor = transform.GetMessageExpression(channelRef, MessageDirection.Backwards)
                };
                string evidenceField = "";
                msgInfo[evidenceField] = mi;
                ForEachOperatorStatement(context.AddStatementBeforeCurrent, methodName, evidenceField, argumentTypes);
            }

            private IConditionStatement GetInnermostStochasticConditionStatement()
            {
                List<IConditionStatement> ifContainers = context.FindAncestors<IConditionStatement>();
                ifContainers.Reverse();
                IConditionStatement ics = null;
                foreach (IConditionStatement ifContainer in ifContainers)
                {
                    if (transform.IsStochasticVariableReference(ifContainer.Condition))
                    {
                        ics = ifContainer;
                        break;
                    }
                }

                return ics;
            }

            private void GivePriorityFromDistributionType(Type targetType)
            {
                // If the target can implement SetToSum exactly, we can use BP gate operators
                // TODO: remove this when the support for auto-generated distribution arrays will be added
                Type targetInnermostElementType = Util.GetInnermostElementType(targetType);
                if (typeof(SettableToWeightedSumExact<>).MakeGenericType(targetInnermostElementType).IsAssignableFrom(targetInnermostElementType))
                {
                    this.factorManager.GivePriorityTo(typeof(BeliefPropagationGateEnterOneOp));
                    this.factorManager.GivePriorityTo(typeof(BeliefPropagationGateEnterOp));
                    this.factorManager.GivePriorityTo(typeof(BeliefPropagationGateEnterPartialOp));
                    this.factorManager.GivePriorityTo(typeof(BeliefPropagationGateEnterPartialTwoOp));
                    this.factorManager.GivePriorityTo(typeof(BeliefPropagationGateExitOp));
                    this.factorManager.GivePriorityTo(typeof(BeliefPropagationGateExitTwoOp));
                }
                else
                {
                    this.factorManager.GivePriorityTo(typeof(GateEnterOneOp<>));
                    this.factorManager.GivePriorityTo(typeof(GateEnterOp<>));
                    this.factorManager.GivePriorityTo(typeof(GateEnterPartialOp<>));
                    this.factorManager.GivePriorityTo(typeof(GateEnterPartialTwoOp));
                    this.factorManager.GivePriorityTo(typeof(GateExitOp<>));
                    this.factorManager.GivePriorityTo(typeof(GateExitTwoOp));
                }

                if (targetInnermostElementType == typeof(StringDistribution))
                {
                    // Don't use ReplicateOp_Divide for StringAutomaton
                    // TODO: replace this workaround with a proper extensibility mechanism
                    this.factorManager.GivePriorityTo(typeof(ReplicateOp_NoDivide));
                }
            }

            private void GivePriorityFromAttributes(IMethodInvokeExpression imie)
            {
                DivideMessages divideMessages;
                // Gibbs must never divide messages
                if (alg is GibbsSampling)
                    divideMessages = new DivideMessages(false);
                else
                    divideMessages = context.InputAttributes.Get<DivideMessages>(imie);
                if (divideMessages != null)
                {
                    if (divideMessages.useDivision)
                    {
                        factorManager.GivePriorityTo(typeof(ReplicateOp_Divide));
                    }
                    else
                    {
                        factorManager.GivePriorityTo(typeof(ReplicateOp_NoDivide));
                    }
                }
                bool isPointMass = context.InputAttributes.Has<ForwardPointMass>(imie);
                if (isPointMass)
                {
                    factorManager.GivePriorityTo(typeof(ReplicatePointOp));
                    factorManager.GivePriorityTo(typeof(GetItemsPointOp<>));
                    factorManager.GivePriorityTo(typeof(GetJaggedItemsPointOp<>));
                    factorManager.GivePriorityTo(typeof(GetDeepJaggedItemsPointOp<>));
                    factorManager.GivePriorityTo(typeof(GetItemsFromJaggedPointOp<>));
                    factorManager.GivePriorityTo(typeof(GetItemsFromDeepJaggedPointOp<>));
                    factorManager.GivePriorityTo(typeof(GetJaggedItemsFromJaggedPointOp<>));
                }
            }

            /// <summary>
            /// Create an operator statement which computes the message to the specified argument, and invoke an action.
            /// May add buffers to msgInfo and argumentTypes.
            /// </summary>
            /// <param name="action">Receives the generated statements</param>
            /// <param name="methodSuffix">Suffix for the operator method</param>
            /// <param name="targetParameter">target argument name</param>
            /// <param name="argumentTypes">Argument types for the operator method.  May be modified with buffers.</param>
            /// <returns></returns>
            public void ForEachOperatorStatement(Action<IStatement> action, string methodSuffix, string targetParameter, Dictionary<string, Type> argumentTypes)
            {
                var customArgumentTypes = new Dictionary<string, Type>(argumentTypes);
                bool useFactor = (targetParameter == info.ParameterNames[0]) && info.OutputIsDeterministic(customArgumentTypes) && !info.IsVoid && !isVariableFactor;

                // For Gibbs, we do not want to allow EP methods which require projection. A crude
                // way of detecting these is to look at the arguments, and reject any method which
                // includes as an argument the incoming message corresponding to the target parameter
                if (alg is GibbsSampling && !useFactor && !isVariableFactor)
                    customArgumentTypes[targetParameter] = typeof(PlaceHolder);

                MessageFcnInfo fninfo;
                if (useFactor) fninfo = info.GetMessageFcnInfoFromFactor();
                else
                {
                    try
                    {
                        fninfo = info.GetMessageFcnInfo(factorManager, methodSuffix, targetParameter, customArgumentTypes, isStochastic);
                    }
                    catch (ArgumentException ex)
                    {
                        ErrorNotSupported(ex);
                        return;
                    }
                    catch (MissingMethodException ex)
                    {
                        ErrorNotSupported(ex);
                        return;
                    }
                    catch (NotSupportedException ex)
                    {
                        ErrorNotSupported(ex);
                        return;
                    }
                    catch (AmbiguousMatchException ex)
                    {
                        ErrorNotSupported(ex);
                        return;
                    }
                    void ErrorNotSupported(Exception ex)
                    {
                        string opErr = "This model is not supported with " + alg.Name + " due to " + info.ToString() +
                                       ". Try using a different algorithm or expressing the model differently";
                        if ((ex is ArgumentException) && (alg is GibbsSampling) && (!isVariableFactor))
                        {
                            transform.Error(opErr + " Gibbs Sampling requires the conditionals to be conjugate", ex);
                        }
                        else
                            transform.Error(opErr, ex);
                    }
                }

                MessageInfo mi = msgInfo[targetParameter];

                if (fninfo.PassResultIndex)
                {
                    Type resultType = argumentTypes[resultName];
                    customArgumentTypes[resultIndexName] = typeof(int);
                    Type resultElementType = Util.GetElementType(resultType);
                    customArgumentTypes[resultName] = resultElementType;
                    // Get the expressions for all the arguments to the operator
                    List<IExpression> args = GetOperatorArguments(fninfo, mi, customArgumentTypes);
                    VariableInformation vi = VariableInformation.GetVariableInformation(context, mi.channelDecl ?? Recognizer.GetVariableDeclaration(mi.messageToFactor));
                    int depth = Recognizer.GetIndexingDepth(mi.messageFromFactor);
                    vi.DefineIndexVarsUpToDepth(context, depth + 1);
                    IVariableDeclaration indexVar = vi.indexVars[depth][0];
                    IExpression size = vi.sizes[depth][0];
                    if (depth == vi.LiteralIndexingDepth - 1)
                    {
                        // generate a separate statement for each array element
                        int sizeAsInt = (int)((ILiteralExpression)size).Value;
                        for (int j = 0; j < sizeAsInt; j++)
                        {
                            // Create the statement which calls the message function
                            IExpression index = Builder.LiteralExpr(j);
                            IStatement st = GetOperatorStatement(fninfo, mi, args, index, isVariableFactor, useFactor);
                            action(st);
                        }
                    }
                    else
                    {
                        // generate a loop that fills in all array elements
                        if (Recognizer.GetLoopForVariable(context, indexVar) != null) indexVar = Builder.VarDecl("_" + resultIndexName, typeof(int));
                        IForStatement fs = Builder.ForStmt(indexVar, size);
                        IExpression index = Builder.VarRefExpr(indexVar);
                        IStatement st = GetOperatorStatement(fninfo, mi, args, index, isVariableFactor, useFactor);
                        fs.Body.Statements.Add(st);
                        action(fs);
                    }
                }
                else
                {
                    customArgumentTypes.Remove(resultIndexName);
                    // Get the expressions for all the arguments to the operator
                    List<IExpression> args = GetOperatorArguments(fninfo, mi, customArgumentTypes);
                    // Create the statement which calls the message function
                    IStatement st = GetOperatorStatement(fninfo, mi, args, null, isVariableFactor, useFactor);
                    action(st);
                }
            }

            /// <summary>
            /// Creates an assignment statement that invokes a message operator.
            /// </summary>
            /// <param name="fninfo">The message operator</param>
            /// <param name="mi">The target of the message</param>
            /// <param name="args">The arguments of the message operator</param>
            /// <param name="index">The index argument of the message operator</param>
            /// <param name="isVariableFactor">True if the factor is a variable factor</param>
            /// <param name="useFactor">True if the message operator is the factor itself.</param>
            /// <returns>An assignment statement that invokes the message operator.</returns>
            private IExpressionStatement GetOperatorStatement(
                MessageFcnInfo fninfo,
                MessageInfo mi,
                List<IExpression> args,
                IExpression index,
                bool isVariableFactor,
                bool useFactor)
            {
                List<IExpression> fullArgs = new List<IExpression>();
                if (fninfo.IsIndexedParameter == null) fullArgs.AddRange(args);
                else
                {
                    for (int i = 0; i < args.Count; i++)
                    {
                        IExpression arg = args[i];
                        if (fninfo.IsIndexedParameter[i]) arg = Builder.ArrayIndex(arg, index);
                        fullArgs.Add(arg);
                    }
                }
                IExpression msg = mi.messageFromFactor;
                if (index != null)
                {
                    if (fninfo.ResultIndexParameterIndex != fullArgs.Count)
                        transform.Error("'" + resultIndexName + "' is not the 2nd to last argument of " + StringUtil.MethodSignatureToString(fninfo.Method));
                    fullArgs.Add(index);
                    msg = Builder.ArrayIndex(msg, index);
                }
                if (fninfo.PassResult)
                {
                    if (fninfo.ResultParameterIndex != fullArgs.Count)
                        transform.Error("'" + resultName + "' is not the last argument of " + StringUtil.MethodSignatureToString(fninfo.Method));
                    fullArgs.Add(msg);
                }
                fullArgs = ConvertArguments(fninfo.Method, fullArgs);
                IExpression operatorMethod = Builder.StaticMethod(fninfo.Method, fullArgs.ToArray());
                if (!context.OutputAttributes.Has<MessageFcnInfo>(fninfo.Method))
                    context.OutputAttributes.Set(fninfo.Method, fninfo);

                IExpression copyOperatorMethod = CheckForCopyOperators(fninfo, args, msg);
                if (copyOperatorMethod != null)
                {
                    operatorMethod = copyOperatorMethod;
                    // this could be removed if Copy has a Trigger attribute
                    isVariableFactor = false;
                }

                // Get the quality band of the operator method and attach an attribute
                // Quality bands on the factor itself are optional, and are handled by
                // the general mechanism in IterativeProcessTransform.
                if (!useFactor && (operatorMethod is IMethodInvokeExpression imie))
                {
                    var mr = Recognizer.GetMethodReference(imie);
                    // Get the quality band of the operator method and attach an attribute
                    QualityBand qualityBand = Quality.GetQualityBand(mr.MethodInfo);
                    context.OutputAttributes.Set(imie, new QualityBandCompilerAttribute(qualityBand));
                }

                operatorMethod = WrapWithTracing(operatorMethod, fninfo.Method.ReturnType, mi.channelDecl, msg);

                IAssignExpression assignExpr = CreateAssignExpression(msg, operatorMethod);

                if (mi.channelDecl != null)
                    context.OutputAttributes.Set(assignExpr, new DescriptionAttribute("Message to '" + mi.channelDecl.Name + "' from " + info.Method.Name + " factor"));
                var st = Builder.ExprStatement(assignExpr);
                bool needAllTriggers = false;
                if (fninfo.AllTriggers) needAllTriggers = true;
                else if (alg is VariationalMessagePassing vmp && !isVariableFactor && !vmp.UseDerivMessages && !fninfo.NoTriggers)
                {
                    needAllTriggers = true;
                }
                else if (alg is GibbsSampling && !fninfo.IsStochastic)
                {
                    // For Gibbs, all factors need triggers. The only edge that should not
                    // be triggered is the one drawing samples.
                    needAllTriggers = true;
                }
                if (needAllTriggers) context.OutputAttributes.Set(operatorMethod, new AllTriggersAttribute());

                context.OutputAttributes.Set(st, new OperatorStatement());
                if (fninfo.IsMultiplyAll) context.OutputAttributes.Set(st, new MultiplyAllCompilerAttribute());
                if (fninfo.IsStochastic) context.OutputAttributes.Set(st, new DependsOnIteration());
                if (mi.channelDecl != null)
                    context.InputAttributes.CopyObjectAttributesTo<InitialiseBackward>(mi.channelDecl, context.OutputAttributes, st);
                return st;
            }

            private IAssignExpression CreateAssignExpression(IExpression msg, IExpression operatorMethod)
            {
                Type lhsType = msg.GetExpressionType();
                Type rhsType = operatorMethod.GetExpressionType();
                if (lhsType == typeof(Bernoulli) && rhsType == typeof(double))
                {
                    operatorMethod = Builder.StaticMethod(new Func<double, Bernoulli>(Bernoulli.FromLogOdds), operatorMethod);
                }
                else if (!lhsType.IsAssignableFrom(rhsType))
                {
                    // if the lhs and rhs have different types, convert from:
                    //   x = f(y);
                    // to:
                    //   x = SetPoint(x,f(y));
                    operatorMethod = transform.SetPointExpression(msg, lhsType, operatorMethod, rhsType);
                }

                return Builder.AssignExpr(msg, operatorMethod);
            }

            private IExpression WrapWithTracing(IExpression operatorMethod, Type returnType, IVariableDeclaration channelDecl, IExpression msg)
            {
                // Support for the 'TraceMessages' and 'ListenToMessages' attributes
                if ((transform.compiler.TraceAllMessages && !ModelCompiler.UseTracingTransform) ||
                    (channelDecl != null && (context.InputAttributes.Has<TraceMessages>(channelDecl) ||
                                               context.InputAttributes.Has<ListenToMessages>(channelDecl))))
                {
                    string msgText = msg.ToString();
                    // Look for TraceMessages attribute that matches this message
                    var trace = channelDecl is null ? null : context.InputAttributes.Get<TraceMessages>(channelDecl);
                    if (trace != null && trace.Containing != null && !msgText.Contains(trace.Containing)) trace = null;

                    // Look for ListenToMessages attribute that matches this message
                    var listenTo = channelDecl is null ? null : context.InputAttributes.Get<ListenToMessages>(channelDecl);
                    if (listenTo != null && listenTo.Containing != null && !msgText.Contains(listenTo.Containing)) listenTo = null;


                    if ((listenTo != null) || (trace != null) || (transform.compiler.TraceAllMessages && !ModelCompiler.UseTracingTransform))
                    {
                        IExpression textExpr = DebuggingSupport.GetExpressionTextExpression(msg);
                        if (listenTo != null)
                        {
                            // Generate code to emit message events
                            var methodRef = new CodeModel.Concrete.XMethodReference(transform.GetMessageUpdatedMethod());
                            var mre = Builder.MethodRefExpr(methodRef, Builder.ThisRefExpr());
                            operatorMethod = Builder.StaticGenericMethod(
                                new Func<PlaceHolder, string, Action<MessageUpdatedEventArgs>, bool, PlaceHolder>(Tracing.FireEvent<PlaceHolder>),
                                new Type[] { returnType }, operatorMethod, textExpr, mre, Builder.LiteralExpr(trace != null));
                        }
                        else
                        {
                            // Add a call to Tracing.Trace() to cause the message to be written to a TraceWriter
                            operatorMethod = Builder.StaticGenericMethod(new Func<PlaceHolder, string, PlaceHolder>(Tracing.Trace<PlaceHolder>),
                                                                         new Type[] { returnType }, operatorMethod, textExpr);
                        }
                    }
                }

                return operatorMethod;
            }

            /// <summary>
            /// Checks to see if this operator is just copying one of its parameters and hence can be
            /// optimised away.
            /// </summary>
            /// <param name="fninfo">The operator function info</param>
            /// <param name="args">The operator arguments</param>
            /// <param name="msg">The expression for the LHS which will store the result of the operator</param>
            /// <returns>The new expression to compute the result of the operation, or null if the operator is not a copy</returns>
            private IExpression CheckForCopyOperators(MessageFcnInfo fninfo, IList<IExpression> args, IExpression msg)
            {
                if (!transform.UseCopyOperators)
                    return null;
                // Find out if the operator is just copying one of its arguments
                int copyArg = fninfo.ReturnedParameterIndex;
                if (copyArg != -1)
                {
                    IExpression argToBeCopied = args[copyArg];
                    Type resultType = msg.GetExpressionType();
                    if (argToBeCopied == null)
                    {
                        transform.Error("method argument is null");
                        return null;
                    }
                    Type sourceType = argToBeCopied.GetExpressionType();
                    if (resultType.IsArray)
                    {
                        Type eltType = resultType.GetElementType();
                        if (Distribution.IsDistributionType(eltType) && Distribution.IsSettableTo(eltType, argToBeCopied.GetExpressionType()))
                        {
                            if (resultType.GetArrayRank() != 1) transform.Error("array rank > 1 is not supported in this context");
                            return Builder.StaticGenericMethod(new Func<Bernoulli[], Bernoulli[], Bernoulli[]>(ArrayHelper.SetTo),
                                                               new Type[] { eltType }, msg, argToBeCopied);
                        }
                        else
                            return Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder>(Clone.Copy<PlaceHolder>),
                                                               new Type[] { sourceType }, argToBeCopied);
                    }
                    else
                    {
                        // must use IsDistributionType here, not IsSettableTo, because only DistributionTypes are guaranteed to be initialized
                        if (Distribution.IsDistributionType(resultType) && Distribution.IsSettableTo(resultType, argToBeCopied.GetExpressionType()))
                            // TODO: find the appropriate generic type parameter, similar to SetPointExpression
                            return Builder.StaticGenericMethod(new Func<Bernoulli, Bernoulli, Bernoulli>(ArrayHelper.SetTo),
                                                               new Type[] { resultType }, msg, argToBeCopied);
                        else
                            return Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder>(Clone.Copy<PlaceHolder>),
                                                               new Type[] { sourceType }, argToBeCopied);
                    }
                }

                // Find out if the operator is copying one of its arguments in every element of the returned value.
                int copyMultipleArg = fninfo.ReturnedInAllElementsParameterIndex;
                if (copyMultipleArg != -1)
                {
                    IExpression argToBeCopied = args[copyMultipleArg];
                    Type resultType = msg.GetExpressionType();
                    if (resultType.IsArray)
                    {
                        if (resultType.GetArrayRank() != 1) transform.Error("array rank > 1 is not supported in this context");
                        return Builder.StaticGenericMethod(new Func<double[], double, double[]>(ArrayHelper.SetAllElementsTo),
                                                           new Type[] { argToBeCopied.GetExpressionType() }, msg, argToBeCopied);
                    }
                    else
                    {
                        return Builder.StaticGenericMethod(new Func<Vector, double, Vector>(ArrayHelper.SetAllElementsTo),
                                                           new Type[] { msg.GetExpressionType(), argToBeCopied.GetExpressionType() }, msg, argToBeCopied);
                    }
                }

                // Special handling of DiscreteUniform.SampleAverageConditional
                if (fninfo.Method.Equals(new Func<int, Discrete, Discrete>(DiscreteUniform.SampleAverageConditional).Method))
                {
                    IExpression size = args[0];
                    // check the marginal prototype of the result argument
                    IExpression result = msg;
                    IVariableDeclaration resultVar = Recognizer.GetVariableDeclaration(result);
                    MessageArrayInformation mai = context.InputAttributes.Get<MessageArrayInformation>(resultVar);
                    IExpression mpe = mai.marginalPrototypeExpression;
                    IExpression prototypeSize = StocAnalysisTransform.GetDiscreteLengthExpression(mpe);
                    if (size.Equals(prototypeSize))
                    {
                        return MakeUniform(mpe);
                    }
                }

                return null;
            }

            /// <summary>
            /// Adds statements that create a buffer for a message operator.
            /// Buffer is added to msgInfo and argumentTypes.
            /// </summary>
            /// <param name="name">Name of the buffer</param>
            /// <param name="type">Type of the buffer</param>
            /// <param name="fcninfo">The message operator that requested the buffer</param>
            /// <param name="miTgt">The target message that requested the buffer</param>
            /// <param name="argumentTypes">Modified to have the buffer</param>
            private MessageInfo CreateBuffer(
                string name,
                Type type,
                MessageFcnInfo fcninfo,
                MessageInfo miTgt,
                Dictionary<string, Type> argumentTypes)
            {
                if (type != null)
                    argumentTypes[name] = type;
                else
                    argumentTypes.Remove(name);
                Type oldResultType = argumentTypes[resultName];
                Type oldResultIndexType = argumentTypes.ContainsKey(resultIndexName) ? argumentTypes[resultIndexName] : null;
                if (type != null)
                    argumentTypes[resultName] = type;
                else
                    argumentTypes.Remove(resultName);
                argumentTypes.Remove(resultIndexName);
                MessageFcnInfo fcninfo2 = info.GetMessageFcnInfo(factorManager, "", name, argumentTypes, isStochastic);
                if (type == null)
                {
                    if (!fcninfo2.PassResultIndex)
                        type = fcninfo2.Method.ReturnType;
                    else
                    {
                        // must get buffer type from Init method
                        MessageFcnInfo fcninfoInit = info.GetMessageFcnInfo(factorManager, "Init", name, argumentTypes, isStochastic);
                        if (fcninfoInit.PassResultIndex)
                        {
                            transform.Error("Init method expects resultIndex but this is not supported");
                        }
                        type = fcninfoInit.Method.ReturnType;
                    }
                    argumentTypes[name] = type;
                    argumentTypes[resultName] = type;
                }
                string prefix = miTgt.messageFromFactor.ToString();
                if (true)
                {
                    // prefix the buffer name with the name of the first argument of its defining function
                    ParameterInfo parameter2 = fcninfo2.Method.GetParameters()[0];
                    string factorParameterName2 = fcninfo2.factorEdgeOfParameter[parameter2.Name].ParameterName;
                    if (msgInfo.TryGetValue(factorParameterName2, out MessageInfo mi2))
                    {
                        prefix = mi2.messageToFactor.ToString();
                    }
                }
                prefix = CodeBuilder.MakeValid(prefix);
                IVariableDeclaration bufferDecl = Builder.VarDecl(VariableInformation.GenerateName(context, prefix + "_" + name), type);
                context.OutputAttributes.Set(bufferDecl, new DescriptionAttribute("Buffer for " + StringUtil.EscapeXmlCharacters(StringUtil.MethodFullNameToString(fcninfo.Method))));
                context.OutputAttributes.Set(bufferDecl, new Containers(context));
                IExpression msg = Builder.VarRefExpr(bufferDecl);
                MessageInfo mi = new MessageInfo
                {
                    ////channelDecl = miTgt.channelDecl,
                    messageFromFactor = msg,
                    messageToFactor = msg
                };
                msgInfo[name] = mi;
                string flippedName = FactorManager.FlipCapitalization(name);
                msgInfo[flippedName] = mi;
                // TODO: fix this in cases where buffers form a cycle and therefore need inits
                if (fcninfo2.PassResult || HasParameter(fcninfo2, name) || HasParameter(fcninfo2, flippedName))
                {
                    context.AddStatementBeforeCurrent(Builder.ExprStatement(Builder.VarDeclExpr(bufferDecl)));
                    void AddInit(IStatement st)
                    {
                        context.OutputAttributes.Remove<OperatorStatement>(st);
                        context.OutputAttributes.Set(st, new Initializer());
                        context.AddStatementBeforeCurrent(st);
                    }
                    ForEachOperatorStatement(AddInit, "Init", name, argumentTypes);
                }
                else
                {
                    var assignSt = Builder.AssignStmt(Builder.VarDeclExpr(bufferDecl), Builder.DefaultExpr(type));
                    context.OutputAttributes.Set(assignSt, new Initializer());
                    context.AddStatementBeforeCurrent(assignSt);
                    context.OutputAttributes.Set(bufferDecl, new DoesNotHaveInitializer());
                }
                ForEachOperatorStatement(context.AddStatementBeforeCurrent, "", name, argumentTypes);
                argumentTypes[resultName] = oldResultType;
                if (oldResultIndexType != null)
                    argumentTypes[resultIndexName] = oldResultIndexType;
                return mi;
            }

            /// <summary>
            /// Get the operator arguments for a specified operator method
            /// </summary>
            /// <param name="fcninfo">Operator method information</param>
            /// <param name="miTgt">The message info for the target argument</param>
            /// <param name="argumentTypes">Argument types for the operator method.  May be modified with buffers.</param>
            /// 
            /// <returns></returns>
            private List<IExpression> GetOperatorArguments(MessageFcnInfo fcninfo, MessageInfo miTgt, Dictionary<string, Type> argumentTypes)
            {
                List<IExpression> args = new List<IExpression>();
                // Get the parameters of the operator method
                ParameterInfo[] parameters = fcninfo.Method.GetParameters();
                for (int i = 0; i < parameters.Length; i++)
                {
                    ParameterInfo parameter = parameters[i];
                    if (parameter.Name == resultName || parameter.Name == resultIndexName) continue;
                    FactorEdge factorEdge;
                    if (!fcninfo.factorEdgeOfParameter.TryGetValue(parameter.Name, out factorEdge))
                    {
                        transform.Error("Parameter name '" + parameter.Name + "' is unrecognized in " + StringUtil.MethodFullNameToString(fcninfo.Method));
                        continue;
                    }
                    string factorParameterName = factorEdge.ParameterName;
                    bool isOutgoingMessage = factorEdge.IsOutgoingMessage;
                    MessageInfo mi;
                    if (!msgInfo.TryGetValue(factorParameterName, out mi))
                    {
                        // create a buffer
                        Type bufferType = parameter.ParameterType;
                        if (fcninfo.IsIndexedParameter != null && fcninfo.IsIndexedParameter[i])
                            bufferType = null;
                        try
                        {
                            mi = CreateBuffer(factorParameterName, bufferType, fcninfo, miTgt, argumentTypes);
                        }
                        catch (Exception ex)
                        {
                            transform.Error("Could not construct operator method", ex);
                            args.Add(null);
                            continue;
                        }
                    }
                    if (mi.hasNonUnitDerivative)
                    {
                        object[] attrs = parameter.GetCustomAttributes(typeof(Stochastic), true);
                        if ((attrs != null) && (attrs.Length > 0))
                        {
                            transform.Error("Argument '" + parameter.Name + "' to operator method '" + StringUtil.MethodFullNameToString(fcninfo.Method) +
                                  "' must not be a derived variable.   Set Compiler.AllowDerivedParents = true if you want to risk running inference in this model.");
                        }
                    }
                    if (isOutgoingMessage && mi.messageFromFactor == null)
                    {
                        transform.Error("The operator method requires a message to " + factorParameterName + " but there is no such message");
                    }
                    IExpression value = isOutgoingMessage ? mi.messageFromFactor : mi.messageToFactor;
                    bool isWeaklyTyped = isVariableFactor && parameter.Name == "init";
                    if (isWeaklyTyped)
                    {
                        value = transform.ConvertInitialiser(value);
                        value = transform.ConvertInitialiser(value, parameter.ParameterType, null, 0, false);
                    }
                    args.Add(value);
                }
                return args;
            }

            private List<IExpression> ConvertArguments(MethodInfo method, IList<IExpression> args)
            {
                List<IExpression> convertedArgs = new List<IExpression>();
                ParameterInfo[] parameters = method.GetParameters();
                for (int i = 0; i < args.Count; i++)
                {
                    IExpression arg = args[i];
                    if (arg == null) continue;
                    Type argType = arg.GetExpressionType();
                    IExpression convertedArg = arg;
                    ParameterInfo parameter = parameters[i];
                    if (!parameter.ParameterType.IsAssignableFrom(argType))
                    {
                        Type distType = parameter.ParameterType;
                        Type domainType = argType;
                        MethodInfo pointMassMethod = FactorManager.GetPointMassMethod(distType, domainType);
                        if (pointMassMethod != null)
                        {
                            convertedArg = Builder.StaticMethod(pointMassMethod, arg);
                        }
                        else
                        {
                            transform.Error("Cannot convert from '" + StringUtil.TypeToString(argType) + "' to '" + StringUtil.TypeToString(distType) + "' for argument of method " +
                                  StringUtil.MethodSignatureToString(method));
                        }
                    }
                    convertedArgs.Add(convertedArg);
                }
                if (convertedArgs.Count != args.Count || convertedArgs.Count != parameters.Length) transform.Error("Some arguments were not converted");
                return convertedArgs;
            }

            /// <summary>
            /// Indicates whether the given factor parameter is an argument of the given message operator.
            /// </summary>
            /// <param name="fcninfo">The message operator</param>
            /// <param name="name">The factor parameter name</param>
            /// <returns>True if the factor parameter is an argument of the message operator</returns>
            /// <exception cref="NotSupportedException">The message operator has an unrecognized argument.</exception>
            private bool HasParameter(MessageFcnInfo fcninfo, string name)
            {
                ParameterInfo[] parameters = fcninfo.Method.GetParameters();
                foreach (ParameterInfo parameter in parameters)
                {
                    if (fcninfo.factorEdgeOfParameter.TryGetValue(parameter.Name, out FactorEdge edge))
                    {
                        if (edge.ParameterName == name) return true;
                    }
                    else transform.Error("Unrecognized parameter " + parameter.Name + " in method " + StringUtil.MethodSignatureToString(fcninfo.Method));
                }
                return false;
            }

            /// <summary>
            /// Describes a message in a message-passing program.
            /// </summary>
            /// <remarks>
            /// A message is a channel paired with a direction (either forwards or backwards).
            /// A message can be labelled as an "output" to mean it is the result of inference.
            /// When a random variable has multiple uses, all uses are labelled with the same MessageInfo.
            /// 
            /// If the channel is in a plate, we need to distinguish between the case of one message per each
            /// plate instance versus one message for the entire plate.
            /// 
            /// If the channel is an array type, then there are three possible message types:
            /// 1. If the channel is in a plate, then so is the message.
            /// 2. If the channel is not in a plate, but individual messages are desired for each array element.
            /// If the channel is inside a plate, in which case it stores for an array of distributions.
            /// </remarks>
            public class MessageInfo
            {
                internal IExpression messageFromFactor, messageToFactor;
                internal readonly IVariableDeclaration channelDecl;

                /// <summary>
                /// True if the factor argument is a derived variable with non-unit derivative.
                /// </summary>
                internal readonly bool hasNonUnitDerivative;

                public MessageInfo()
                {
                }

                public MessageInfo(MessageTransform transform, IExpression channelRef)
                {
                    IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(channelRef);
                    if (ivd == null)
                        return;
                    channelDecl = ivd;
                    hasNonUnitDerivative = transform.context.InputAttributes.Has<DerivedVariable>(ivd) && transform.hasNonUnitDerivative.Contains(ivd);
                    if (transform.AllowDerivedParents)
                        hasNonUnitDerivative = false;
                }

                public override string ToString()
                {
                    return $"MessageInfo(messageFromFactor: {messageFromFactor}, messageToFactor: {messageToFactor}, channelDecl: {channelDecl}, hasNonUnitDerivative: {hasNonUnitDerivative})";
                }
            }

            private class MessageInfoDict : ICompilerAttribute
            {
                public IReadOnlyDictionary<string, MessageInfo> msgInfo;

                public override string ToString()
                {
                    return StringUtil.DictionaryToString(msgInfo, " ");
                }
            }
        }

        private static bool HasBuffer(MessageFcnInfo fcninfo, string name)
        {
            object[] bufferAttrs = fcninfo.Method.DeclaringType.GetCustomAttributes(typeof(BuffersAttribute), true);
            foreach (BuffersAttribute bufferAttr in bufferAttrs)
            {
                foreach (string bufferName in bufferAttr.BufferNames)
                {
                    if (name == bufferName) return true;
                }
            }
            return false;
        }

        private IMethodDeclaration GetMessageUpdatedMethod()
        {
            if (messageUpdatedMethod == null)
            {
                // Generate message updated event on generated class
                var td = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
                var messageUpdatedEvent = Builder.EventDecl(DebuggingSupport.MessageEventName,
                                                            (ITypeReference)Builder.TypeRef(typeof(EventHandler<MessageUpdatedEventArgs>)), td);
                messageUpdatedEvent.Documentation = "Event that is fired when a message that is being monitored is updated.";
                td.Events.Add(messageUpdatedEvent);
                // Build a wrapper function that allows clients to fire the event
                messageUpdatedMethod = Builder.FireEventDecl(MethodVisibility.Private, "On" + DebuggingSupport.MessageEventName, messageUpdatedEvent);
            }
            return messageUpdatedMethod;
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            if (iae.Target is IVariableDeclarationExpression ivde)
            {
                if (IsStochasticVariableReference(ivde))
                {
                    ConvertExpression(iae.Target);
                    IAssignExpression ae2 = Builder.AssignExpr();
                    ae2.Target = Builder.VarRefExpr(ivde.Variable);
                    ae2.Expression = iae.Expression;
                    return ConvertExpression(ae2);
                }
            }

            // This will automatically convert the lhs into a forward message.
            IAssignExpression ae = (IAssignExpression)base.ConvertAssign(iae);
            if (ae == null || ae.Target == null || ae.Expression == null) return null;
            Type lhsType = ae.Target.GetExpressionType();
            Type rhsType = ae.Expression.GetExpressionType();
            if (lhsType == null)
            {
                Error("Cannot determine type of " + ae.Target);
                return null;
            }
            if (!lhsType.IsAssignableFrom(rhsType))
            {
                // if the lhs and rhs have different types, convert from:
                //   x = f(y);
                // to:
                //   x = SetPoint(x,f(y));
                ae.Expression = SetPointExpression(ae.Target, lhsType, ae.Expression, rhsType);
            }
            return ae;
        }

        private IExpression SetPointExpression(IExpression target, Type targetType, IExpression expression, Type expressionType)
        {
            Type hasPoint = targetType.GetInterface(typeof(HasPoint<>).Name);
            if (hasPoint == null)
            {
                Error("Cannot call SetPoint on " + StringUtil.TypeToString(targetType));
                return expression;
            }
            expressionType = hasPoint.GetGenericArguments()[0];
            return Builder.StaticGenericMethod(new Func<Bernoulli, bool, Bernoulli>(Distribution.SetPoint),
                                                        new Type[] { targetType, expressionType }, target, expression);
        }

        protected IExpression ConvertInfer(IMethodInvokeExpression imie)
        {
            IStatement ist = context.FindAncestor<IStatement>();
            context.OutputAttributes.Set(ist, new OperatorStatement());
            object decl = Recognizer.GetDeclaration(imie.Arguments[0]);
            // This will automatically convert the argument into a forward message
            // make a copy if we will modify later
            this.ShallowCopy = (decl != null);
            IMethodInvokeExpression mie = (IMethodInvokeExpression)base.ConvertMethodInvoke(imie);
            this.ShallowCopy = false;
            if (decl != null)
            {
                QueryType query = null;
                if (mie.Arguments.Count >= 3)
                {
                    ExpressionEvaluator eval = new ExpressionEvaluator();
                    query = (QueryType)eval.Evaluate(mie.Arguments[2]);
                }
                ChannelToMessageInfo ctmi = context.InputAttributes.Get<ChannelToMessageInfo>(decl);
                if (ctmi == null)
                {
                    ctmi = context.InputAttributes.Get<ObservedVariableMessages>(decl)?.ctmi;
                }
                if (ctmi != null)
                {
                    MessageDirection direction = (query == QueryTypes.MarginalDividedByPrior) ? MessageDirection.Backwards : MessageDirection.Forwards;
                    MessageArrayInformation mai = (direction == MessageDirection.Backwards) ? ctmi.bck : ctmi.fwd;
                    var arguments = mie.Arguments.ToArray();
                    arguments[0] = Builder.VarRefExpr(mai.decl);
                    var type = mai.decl.VariableType.DotNetType;
                    var method = ((MethodInfo)mie.Method.Method.MethodInfo).GetGenericMethodDefinition().MakeGenericMethod(type);
                    mie = Builder.StaticGenericMethod(method, arguments);
                }
                if (mie.Arguments.Count == 1)
                {
                    string varName;
                    ChannelInfo ci = context.InputAttributes.Get<ChannelInfo>(decl);
                    if (ci != null) varName = ci.varInfo.Name;
                    else if (decl is IParameterDeclaration ipd) varName = ipd.Name;
                    else varName = ((IVariableDeclaration)decl).Name;
                    mie = Builder.StaticGenericMethod(new Action<object, string>(InferNet.Infer), mie.Arguments[0], Builder.LiteralExpr(varName));
                }
            }
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(mie.Arguments[0]);
            if (ivd != null && !context.InputAttributes.Has<IsInferred>(ivd))
                context.OutputAttributes.Set(ivd, new IsInferred());
            return mie;
        }

        /// <summary>
        /// Returns true if the supplied expression is stochastic.  
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        protected bool IsStochasticVariableReference(IExpression expr)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
            return (ivd != null) && context.InputAttributes.Has<ChannelInfo>(ivd);
        }

        /// <summary>
        /// Create the statements for creating forward and backward message arrays.
        /// These are the _F and _B variables in the generated code. The MessageArrayInformation
        /// instances record information about these variables.
        /// </summary>
        private MessageArrayInformation CreateMessageVariable(
            string name,
            ChannelInfo channelInfo,
            ChannelToMessageInfo ctmi,
            MessageDirection direction,
            ChannelPathAttribute[] cpas)
        {
            string messageName = name + (direction == MessageDirection.Backwards ? "_B" : "_F");
            // Ask the algorithm for the message prototype in this direction
            // Message types may be different in each direction
            VariableInformation vi;
            if (channelInfo.decl != null)
            {
                vi = VariableInformation.GetVariableInformation(context, channelInfo.decl);
            }
            else vi = channelInfo.varInfo;
            if (vi.marginalPrototypeExpression == null)
            {
                try
                {
                    vi.SetMarginalPrototypeFromAttribute(null);
                }
                catch (ArgumentException ex)
                {
                    Error(ex.Message);
                }
            }
            IExpression prototypeExpression = vi.marginalPrototypeExpression;
            IExpression mpe = GetMarginalPrototypeExpression(messageName, channelInfo, direction, cpas, prototypeExpression);
            Type innermostType = mpe.GetExpressionType();
            // See if this is a distribution
            bool isDistribution = Distribution.IsDistributionType(innermostType) &&
                Distribution.GetDomainType(innermostType).Equals(Distribution.GetDomainType(prototypeExpression.GetExpressionType()));

            // Cache the message type
            bool isGibbsMarginal = false;
            Type messageType;
            if (isDistribution)
            {
                messageType = GetMessageType(channelInfo, innermostType);
            }
            else if (algorithm is GibbsSampling && channelInfo.IsMarginal)
            {
                messageType = innermostType;
                isGibbsMarginal = true;
            }
            else if (innermostType.Equals(channelInfo.channelType))
            {
                messageType = channelInfo.channelType;
            }
            else 
            {
                messageType = JaggedArray.GetTypes(
                    channelInfo.channelType,
                    JaggedArray.GetInnermostType(channelInfo.channelType),
                    innermostType)[0];
            }
            // Cache the message array information, including the declaration
            MessageArrayInformation mai = new MessageArrayInformation
            {
                // Create the variable declaration of the message array variable
                decl = Builder.VarDecl(messageName, messageType),
                ci = channelInfo,
                isDistribution = isDistribution,
                marginalPrototypeExpression = mpe
            };
            // Attach the message array information as an attribute on the variable declaration
            context.OutputAttributes.Set(mai.decl, mai);
            if (direction == MessageDirection.Backwards)
            {
                ctmi.bck = mai;
            }
            else
            {
                ctmi.fwd = mai;
            }
            VariableInformation messageInfo = VariableInformation.GetVariableInformation(context, mai.decl);
            if (!isGibbsMarginal)
            {
                messageInfo.indexVars = vi.indexVars;
                messageInfo.sizes = vi.sizes;
            }

            if (channelInfo.decl != null)
            {
                AttachAttributes(mai.decl, channelInfo, direction);
            }

            return mai;
        }

        private IExpression GetMarginalPrototypeExpression(string messageName, ChannelInfo channelInfo, MessageDirection direction, ChannelPathAttribute[] cpas, IExpression prototypeExpression)
        {
            List<QueryType> qtlist;
            if (channelInfo.decl != null && !context.InputAttributes.Has<IsInferred>(channelInfo.decl)) qtlist = new List<QueryType>();
            else qtlist = context.InputAttributes.GetAll<QueryTypeCompilerAttribute>(channelInfo.varInfo.declaration).Select(attr => attr.QueryType).ToList();
            IExpression mpe = algorithm.GetMessagePrototype(channelInfo, direction, prototypeExpression, null, qtlist);
            if (direction == MessageDirection.Forwards && !channelInfo.IsMarginal && !channelInfo.varInfo.IsStochastic)
            {
                return Builder.DefaultExpr(channelInfo.channelType);
            }

            string path = "";
            // Check for any path attributes for the channel
            bool foundOne = false;
            bool foundNonDefault = false;

            foreach (ChannelPathAttribute cpa in cpas)
            {
                if (cpa.Direction != direction)
                    continue;

                // If we've already found a path for this direction
                // and this is default path, continue
                if (foundOne && cpa.FromDefault)
                    continue;

                bool notDefault = !cpa.FromDefault;

                // Flag an error if there are two inconsistent non-default paths
                if (foundNonDefault && notDefault && cpa.Path != path)
                    Error($"Inconsistent message types for {messageName}. If you have manually specified groups, try changing the root variable.");

                if (!foundNonDefault)
                {
                    path = cpa.Path;
                    mpe = algorithm.GetMessagePrototype(channelInfo, direction, prototypeExpression, path, qtlist);
                    foundOne = true;
                    foundNonDefault = notDefault;
                }
            }

            return mpe;
        }

        private MessageArrayInformation CreateMessageVariable2(string name, ChannelInfo channelInfo, ChannelToMessageInfo ctmi, MessageDirection direction,
                                                                ChannelPathAttribute[] cpas)
        {
            Dictionary<IVariableDeclaration, IVariableDeclaration> messageVars = (direction == MessageDirection.Forwards) ? analysis.fwdMessageVars : analysis.bckMessageVars;
            IVariableDeclaration messageVar;
            if (!messageVars.TryGetValue(channelInfo.decl, out messageVar)) return null;
            Type messageType = messageVar.VariableType.DotNetType;

            // See if this is a distribution
            bool isDistribution = Distribution.IsDistributionType(messageType);

            // Cache the message array information, including the declaration
            MessageArrayInformation mai = new MessageArrayInformation
            {
                // Create the variable declaration of the message array variable
                decl = messageVar,
                ci = channelInfo,
                isDistribution = isDistribution
            };
            //mai.marginalPrototypeExpression = initExpr;
            // Attach the message array information as an attribute on the variable declaration
            context.OutputAttributes.Set(mai.decl, mai);
            if (direction == MessageDirection.Backwards)
            {
                ctmi.bck = mai;
            }
            else
            {
                ctmi.fwd = mai;
            }

            AttachAttributes(mai.decl, channelInfo, direction);

            return mai;
        }

        private void AttachAttributes(IVariableDeclaration ivd, ChannelInfo channelInfo, MessageDirection direction)
        {
            // Description of this message array variable - this will eventually get built into a code comment
            DescriptionAttribute da = context.InputAttributes.Get<DescriptionAttribute>(channelInfo.decl);
            if (da != null)
            {
                string s = "Message";
                if (ivd.VariableType.DotNetType.IsArray) s += "s";
                bool isFrom = (direction == MessageDirection.Backwards);
                if (channelInfo.IsDef) isFrom = !isFrom;
                if (isFrom) s += " from ";
                else s += " to ";
                context.OutputAttributes.Set(ivd, new DescriptionAttribute(s + da.Description));
            }

            // Propagate InitialiseTo attribute
            InitialiseTo it = context.InputAttributes.Get<InitialiseTo>(channelInfo.decl);
            if (it != null)
            {
                if (channelInfo.IsMarginal || direction == MessageDirection.Forwards)
                {
                    context.OutputAttributes.Set(ivd, it);
                }
            }

            // Propagate InitialiseBackward
            InitialiseBackward ib = context.InputAttributes.Get<InitialiseBackward>(channelInfo.decl);
            if (ib != null)
            {
                if (direction == MessageDirection.Backwards)
                {
                    context.OutputAttributes.Set(ivd, ib);
                }
            }

            // Propagate InitialiseBackwardTo
            InitialiseBackwardTo ibt = context.InputAttributes.Get<InitialiseBackwardTo>(channelInfo.decl);
            if (ibt != null)
            {
                if (direction == MessageDirection.Backwards)
                {
                    context.OutputAttributes.Set(ivd, new InitialiseTo(ibt.initialMessagesExpression));
                }
            }
        }

        /// <summary>
        /// Convert a variable declaration expression. Each channel declaration expression
        /// is converted to a forward and a backward message declaration expression. Message
        /// array information for both directions is recorded
        /// </summary>
        /// <param name="ivde">The channel variable declaration expression</param>
        /// <returns>null</returns>
        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            IVariableDeclaration ivd = ivde.Variable;
            context.InputAttributes.Remove<Containers>(ivd);
            context.OutputAttributes.Set(ivd, new Containers(context));
            ChannelInfo channelInfo = context.InputAttributes.Get<ChannelInfo>(ivd);
            // If a variable is stochastic it will get turned into channels, so channelInfo==null
            // identifies a non-stochastic variable
            if (channelInfo == null)
            {
                return ivde; // The variable is constant.
            }
            // ChannelToMessageInfo records any info needed to convert channels to messages
            // including references to the forward and backward message array information
            ChannelToMessageInfo ctmi = context.InputAttributes.Get<ChannelToMessageInfo>(ivd);
            if (ctmi == null)
            {
                ctmi = new ChannelToMessageInfo();
                // Attach the info to declaration of channel variable
                context.InputAttributes.Set(ivd, ctmi);
            }
            var cpas = context.InputAttributes.GetAll<ChannelPathAttribute>(ivd).ToArray();
            foreach (MessageDirection direction in directions)
            {
                MessageArrayInformation mai = UseMessageAnalysis
                                                  ? CreateMessageVariable2(ivd.Name, channelInfo, ctmi, direction, cpas)
                                                  : CreateMessageVariable(ivd.Name, channelInfo, ctmi, direction, cpas);
                if (mai == null) continue;
                context.OutputAttributes.Set(mai.decl, new Containers(context));
                // Make the declaration expression for the message array variable
                IExpressionStatement stmt = MakeDeclStatement(channelInfo, mai);
                // This will be a declaration expression if this is an array and
                // a declaration with assignment otherwise
                if (InitializeOnSeparateLine && stmt.Expression is IAssignExpression iae)
                {
                    context.AddStatementBeforeCurrent(Builder.ExprStatement(iae.Target));
                    iae.Target = Builder.VarRefExpr(mai.decl);
                    context.AddStatementBeforeCurrent(Builder.ExprStatement(iae));
                }
                else context.AddStatementBeforeCurrent(stmt);
            }
            return null;
        }

        /// <summary>
        /// Make the declaration statement for a message array variable
        /// </summary>
        /// <param name="channelInfo">The channel information</param>
        /// <param name="mai">The information for the message array variable</param>
        /// <returns>An assign expression or a declaration expression</returns>
        protected IExpressionStatement MakeDeclStatement(ChannelInfo channelInfo, MessageArrayInformation mai)
        {
            // Build the declaration expression
            IVariableDeclarationExpression ivde = Builder.VarDeclExpr(mai.decl);
            // Attach the quality band for the declaration expression
            Type msgType = mai.messageArrayType;
            if (Distribution.HasDistributionType(msgType))
            {
                QualityBand qb = Distribution.GetQualityBand(msgType);
                Context.OutputAttributes.Set(ivde, new QualityBandCompilerAttribute(qb));
            }

            VariableInformation channelVarInfo = (channelInfo.decl != null)
                ? context.InputAttributes.Get<VariableInformation>(channelInfo.decl)
                : channelInfo.varInfo;
            // Note: sizes.Count != ArrayDepth in general
            // If this variable has sizes, then it must have array create expressions, which will be handled by DoConvertArray.
            // However, !mai.isDistribution implies DoConvertArray will always have elementInit==null and generates no statements.
            if (channelVarInfo.sizes.Count == 0 || (!mai.isDistribution && !(algorithm is GibbsSampling)))
            {
                InitialiseTo it = context.InputAttributes.Get<InitialiseTo>(mai.decl);
                bool userInitialized = (it != null);
                IExpression expr;
                if (it != null)
                {
                    if (IsGibbsMarginal(msgType))
                    {
                        AddGibbsMarginalInitStatements(Builder.VarRefExpr(mai.decl), it.initialMessagesExpression, channelVarInfo);
                        expr = mai.marginalPrototypeExpression;
                    }
                    else
                    {
                        expr = ConvertInitialiser(it.initialMessagesExpression);
                        expr = ConvertInitialiser(expr, msgType, channelVarInfo);
                    }
                }
                else if (!UseMessageAnalysis && mai.isDistribution) expr = MakeUniform(mai.marginalPrototypeExpression);
                else expr = mai.marginalPrototypeExpression;
                if (expr != null)
                {
                    if (!msgType.IsAssignableFrom(expr.GetExpressionType()))
                        Error($"Invalid initializer: {ivde} = {expr}");
                    IExpressionStatement init = Builder.AssignStmt(ivde, expr);
                    context.OutputAttributes.Set(init, new Initializer() { UserInitialized = userInitialized });
                    return init;
                }
            }
            IExpressionStatement declSt = Builder.ExprStatement(ivde);
            context.OutputAttributes.Set(declSt, new Initializer());
            return declSt;
        }

        protected override IExpression ConvertArrayCreate(IArrayCreateExpression iace)
        {
            // Get the index of the assignment expression whose RHS is this array create expression 
            int assignIndex = context.FindAncestorIndex<IAssignExpression>();
            // In MSL, random variable array creation can only occur on the RHS of an assignment.  
            if (assignIndex == context.InputStack.Count - 2)
            {
                IAssignExpression iae = (IAssignExpression)context.GetAncestor(assignIndex);
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
                if (ivd != null)
                {
                    ChannelInfo ci = context.InputAttributes.Get<ChannelInfo>(ivd);
                    ChannelToMessageInfo ctmi = context.InputAttributes.Get<ChannelToMessageInfo>(ivd);
                    if (ctmi != null)
                    {
                        int depth = Recognizer.GetIndexingDepth(iae.Target);
                        foreach (var direction in directions)
                        {
                            CreateArray(direction, iace, iae.Target, depth, ivd, ci, ctmi);
                        }
                        return null;
                    }
                }
            }
            return base.ConvertArrayCreate(iace);
        }

        private void CreateArray(MessageDirection direction, IArrayCreateExpression iace, IExpression lhs, int depth, IVariableDeclaration ivd, ChannelInfo ci, ChannelToMessageInfo ctmi)
        {
            //lhs = Builder.ReplaceVariable(lhs, ivd, mai.decl);
            lhs = GetMessageExpression(lhs, direction);
            IExpression rhs = UseMessageAnalysis
                                  ? DoConvertArray2(direction, ivd, lhs, iace, depth, ctmi)
                                  : DoConvertArray(direction, ivd, lhs, iace, depth, ci, ctmi);
            if (rhs != null)
            {
                IExpressionStatement st = Builder.AssignStmt(lhs, rhs);
                context.OutputAttributes.Set(st.Expression, new DescriptionAttribute("Create array for '" + ci.decl.Name + "' " + direction + " messages."));
                Initializer ia = context.OutputAttributes.Get<Initializer>(rhs);
                if (ia != null)
                {
                    context.OutputAttributes.Remove<Initializer>(rhs);
                    context.OutputAttributes.Set(st, ia);
                }
                context.AddStatementBeforeCurrent(st);
            }
        }

        private IExpression DoConvertArray2(
            MessageDirection direction,
            IVariableDeclaration inputTarget,
            IExpression outputLhs,
            IArrayCreateExpression iace, int depth,
            ChannelToMessageInfo ctmi)
        {
            // mai is also an attribute of outputLhs
            MessageArrayInformation mai = (direction == MessageDirection.Forwards) ? ctmi.fwd : ctmi.bck;
            if (mai == null) return null;
            Type messageType = mai.messageArrayType; // type of the entire message array at the top level

            // arrayType must be a distribution array
            IList<IExpression> args = Builder.ExprCollection();
            //if (ci.ArrayDepth == depth + 1) args.Add(elementInit);
            args.AddRange(iace.Dimensions);
            Type arrayType = messageType; // type of this array
            if (!IsGibbsMarginal(arrayType))
            {
                for (int i = 0; i < depth; i++)
                {
                    arrayType = Util.GetElementType(arrayType);
                }
            }
            return Builder.NewObject(arrayType, args);
        }

        /// <summary>
        /// Construct an array creation expression and insert initialization statements
        /// </summary>
        /// <param name="direction">The message direction</param>
        /// <param name="inputTarget">Declaration of the variable on the LHS</param>
        /// <param name="outputLhs">Expression on the LHS</param>
        /// <param name="iace">Expression on the RHS</param>
        /// <param name="depth">Number of indexing brackets on the LHS</param>
        /// <param name="ci">Attribute of ivd</param>
        /// <param name="ctmi">Attribute of ivd</param>
        /// <returns></returns>
        private IExpression DoConvertArray(
            MessageDirection direction,
            IVariableDeclaration inputTarget,
            IExpression outputLhs,
            IArrayCreateExpression iace, int depth,
            ChannelInfo ci,
            ChannelToMessageInfo ctmi)
        {
            // mai is also an attribute of outputLhs
            MessageArrayInformation mai = (direction == MessageDirection.Forwards) ? ctmi.fwd : ctmi.bck;
            VariableInformation channelVarInfo = context.InputAttributes.Get<VariableInformation>(inputTarget);
            IExpression elementInit;
            Type messageType = mai.messageArrayType; // type of the entire message array at the top level

            IExpression mpe = mai.marginalPrototypeExpression;
            try
            {
                List<IList<IExpression>> indices = Recognizer.GetIndices(outputLhs);
                if (indices.Count > 0)
                {
                    mpe = channelVarInfo.ReplaceIndexVars(context, mpe, indices, null, out int replaceCount);
                }
                if (mai.isDistribution)
                {
                    elementInit = MakeUniform(mpe);
                }
                else
                {
                    elementInit = null;
                }
            }
            catch (ArgumentException ex)
            {
                Error("Invalid marginal prototype expression: " + mpe, ex);
                return iace;
            }
            if (algorithm is GibbsSampling && ci.IsMarginal)
            {
                if (direction == MessageDirection.Forwards)
                {
                    if (mpe is IObjectCreateExpression ioce && IsGibbsMarginal(messageType))
                    {
                        if (depth > 0) return null;
                        InitialiseTo it = context.InputAttributes.Get<InitialiseTo>(mai.decl);
                        if (it != null)
                            AddGibbsMarginalInitStatements(Builder.VarRefExpr(mai.decl), it.initialMessagesExpression, channelVarInfo);
                        if (ioce.Arguments[0] is IDefaultExpression)
                        {
                            // replace the first argument of "new GibbsMarginal" with marginal_B
                            IExpression messageExpression = Builder.VarRefExpr(Builder.VarRef(ctmi.bck.decl));
                            ioce.Arguments[0] = messageExpression;
                            List<IExpression> newArgs = new List<IExpression>();
                            newArgs.Add(messageExpression);
                            for (int i = 1; i < ioce.Arguments.Count; i++)
                            {
                                newArgs.Add(ioce.Arguments[i]);
                            }
                            IExpression initExpr = Builder.NewObject(ioce.Type, newArgs);
                            context.OutputAttributes.Set(initExpr, new Initializer());
                            return initExpr;
                        }
                        else return ioce;
                    }
                }
                if (!mai.isDistribution)
                {
                    if (direction == MessageDirection.Forwards && depth == 0) return mpe;
                    else return null;
                }
            }

            bool isInitialiseTo = false;
            // If the user has specified an initializer, use it for elementInit.
            if (!(ci.IsMarginal && algorithm is GibbsSampling))
            {
                InitialiseTo it = context.InputAttributes.Get<InitialiseTo>(mai.decl);
                if (it != null)
                {
                    elementInit = it.initialMessagesExpression;
                    isInitialiseTo = true;
                }
                // fall through
            }
            int arrayDepth;
            Type domainType;
            if (elementInit != null)
            {
                if (isInitialiseTo)
                {
                    domainType = GetInitializerDomainType(elementInit, out int minusDepth);
                    Assert.IsTrue(minusDepth >= 0);
                    arrayDepth = Util.GetArrayDepth(ci.channelType, domainType) - minusDepth;
                    bool addBracketsToInitialiser = true;
                    if (addBracketsToInitialiser)
                    {
                        // add brackets to avoid the creation of unnecessary arrays
                        elementInit = AddBracketsToInitialiser(context, elementInit, channelVarInfo, ref arrayDepth, minusDepth);
                    }
                }
                else
                {
                    domainType = Distribution.GetDomainType(elementInit.GetExpressionType());
                    arrayDepth = Util.GetArrayDepth(ci.channelType, domainType);
                }
            }
            else
            {
                Type marginalType = mpe.GetExpressionType();
                // If marginalType is PointMass<innermost channelType>, we want GetDomainType.
                // If marginalType is innermost channelType, we do not want GetDomainType.
                domainType = mai.isDistribution ? Distribution.GetDomainType(marginalType) : marginalType;
                arrayDepth = Util.GetArrayDepth(ci.channelType, domainType);
            }
            bool depthZeroInit = arrayDepth == 0 && depth == 0 && elementInit != null;
            // if arrayDepth < depth+1, then DoConvertArray will have already created an initializer
            if (arrayDepth < depth + 1 && !depthZeroInit) return null;
            Type arrayType = messageType; // type of this array
            if (!IsGibbsMarginal(arrayType))
            {
                for (int i = 0; i < depth; i++)
                {
                    arrayType = Util.GetElementType(arrayType);
                }
            }
            // arrayType is now the message type of the LHS
            if (depthZeroInit)
            {
                if (isInitialiseTo)
                {
                    elementInit = ConvertInitialiser(elementInit);
                    elementInit = ConvertInitialiser(elementInit, arrayType, channelVarInfo, 0, false);
                    Type outputLhsType = outputLhs.GetExpressionType();
                    if (Distribution.IsSettableTo(outputLhsType, outputLhsType))
                    {
                        elementInit = Builder.StaticGenericMethod(
                            new Func<PlaceHolder, PlaceHolder, PlaceHolder>(ArrayHelper.SetTo<PlaceHolder>),
                            new Type[] { outputLhsType }, outputLhs, elementInit);
                    }
                    IStatement init = Builder.AssignStmt(outputLhs, elementInit);
                    context.OutputAttributes.Set(init, new Initializer()
                    {
                        UserInitialized = isInitialiseTo
                    });
                    context.AddStatementAfterCurrent(init);
                    // fall through
                }
                else
                {
                    context.OutputAttributes.Set(elementInit, new Initializer()
                    {
                        UserInitialized = isInitialiseTo
                    });
                    return elementInit;
                }
            }
            else if (arrayDepth == depth + 1 && elementInit != null)
            {
                if (isInitialiseTo)
                {
                    elementInit = ConvertInitialiser(elementInit);
                    Type elementType = Util.GetElementType(arrayType);
                    elementInit = ConvertInitialiser(elementInit, elementType, channelVarInfo);
                }
                foreach (IStatement init in FillArray(outputLhs, channelVarInfo, depth, (IReadOnlyList<IExpression>)iace.Dimensions, elementInit))
                {
                    context.OutputAttributes.Set(init, new Initializer() { UserInitialized = isInitialiseTo });
                    context.AddStatementAfterCurrent(init);
                }
                //args.Add(MakeArrayInitDelegate(elementInit, margType, ci.varInfo, iace.Dimensions.Count));
            }
            IExpression arrayCreate = GetArrayCreateExpression(outputLhs, arrayType, iace.Dimensions);
            context.OutputAttributes.Set(arrayCreate, new Initializer() { UserInitialized = isInitialiseTo });
            return arrayCreate;
        }

        internal static IExpression GetArrayCreateExpression(IExpression outputLhs, Type arrayType, IEnumerable<IExpression> dimensions)
        {
            if (arrayType.IsArray)
            {
                Type elementType = arrayType.GetElementType();
                return Builder.ArrayCreateExpr(elementType, dimensions);
            }
            // arrayType must be a distribution array
            IList<IExpression> args = Builder.ExprCollection();
            if (IsFileArray(arrayType))
            {
                bool parentIsDFA = false;
                IArrayIndexerExpression iaieLhs = outputLhs as IArrayIndexerExpression;
                if (iaieLhs != null)
                {
                    Type targetType = iaieLhs.Target.GetExpressionType();
                    parentIsDFA = IsFileArray(targetType);
                }
                if (parentIsDFA)
                {
                    args.Add(iaieLhs.Target);
                    args.AddRange(iaieLhs.Indices);
                }
                else
                {
                    string name = GetName(outputLhs);
                    args.Add(Builder.StaticMethod(new Func<string, string>(FileArray<bool>.GetTempFolder), Builder.LiteralExpr(name)));
                }
            }
            args.AddRange(dimensions);
            return Builder.NewObject(arrayType, args);
        }

        internal static bool IsFileArray(Type type)
        {
            return (type.Name == typeof(DistributionFileArray<,>).Name || type.Name == typeof(FileArray<>).Name);
        }

        internal static string GetName(IExpression expr)
        {
            if (expr is IArrayIndexerExpression iaie) return GetName(iaie);
            else if (expr is IVariableReferenceExpression ivre) return ivre.Variable.Variable.Name;
            else if (expr is IVariableDeclarationExpression ivde) return ivde.Variable.Name;
            else return CodeBuilder.MakeValid(expr.ToString());
        }

        internal static string GetName(IArrayIndexerExpression iaie)
        {
            string name = GetName(iaie.Target);
            IList<IExpression> bracket = iaie.Indices;
            for (int j = 0; j < bracket.Count; j++)
            {
                name += "_" + GetName(bracket[j]);
            }
            return name;
        }

        private Type GetInitializerDomainType(IExpression initialiser, out int minusDepth)
        {
            if (initialiser is IArrayIndexerExpression iaie)
            {
                Type type = GetInitializerDomainType(iaie.Target, out minusDepth);
                if (minusDepth > 0)
                {
                    minusDepth--;
                    return type;
                }
                else return Util.GetElementType(type);
            }
            else
            {
                if (initialiser is IArrayCreateExpression iace)
                {
                    if (iace.Type.DotNetType.Equals(typeof(PlaceHolder)))
                    {
                        Type type = GetInitializerDomainType(iace.Initializer.Expressions[0], out minusDepth);
                        minusDepth++;
                        return type;
                    }
                }
                Type initType = initialiser.GetExpressionType();
                minusDepth = 0;
                while (Util.IsIList(initType))
                {
                    minusDepth++;
                    initType = Util.GetElementType(initType);
                }
                return Distribution.GetDomainType(initType);
            }
        }

        /// <summary>
        /// Convert a raw initialiser expression (which may not be valid code) into a well-typed expression, possibly with "new PlaceHolder" on the outside.
        /// </summary>
        /// <param name="initialiser"></param>
        /// <returns></returns>
        private IExpression ConvertInitialiser(IExpression initialiser)
        {
            if (initialiserType.TryGetValue(initialiser, out Type type)) return Builder.CastExpr(initialiser, type);
            if (initialiser is IArrayIndexerExpression iaie)
            {
                IExpression target = ConvertInitialiser(iaie.Target);
                if (target is IArrayCreateExpression iace)
                {
                    if (iace.Type.DotNetType.Equals(typeof(PlaceHolder)) && iace.Initializer != null && iace.Initializer.Expressions.Count == 1)
                    {
                        // VariableInformation.DeriveArrayVariable has created an initializer expression of the form:
                        // new PlaceHolder[wildcard0,wildcard1] { initExpr[wildcard0,wildcard1] }
                        IExpression initExpr = iace.Initializer.Expressions[0];
                        // replace index variables with the given indices
                        Dictionary<IExpression, IExpression> replacements = new Dictionary<IExpression, IExpression>();
                        for (int dim = 0; dim < iace.Dimensions.Count; dim++)
                        {
                            replacements.Add(iace.Dimensions[dim], iaie.Indices[dim]);
                        }
                        return Builder.ReplaceSubexpressions(initExpr, replacements);
                    }
                }
                return Builder.ArrayIndex(target, iaie.Indices);
            }
            else if (initialiser is IArrayCreateExpression iace)
            {
                if (iace.Initializer == null) return initialiser;
                IArrayCreateExpression ace = Builder.ArrayCreateExpr(iace.Type, iace.Dimensions);
                ace.Initializer = Builder.BlockExpr();
                foreach (IExpression expr in iace.Initializer.Expressions)
                {
                    ace.Initializer.Expressions.Add(ConvertInitialiser(expr));
                }
                return ace;
            }
            else return initialiser;
        }

        /// <summary>
        /// Create an initialiser expression with additional indexing brackets
        /// </summary>
        /// <param name="context"></param>
        /// <param name="initialiser"></param>
        /// <param name="varInfo"></param>
        /// <param name="depth">Increased by to the number of brackets added</param>
        /// <param name="numBracketsToAdd">Desired number of brackets to add</param>
        /// <returns></returns>
        private IExpression AddBracketsToInitialiser(BasicTransformContext context, IExpression initialiser, VariableInformation varInfo, ref int depth, int numBracketsToAdd)
        {
            for (int bracket = 0; bracket < numBracketsToAdd; bracket++)
            {
                if (varInfo.sizes.Count < depth + 1) break;
                varInfo.DefineIndexVarsUpToDepth(context, depth + 1);
                var depthCopy = depth;
                IExpression[] indices = Util.ArrayInit(varInfo.indexVars[depth].Length,
                    i => Builder.VarRefExpr(varInfo.indexVars[depthCopy][i]));
                initialiser = Builder.ArrayIndex(initialiser, indices);
                depth++;
            }
            return initialiser;
        }

        /// <summary>
        /// Convert an initialiser expression (which may have "new PlaceHolder" on the outside) into a valid expression of the desired type.
        /// </summary>
        /// <param name="initialiser"></param>
        /// <param name="desiredType"></param>
        /// <param name="varInfo"></param>
        /// <param name="depth"></param>
        /// <param name="makeCopy"></param>
        /// <returns></returns>
        private IExpression ConvertInitialiser(IExpression initialiser, Type desiredType, VariableInformation varInfo, int depth = 0, bool makeCopy = true)
        {
            if (initialiser is IArrayCreateExpression iace)
            {
                if (iace.Initializer != null && iace.Initializer.Expressions.Count == 1)
                {
                    Type desiredElementType = Util.GetElementType(desiredType);
                    IExpression elementInit = ConvertInitialiser(iace.Initializer.Expressions[0], desiredElementType, varInfo, depth + 1);
                    if (desiredType.IsArray)
                    {
                        return GetArrayCreateExpression(desiredType, elementInit, varInfo, depth);
                    }
                    else
                    {
                        return GetDistributionArrayCreateExpression(desiredType, desiredElementType, elementInit, varInfo, depth);
                    }
                }
            }
            Type initType = initialiser.GetExpressionType();
            if (!desiredType.IsAssignableFrom(initType))
            {
                if (initType != null && Distribution.IsDistributionType(initType) && desiredType.IsAssignableFrom(Distribution.GetDomainType(initType)))
                {
                    return GetSamplingExpression(initialiser);
                }
                if (initialiser is IArrayIndexerExpression iaie)
                {
                    Type listType = typeof(IList<>).MakeGenericType(desiredType);
                    IExpression target = ConvertInitialiser(iaie.Target, listType, varInfo, depth - 1, false);
                    initialiser = Builder.ArrayIndex(target, iaie.Indices);
                }
                else if (!initType.IsAssignableFrom(desiredType))
                {
                    throw new InferCompilerException($"Initialiser {initialiser} has type {initType} instead of {desiredType}");
                }
                else
                {
                    initialiserType[initialiser] = desiredType;
                    initialiser = Builder.CastExpr(initialiser, desiredType);
                }
            }
            if (makeCopy && Distribution.IsDistributionType(desiredType))
            {
                initialiser = Builder.StaticGenericMethod(
                    new Func<PlaceHolder, PlaceHolder>(ArrayHelper.MakeCopy),
                    new Type[] { desiredType },
                    initialiser);
            }
            return initialiser;
        }

        public static IExpression GetDistributionArrayCreateExpression(Type arrayType, Type innermostElementType, IExpression innermostElementInit, VariableInformation varInfo,
                                                                       int depth = 0)
        {
            // new DistributionRefArray<>(sourceArray.Length, index0 => new DistributionArray(sourceArray[index0].Length, ...))
            if (innermostElementType.IsAssignableFrom(arrayType)) return innermostElementInit;
            Type elementType = Util.GetElementType(arrayType, out int rank);
            if (elementType == null) throw new ArgumentException(arrayType + " is not an array type with innermost element type " + innermostElementType);
            IExpression elementInit = GetDistributionArrayCreateExpression(elementType, innermostElementType, innermostElementInit, varInfo, depth + 1);
            IExpression initDelegate = MakeArrayInitDelegate(elementInit, varInfo.indexVars[depth]);
            Type innerType = elementInit.GetExpressionType();
            Type distributionType = Distribution.MakeDistributionArrayType(innerType, rank);
            IList<IExpression> args = Builder.ExprCollection();
            for (int i = 0; i < rank; i++)
            {
                args.Add(varInfo.sizes[depth][i]);
            }
            args.Add(initDelegate);
            return Builder.NewObject(distributionType, args);
        }

        public static IAnonymousMethodExpression MakeArrayInitDelegate(IExpression elementInit, IVariableDeclaration[] indexVars)
        {
            // result has the form:  
            //   delegate(int index1, int index2) { return elementInit; })
            int rank = indexVars.Length;
            Type[] typeArgs = new Type[rank + 1];
            for (int i = 0; i < rank; i++)
            {
                typeArgs[i] = typeof(int);
            }
            typeArgs[typeArgs.Length - 1] = elementInit.GetExpressionType();
            Type delegateType;
            if (rank == 1) delegateType = typeof(Func<,>);
            else if (rank == 2) delegateType = typeof(Func<,,>);
            else if (rank == 3) delegateType = typeof(Func<,,,>);
            else if (rank == 4) delegateType = typeof(Func<,,,,>);
            else throw new NotImplementedException("Cannot initialize array of rank " + rank);
            IAnonymousMethodExpression iame = Builder.AnonMethodExpr(delegateType.MakeGenericType(typeArgs));
            iame.Body = Builder.BlockStmt();
            for (int i = 0; i < rank; i++)
            {
                IVariableDeclaration indexVar = indexVars[i];
                IParameterDeclaration param = Builder.Param(indexVar.Name, typeof(int));
                iame.Parameters.Add(param);
                elementInit = Builder.ReplaceExpression(elementInit, Builder.VarRefExpr(indexVar), Builder.ParamRef(param));
            }
            iame.Body.Statements.Add(Builder.Return(elementInit));
            return iame;
        }

        private IExpression GetSamplingExpression(IExpression distributionExpr)
        {
            Type srcType = distributionExpr.GetExpressionType();
            Type sampleableType = typeof(Sampleable<>).MakeGenericType(Distribution.GetDomainType(srcType));
            MethodInfo smplMthd = sampleableType.GetMethod("Sample", new Type[] { });
            if (!sampleableType.IsAssignableFrom(srcType))
            {
                distributionExpr = Builder.CastExpr(distributionExpr, sampleableType);
            }
            IMethodInvokeExpression imie = Builder.MethodInvkExpr();
            imie.Method = Builder.MethodRefExpr(Builder.MethodRef(smplMthd), distributionExpr);
            return imie;
        }

        private void AddGibbsMarginalInitStatements(IExpression gibbsMargExpr, IExpression distExpr, VariableInformation varInfo)
        {
            if (gibbsMargExpr != null && distExpr != null)
            {
                Type gmType = gibbsMargExpr.GetExpressionType();
                IFieldReferenceExpression ipr = Builder.FieldRefExpr(
                    gibbsMargExpr, gmType, "LastConditional");
                Type iprType = ipr.GetExpressionType();
                distExpr = ConvertInitialiser(distExpr);
                distExpr = ConvertInitialiser(distExpr, iprType, varInfo);
                IStatement is1 = Builder.AssignStmt(ipr, distExpr);
                MethodInfo postUpdate = gmType.GetMethod("PostUpdate");
                IMethodInvokeExpression imie = Builder.MethodInvkExpr();
                imie.Method = Builder.MethodRefExpr(Builder.MethodRef(postUpdate), gibbsMargExpr);
                IStatement is2 = Builder.ExprStatement(imie);
                IBlockStatement ibs = Builder.BlockStmt();
                context.OutputAttributes.Set(ibs, new Initializer() { UserInitialized = true });
                ibs.Statements.Add(is1);
                ibs.Statements.Add(is2);
                context.AddStatementAfterCurrent(ibs);
            }
        }

        /// <summary>
        /// Constructs statements that initialize the elements of an array in a single indexing bracket.
        /// </summary>
        /// <param name="outputLhs">The array to be initialized.</param>
        /// <param name="varInfo">variable info for the array.</param>
        /// <param name="depth">Number of indexing brackets in <paramref name="outputLhs"/></param>
        /// <param name="dimensions">The dimensions of the array.</param>
        /// <param name="elementInit">An expression for the array elements.  May contain references to the array's index variables, as stored in <paramref name="varInfo"/>.</param>
        /// <returns></returns>
        /// <remarks>
        /// Because elementInit may refer to the array's index variables, the generated loop will use these variables, if they are available in <paramref name="varInfo"/>.
        /// </remarks>
        protected IStatement[] FillArray(IExpression outputLhs, VariableInformation varInfo, int depth, IReadOnlyList<IExpression> dimensions, IExpression elementInit)
        {
            if (outputLhs is IVariableDeclarationExpression ivde) outputLhs = Builder.VarRefExpr(ivde.Variable);
            int indexingDepth = depth;
            if (indexingDepth == varInfo.LiteralIndexingDepth - 1)
            {
                // generate a separate statement for each array element
                if (dimensions.Count != 1) throw new ArgumentException("dimensions.Count != 1", nameof(dimensions));
                int sizeAsInt = (int)((ILiteralExpression)dimensions[0]).Value;
                return Util.ArrayInit(sizeAsInt, j =>
                {
                    IExpression index = Builder.LiteralExpr(j);
                    return Builder.AssignStmt(Builder.ArrayIndex(outputLhs, index), elementInit);
                });
            }
            // output a loop that explicitly initializes the array.
            IExpression[] indices = new IExpression[dimensions.Count];
            IVariableDeclaration[] indexVars = new IVariableDeclaration[indices.Length];
            if (varInfo.indexVars.Count > indexingDepth)
            {
                // this array may contain nulls, for indices that were not variables.
                varInfo.indexVars[indexingDepth].CopyTo(indexVars, 0);
            }
            for (int i = 0; i < indices.Length; i++)
            {
                // the same loop index may appear more than once in varInfo.  in this case, must create new indexVars.
                // this requires GetLoopForVariable to match based on name, since the IVariableDeclaration may not be the same object
                if (indexVars[i] == null || Recognizer.GetLoopForVariable(context, indexVars[i]) != null)
                {
                    indexVars[i] = Builder.VarDecl("_ind" + i, typeof(int));
                }
            }
            for (int i = 0; i < indexVars.Length; i++)
            {
                indices[i] = Builder.VarRefExpr(indexVars[i]);
            }
            IExpression elementLhs = Builder.ArrayIndex(outputLhs, indices);
            // if elementInit is MakeCopy, break into allocation followed by SetTo
            IStatement arrayCreate = null;
            if (varInfo.sizes.Count > indexingDepth + 1 &&
                Recognizer.IsStaticGenericMethod(elementInit, new Func<PlaceHolder, PlaceHolder>(ArrayHelper.MakeCopy<PlaceHolder>)))
            {
                Type type = elementLhs.GetExpressionType();
                var arrayCreateExpr = GetArrayCreateExpression(elementLhs, type, varInfo.sizes[indexingDepth + 1]);
                arrayCreate = Builder.AssignStmt(elementLhs, arrayCreateExpr);
                elementInit = ((IMethodInvokeExpression)elementInit).Arguments[0];
                elementInit = Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder, PlaceHolder>(ArrayHelper.SetTo),
                    new Type[] { type }, elementLhs, elementInit);
            }
            IStatement ist = Builder.AssignStmt(elementLhs, elementInit);
            var fs = Builder.NestedForStmt(indexVars, dimensions, out IForStatement innerForStatement);
            if (arrayCreate != null)
                innerForStatement.Body.Statements.Add(arrayCreate);
            innerForStatement.Body.Statements.Add(ist);
            return new[] { fs };
        }

        public static IExpression GetArrayCreateExpression(Type arrayType, IExpression elementInit, VariableInformation varInfo, int depth = 0)
        {
            Type elementType = Util.GetElementType(arrayType, out int rank);
            IExpression initDelegate = MakeArrayInitDelegate(elementInit, varInfo.indexVars[depth]);
            IExpression[] args = new IExpression[rank + 1];
            for (int i = 0; i < rank; i++)
            {
                args[i] = varInfo.sizes[depth][i];
            }
            args[rank] = initDelegate;
            if (rank == 1)
            {
                return Builder.StaticGenericMethod(new Func<int, Converter<int, PlaceHolder>, PlaceHolder[]>(Util.ArrayInit<PlaceHolder>), new Type[] { elementType }, args);
            }
            else if (rank == 2)
            {
                return Builder.StaticGenericMethod(new Func<int, int, Func<int, int, PlaceHolder>, PlaceHolder[,]>(Util.ArrayInit<PlaceHolder>), new Type[] { elementType }, args);
            }
            else
                throw new NotImplementedException("rank > 2");
        }

        protected static IExpression MakeUniform(IExpression expression)
        {
            Type type = expression.GetExpressionType();
            if (IsPointMass(type)) return expression;
            if (Quoter.TryQuoteConstructable(null, out IExpression uniformExpr, "IsUniform", type)) return uniformExpr;
            return Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder>(ArrayHelper.MakeUniform<PlaceHolder>),
                                               new Type[] { type }, expression);
        }

        protected IExpression NewArrayFilled(Type elementType, IList<IExpression> sizes, IExpression elementExpr)
        {
            if (sizes.Count > 2) throw new NotImplementedException("Arrays of more than two dimensions are not yet supported.");
            IExpression[] args = new IExpression[sizes.Count + 1];
            Type[] argTypes = new Type[sizes.Count + 1];
            for (int i = 0; i < sizes.Count; i++)
            {
                args[i] = sizes[i];
                argTypes[i] = typeof(int);
            }
            args[args.Length - 1] = elementExpr;
            argTypes[argTypes.Length - 1] = elementType;
            Type returnType = CodeBuilder.MakeArrayType(elementType, sizes.Count);
            Delegate d;
            if (sizes.Count == 1)
            {
                d = new Func<PlaceHolder[], PlaceHolder, PlaceHolder[]>(ArrayHelper.Fill<PlaceHolder>);
            }
            else
            {
                d = new Func<PlaceHolder[,], PlaceHolder, PlaceHolder[,]>(ArrayHelper.Fill2D<PlaceHolder>);
            }
            return Builder.StaticGenericMethod(d, new Type[] { elementType }, args);
        }

        /// <summary>
        /// The current direction to use when converting variable references.
        /// </summary>
        protected MessageDirection? messageDirection;

        protected IExpression GetMessageExpression(IExpression channelRef, MessageDirection direction)
        {
            messageDirection = direction;
            IExpression expr = ConvertExpression(channelRef);
            messageDirection = null;
            return expr;
        }

        /// <summary>
        /// Converts a variable reference into a message, by replacing it with a reference to
        /// the forward or backward message arrays.  Also retrieves information about the message array.
        /// </summary>
        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            IVariableDeclaration ivd = ivre.Variable.Resolve();
            ChannelToMessageInfo ctmi = context.InputAttributes.Get<ChannelToMessageInfo>(ivd);
            if (ctmi == null) return ivre;
            MessageArrayInformation mai = (messageDirection == MessageDirection.Backwards) ? ctmi.bck : ctmi.fwd;
            if (mai == null) return null;
            IVariableDeclaration ivd2 = mai.decl;
            IExpression vre = Builder.VarRefExpr(ivd2);
            return vre;
        }

        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            MessageDirection? oldDirection = messageDirection;
            messageDirection = MessageDirection.Forwards;
            IList<IExpression> newIndices = ConvertCollection(iaie.Indices);
            messageDirection = oldDirection;
            IExpression newTarget = ConvertExpression(iaie.Target);
            if (ReferenceEquals(newTarget, iaie.Target) &&
                ReferenceEquals(newIndices, iaie.Indices))
                return iaie;
            IArrayIndexerExpression aie = Builder.ArrayIndxrExpr();
            aie.Target = newTarget;
            aie.Indices.AddRange(newIndices);
            Context.InputAttributes.CopyObjectAttributesTo(iaie, context.OutputAttributes, aie);
            return aie;
        }

        /// <summary>
        /// Convert an array type into a distribution type.
        /// </summary>
        /// <param name="arrayType">A scalar, array, multidimensional array, or IList type.</param>
        /// <param name="innermostElementType">Type of innermost array element (may be itself an array, if the array is compound).</param>
        /// <param name="newInnermostElementType">Distribution type to use for the innermost array elements.</param>
        /// <param name="useDistributionArrays">Convert outer arrays to DistributionArrays.</param>
        /// <returns>A distribution type with the same structure as <paramref name="arrayType"/> but whose element type is <paramref name="newInnermostElementType"/>.</returns>
        /// <remarks>
        /// Similar to <see cref="Util.ChangeElementTypeAndRank"/> but converts arrays to DistributionArrays.
        /// </remarks>
        public static Type GetDistributionType(Type arrayType, Type innermostElementType, Type newInnermostElementType, bool useDistributionArrays)
        {
            if (innermostElementType.IsAssignableFrom(arrayType)) return newInnermostElementType;
            if (arrayType.IsArray)
            {
                Type elementType = Util.GetElementType(arrayType, out int rank);
                if (elementType == null) throw new ArgumentException(arrayType + " is not an array type with innermost element type " + innermostElementType);
                Type innerType = GetDistributionType(elementType, innermostElementType, newInnermostElementType, useDistributionArrays);
                if (useDistributionArrays)
                {
                    return Distribution.MakeDistributionArrayType(innerType, rank);
                }
                else
                {
                    return CodeBuilder.MakeArrayType(innerType, rank);
                }
            }
            else
            {
                return typeof(PointMass<>).MakeGenericType(arrayType);
            }
        }

        /// <summary>
        /// Convert an array type into a distribution type, converting a specified number of inner arrays to DistributionArrays.
        /// </summary>
        /// <param name="arrayType"></param>
        /// <param name="innermostElementType"></param>
        /// <param name="newInnermostElementType"></param>
        /// <param name="depth">The current depth from the declaration type.</param>
        /// <param name="useDistributionArraysDepth">The depth at which to start converting to DistributionArrays.  This can be overridden by <paramref name="useFileArrayAtDepth"/></param>
        /// <param name="useFileArrayAtDepth"></param>
        /// <returns></returns>
        public static Type GetDistributionType(Type arrayType, Type innermostElementType, Type newInnermostElementType, int depth, int useDistributionArraysDepth,
                                               Predicate<int> useFileArrayAtDepth)
        {
            if (innermostElementType.IsAssignableFrom(arrayType)) return newInnermostElementType;
            Type elementType = Util.GetElementType(arrayType, out int rank);
            if (elementType == null) throw new ArgumentException(arrayType + " is not an array type.");
            if (arrayType.IsAssignableFrom(Util.MakeArrayType(elementType, rank)))
            {
                if (useFileArrayAtDepth(depth))
                {
                    Type innerType = GetDistributionType(elementType, innermostElementType, newInnermostElementType, depth + 1, depth, useFileArrayAtDepth);
                    return Distribution.MakeDistributionFileArrayType(innerType, rank);
                }
                else
                {
                    Type innerType = GetDistributionType(elementType, innermostElementType, newInnermostElementType, depth + 1, useDistributionArraysDepth, useFileArrayAtDepth);
                    if ((depth >= useDistributionArraysDepth) && (useDistributionArraysDepth >= 0))
                    {
                        return Distribution.MakeDistributionArrayType(innerType, rank);
                    }
                    else
                    {
                        return CodeBuilder.MakeArrayType(innerType, rank);
                    }
                }
            }
            else
            {
                return typeof(PointMass<>).MakeGenericType(arrayType);
            }
        }

        public static Type GetArrayType(Type arrayType, Type innermostElementType, int depth, Predicate<int> useFileArrayAtDepth)
        {
            if (arrayType == innermostElementType) return innermostElementType;
            Type elementType = Util.GetElementType(arrayType, out int rank);
            if (elementType == null) throw new ArgumentException(arrayType + " is not an array type.");
            Type innerType = GetArrayType(elementType, innermostElementType, depth + 1, useFileArrayAtDepth);
            if (useFileArrayAtDepth(depth))
            {
                return MakeFileArrayType(innerType, rank);
            }
            else
            {
                return CodeBuilder.MakeArrayType(innerType, rank);
            }
        }

        public static Type MakeFileArrayType(Type elementType, int rank)
        {
            if (rank < 1) throw new ArgumentException("rank (" + rank + ") < 1");
            if (rank == 1)
            {
                return typeof(FileArray<>).MakeGenericType(elementType);
            }
            else
            {
                throw new ArgumentException("FileArray rank > 1 not yet implemented");
            }
        }

        public Type GetMessageType(ChannelInfo ci, Type marginalType)
        {
            VariableInformation vi = (ci.decl != null)
                ? VariableInformation.GetVariableInformation(context, ci.decl)
                : ci.varInfo;
            // leading array is [] up to distArraysDepth, then distribution arrays
            int distArraysDepth = System.Math.Max((algorithm is GibbsSampling) ? 0 : vi.DistArrayDepth, vi.LiteralIndexingDepth);
            bool useFileArrayAtDepth(int depth) => vi.IsPartitionedAtDepth(context, depth);
            for (int depth = 0; depth < distArraysDepth; depth++)
            {
                if (useFileArrayAtDepth(depth))
                {
                    // in order to use DistributionFileArray, the element type must be a distribution type
                    distArraysDepth = depth + 1;
                    break;
                }
            }
            Type domainType = Distribution.GetDomainType(marginalType);
            try
            {
                return GetDistributionType(ci.channelType, domainType, marginalType, 0, distArraysDepth, useFileArrayAtDepth);
            }
            catch (Exception ex)
            {
                Error(ex.Message);
                return ci.channelType;
            }
        }

        private class ChannelToMessageInfo : ICompilerAttribute
        {
            /// <summary>
            /// Declaration of the forward message array/variable
            /// </summary>
            internal MessageArrayInformation fwd;

            /// <summary>
            /// Declaration of the backward message array/variable
            /// </summary>
            internal MessageArrayInformation bck;

            public override string ToString()
            {
                return $"ChannelToMessageInfo fwd = {fwd}, bck = {bck}";
            }
        }

        private class ObservedVariableMessages : ICompilerAttribute
        {
            internal readonly ChannelToMessageInfo ctmi;

            internal ObservedVariableMessages(ChannelToMessageInfo ctmi)
            {
                this.ctmi = ctmi;
            }

            public override string ToString()
            {
                return $"ObservedVariableMessages {ctmi}";
            }
        }
    }

    /// <summary>
    /// Information about either the forwards or backwards message array
    /// </summary>
    internal class MessageArrayInformation : ICompilerAttribute
    {
        internal ChannelInfo ci;
        /// <summary>
        /// Declaration of the message array variable
        /// </summary>
        internal IVariableDeclaration decl;
        internal IExpression marginalPrototypeExpression = null;

        internal Type messageArrayType
        {
            get { return decl.VariableType.DotNetType; }
        }

        /// <summary>
        /// Indicates whether the message is a distribution over the channel.  
        /// (This can only be false when the algorithm overrides GetMessagePrototype.)
        /// </summary>
        internal bool isDistribution = true;

        /// <summary>
        /// Number of times this variable is used as an argument to a message operator.
        /// Only used by AccumulationTransform.
        /// </summary>
        internal int useCount;

        /// <summary>
        /// Set by LoopCuttingTransform, used by AccumulationTransform
        /// </summary>
        internal LoopVarInfo loopVarInfo;

        public override string ToString()
        {
            return $"MessageArrayInformation({decl},useCount={useCount},{ci},{marginalPrototypeExpression})";
        }
    }

    /// <summary>
    /// Enumeration for labelling forward and backward messages.
    /// </summary>
    public enum MessageDirection
    {
        Backwards,
        Forwards
    };

    /// <summary>
    /// Attached to a statement to indicate that it may appear in a while loop, i.e. it updates a message.  A statement without this attribute cannot appear in a while loop.
    /// </summary>
    internal class OperatorStatement : ICompilerAttribute
    {
    }

    /// <summary>
    /// Attribute used to mark methods which contain operator statements (i.e. statements which are to be processed by the scheduler).
    /// </summary>
    internal class OperatorMethod : ICompilerAttribute
    {
    }

    /// <summary>
    /// Attribute used to mark methods which assert deterministic constraints
    /// </summary>
    internal class DeterministicConstraint : ICompilerAttribute
    {
    }

    /// <summary>
    /// Attached to statements to indicate that they will be overwritten by later statements, i.e. this is not the final value of the variable.
    /// </summary>
    internal class Initializer : ICompilerAttribute
    {
        /// <summary>
        /// If true, this statement is initialized by the user.
        /// </summary>
        public bool UserInitialized;

        public override string ToString()
        {
            return "Initializer" + (UserInitialized ? "(UserInitialized)" : "");
        }
    }


    /// <summary>
    /// Attribute used to mark variables that appear as arguments to Infer()
    /// </summary>
    internal class IsInferred : ICompilerAttribute
    {
    }

    /// <summary>
    /// When applied to a variable, indicates that the forward message is always a point mass (but not necessarily constant).
    /// </summary>
    internal class ForwardPointMass : ICompilerAttribute
    {
    }

    /// <summary>
    /// When applied to a method invoke expression, indicates that the execution of this
    /// expression depends on the iteration counter
    /// </summary>
    internal class DependsOnIteration : ICompilerAttribute
    {
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "DependsOnIteration";
        }
    }

    /// <summary>
    /// When attached to a message declaration, indicates that the message has no initializer statement, only update statements.
    /// </summary>
    internal class DoesNotHaveInitializer : ICompilerAttribute
    {
    }

    /// <summary>
    /// A type which can be used as a placeholder in a generic method reference.  It will be replaced
    /// by the dynamically specified type argument provided separately.
    /// </summary>
    internal interface PlaceHolder : IDistribution<object>, SettableToProduct<PlaceHolder>, SettableTo<PlaceHolder>
    {
    }

    internal interface PlaceHolder2 : IDistribution<object>, SettableToProduct<PlaceHolder>
    {
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}