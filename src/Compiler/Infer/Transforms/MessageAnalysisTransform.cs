// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using NodeIndex = System.Int32;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    internal class MessageAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "MessageAnalysisTransform"; }
        }

        protected IAlgorithm algorithm;
        protected FactorManager factorManager;

        /// <summary>
        /// Collects all factors (method invokes) in the program
        /// </summary>
        protected List<IExpression> factorExprs = new List<IExpression>();

        /// <summary>
        /// The variable for each type factory
        /// </summary>
        public Dictionary<Type, IVariableDeclaration> factoryVars = new Dictionary<Type, IVariableDeclaration>();

        /// <summary>
        /// Gives the expression used to initialize a factory
        /// </summary>
        public Dictionary<IVariableDeclaration, IExpression> factoryInitExprs = new Dictionary<IVariableDeclaration, IExpression>();

        /// <summary>
        /// The forward message for a given channel
        /// </summary>
        public Dictionary<IVariableDeclaration, IVariableDeclaration> fwdMessageVars = new Dictionary<IVariableDeclaration, IVariableDeclaration>();

        /// <summary>
        /// The backward message for a given channel
        /// </summary>
        public Dictionary<IVariableDeclaration, IVariableDeclaration> bckMessageVars = new Dictionary<IVariableDeclaration, IVariableDeclaration>();

        /// <summary>
        /// Gives the expression used to initialize a message
        /// </summary>
        public Dictionary<KeyValuePair<IVariableDeclaration, IExpression>, IExpression> messageInitExprs =
            new Dictionary<KeyValuePair<IVariableDeclaration, IExpression>, IExpression>();

        public MessageAnalysisTransform(IAlgorithm algorithm, FactorManager factorManager)
        {
            this.algorithm = algorithm;
            this.factorManager = factorManager;
        }

        protected override IStatement ConvertExpressionStatement(IExpressionStatement ies)
        {
            if (ies.Expression is IAssignExpression iae)
            {
                if (iae.Expression is IMethodInvokeExpression)
                    factorExprs.Add(iae);
            }
            else if (ies.Expression is IMethodInvokeExpression imie)
            {
                if (CodeRecognizer.IsInfer(imie)) return ies;
                factorExprs.Add(ies.Expression);
            }
            return ies;
        }

        protected override IMethodDeclaration ConvertMethod(IMethodDeclaration imd)
        {
            base.ConvertMethod(imd);
            PostProcess();
            return imd;
        }

        private void PostProcess()
        {
            // create a dependency graph between MethodInvokes
            Dictionary<IVariableDeclaration, List<int>> mutationsOfVariable = new Dictionary<IVariableDeclaration, List<NodeIndex>>();
            IndexedGraph g = new IndexedGraph(factorExprs.Count);
            foreach (NodeIndex node in g.Nodes)
            {
                IExpression factor = factorExprs[node];
                NodeInfo info = GetNodeInfo(factor);
                for (int i = 0; i < info.arguments.Count; i++)
                {
                    if (info.isReturnOrOut[i])
                    {
                        // this is a mutation.  add to mutationsOfVariable.
                        IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(info.arguments[i]);
                        if (ivd != null && CodeRecognizer.IsStochastic(context, ivd))
                        {
                            List<int> nodes;
                            if (!mutationsOfVariable.TryGetValue(ivd, out nodes))
                            {
                                nodes = new List<NodeIndex>();
                                mutationsOfVariable[ivd] = nodes;
                            }
                            nodes.Add(node);
                        }
                    }
                }
            }
            foreach (NodeIndex node in g.Nodes)
            {
                IExpression factor = factorExprs[node];
                NodeInfo info = GetNodeInfo(factor);
                for (int i = 0; i < info.arguments.Count; i++)
                {
                    if (!info.isReturnOrOut[i])
                    {
                        // not a mutation.  create a dependency on all mutations.
                        IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(info.arguments[i]);
                        if (ivd != null && CodeRecognizer.IsStochastic(context, ivd))
                        {
                            foreach (NodeIndex source in mutationsOfVariable[ivd])
                            {
                                g.AddEdge(source, node);
                            }
                        }
                    }
                }
            }
            List<NodeIndex> topo_nodes = new List<NodeIndex>();
            DepthFirstSearch<NodeIndex> dfs = new DepthFirstSearch<NodeIndex>(g.SourcesOf, g);
            dfs.FinishNode += delegate(NodeIndex node)
                {
                    IExpression factor = factorExprs[node];
                    ProcessFactor(factor, MessageDirection.Forwards);
                    topo_nodes.Add(node);
                };
            // process nodes forward
            dfs.SearchFrom(g.Nodes);
            // process nodes backward
            for (int i = topo_nodes.Count - 1; i >= 0; i--)
            {
                NodeIndex node = topo_nodes[i];
                IExpression factor = factorExprs[node];
                ProcessFactor(factor, MessageDirection.Backwards);
            }
        }

        /// <summary>
        /// Describes an instance of a factor in the program
        /// </summary>
        private class NodeInfo
        {
            public FactorManager.FactorInfo info;
            public IMethodInvokeExpression imie;
            public List<bool> isReturnOrOut = new List<bool>();
            public List<IExpression> arguments = new List<IExpression>();

            public NodeInfo(IMethodInvokeExpression imie)
            {
                this.imie = imie;
            }
        }

        private NodeInfo GetNodeInfo(IExpression factor)
        {
            IExpression target = null;
            IMethodInvokeExpression imie;
            if (factor is IAssignExpression iae)
            {
                target = iae.Target;
                if (target is IVariableDeclarationExpression)
                {
                    IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
                    target = Builder.VarRefExpr(ivd);
                }
                imie = (IMethodInvokeExpression)iae.Expression;
            }
            else imie = (IMethodInvokeExpression)factor;
            NodeInfo info = new NodeInfo(imie)
            {
                info = CodeRecognizer.GetFactorInfo(context, imie)
            };
            if (target != null)
            {
                info.isReturnOrOut.Add(true);
                info.arguments.Add(target);
            }
            if (!info.info.Method.IsStatic)
            {
                info.isReturnOrOut.Add(false);
                info.arguments.Add(imie.Method.Target);
            }
            foreach (IExpression arg in imie.Arguments)
            {
                bool isOut = (arg is IAddressOutExpression);
                info.isReturnOrOut.Add(isOut);
                info.arguments.Add(isOut ? ((IAddressOutExpression)arg).Expression : arg);
            }
            return info;
        }

        private MessageFcnInfo GetMessageFcnInfo(FactorManager.FactorInfo info, string methodSuffix, string targetParameter, IReadOnlyDictionary<string, Type> parameterTypes)
        {
            var factoryParameters = new List<KeyValuePair<string, Type>>();
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, methodSuffix, targetParameter, parameterTypes);
            ParameterInfo[] parameters = fcninfo.Method.GetParameters();
            foreach (ParameterInfo parameter in parameters)
            {
                if (parameterTypes.ContainsKey(parameter.Name)) continue;
                if (IsFactoryType(parameter.ParameterType))
                {
                    Type[] typeArgs = parameter.ParameterType.GetGenericArguments();
                    Type itemType = typeArgs[0];
                    Type arrayType = Distribution.MakeDistributionArrayType(itemType, 1);
                    Type factoryType = typeof(IArrayFactory<,>).MakeGenericType(itemType, arrayType);
                    factoryParameters.Add(new KeyValuePair<string,Type>(parameter.Name, factoryType));
                }
            }
            if (factoryParameters.Count > 0)
            {
                return GetMessageFcnInfo(info, methodSuffix, targetParameter, Append(parameterTypes, factoryParameters));
            }
            else
            {
                return fcninfo;
            }
        }

        private static Dictionary<TKey, TValue> Append<TKey, TValue>(IReadOnlyDictionary<TKey, TValue> dict, IEnumerable<KeyValuePair<TKey, TValue>> keyValuePairs)
        {
            var result = new Dictionary<TKey, TValue>();
            result.AddRange(dict);
            result.AddRange(keyValuePairs);
            return result;
        }

        private IVariableDeclaration GetFactoryVariable(Type type)
        {
            IVariableDeclaration ivd;
            if (factoryVars.TryGetValue(type, out ivd)) return ivd;
            string name = VariableInformation.GenerateName(context, "_factory");
            ivd = Builder.VarDecl(name, type);
            factoryVars[type] = ivd;
            Type[] typeArgs = type.GetGenericArguments();
            Type arrayType = typeArgs[1];
            IExpression factoryExpr = Builder.NewObject(arrayType, Builder.LiteralExpr(0));
            factoryInitExprs[ivd] = factoryExpr;
            return ivd;
        }

        /// <summary>
        /// Returns true if type is IArrayFactory
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        private bool IsFactoryType(Type type)
        {
            return type.Name == typeof (IArrayFactory<,>).Name;
        }

        // Determine the message type of target from the message type of the factor arguments
        protected void ProcessFactor(IExpression factor, MessageDirection direction)
        {
            NodeInfo info = GetNodeInfo(factor);
            // fill in argumentTypes
            Dictionary<string, Type> argumentTypes = new Dictionary<string, Type>();
            Dictionary<string, IExpression> arguments = new Dictionary<string, IExpression>();
            for (int i = 0; i < info.info.ParameterNames.Count; i++)
            {
                string parameterName = info.info.ParameterNames[i];
                // Create message info. 'isForward' says whether the message
                // out is in the forward or backward direction
                bool isChild = info.isReturnOrOut[i];
                IExpression arg = info.arguments[i];
                bool isConstant = !CodeRecognizer.IsStochastic(context, arg);
                if (isConstant)
                {
                    arguments[parameterName] = arg;
                    Type inwardType = arg.GetExpressionType();
                    argumentTypes[parameterName] = inwardType;
                }
                else if (!isChild)
                {
                    IExpression msgExpr = GetMessageExpression(arg, fwdMessageVars);
                    if (msgExpr == null) return;
                    arguments[parameterName] = msgExpr;
                    Type inwardType = msgExpr.GetExpressionType();
                    if (inwardType == null)
                    {
                        Error("inferred an incorrect message type for " + arg);
                        return;
                    }
                    argumentTypes[parameterName] = inwardType;
                }
                else if (direction == MessageDirection.Backwards)
                {
                    IExpression msgExpr = GetMessageExpression(arg, bckMessageVars);
                    if (msgExpr == null)
                    {
                        //Console.WriteLine("creating backward message for "+arg);
                        CreateBackwardMessageFromForward(arg, null);
                        msgExpr = GetMessageExpression(arg, bckMessageVars);
                        if (msgExpr == null) return;
                    }
                    arguments[parameterName] = msgExpr;
                    Type inwardType = msgExpr.GetExpressionType();
                    if (inwardType == null)
                    {
                        Error("inferred an incorrect message type for " + arg);
                        return;
                    }
                    argumentTypes[parameterName] = inwardType;
                }
            }
            IAlgorithm alg = algorithm;
            Algorithm algAttr = context.InputAttributes.Get<Algorithm>(info.imie);
            if (algAttr != null) alg = algAttr.algorithm;
            List<ICompilerAttribute> factorAttributes = context.InputAttributes.GetAll<ICompilerAttribute>(info.imie);
            string methodSuffix = alg.GetOperatorMethodSuffix(factorAttributes);
            // infer types of children
            for (int i = 0; i < info.info.ParameterNames.Count; i++)
            {
                string parameterName = info.info.ParameterNames[i];
                bool isChild = info.isReturnOrOut[i];
                if (isChild != (direction == MessageDirection.Forwards)) continue;
                IExpression target = info.arguments[i];
                bool isConstant = !CodeRecognizer.IsStochastic(context, target);
                if (isConstant) continue;
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
                if (ivd == null) continue;
                Type targetType = null;
                MessageFcnInfo fcninfo = null;
                if (direction == MessageDirection.Forwards)
                {
                    try
                    {
                        fcninfo = GetMessageFcnInfo(info.info, "Init", parameterName, argumentTypes);
                    }
                    catch (Exception)
                    {
                        try
                        {
                            fcninfo = GetMessageFcnInfo(info.info, methodSuffix + "Init", parameterName, argumentTypes);
                        }
                        catch (Exception ex)
                        {
                            //Error("could not determine message type of "+ivd.Name, ex);
                            try
                            {
                                fcninfo = GetMessageFcnInfo(info.info, methodSuffix, parameterName, argumentTypes);
                                if (fcninfo.PassResult)
                                    throw new MissingMethodException(StringUtil.MethodFullNameToString(fcninfo.Method) +
                                                                     " is not suitable for initialization since it takes a result parameter.  Please provide a separate Init method.");
                                if (fcninfo.PassResultIndex)
                                    throw new MissingMethodException(StringUtil.MethodFullNameToString(fcninfo.Method) +
                                                                     " is not suitable for initialization since it takes a resultIndex parameter.  Please provide a separate Init method.");
                            }
                            catch (Exception ex2)
                            {
                                if (direction == MessageDirection.Forwards)
                                {
                                    Error("could not determine " + direction + " message type of " + ivd.Name + ": " + ex.Message, ex2);
                                    continue;
                                }
                                fcninfo = null;
                            }
                        }
                    }
                    if (fcninfo != null)
                    {
                        targetType = fcninfo.Method.ReturnType;
                        if (targetType.IsGenericParameter)
                        {
                            if (direction == MessageDirection.Forwards)
                            {
                                Error("could not determine " + direction + " message type of " + ivd.Name + " in " + StringUtil.MethodFullNameToString(fcninfo.Method));
                                continue;
                            }
                            fcninfo = null;
                        }
                    }
                    if (fcninfo != null)
                    {
                        VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
                        try
                        {
                            targetType = MessageTransform.GetDistributionType(ivd.VariableType.DotNetType, target.GetExpressionType(), targetType, true);
                        }
                        catch (Exception ex)
                        {
                            if (direction == MessageDirection.Forwards)
                            {
                                Error("could not determine " + direction + " message type of " + ivd.Name, ex);
                                continue;
                            }
                            fcninfo = null;
                        }
                    }
                }
                Dictionary<IVariableDeclaration, IVariableDeclaration> messageVars = (direction == MessageDirection.Forwards) ? fwdMessageVars : bckMessageVars;
                if (fcninfo != null)
                {
                    string name = ivd.Name + (direction == MessageDirection.Forwards ? "_F" : "_B");
                    IVariableDeclaration msgVar;
                    if (!messageVars.TryGetValue(ivd, out msgVar))
                    {
                        msgVar = Builder.VarDecl(name, targetType);
                    }
                    if (true)
                    {
                        // construct the init expression
                        List<IExpression> args = new List<IExpression>();
                        ParameterInfo[] parameters = fcninfo.Method.GetParameters();
                        foreach (ParameterInfo parameter in parameters)
                        {
                            string argName = parameter.Name;
                            if (IsFactoryType(parameter.ParameterType))
                            {
                                IVariableDeclaration factoryVar = GetFactoryVariable(parameter.ParameterType);
                                args.Add(Builder.VarRefExpr(factoryVar));
                            }
                            else
                            {
                                FactorEdge factorEdge = fcninfo.factorEdgeOfParameter[parameter.Name];
                                string factorParameterName = factorEdge.ParameterName;
                                bool isOutgoingMessage = factorEdge.IsOutgoingMessage;
                                if (!arguments.ContainsKey(factorParameterName))
                                {
                                    if (direction == MessageDirection.Forwards)
                                    {
                                        Error(StringUtil.MethodFullNameToString(fcninfo.Method) + " is not suitable for initialization since it requires '" + parameter.Name +
                                              "'.  Please provide a separate Init method.");
                                    }
                                    fcninfo = null;
                                    break;
                                }
                                IExpression arg = arguments[factorParameterName];
                                args.Add(arg);
                            }
                        }
                        if (fcninfo != null)
                        {
                            IMethodInvokeExpression imie = Builder.StaticMethod(fcninfo.Method, args.ToArray());
                            //IExpression initExpr = MessageTransform.GetDistributionArrayCreateExpression(ivd.VariableType.DotNetType, target.GetExpressionType(), imie, vi);
                            IExpression initExpr = imie;
                            KeyValuePair<IVariableDeclaration, IExpression> key = new KeyValuePair<IVariableDeclaration, IExpression>(msgVar, factor);
                            messageInitExprs[key] = initExpr;
                        }
                    }
                    if (fcninfo != null) messageVars[ivd] = msgVar;
                }
                if (fcninfo == null)
                {
                    if (direction == MessageDirection.Forwards) continue;
                    //Console.WriteLine("creating backward message for "+target);
                    CreateBackwardMessageFromForward(target, factor);
                }
                IExpression msgExpr = GetMessageExpression(target, messageVars);
                arguments[parameterName] = msgExpr;
                Type inwardType = msgExpr.GetExpressionType();
                argumentTypes[parameterName] = inwardType;
            }
        }

        private void CreateBackwardMessageFromForward(IExpression expr, IExpression factor)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
            IVariableDeclaration forwardVar;
            if (!fwdMessageVars.TryGetValue(ivd, out forwardVar)) return;
            string name = ivd.Name + "_B";
            IVariableDeclaration backwardVar = Builder.VarDecl(name, forwardVar.VariableType);
            bckMessageVars[ivd] = backwardVar;
            KeyValuePair<IVariableDeclaration, IExpression> keyB = new KeyValuePair<IVariableDeclaration, IExpression>(backwardVar, factor);
            messageInitExprs[keyB] = MakeUniform(GetMessageExpression(expr, fwdMessageVars));
        }

        protected IExpression MakeUniform(IExpression expression)
        {
            Type type = expression.GetExpressionType();
            if (type.Name == typeof (GibbsMarginal<,>).Name) return expression;
            return Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder>(ArrayHelper.MakeUniform<PlaceHolder>),
                                               new Type[] {type}, expression);
        }

        private IExpression GetMessageExpression(IExpression expr, IDictionary<IVariableDeclaration, IVariableDeclaration> msgVars)
        {
            if (expr is IVariableReferenceExpression ivre)
            {
                IVariableDeclaration ivd = ivre.Variable.Resolve();
                if (msgVars.TryGetValue(ivd, out IVariableDeclaration msgVar)) return Builder.VarRefExpr(msgVar);
                else return null;
            }
            else if (expr is IArrayIndexerExpression iaie)
            {
                IExpression targetMsg = GetMessageExpression(iaie.Target, msgVars);
                if (targetMsg == null) return null;
                else return Builder.ArrayIndex(targetMsg, iaie.Indices);
            }
            else
            {
                throw new ArgumentException("Unrecognized method argument expression");
            }
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}