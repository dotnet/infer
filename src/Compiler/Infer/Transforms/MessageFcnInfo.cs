// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    using System;
    using System.Collections.Generic;
    using System.Reflection;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Compiler.Attributes;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Compiler;
    using System.Linq;

    /// <summary>
    /// Records information about the operator method 
    /// </summary>
    internal class MessageFcnInfo : ICompilerAttribute
    {
        public readonly MethodInfo Method;

        /// <summary>
        /// The name of the factor edge being computed.
        /// </summary>
        public string TargetParameter;

        public string Suffix;

        /// <summary>
        /// True if the method has a result parameter
        /// </summary>
        public bool PassResult
        {
            get { return (ResultParameterIndex != -1); }
        }

        /// <summary>
        /// True if the method has a resultIndex parameter
        /// </summary>
        public bool PassResultIndex
        {
            get { return (ResultIndexParameterIndex != -1); }
        }

        /// <summary>
        /// True if the message function performs sampling.
        /// </summary>
        public readonly bool IsStochastic;

        public IReadOnlyList<FactorEdge> Dependencies, Requirements, Triggers;

        /// <summary>
        /// Includes all parameters, including buffers, result, and resultIndex.
        /// </summary>
        public readonly IReadOnlyDictionary<string, FactorEdge> factorEdgeOfParameter;

        public DependencyInformation DependencyInfo;

        /// <summary>
        /// Indicates whether to skip the message if all arguments are uniform.
        /// </summary>
        /// <remarks>
        /// Only meaningful if Requirements.Count == 0.
        /// </remarks>
        public bool SkipIfAllUniform;

        /// <summary>
        /// Indicates whether the method has AllTriggersAttribute.
        /// </summary>
        public readonly bool AllTriggers;

        /// <summary>
        /// Indicates whether the method has NoTriggersAttribute.
        /// </summary>
        public bool NoTriggers;

        /// <summary>
        /// If the message is not supported, provides the explanation.
        /// </summary>
        /// <remarks>
        /// Will be null if the message is supported.
        /// </remarks>
        public readonly string NotSupportedMessage;

        /// <summary>
        /// True if the function returns the product of all arguments.
        /// </summary>
        public readonly bool IsMultiplyAll;

        /// <summary>
        /// Index of the IsReturned parameter, or -1 if none.
        /// </summary>
        public readonly int ReturnedParameterIndex = -1;

        /// <summary>
        /// Index of the IsReturnedInAllElements parameter, or -1 if none.
        /// </summary>
        public readonly int ReturnedInAllElementsParameterIndex = -1;

        /// <summary>
        /// Index of the result parameter, or -1 if none.
        /// </summary>
        public readonly int ResultParameterIndex = -1;

        /// <summary>
        /// Index of the resultIndex parameter, or -1 if none.
        /// </summary>
        public readonly int ResultIndexParameterIndex = -1;

        /// <summary>
        /// True if the parameter has IndexedAttribute.  Array may be null if no parameters have the attribute.
        /// </summary>
        public readonly bool[] IsIndexedParameter;

        public MessageFcnInfo(MethodInfo method, IReadOnlyCollection<ParameterInfo> parameters, IReadOnlyDictionary<string, FactorEdge> factorEdgeOfParameter = null)
        {
            this.Method = method;
            this.factorEdgeOfParameter = factorEdgeOfParameter;
            int parameterIndex = 0;
            foreach (ParameterInfo parameter in parameters)
            {
                if (parameter.Name == "result" || parameter.Name == "Result")
                {
                    if (ResultParameterIndex != -1) throw new Exception("Only one parameter of a method can be the result");
                    ResultParameterIndex = parameterIndex;
                }
                else if (parameter.Name == "resultIndex" || parameter.Name == "ResultIndex")
                {
                    if (ResultIndexParameterIndex != -1) throw new Exception("Only one parameter of a method can be the resultIndex");
                    ResultIndexParameterIndex = parameterIndex;
                }
                else if (parameter.IsDefined(typeof(IsReturnedAttribute), false))
                {
                    if (ReturnedParameterIndex != -1) throw new Exception("IsReturnedAttribute can only be attached to one parameter of a method");
                    // MessageTransform must not replace this method if it has a NoInit attribute.
                    if (!parameter.IsDefined(typeof(NoInitAttribute), false) && !parameter.IsDefined(typeof(DiodeAttribute), false))
                        ReturnedParameterIndex = parameterIndex;
                }
                else if (parameter.IsDefined(typeof(IsReturnedInEveryElementAttribute), false))
                {
                    if (ReturnedInAllElementsParameterIndex != -1) throw new Exception("IsReturnedInAllElementsAttribute can only be attached to one parameter of a method");
                    ReturnedInAllElementsParameterIndex = parameterIndex;
                }
                if (parameter.IsDefined(typeof(IndexedAttribute), false))
                {
                    if (IsIndexedParameter == null) IsIndexedParameter = new bool[parameters.Count];
                    IsIndexedParameter[parameterIndex] = true;
                }
                parameterIndex++;
            }
            IsMultiplyAll = Method.IsDefined(typeof(MultiplyAllAttribute), false);
            IsStochastic = Method.IsDefined(typeof(Stochastic), false);
            object[] attrs = Method.GetCustomAttributes(typeof(NotSupportedAttribute), false);
            if (attrs.Length > 0)
            {
                NotSupportedMessage = ((NotSupportedAttribute)attrs[0]).Message;
            }
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public MessageFcnInfo(MessageFcnInfo that)
        {
            Method = that.Method;
            TargetParameter = that.TargetParameter;
            Suffix = that.Suffix;
            IsStochastic = that.IsStochastic;
            Dependencies = that.Dependencies;
            Requirements = that.Requirements;
            Triggers = that.Triggers;
            SkipIfAllUniform = that.SkipIfAllUniform;
            NotSupportedMessage = that.NotSupportedMessage;
            factorEdgeOfParameter = that.factorEdgeOfParameter;
            DependencyInfo = that.DependencyInfo;
            IsMultiplyAll = that.IsMultiplyAll;
            ReturnedParameterIndex = that.ReturnedParameterIndex;
            ReturnedInAllElementsParameterIndex = that.ReturnedInAllElementsParameterIndex;
            ResultParameterIndex = that.ResultParameterIndex;
            ResultIndexParameterIndex = that.ResultIndexParameterIndex;
        }

        /// <summary>
        /// Get the field names and types of message function parameters.
        /// </summary>
        /// <returns>A list of (field name, parameter type) in the same order as the parameters of the function.</returns>
        /// <remarks>
        /// The field names are not necessarily the same as the parameter names of the method.
        /// The field names are a subset of the entries in the FactorInfo of the factor.
        /// Additionally, a parameter may have the special field name "result" or "resultIndex".
        /// </remarks>
        public List<KeyValuePair<string, Type>> GetParameterTypes()
        {
            List<KeyValuePair<string, Type>> result = new List<KeyValuePair<string, Type>>();
            ParameterInfo[] parameters = Method.GetParameters();
            for (int i = 0; i < parameters.Length; i++)
            {
                ParameterInfo parameter = parameters[i];
                if (factorEdgeOfParameter.ContainsKey(parameter.Name))
                {
                    result.Add(new KeyValuePair<string, Type>(factorEdgeOfParameter[parameter.Name].ToString(), parameter.ParameterType));
                }
                else
                {
                    result.Add(new KeyValuePair<string, Type>(FactorManager.Uncapitalize(parameter.Name), parameter.ParameterType));
                }
            }
            return result;
        }

        public override string ToString()
        {
            System.Text.StringBuilder s = new System.Text.StringBuilder(StringUtil.MethodSignatureToString(Method));
            s.AppendLine();
            if (TargetParameter != null)
            {
                s.Append("  target: ");
                s.Append(TargetParameter);
                s.AppendLine();
            }
            if (Suffix != null)
            {
                s.Append("  suffix: ");
                s.AppendLine(Suffix);
            }
            s.AppendLine(DependencyInfo.ToString());
            return s.ToString();
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}