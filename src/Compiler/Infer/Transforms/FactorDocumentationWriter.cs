// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Reflection;
    using System.Text;
    using System.Xml;

    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Automatically generates XML documentation for message operators.
    /// </summary>
    internal static class FactorDocumentationWriter
    {
        /// <summary>
        /// Writes the XML documentation for all message operators in Microsoft.ML.Probabilistic to a given file.
        /// </summary>
        /// <param name="fileName">The name of the file to write the documentation to.</param>
        public static void WriteFactorDocumentation(string fileName)
        {
            using (var writer = new XmlTextWriter(fileName, Encoding.UTF8) { Formatting = Formatting.Indented })
            {
                writer.WriteStartDocument();
                writer.WriteStartElement("factor_docs");
                WriteFactorDocumentation(writer);
                writer.WriteEndElement();
                writer.WriteEndDocument();
            }
        }

        /// <summary>
        /// Writes the XML documentation for all message operators in Microsoft.ML.Probabilistic using a given XML writer.
        /// </summary>
        /// <param name="writer">The XML writer.</param>
        public static void WriteFactorDocumentation(XmlWriter writer)
        {
            var typeToFactors = new Dictionary<Type, HashSet<FactorInfoWrapper>>();
            var typeToMessageFunctions = new Dictionary<Type, HashSet<MessageFunctionInfoWrapper>>();

            foreach (FactorManager.FactorInfo factorInfo in FactorManager.GetFactorInfos())
            {
                foreach (MessageFcnInfo messageFunctionInfo in factorInfo.GetMessageFcnInfos())
                {
                    Type declaringType = messageFunctionInfo.Method.DeclaringType;
                    if (!typeToFactors.ContainsKey(declaringType))
                    {
                        typeToFactors.Add(declaringType, new HashSet<FactorInfoWrapper>());
                        typeToMessageFunctions.Add(declaringType, new HashSet<MessageFunctionInfoWrapper>());
                    }

                    typeToFactors[declaringType].Add(new FactorInfoWrapper(factorInfo));
                    // TODO: sometimes the same message operator is used by more than one factor.
                    // TODO: Can we pick any factor to generate the doc?
                    typeToMessageFunctions[declaringType].Add(new MessageFunctionInfoWrapper(messageFunctionInfo, factorInfo));
                }
            }

            foreach (Type type in typeToFactors.Keys)
            {
                if (type.Assembly.GetCustomAttributes(typeof(HasMessageFunctionsAttribute), true).Length > 0)
                {
                    WriteTypeDocumentation(writer, type, typeToFactors[type], typeToMessageFunctions[type]);
                }
            }
        }

        /// <summary>
        /// Writes the XML documentation for a given type containing message operators using a given XML writer.
        /// </summary>
        /// <param name="writer">The XML writer.</param>
        /// <param name="type">The type containing the message operators.</param>
        /// <param name="factors">The list of factor <paramref name="type"/> provides message operators for.</param>
        /// <param name="messageFunctions">The list of message operators provided by <paramref name="type"/>.</param>
        private static void WriteTypeDocumentation(
            XmlWriter writer,
            Type type,
            IEnumerable<FactorInfoWrapper> factors,
            IEnumerable<MessageFunctionInfoWrapper> messageFunctions)
        {
            string className = QuoteCodeElementName(StringUtil.TypeToString(type));

            writer.WriteStartElement("message_op_class");
            writer.WriteAttributeString("name", className);

            writer.WriteStartElement("doc");
            writer.WriteStartElement("summary");
            writer.WriteString("Provides outgoing messages for ");
            if (factors.Count() == 1)
            {
                WriteMethodReference(writer, factors.Single().FactorInfo.Method);
            }
            else
            {
                writer.WriteString("the following factors:");
                writer.WriteStartElement("list");
                writer.WriteAttributeString("type", "bullet");
                foreach (FactorInfoWrapper factorInfoWrapper in factors)
                {
                    writer.WriteStartElement("item");
                    writer.WriteStartElement("description");
                    WriteMethodReference(writer, factorInfoWrapper.FactorInfo.Method);
                    writer.WriteEndElement();
                    writer.WriteEndElement();
                }

                writer.WriteEndElement();
            }

            writer.WriteString(", given random arguments to the function.");
            writer.WriteEndElement();
            writer.WriteEndElement();

            foreach (MessageFunctionInfoWrapper messageFunctionInfoWrapper in messageFunctions)
            {
                WriteMessageFunctionDocumentation(writer, messageFunctionInfoWrapper.FactorInfo, messageFunctionInfoWrapper.MessageFunctionInfo);
            }

            writer.WriteEndElement();
        }

        /// <summary>
        /// Writes a method reference in XML documentation format.
        /// </summary>
        /// <param name="writer">The writer.</param>
        /// <param name="method">The method to reference.</param>
        private static void WriteMethodReference(XmlWriter writer, MethodInfo method)
        {
            if (method.IsGenericMethod)
            {
                method = method.GetGenericMethodDefinition();
            }
            
            string methodName = QuoteCodeElementName(
                StringUtil.MethodSignatureToString(method, useFullName: true, omitParameterNames: true));

            // some changes to prevent System.Math and Microsoft.ML.Probabilistic.Math conflict
            if(methodName.StartsWith("Math."))
            {
                methodName = "System." + methodName;
            }
            writer.WriteElementAttributeString("see", "cref", methodName);
        }

        /// <summary>
        /// Writes the XML documentation for a given message operator using a given XML writer.
        /// </summary>
        /// <param name="writer">The XML writer.</param>
        /// <param name="factorInfo">The factor the message operator is for.</param>
        /// <param name="messageFunctionInfo">The message operator.</param>
        private static void WriteMessageFunctionDocumentation(
            XmlWriter writer, FactorManager.FactorInfo factorInfo, MessageFcnInfo messageFunctionInfo)
        {
            string methodName = QuoteCodeElementName(StringUtil.MethodSignatureToString(
                messageFunctionInfo.Method, useFullName: false, omitParameterNames: true));
            
            writer.WriteStartElement("message_doc");
            writer.WriteAttributeString("name", methodName);

            WriteMessageOperatorSummary(writer, messageFunctionInfo);
            
            foreach (ParameterInfo parameter in messageFunctionInfo.Method.GetParameters())
            {
                WriteMessageOperatorParameterDescription(writer, factorInfo, messageFunctionInfo, parameter);
            }

            WriteMessageOperatorReturns(writer, factorInfo, messageFunctionInfo);
            WriteMessageOperatorRemarks(writer, factorInfo, messageFunctionInfo);
            WriteMessageOperatorExceptionSpec(writer, factorInfo, messageFunctionInfo);

            writer.WriteEndElement();
        }

        /// <summary>
        /// Writes the exception specification section of the XML documentation for a given message operator using a given XML writer.
        /// </summary>
        /// <param name="writer">The XML writer.</param>
        /// <param name="factorInfo">The factor the message operator is for.</param>
        /// <param name="messageFunctionInfo">The message operator.</param>
        private static void WriteMessageOperatorExceptionSpec(
            XmlWriter writer, FactorManager.FactorInfo factorInfo, MessageFcnInfo messageFunctionInfo)
        {
            var required = new List<FactorEdge>();
            ParameterInfo[] parameters = messageFunctionInfo.Method.GetParameters();
            var fieldToParameter = new Dictionary<string, string>();
            foreach (ParameterInfo parameter in parameters)
            {
                if (parameter.Name != "result" && parameter.Name != "resultIndex")
                {
                    FactorEdge edge;
                    if (messageFunctionInfo.factorEdgeOfParameter.TryGetValue(parameter.Name, out edge))
                    {
                        string field = edge.ToString();
                        fieldToParameter.Add(field, parameter.Name);
                        Type type = parameter.ParameterType;
                        if (!edge.IsOutgoingMessage && factorInfo.ParameterTypes.ContainsKey(field) && type != factorInfo.ParameterTypes[field])
                        {
                            required.AddRange(messageFunctionInfo.Requirements.Where(range => range.ParameterName == field));
                        }
                    }
                }
            }
            
            foreach (FactorEdge range in required)
            {
                writer.WriteStartElement("exception");
                writer.WriteAttributeString("cref", "ImproperMessageException");
                writer.WriteElementAttributeString("paramref", "name", fieldToParameter[range.ParameterName]);
                writer.WriteString(" is not a proper distribution.");
                writer.WriteEndElement();
            }
        }

        /// <summary>
        /// Writes the remarks section of the XML documentation for a given message operator using a given XML writer.
        /// </summary>
        /// <param name="writer">The XML writer.</param>
        /// <param name="factorInfo">The factor the message operator is for.</param>
        /// <param name="messageFunctionInfo">The message operator.</param>
        private static void WriteMessageOperatorRemarks(
            XmlWriter writer, FactorManager.FactorInfo factorInfo, MessageFcnInfo messageFunctionInfo)
        {
            bool isConstraint = factorInfo.Method.ReturnType == typeof(void);

            // must use the parameter names of the factor, not the operator method, because the operator method may not have parameters for all of the factor edges
            string argsString = StringUtil.CollectionToString(factorInfo.ParameterNames, ",");
            string childString = isConstraint ? string.Empty : factorInfo.ParameterNames[0];

            bool childIsRandom = false;
            ParameterInfo[] parameters = messageFunctionInfo.Method.GetParameters();
            string randomFieldsString = string.Empty;
            string randomFieldsMinusTarget = string.Empty;
            string randomFieldsMinusTargetAndChild = string.Empty;
            foreach (ParameterInfo parameter in parameters)
            {
                if (parameter.Name != "result" && parameter.Name != "resultIndex")
                {
                    FactorEdge edge;
                    if (messageFunctionInfo.factorEdgeOfParameter.TryGetValue(parameter.Name, out edge))
                    {
                        string field = edge.ToString();
                        Type type = parameter.ParameterType;
                        if (!edge.IsOutgoingMessage && factorInfo.ParameterTypes.ContainsKey(field) && type != factorInfo.ParameterTypes[field])
                        {
                            if (!isConstraint && factorInfo.ParameterNames[0] == field)
                            {
                                childIsRandom = true;
                            }

                            randomFieldsString += randomFieldsString == string.Empty ? field : "," + field;
                            if (field != messageFunctionInfo.TargetParameter)
                            {
                                randomFieldsMinusTarget += randomFieldsMinusTarget == string.Empty ? field : "," + field;
                                if (factorInfo.ParameterNames[0] != field)
                                {
                                    randomFieldsMinusTargetAndChild += randomFieldsMinusTargetAndChild == string.Empty ? field : "," + field;
                                }
                            }
                        }
                    }
                }
            }
            
            writer.WriteStartElement("remarks");
            writer.WriteStartElement("para");
            if (messageFunctionInfo.Method.Name == "AverageLogFactor")
            {
                if (randomFieldsString == string.Empty)
                {
                    writer.WriteString("The formula for the result is ");
                    writer.WriteElementFormatString("c", "log(factor({0}))", argsString);
                }
                else
                {
                    if (factorInfo.IsDeterministicFactor)
                    {
                        writer.WriteString("In Variational Message Passing, the evidence contribution of a deterministic factor is zero");
                    }
                    else
                    {
                        writer.WriteString("The formula for the result is ");
                        writer.WriteElementFormatString("c", "sum_({0}) p({0}) log(factor({1}))", randomFieldsString, argsString);
                    }
                }

                writer.WriteString(". Adding up these values across all factors and variables gives the log-evidence estimate for VMP.");
            }
            else if (messageFunctionInfo.Method.Name == "LogAverageFactor")
            {
                writer.WriteString("The formula for the result is ");

                if (randomFieldsString == string.Empty)
                {
                    writer.WriteElementFormatString("c", "log(factor({0}))", argsString);
                }
                else
                {
                    writer.WriteElementFormatString("c", "log(sum_({0}) p({0}) factor({1}))", randomFieldsString, argsString);
                }

                writer.WriteString(".");
            }
            else if (messageFunctionInfo.Method.Name == "LogEvidenceRatio")
            {
                writer.WriteString("The formula for the result is ");

                if (randomFieldsString == string.Empty)
                {
                    writer.WriteElementFormatString("c", "log(factor({0}))", argsString);
                }
                else
                {
                    if (childIsRandom)
                    {
                        writer.WriteElementFormatString("c", "log(sum_({0}) p({0}) factor({1}) / sum_{2} p({2}) messageTo({2}))", randomFieldsString, argsString, childString);
                    }
                    else
                    {
                        writer.WriteElementFormatString("c", "log(sum_({0}) p({0}) factor({1}))", randomFieldsString, argsString);
                    }
                }

                writer.WriteString(". Adding up these values across all factors and variables gives the log-evidence estimate for EP.");
            }
            else if (messageFunctionInfo.Suffix == "Conditional")
            {
                writer.WriteString("The outgoing message is the factor viewed as a function of ");
                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                writer.WriteString(" conditioned on the given values.");
            }
            else if (messageFunctionInfo.Suffix == "AverageConditional")
            {
                if (randomFieldsMinusTarget == string.Empty)
                {
                    writer.WriteString("The outgoing message is the factor viewed as a function of ");
                    writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                    writer.WriteString(" conditioned on the given values.");
                }
                else
                {
                    writer.WriteString("The outgoing message is a distribution matching the moments of ");
                    writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                    writer.WriteString(" as the random arguments are varied. The formula is ");
                    writer.WriteElementFormatString("c", "proj[p({2}) sum_({1}) p({1}) factor({0})]/p({2})", argsString, randomFieldsMinusTarget, messageFunctionInfo.TargetParameter);
                    writer.WriteString(".");
                }
            }
            else if (messageFunctionInfo.Suffix == "AverageLogarithm")
            {
                if (randomFieldsMinusTarget == string.Empty)
                {
                    writer.WriteString("The outgoing message is the factor viewed as a function of ");
                    writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                    writer.WriteString(" conditioned on the given values.");
                }
                else
                {
                    if (factorInfo.IsDeterministicFactor)
                    {
                        if (childString == messageFunctionInfo.TargetParameter)
                        {
                            writer.WriteString("The outgoing message is a distribution matching the moments of ");
                            writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                            writer.WriteString(" as the random arguments are varied. The formula is ");
                            writer.WriteElementFormatString("c", "proj[sum_({1}) p({1}) factor({0})]", argsString, randomFieldsMinusTarget);
                        }
                        else
                        {
                            if (childIsRandom && randomFieldsMinusTargetAndChild == string.Empty)
                            {
                                writer.WriteString("The outgoing message is the factor viewed as a function of ");
                                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                                writer.WriteString(" with ");
                                writer.WriteElementFormatString("c", childString);
                                writer.WriteString(" integrated out. The formula is ");
                                writer.WriteElementFormatString("c", "sum_{1} p({1}) factor({0})", argsString, childString);
                            }
                            else
                            {
                                writer.WriteString("The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except ");
                                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                                writer.WriteString(". ");

                                if (childIsRandom)
                                {
                                    writer.WriteString("Because the factor is deterministic, ");
                                    writer.WriteElementFormatString("c", childString);
                                    writer.WriteString(" is integrated out before taking the logarithm. The formula is ");
                                    writer.WriteElementFormatString(
                                        "c", "exp(sum_({1}) p({1}) log(sum_{2} p({2}) factor({0})))", argsString, randomFieldsMinusTargetAndChild, childString);
                                }
                                else
                                {
                                    writer.WriteString("The formula is ");
                                    writer.WriteElementFormatString("c", "exp(sum_({1}) p({1}) log(factor({0})))", argsString, randomFieldsMinusTarget);
                                }
                            }
                        }
                    }
                    else
                    {
                        writer.WriteString(
                            "The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except ");
                        writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                        writer.WriteString(". The formula is ");
                        writer.WriteElementFormatString("c", "exp(sum_({1}) p({1}) log(factor({0})))", argsString, randomFieldsMinusTarget);
                    }

                    writer.WriteString(".");
                }
            }

            writer.WriteEndElement();
            writer.WriteEndElement();
        }

        /// <summary>
        /// Writes the returns section of the XML documentation for a given message operator using a given XML writer.
        /// </summary>
        /// <param name="writer">The XML writer.</param>
        /// <param name="factorInfo">The factor the message operator is for.</param>
        /// <param name="messageFunctionInfo">The message operator.</param>
        private static void WriteMessageOperatorReturns(
            XmlWriter writer, FactorManager.FactorInfo factorInfo, MessageFcnInfo messageFunctionInfo)
        {
            writer.WriteStartElement("returns");

            if (messageFunctionInfo.PassResult)
            {
                writer.WriteElementAttributeString("paramref", "name", "result");
            }
            else if (messageFunctionInfo.Method.Name == "LogFactorValue")
            {
                writer.WriteString("Logarithm of the factor's value at the given arguments.");
            }
            else if (messageFunctionInfo.Method.Name == "LogAverageFactor")
            {
                writer.WriteString("Logarithm of the factor's average value across the given argument distributions.");
            }
            else if (messageFunctionInfo.Method.Name == "LogEvidenceRatio")
            {
                writer.WriteString("Logarithm of the factor's contribution the EP model evidence.");
            }
            else if (messageFunctionInfo.Method.Name == "AverageLogFactor")
            {
                writer.WriteString(factorInfo.IsDeterministicFactor ? "Zero." : "Average of the factor's log-value across the given argument distributions.");
            }
            else if (messageFunctionInfo.Suffix == "Conditional")
            {
                writer.WriteString("The outgoing Gibbs message to the ");
                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                writer.WriteString(" argument.");
            }
            else if (messageFunctionInfo.Suffix == "AverageConditional")
            {
                writer.WriteString("The outgoing EP message to the ");
                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                writer.WriteString(" argument.");
            }
            else if (messageFunctionInfo.Suffix == "AverageLogarithm")
            {
                writer.WriteString("The outgoing VMP message to the ");
                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                writer.WriteString(" argument.");
            }
            else if (messageFunctionInfo.Suffix == "Init")
            {
                writer.WriteString("Initial value of buffer ");
                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                writer.WriteString(".");
            }
            else if (messageFunctionInfo.Suffix == string.Empty)
            {
                writer.WriteString("New value of buffer ");
                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                writer.WriteString(".");
            }

            writer.WriteEndElement();
        }

        /// <summary>
        /// Writes the XML documentation for a parameter of given message operator using a given XML writer.
        /// </summary>
        /// <param name="writer">The XML writer.</param>
        /// <param name="factorInfo">The factor the message operator is for.</param>
        /// <param name="messageFunctionInfo">The message operator.</param>
        /// <param name="parameter">The parameter.</param>
        private static void WriteMessageOperatorParameterDescription(
            XmlWriter writer, FactorManager.FactorInfo factorInfo, MessageFcnInfo messageFunctionInfo, ParameterInfo parameter)
        {
            writer.WriteStartElement("param");
            writer.WriteAttributeString("name", parameter.Name);

            if (parameter.Name == "result")
            {
                writer.WriteString("Modified to contain the outgoing message.");
            }
            else if (parameter.Name == "resultIndex")
            {
                writer.WriteString("Index of the ");
                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                writer.WriteString(" for which a message is desired.");
            }
            else
            {
                FactorEdge edge;
                if (!messageFunctionInfo.factorEdgeOfParameter.TryGetValue(parameter.Name, out edge))
                {
                    // TODO: skip methods which aren't message operators
                }
                else
                {
                    string field = edge.ToString();
                    Type type = parameter.ParameterType;
                    if (edge.IsOutgoingMessage)
                    {
                        bool isFresh = parameter.IsDefined(typeof(FreshAttribute), false);
                        writer.WriteFormatString("{0} message to ", isFresh ? "Outgoing" : "Previous outgoing");
                        writer.WriteElementFormatString("c", parameter.Name.Substring(3));
                        writer.WriteString(".");
                    }
                    else if (factorInfo.ParameterTypes.ContainsKey(field))
                    {
                        if (type == factorInfo.ParameterTypes[field])
                        {
                            writer.WriteString("Constant value for ");
                            writer.WriteElementFormatString("c", field);
                            writer.WriteString(".");
                        }
                        else
                        {
                            writer.WriteString("Incoming message from ");
                            writer.WriteElementFormatString("c", field);
                            writer.WriteString(".");
                            foreach (FactorEdge rrange in messageFunctionInfo.Requirements)
                            {
                                FactorEdge range = rrange;
                                if (range.ParameterName == field)
                                {
                                    writer.WriteString(" Must be a proper distribution. If ");
                                    if (Util.IsIList(factorInfo.ParameterTypes[field]))
                                    {
                                        foreach (FactorEdge range2 in messageFunctionInfo.Dependencies)
                                        {
                                            if (range2.ParameterName == field)
                                            {
                                                range = range.Intersect(range2);
                                            }
                                        }

                                        if (range.MinCount == 1)
                                        {
                                            writer.WriteString(range.ContainsIndex ? "the element at resultIndex is " : "all elements are ");
                                        }
                                        else if (range.ContainsAllOthers)
                                        {
                                            writer.WriteString(range.ContainsIndex ? "any element is " : "any element besides resultIndex is ");
                                        }
                                    }

                                    writer.WriteString("uniform, the result will be uniform.");
                                }
                            }
                        }
                    }
                    else
                    {
                        writer.WriteString("Buffer ");
                        writer.WriteElementFormatString("c", parameter.Name);
                        writer.WriteString(".");
                    }
                }
            }

            writer.WriteEndElement();
        }

        /// <summary>
        /// Writes the summary section of the XML documentation for a given message operator using a given XML writer.
        /// </summary>
        /// <param name="writer">The XML writer.</param>
        /// <param name="messageFunctionInfo">The message operator.</param>
        private static void WriteMessageOperatorSummary(XmlWriter writer, MessageFcnInfo messageFunctionInfo)
        {
            writer.WriteStartElement("summary");

            if (messageFunctionInfo.Method.Name == "LogFactorValue")
            {
                writer.WriteString("Evidence message for Gibbs.");
            }
            else if (messageFunctionInfo.Method.Name == "LogAverageFactor")
            {
                writer.WriteString("Evidence message for EP.");
            }
            else if (messageFunctionInfo.Method.Name == "LogEvidenceRatio")
            {
                writer.WriteString("Evidence message for EP.");
            }
            else if (messageFunctionInfo.Method.Name == "AverageLogFactor")
            {
                writer.WriteString("Evidence message for VMP.");
            }
            else if (messageFunctionInfo.Suffix == "Conditional")
            {
                writer.WriteString("Gibbs message to ");
                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                writer.WriteString(".");
            }
            else if (messageFunctionInfo.Suffix == "AverageConditional")
            {
                writer.WriteString("EP message to ");
                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                writer.WriteString(".");
            }
            else if (messageFunctionInfo.Suffix == "AverageLogarithm")
            {
                writer.WriteString("VMP message to ");
                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                writer.WriteString(".");
            }
            else if (messageFunctionInfo.Suffix == "Init")
            {
                writer.WriteString("Initialize the buffer ");
                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                writer.WriteString(".");
            }
            else if (messageFunctionInfo.Suffix == string.Empty)
            {
                writer.WriteString("Update the buffer ");
                writer.WriteElementFormatString("c", messageFunctionInfo.TargetParameter);
                writer.WriteString(".");
            }

            writer.WriteEndElement();
        }

        /// <summary>
        /// Quotes a given type or method name so that it can be put in an XML file.
        /// </summary>
        /// <param name="name">The code element name to quote.</param>
        /// <returns>The quoted code element name.</returns>
        private static string QuoteCodeElementName(string name)
        {
            return name.Replace('<', '{').Replace('>', '}');
        }

        /// <summary>
        /// Wraps <see cref="FactorManager.FactorInfo"/>, providing comparison based on factor method equality.
        /// </summary>
        private class FactorInfoWrapper
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="FactorInfoWrapper"/> class.
            /// </summary>
            /// <param name="factorInfo">The wrapped factor info.</param>
            public FactorInfoWrapper(FactorManager.FactorInfo factorInfo)
            {
                Debug.Assert(factorInfo != null, "The given factor info cannot be null.");
                this.FactorInfo = factorInfo;
            }

            /// <summary>
            /// Gets the wrapped factor info.
            /// </summary>
            public FactorManager.FactorInfo FactorInfo { get; private set; }

            /// <summary>
            /// Compares this object with a given one.
            /// </summary>
            /// <param name="obj">The object to compare with.</param>
            /// <returns><see langword="true"/> if the objects are equal, <see langword="false"/> otherwise.</returns>
            /// <remarks>The two wrappers are considered equal, if they wrap factor info for the same factor method.</remarks>
            public override bool Equals(object obj)
            {
                if (obj == null || GetType() != obj.GetType())
                {
                    return false;
                }

                return this.FactorInfo.Method.Equals(((FactorInfoWrapper)obj).FactorInfo.Method);
            }

            /// <summary>
            /// Computes the hash code of this object.
            /// </summary>
            /// <returns>The computed hash code.</returns>
            /// <remarks>Only the factor method is used to compute the hash code.</remarks>
            public override int GetHashCode()
            {
                return this.FactorInfo.Method.GetHashCode();
            }
        }

        /// <summary>
        /// Wraps <see cref="FactorManager.FactorInfo"/>, providing comparison based on message operator method equality
        /// and augmenting message info with the info about a factor it is for.
        /// </summary>
        private class MessageFunctionInfoWrapper
        {
            /// <summary>
            /// The method info for the message operator method obtained from the declaring type.
            /// Used for equality comparison and hash code computation.
            /// </summary>
            private readonly MethodInfo declaringTypeMethodInfo;

            /// <summary>
            /// Initializes a new instance of the <see cref="MessageFunctionInfoWrapper"/> class.
            /// </summary>
            /// <param name="messageFunctionInfo">The wrapped message function info.</param>
            /// <param name="factorInfo">The factor info of a factor the message function is for.</param>
            public MessageFunctionInfoWrapper(MessageFcnInfo messageFunctionInfo, FactorManager.FactorInfo factorInfo)
            {
                this.MessageFunctionInfo = messageFunctionInfo;
                this.FactorInfo = factorInfo;

                // MethodInfo instances obtained from the declaring type and the actual type are different
                Type messageFuncDeclaringType = this.MessageFunctionInfo.Method.DeclaringType;
                this.declaringTypeMethodInfo = messageFuncDeclaringType.GetMethod(
                    this.MessageFunctionInfo.Method.Name, this.MessageFunctionInfo.Method.GetParameters().Select(p => p.ParameterType).ToArray());
            }

            /// <summary>
            /// Gets the wrapped message function info.
            /// </summary>
            public MessageFcnInfo MessageFunctionInfo { get; private set; }

            /// <summary>
            /// Gets the information about a factor the message function is for.
            /// </summary>
            public FactorManager.FactorInfo FactorInfo { get; private set; }

            /// <summary>
            /// Compares this object with a given one.
            /// </summary>
            /// <param name="obj">The object to compare with.</param>
            /// <returns><see langword="true"/> if the objects are equal, <see langword="false"/> otherwise.</returns>
            /// <remarks>The two wrappers are considered equal, if they wrap message info for the same message function.</remarks>
            public override bool Equals(object obj)
            {
                if (obj == null || GetType() != obj.GetType())
                {
                    return false;
                }

                return this.declaringTypeMethodInfo.Equals(((MessageFunctionInfoWrapper)obj).declaringTypeMethodInfo);
            }

            /// <summary>
            /// Computes the hash code of this object.
            /// </summary>
            /// <returns>The computed hash code.</returns>
            /// <remarks>Only the message function method is used to compute the hash code.</remarks>
            public override int GetHashCode()
            {
                return this.declaringTypeMethodInfo.GetHashCode();
            }
        }
    }
}
