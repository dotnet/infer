// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    /// <summary>
    /// A simple command line parser.
    /// </summary>
    public class CommandLineParser
    {
        /// <summary>
        /// The registered parameter descriptions.
        /// </summary>
        private readonly Dictionary<string, ParameterDescription> parameterDescriptions = new Dictionary<string, ParameterDescription>();

        /// <summary>
        /// Tries to parse a given command line string. If the string is invalid, reports the error and prints usage information if necessary.
        /// </summary>
        /// <param name="args">The command line string represented as an array of arguments.</param>
        /// <param name="usagePrefix">The prefix to print in front of the usage string.</param>
        /// <returns>True if the string was successfully parsed, false otherwise.</returns>
        public bool TryParse(string[] args, string usagePrefix)
        {
            if (args == null)
            {
                throw new ArgumentNullException(nameof(args));
            }

            var encounteredParameters = new HashSet<string>();
            for (int i = 0; i < args.Length; i += 2)
            {
                string parameterName = args[i];

                if (!this.parameterDescriptions.ContainsKey(parameterName))
                {
                    Console.WriteLine("Usage error: parameter {0} is unknown.", parameterName);
                    this.PrintUsage(usagePrefix);
                    return false;
                }

                ParameterDescription parameterDescription = this.parameterDescriptions[parameterName];

                if (encounteredParameters.Contains(parameterName))
                {
                    Console.WriteLine("Usage error: parameter {0} was specified multiple times.", parameterName);
                    return false;
                }
                
                if (parameterDescription.IsFlag)
                {
                    parameterDescription.Handler(string.Empty);
                    --i;
                }
                else
                {
                    if (args.Length == i + 1)
                    {
                        Console.WriteLine("Usage error: parameter {0} has no corresponding value.", parameterName);
                        return false;
                    }

                    string parameterValue = args[i + 1];
                    
                    if (!parameterDescription.Handler(parameterValue))
                    {
                        Console.WriteLine("Usage error: value '{0}' of the parameter {1} has invalid format.", parameterValue, parameterName);
                        return false;
                    }    
                }

                encounteredParameters.Add(parameterName);
            }

            var missingRequiredParameters =
                this.parameterDescriptions.Where(kv => kv.Value.Type == CommandLineParameterType.Required && !encounteredParameters.Contains(kv.Key));
            if (missingRequiredParameters.Any())
            {
                Console.WriteLine("Usage error: required parameter {0} is missing.", missingRequiredParameters.First().Key);
                this.PrintUsage(usagePrefix);
                return false;
            }

            return true;
        }

        /// <summary>
        /// Registers a new handler for the command-line flag parameter.
        /// </summary>
        /// <param name="parameterName">The name of the parameter.</param>
        /// <param name="descriptionString">The description string for the parameter.</param>
        /// <param name="handler">The handler.</param>
        public void RegisterParameterHandler(string parameterName, string descriptionString, Action handler)
        {
            if (handler == null)
            {
                throw new ArgumentNullException(nameof(handler));
            }

            Func<string, bool> handlerWrapper = parameterValueString =>
            {
                handler();
                return true;
            };

            // Other parameters are checked inside the RegisterParameterHandlerImpl
            this.RegisterParameterHandlerImpl(parameterName, string.Empty, descriptionString, true, handlerWrapper, CommandLineParameterType.Optional);
        }

        /// <summary>
        /// Registers a new handler for the command-line parameter of string type.
        /// </summary>
        /// <param name="parameterName">The name of the parameter.</param>
        /// <param name="valueDescriptionString">The description string for the parameter value.</param>
        /// <param name="descriptionString">The description string for the parameter.</param>
        /// <param name="handler">The handler.</param>
        /// <param name="parameterType">Whether the parameter is required or optional.</param>
        public void RegisterParameterHandler(
            string parameterName, string valueDescriptionString, string descriptionString, Action<string> handler, CommandLineParameterType parameterType)
        {
            if (handler == null)
            {
                throw new ArgumentNullException(nameof(handler));
            }

            Func<string, bool> handlerWrapper = parameterValueString =>
            {
                handler(parameterValueString);
                return true;
            };

            // Other parameters are checked inside the RegisterParameterHandlerImpl
            this.RegisterParameterHandlerImpl(parameterName, valueDescriptionString, descriptionString, false, handlerWrapper, parameterType);
        }

        /// <summary>
        /// Registers a new handler for the command-line parameter of integer type.
        /// </summary>
        /// <param name="parameterName">The name of the parameter.</param>
        /// <param name="valueDescriptionString">The description string for the parameter value.</param>
        /// <param name="descriptionString">The description string for the parameter.</param>
        /// <param name="handler">The handler.</param>
        /// <param name="parameterType">Whether the parameter is required or optional.</param>
        public void RegisterParameterHandler(
            string parameterName, string valueDescriptionString, string descriptionString, Action<int> handler, CommandLineParameterType parameterType)
        {
            if (handler == null)
            {
                throw new ArgumentNullException(nameof(handler));
            }

            Func<string, bool> handlerWrapper = parameterValueString =>
            {
                int parameterValue;
                if (!int.TryParse(parameterValueString, out parameterValue))
                {
                    return false;
                }

                handler(parameterValue);
                return true;
            };

            // Other parameters are checked inside the RegisterParameterHandlerImpl
            this.RegisterParameterHandlerImpl(parameterName, valueDescriptionString, descriptionString, false, handlerWrapper, parameterType);
        }

        /// <summary>
        /// Registers a new handler for the parameter of floating-point type.
        /// </summary>
        /// <param name="parameterName">The name of the parameter.</param>
        /// <param name="valueDescriptionString">The description string for the parameter value.</param>
        /// <param name="descriptionString">The description string for the parameter.</param>
        /// <param name="handler">The handler.</param>
        /// <param name="parameterType">Whether the parameter is required or optional.</param>
        public void RegisterParameterHandler(
            string parameterName, string valueDescriptionString, string descriptionString, Action<double> handler, CommandLineParameterType parameterType)
        {
            if (handler == null)
            {
                throw new ArgumentNullException(nameof(handler));
            }

            Func<string, bool> handlerWrapper = parameterValueString =>
            {
                double parameterValue;
                if (!double.TryParse(parameterValueString, out parameterValue))
                {
                    return false;
                }

                handler(parameterValue);
                return true;
            };

            // Other parameters are checked inside the RegisterParameterHandlerImpl
            this.RegisterParameterHandlerImpl(parameterName, valueDescriptionString, descriptionString, false, handlerWrapper, parameterType);
        }

        /// <summary>
        /// Prints information about the usage of a given parameter.
        /// </summary>
        /// <param name="parameterName">The name of the parameter.</param>
        /// <param name="description">The description of the parameter.</param>
        private static void PrintParameterUsage(string parameterName, ParameterDescription description)
        {
            const int ParameterNameValuePadding = 35;
            string parameterNameValue = description.IsFlag ? parameterName : string.Format("{0} {1}", parameterName, description.ValueDescriptionString);
            Console.WriteLine("  {0} {1}", parameterNameValue.PadRight(ParameterNameValuePadding), description.DescriptionString);
        }

        /// <summary>
        /// Prints the usage description.
        /// </summary>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        private void PrintUsage(string usagePrefix)
        {
            Console.WriteLine("Usage: {0} <parameters>", usagePrefix);

            Console.WriteLine();
            Console.WriteLine("Required parameters:");
            foreach (KeyValuePair<string, ParameterDescription> parameterWithDescription in this.parameterDescriptions)
            {
                if (parameterWithDescription.Value.Type == CommandLineParameterType.Required)
                {
                    PrintParameterUsage(parameterWithDescription.Key, parameterWithDescription.Value);
                }
            }

            Console.WriteLine();
            Console.WriteLine("Optional parameters:");
            foreach (KeyValuePair<string, ParameterDescription> parameterWithDescription in this.parameterDescriptions)
            {
                if (parameterWithDescription.Value.Type == CommandLineParameterType.Optional)
                {
                    PrintParameterUsage(parameterWithDescription.Key, parameterWithDescription.Value);
                }
            }
        }

        /// <summary>
        /// Actually registers a new parameter handler.
        /// </summary>
        /// <param name="parameterName">The name of the parameter.</param>
        /// <param name="valueDescriptionString">The description string for the parameter value.</param>
        /// <param name="descriptionString">The description string for the parameter.</param>
        /// <param name="isFlag">The value indicating whether the parameter represents a flag and, thus, doesn't need a value.</param>
        /// <param name="handler">The delegate to handle the parameter.</param>
        /// <param name="parameterType">Whether the parameter is required or optional.</param>
        private void RegisterParameterHandlerImpl(
            string parameterName,
            string valueDescriptionString,
            string descriptionString,
            bool isFlag,
            Func<string, bool> handler,
            CommandLineParameterType parameterType)
        {
            Debug.Assert(handler != null, "A valid handler should be specified.");
            
            if (string.IsNullOrEmpty(parameterName))
            {
                throw new ArgumentException("A parameter name should be a valid non-empty string.", nameof(parameterName));
            }

            if (this.parameterDescriptions.ContainsKey(parameterName))
            {
                throw new ArgumentException("Given parameter was already registered.", nameof(parameterName));
            }

            if (!isFlag && string.IsNullOrEmpty(valueDescriptionString))
            {
                throw new ArgumentException("A parameter value description should be a valid non-empty string.", nameof(valueDescriptionString));
            }
            
            if (string.IsNullOrEmpty(descriptionString))
            {
                throw new ArgumentException("A parameter description should be a valid non-empty string.", nameof(descriptionString));
            }
            
            if (handler == null)
            {
                throw new ArgumentNullException(nameof(handler));
            }

            this.parameterDescriptions.Add(
                parameterName, new ParameterDescription(valueDescriptionString, descriptionString, isFlag, handler, parameterType));
        }

        /// <summary>
        /// Represents the description of a command-line parameter.
        /// </summary>
        private class ParameterDescription
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="ParameterDescription"/> class.
            /// </summary>
            /// <param name="valueDescriptionString">The value description string for the parameter.</param>
            /// <param name="descriptionString">The description string for the parameter.</param>
            /// <param name="isFlag">The value indicating whether the parameter represents a flag and, thus, doesn't need a value.</param>
            /// <param name="handler">The handler for the parameter value.</param>
            /// <param name="type">The type.</param>
            public ParameterDescription(
                string valueDescriptionString, string descriptionString, bool isFlag, Func<string, bool> handler, CommandLineParameterType type)
            {
                Debug.Assert(!(!isFlag && string.IsNullOrEmpty(valueDescriptionString)), "A valid value description string should be specified for non-flag parameters.");
                Debug.Assert(!string.IsNullOrEmpty(descriptionString), "A valid description string should be specified.");
                Debug.Assert(!(isFlag && type == CommandLineParameterType.Required), "A flag can not be required.");
                Debug.Assert(handler != null, "A valid handler should be specified.");
                
                this.ValueDescriptionString = valueDescriptionString;
                this.DescriptionString = descriptionString;
                this.IsFlag = isFlag;
                this.Handler = handler;
                this.Type = type;
            }

            /// <summary>
            /// Gets the value description string for the parameter.
            /// </summary>
            public string ValueDescriptionString { get; private set; }
            
            /// <summary>
            /// Gets the description string for the parameter.
            /// </summary>
            public string DescriptionString { get; private set; }

            /// <summary>
            /// Gets a value indicating whether the parameter represents a flag and, thus, doesn't need a value.
            /// </summary>
            public bool IsFlag { get; private set; }
            
            /// <summary>
            /// Gets the handler for the parameter value.
            /// </summary>
            public Func<string, bool> Handler { get; private set; }

            /// <summary>
            /// Gets a value indicating whether the parameter is required or optional.
            /// </summary>
            public CommandLineParameterType Type { get; private set; }
        }
    }
}
