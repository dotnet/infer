// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Command line module acting as a collection of nested modules.
    /// </summary>
    public class CommandLineModuleSelector : CommandLineModule
    {
        /// <summary>
        /// A dictionary to retrieve nested modules by name efficiently.
        /// </summary>
        private readonly Dictionary<string, CommandLineModule> registeredModulesDictionary = new Dictionary<string, CommandLineModule>();

        /// <summary>
        /// The list of the nested module names with the registration order preserved.
        /// </summary>
        private readonly List<string> registeredModuleNames = new List<string>();

        /// <summary>
        /// Registers a new module.
        /// </summary>
        /// <param name="moduleName">The name of the module, which will be used to invoke it from the command line.</param>
        /// <param name="module">The module.</param>
        public void RegisterModule(string moduleName, CommandLineModule module)
        {
            if (string.IsNullOrEmpty(moduleName))
            {
                throw new ArgumentException("Valid module name should be specified.", nameof(moduleName));
            }

            if (moduleName.Contains(' ') || moduleName.Contains('\t'))
            {
                throw new ArgumentException("Module name should not contains spaces or tabs.", nameof(moduleName));
            }

            if (this.registeredModulesDictionary.ContainsKey(moduleName))
            {
                throw new ArgumentException("A module with the given name has already been registered.", nameof(moduleName));
            }

            if (module == null)
            {
                throw new ArgumentNullException(nameof(module));
            }
            
            this.registeredModulesDictionary.Add(moduleName, module);
            this.registeredModuleNames.Add(moduleName);
        }

        /// <summary>
        /// Determines which one of the nested modules should be invoked and runs it.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public override bool Run(string[] args, string usagePrefix)
        {
            if (args == null)
            {
                throw new ArgumentNullException(nameof(args));
            }
            
            if (args.Length == 0)
            {
                this.PrintUsage(usagePrefix);
                return false;
            }

            string moduleName = args[0];
            if (!this.registeredModulesDictionary.ContainsKey(moduleName))
            {
                Console.WriteLine("Unknown module: {0}.", moduleName);
                this.PrintUsage(usagePrefix);
                return false;
            }

            CommandLineModule invokedSubmodule = this.registeredModulesDictionary[moduleName];
            return invokedSubmodule.Run(args.Skip(1).ToArray(), string.Format("{0} {1}", usagePrefix, moduleName));
        }

        /// <summary>
        /// Prints usage information for all the nested modules.
        /// </summary>
        /// <param name="usagePrefix">String that should be printed before the usage.</param>
        private void PrintUsage(string usagePrefix)
        {
            Console.WriteLine("Usage: {0} <module> <parameters>", usagePrefix);
            Console.WriteLine("Available modules:");

            for (int i = 0; i < this.registeredModuleNames.Count; ++i)
            {
                Console.WriteLine("  {0}", this.registeredModuleNames[i]);
            }
        }
    }
}