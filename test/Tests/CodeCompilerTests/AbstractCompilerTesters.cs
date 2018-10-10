// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests.CodeCompilerTests
{
    /// <summary>
    /// Abstract class for testing code compiler.
    /// </summary>
    public abstract class AbstractCompilerTester
    {
        protected List<string> fileNames = null;
        protected List<string> sourceCodes = null;

        protected List<Assembly> assemblies;
        protected CodeCompiler compiler;

        public AbstractCompilerTester()
        {
            assemblies = AppDomain.CurrentDomain.GetAssemblies()
                .Where(a => !a.IsDynamic && !string.IsNullOrEmpty(a.Location))
                .GroupBy(x => x.FullName)
                .Select(x => x.First())
                .ToList();
            compiler = new CodeCompiler();
        }

        abstract protected void CheckCompiling();
    }



    /// <summary>
    /// Abstract class for testing compiler using valid source code.
    /// </summary>
    public abstract class AbstractCompilerValidSourceTester : AbstractCompilerTester
    {
        public AbstractCompilerValidSourceTester()
        {
            sourceCodes = new List<string>
            {
                @"using System;
                class GeneratedTestProgram
                {
                    public static void Main()
                    {
                        int fibo1 = 1, fibo2 = 1;
                        for(int i = 0; i < 20; i++)
                        {
                            int tmp = fibo2;
                            fibo2 = fibo1 + tmp;
                            fibo1 = tmp;
                        }
                        Console.WriteLine(""F21 = {0}"", fibo2);
                    }
                }",

                @"using System;
                class GeneratedTestProgram2
                {
                    public static void Main()
                    {
                        double e = 1;
                        int n = 1;
                        for (int i = 2; i < 20; i++)
                        {
                            e += 1.0 / n;
                            n *= i;
                        }
                        Console.WriteLine(""e ~ {0}"", e);
                    }
                }"
            };

            fileNames = new List<string>(sourceCodes.Count);
            for (int i = 0; i < sourceCodes.Count; i++)
            {
                fileNames.Add(Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".cs"));
            }
        }

        override protected void CheckCompiling()
        {
            compiler.generateInMemory = false;
            var crs = compiler.Compile(fileNames, sourceCodes, assemblies);
            Assert.True(crs.Success);
            Assert.Empty(crs.Errors);

            compiler.generateInMemory = true;
            crs = compiler.Compile(fileNames, sourceCodes, assemblies);
            Assert.True(crs.Success);
            Assert.Empty(crs.Errors);
        }
    }



    /// <summary>
    /// Abstract class for testing compiler using valid source code.
    /// </summary>
    public abstract class AbstractCompilerInvalidSourceTester : AbstractCompilerTester
    {
        public AbstractCompilerInvalidSourceTester()
        {
            sourceCodes = new List<string>
            {
                @"using System;
                class GeneratedTestProgram3
                {
                    public static void Main()
                    {
                        for (int i = 2; i < 20)
                        {
                            Console.WriteLine(""ERROR!"");
                        }
                    }
                }",

                @"using System;
                class GeneratedTestProgram4
                {
                    public static void Main()
                    {
                        Console.WriteLine(""{0}"", Math.Sin(3.1415926))
                    }
                }"
            };

            fileNames = new List<string>(sourceCodes.Count);
            for (int i = 0; i < sourceCodes.Count; i++)
            {
                fileNames.Add(Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".cs"));
            }
        }

        override protected void CheckCompiling()
        {
            compiler.generateInMemory = false;
            var crs = compiler.Compile(fileNames, sourceCodes, assemblies);
            Assert.False(crs.Success);
            Assert.True(crs.Errors.Count == 2);

            compiler.generateInMemory = true;
            crs = compiler.Compile(fileNames, sourceCodes, assemblies);
            Assert.False(crs.Success);
            Assert.True(crs.Errors.Count == 2);
        }
    }
}
