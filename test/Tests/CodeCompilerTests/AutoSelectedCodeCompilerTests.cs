// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler;
using System.IO;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests.CodeCompilerTests
{
    /// <summary>
    /// Tests for auto selected compiler for valid source code files.
    /// </summary>
    public class AutoSelectedCodeCompilerValidSourceFileTests : AbstractCompilerValidSourceTester
    {
        public AutoSelectedCodeCompilerValidSourceFileTests()
        {
            for (int i = 0; i < sourceCodes.Count; i++)
            {
                using (var writer = new StreamWriter(fileNames[i]))
                {
                    writer.Write(sourceCodes[i]);
                }
            }

            compiler.compilerChoice = CompilerChoice.Auto;

            //this flag let use the code from given source files
            compiler.writeSourceFiles = true;
        }

        /// <summary>
        /// Check default parameters of compiling.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void DefaultCompileSourceFileTest()
        {
            CheckCompiling();
        }

        /// <summary>
        /// Check optimized and nonoptimized compiling.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void OptimizedCompileSourceFileTest()
        {
            compiler.optimizeCode = true;
            CheckCompiling();

            compiler.optimizeCode = false;
            CheckCompiling();
        }

        /// <summary>
        /// Check compiling with and without making debug information.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void DebugInfoCompileSourceFileTest()
        {
            compiler.includeDebugInformation = true;
            CheckCompiling();

            compiler.includeDebugInformation = false;
            CheckCompiling();
        }

        /// <summary>
        /// Check compiling with and without showing progress.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void ShowProgressCompileSourceFileTest()
        {
            compiler.showProgress = true;
            CheckCompiling();

            compiler.showProgress = false;
            CheckCompiling();
        }
    }



    /// <summary>
    /// Tests for auto selected compiler for valid source code files.
    /// </summary>
    public class AutoSelectedCodeCompilerInvalidSourceFileTests : AbstractCompilerInvalidSourceTester
    {
        public AutoSelectedCodeCompilerInvalidSourceFileTests()
        {
            for (int i = 0; i < sourceCodes.Count; i++)
            {
                using (var writer = new StreamWriter(fileNames[i]))
                {
                    writer.Write(sourceCodes[i]);
                }
            }

            compiler.compilerChoice = CompilerChoice.Auto;

            //this flag let use the code from given source files
            compiler.writeSourceFiles = true;
        }

        /// <summary>
        /// Check default parameters of compiling.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void DefaultCompileSourceFileTest()
        {
            CheckCompiling();
        }

        /// <summary>
        /// Check optimized and nonoptimized compiling.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void OptimizedCompileSourceFileTest()
        {
            compiler.optimizeCode = true;
            CheckCompiling();

            compiler.optimizeCode = false;
            CheckCompiling();
        }

        /// <summary>
        /// Check compiling with and without making debug information.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void DebugInfoCompileSourceFileTest()
        {
            compiler.includeDebugInformation = true;
            CheckCompiling();

            compiler.includeDebugInformation = false;
            CheckCompiling();
        }

        /// <summary>
        /// Check compiling with and without showing progress.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void ShowProgressCompileSourceFileTest()
        {
            compiler.showProgress = true;
            CheckCompiling();

            compiler.showProgress = false;
            CheckCompiling();
        }
    }



    /// <summary>
    /// Tests for auto selected compiler for valid source code that is described in a string.
    /// </summary>
    public class AutoSelectedCodeCompilerValidSourceStringTests : AbstractCompilerValidSourceTester
    {
        public AutoSelectedCodeCompilerValidSourceStringTests()
        {
            compiler.compilerChoice = CompilerChoice.Auto;

            //turn off using existing source files
            compiler.writeSourceFiles = false;
        }

        /// <summary>
        /// Check default parameters of compiling.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void DefaultCompileSourceStringTest()
        {
            CheckCompiling();
        }

        /// <summary>
        /// Check optimized and nonoptimized compiling.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void OptimizedCompileSourceStringTest()
        {
            compiler.optimizeCode = true;
            CheckCompiling();

            compiler.optimizeCode = false;
            CheckCompiling();
        }

        /// <summary>
        /// Check compiling with and without making debug information.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void DebugInfoCompileSourceStringTest()
        {
            compiler.includeDebugInformation = true;
            CheckCompiling();

            compiler.includeDebugInformation = false;
            CheckCompiling();
        }

        /// <summary>
        /// Check compiling with and without showing progress.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void ShowProgressCompileSourceStringTest()
        {
            compiler.showProgress = true;
            CheckCompiling();

            compiler.showProgress = false;
            CheckCompiling();
        }
    }



    /// <summary>
    /// Tests for auto selected compiler for invalid source code that is described in a string.
    /// </summary>
    public class AutoSelectedCodeCompilerInvalidSourceStringTests : AbstractCompilerInvalidSourceTester
    {
        public AutoSelectedCodeCompilerInvalidSourceStringTests()
        {
            compiler.compilerChoice = CompilerChoice.Auto;

            //turn off using source code files
            compiler.writeSourceFiles = false;
        }

        /// <summary>
        /// Check default parameters of compiling.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void DefaultCompileSourceStringTest()
        {
            CheckCompiling();
        }

        /// <summary>
        /// Check optimized and nonoptimized compiling.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void OptimizedCompileSourceStringTest()
        {
            compiler.optimizeCode = true;
            CheckCompiling();

            compiler.optimizeCode = false;
            CheckCompiling();
        }

        /// <summary>
        /// Check compiling with and without making debug information.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void DebugInfoCompileSourceStringTest()
        {
            compiler.includeDebugInformation = true;
            CheckCompiling();

            compiler.includeDebugInformation = false;
            CheckCompiling();
        }

        /// <summary>
        /// Check compiling with and without showing progress.
        /// </summary>
        [Fact]
        [Trait("Category", "CodeCompilerTest")]
        public void ShowProgressCompileSourceStringTest()
        {
            compiler.showProgress = true;
            CheckCompiling();

            compiler.showProgress = false;
            CheckCompiling();
        }
    }
}