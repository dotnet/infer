// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler;
using System;
using System.IO;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests.CodeCompilerTests
{
    /// <summary>
    /// Tests for Code DOM compiling of valid source code files.
    /// </summary>
    public class CodeDomCodeCompilerValidSourceFileTests : AbstractCompilerValidSourceTester
    {
        public CodeDomCodeCompilerValidSourceFileTests()
        {
            for (int i = 0; i < sourceCodes.Count; i++)
            {
                using (var writer = new StreamWriter(fileNames[i]))
                {
                    writer.Write(sourceCodes[i]);
                }
            }

            compiler.compilerChoice = CompilerChoice.CodeDom;

            //this flag let use the code from given source files
            compiler.writeSourceFiles = true;
        }

        protected override void CheckCompiling()
        {
#if NETFRAMEWORK
            base.CheckCompiling();
#else
            Assert.Throws<PlatformNotSupportedException>(() => compiler.Compile(fileNames, sourceCodes, assemblies));
#endif
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
    /// Tests for Code DOM compiling of valid source code files.
    /// </summary>
    public class CodeDomCodeCompilerInvalidSourceFileTests : AbstractCompilerInvalidSourceTester
    {
        public CodeDomCodeCompilerInvalidSourceFileTests()
        {
            for (int i = 0; i < sourceCodes.Count; i++)
            {
                using (var writer = new StreamWriter(fileNames[i]))
                {
                    writer.Write(sourceCodes[i]);
                }
            }

            compiler.compilerChoice = CompilerChoice.CodeDom;

            //this flag let use the code from given source files
            compiler.writeSourceFiles = true;
        }

        protected override void CheckCompiling()
        {
#if NETFRAMEWORK
            base.CheckCompiling();
#else
            Assert.Throws<PlatformNotSupportedException>(() => compiler.Compile(fileNames, sourceCodes, assemblies));
#endif
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
    /// Tests for Code DOM compiling of valid source code that is described in a string.
    /// </summary>
    public class CodeDomCodeCompilerValidSourceStringTests : AbstractCompilerValidSourceTester
    {
        public CodeDomCodeCompilerValidSourceStringTests()
        {
            compiler.compilerChoice = CompilerChoice.CodeDom;

            //turn off using existing source files
            compiler.writeSourceFiles = false;
        }

        protected override void CheckCompiling()
        {
#if NETFRAMEWORK
            base.CheckCompiling();
#else
            Assert.Throws<PlatformNotSupportedException>(() => compiler.Compile(fileNames, sourceCodes, assemblies));
#endif
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
    /// Tests for Code DOM compiling of invalid source code that is described in a string.
    /// </summary>
    public class CodeDomCodeCompilerInvalidSourceStringTests : AbstractCompilerInvalidSourceTester
    {
        public CodeDomCodeCompilerInvalidSourceStringTests()
        {
            compiler.compilerChoice = CompilerChoice.CodeDom;

            //turn off using source code files
            compiler.writeSourceFiles = false;
        }

        protected override void CheckCompiling()
        {
#if NETFRAMEWORK
            base.CheckCompiling();
#else
            Assert.Throws<PlatformNotSupportedException>(() => compiler.Compile(fileNames, sourceCodes, assemblies));
#endif
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