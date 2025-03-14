#if NET
using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using Microsoft.ML.Probabilistic.Models;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests.CodeCompilerTests
{
    public class AssemblyLoadingTests
    {
        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void AssemblyUnloadingTest()
        {
            var assemblies = AppDomain.CurrentDomain.GetAssemblies()
                .Where(a => a.GetName().GetPublicKeyToken().Length == 0)
                .Select(a => a.FullName)
                .ToHashSet();

            RunWithCustomAssemblyLoader();
            for (int i = 0; i < 10; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            var assemblies2 = AppDomain.CurrentDomain.GetAssemblies()
                .Where(a => a.GetName().GetPublicKeyToken().Length == 0)
                .Select(a => a.FullName)
                .ToHashSet();
            var extraAssemblyCount = assemblies2.Except(assemblies).Count();
            Assert.True(extraAssemblyCount == 0, $"There are {extraAssemblyCount} extra assemblies loaded in the current AppDomain. This is likely due to a failure to unload an assembly. Please check the code for any static references to types or methods that may prevent unloading.");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private void RunWithCustomAssemblyLoader()
        {
            var savedAssemblyLoader = InferenceEngine.DefaultEngine.Compiler.AssemblyLoader;
            try
            {
                InferenceEngine.DefaultEngine.Compiler.AssemblyLoader = LoadAssemblyIntoLocalContext;

                new ModelTests().BetaConstructionTest();
            }
            finally
            {
                InferenceEngine.DefaultEngine.Compiler.AssemblyLoader = savedAssemblyLoader;
            }
        }

        /// <summary>
        /// When used as the AssemblyLoader, this method loads the generated assembly into a new AssemblyLoadContext.  Once the InferenceEngine goes out of scope, all generated assemblies will be unloaded.
        /// </summary>
        Assembly LoadAssemblyIntoLocalContext(Stream assemblyStream, Stream pdbStream)
        {
            AssemblyLoadContext assemblyLoadContext = new SimpleAssemblyLoadContext();
            return assemblyLoadContext.LoadFromStream(assemblyStream, pdbStream);
        }

        /// <summary>
        /// An AssemblyLoadContext that loads all dependent assemblies into the default context.
        /// Taken from https://learn.microsoft.com/en-us/dotnet/standard/assembly/unloadability
        /// </summary>
        class SimpleAssemblyLoadContext : AssemblyLoadContext
        {
            public SimpleAssemblyLoadContext() : base(isCollectible: true)
            {
            }

            protected override Assembly Load(AssemblyName name)
            {
                return null;
            }
        }
    }
}
#endif