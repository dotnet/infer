// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.CodeDom.Compiler;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Collections;
#if ROSLYN
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;
#endif

namespace Microsoft.ML.Probabilistic.Compiler
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Transforms type declarations.
    /// </summary>
    /// TODO: Add GeneratedCode attribute to results.
    public class CodeTransformer
    {
        public Dictionary<ITypeDeclaration, ITypeDeclaration> transformMap = new Dictionary<ITypeDeclaration, ITypeDeclaration>();
        public ICodeTransform Transform { get; }

        public bool TrackTransform
        {
            get { return ((BasicTransformContext)Transform.Context).trackTransform; }
            set { ((BasicTransformContext)Transform.Context).trackTransform = value; }
        }

        public CodeTransformer(ICodeTransform t)
        {
            Transform = t;
        }

        public string GetFriendlyName()
        {
            string text = Transform.Name;
            if (text.EndsWith("Transform")) text = text.Substring(0, text.Length - 9); // +" Transform";
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < text.Length; i++)
            {
                if ((i > 0) && Char.IsUpper(text[i]) && Char.IsLower(text[i - 1])) sb.Append(" ");
                sb.Append(text[i]);
            }
            return sb.ToString();
        }

        public List<ITypeDeclaration> TransformToDeclaration(ITypeDeclaration typeDecl)
        {
            Stack<ITypeDeclaration> typesToTransform = new Stack<ITypeDeclaration>();
            typesToTransform.Push(typeDecl);

            HashSet<ITypeDeclaration> typesTransformed = new HashSet<ITypeDeclaration>();
            List<ITypeDeclaration> typeDeclarations = new List<ITypeDeclaration>();
            while (typesToTransform.Count > 0)
            {
                ITypeDeclaration itd = typesToTransform.Pop();
                ITypeDeclaration td = Transform.Transform(itd);
                transformMap[itd] = td;
                typeDeclarations.Add(td);
                typesTransformed.Add(itd);
                foreach (ITypeDeclaration t2 in Transform.Context.TypesToTransform)
                    if (!typesTransformed.Contains(t2))
                        typesToTransform.Push(t2);
            }
            return typeDeclarations;
        }

        public bool OutputEqualsInput
        {
            get
            {
                bool areEqual = true;
                foreach (KeyValuePair<ITypeDeclaration, ITypeDeclaration> entry in transformMap)
                {
                    if (!ReferenceEquals(entry.Key, entry.Value)) areEqual = false;
                }
                return areEqual;
            }
        }
    }

    public class CodeCompiler
    {
        /// <summary>
        /// The language that we'll compile
        /// </summary>
        protected string language = "C#";
        protected bool runningOnNetCore = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription.StartsWith(".NET Core", StringComparison.OrdinalIgnoreCase)
            || System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription.StartsWith(".NET 5", StringComparison.OrdinalIgnoreCase)
            || System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription.StartsWith(".NET 6", StringComparison.OrdinalIgnoreCase)
            || System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription.StartsWith(".NET 7", StringComparison.OrdinalIgnoreCase)
            || System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription.StartsWith(".NET 8", StringComparison.OrdinalIgnoreCase);
        protected bool runningOnMono = Type.GetType("Mono.Runtime") != null;

        /// <summary>
        /// If true, source code files are written out for each transformed class.
        /// </summary>
        public bool writeSourceFiles = true;

        /// <summary>
        /// If true, existing source code files are used.
        /// </summary>
        public bool useExistingFiles = false;

        /// <summary>
        /// If true, source code files are written out for each transformed class.
        /// </summary>
        public bool generateInMemory = true;

        /// <summary>
        /// The absolute or relative path to the generated source code.
        /// </summary>
        public string GeneratedSourceFolder = "GeneratedSource";

        /// <summary>
        /// If true, print messages about compilation progress.
        /// </summary>
        public bool showProgress;


        /// <summary>
        /// If true, causes the C# compiler to generate optimized code
        /// </summary>
        public bool optimizeCode = false;

        /// <summary>
        /// If true, causes the C# compiler to include debug information in the generated DLL.
        /// </summary>
        public bool includeDebugInformation = true;

        /// <summary>
        /// Compiler to use: Roslyn or CodeDom.Compiler. Auto means CodeDom.Compiler on .NET full / Windows, and Roslyn on .NET Core and Mono.
        /// </summary>
        public CompilerChoice compilerChoice = CompilerChoice.Auto;

        public List<string> WriteSource(List<ITypeDeclaration> typeDeclarations, IList<string> filenames, out ICollection<Assembly> referencedAssemblies)
        {
            Stopwatch watch = null;
            if (showProgress)
            {
                watch = new Stopwatch();
                watch.Start();
            }
            List<string> sources = new List<string>();
            bool needFilenames = (!this.generateInMemory) || this.writeSourceFiles;
            if (needFilenames)
            {
                string dirname = Path.GetFullPath(GeneratedSourceFolder);
                Directory.CreateDirectory(dirname);
            }
            referencedAssemblies = new Set<Assembly>();
            foreach (ITypeDeclaration td in typeDeclarations)
            {
                StringWriter sw = new StringWriter();
                ILanguageWriter lw = new CSharpWriter() as ILanguageWriter;
                SourceNode sn = lw.GenerateSource(td);
                LanguageWriter.WriteSourceNode(sw, sn);
                referencedAssemblies.AddRange(lw.ReferencedAssemblies);
                String sourceCode = sw.ToString();
                sources.Add(sourceCode);
                if (needFilenames)
                {
                    string filename = Path.GetFullPath(GetFilenameForType(GeneratedSourceFolder, td, ".cs"));
                    filenames.Add(filename);
                    if (writeSourceFiles)
                    {
                        if (useExistingFiles && File.Exists(filename))
                        {
                            if (showProgress)
                                Console.Write("Using existing source file: {0} ", filename);
                        }
                        else
                        {
                            if (showProgress)
                                Console.Write("Writing source file: {0} ", filename);
                            StreamWriter stw = new StreamWriter(filename);
                            stw.Write(sourceCode);
                            stw.Close();
                        }
                    }
                }
            }
            // sometimes we need to reference assemblies that are not directly referenced in the code,
            // as explained at: http://msdn.microsoft.com/en-us/library/yabyz3h4.aspx
            // If you reference an assembly (Assembly A) that references another assembly (Assembly B), you will need to reference Assembly B if: 
            //   1. A type you use from Assembly A inherits from a type or implements an interface from Assembly B.
            //   2. You invoke a field, property, event, or method that has a return type or parameter type from Assembly B.
            // Rather than check for all this, we add references to all loaded assemblies referenced by referenced assemblies.
            Set<AssemblyName> loadedAssemblies = new Set<AssemblyName>(new AssemblyNameComparer());
            loadedAssemblies.AddRange(AppDomain.CurrentDomain.GetAssemblies().ListSelect(assembly => assembly.GetName()));
            Stack<Assembly> assemblyStack = new Stack<Assembly>();
            foreach (Assembly assembly in referencedAssemblies)
            {
                assemblyStack.Push(assembly);
            }
            while (assemblyStack.Count != 0)
            {
                Assembly assembly = assemblyStack.Pop();
                foreach (AssemblyName name in assembly.GetReferencedAssemblies())
                {
                    if (loadedAssemblies.Contains(name))
                    {
                        Assembly referenced = Assembly.Load(name);
                        if (!referencedAssemblies.Contains(referenced))
                        {
                            referencedAssemblies.Add(referenced);
                            assemblyStack.Push(referenced);
                        }
                    }
                }
            }
            if (showProgress)
            {
                watch.Stop();
                Console.WriteLine("({0}ms)", watch.ElapsedMilliseconds);
            }
            return sources;
        }

        public class AssemblyNameComparer : IEqualityComparer<AssemblyName>
        {
            public bool Equals(AssemblyName x, AssemblyName y)
            {
                return AssemblyName.ReferenceMatchesDefinition(x, y);
            }

            public int GetHashCode(AssemblyName obj)
            {
                return obj.Name.GetHashCode();
            }
        }

        public static string GetFilenameForType(string folder, ITypeDeclaration td, string suffix)
        {
            //string fname = td.Namespace + "." + td.Name + suffix;
            //       dropped the namespace from the generated filename, 
            //       since this is always the same through the modelling API.
            string fname = /*td.Namespace + "." +*/ td.Name + suffix;
            foreach (char ch in Path.GetInvalidFileNameChars())
            {
                fname = fname.Replace(ch, '_');
            }
            string filename = Path.Combine(folder, fname);
            return filename;
        }

        public CompilerResults WriteAndCompile(List<ITypeDeclaration> typeDeclarations)
        {
            List<string> filenames = new List<string>();
            ICollection<Assembly> referencedAssemblies;
            List<string> sources = WriteSource(typeDeclarations, filenames, out referencedAssemblies);
            return Compile(filenames, sources, referencedAssemblies);
        }

        public static Type GetCompiledType(CompilerResults cr, ITypeDeclaration decl)
        {
            //string s = Builder.ToTypeName(decl);
            string s = decl.Namespace + "." + decl.Name;
            Type type = cr.CompiledAssembly.GetType(s);
            if (type == null) throw new CompilationFailedException("Type '" + s + "' not found in the compiled assembly "+cr.CompiledAssembly);
            return type;
        }

        public CompilerResults Compile(List<string> filenames, List<string> sources, ICollection<Assembly> referencedAssemblies)
        {
            Stopwatch watch = null;
            if (showProgress)
            {
                Console.Write("Invoking " + language + " compiler ");
                watch = new Stopwatch();
                watch.Start();
            }
            CompilerResults cr;
            try
            {
                switch (compilerChoice)
                {
                    case CompilerChoice.CodeDom:
                        cr = CompileWithCodeDom(filenames, sources, referencedAssemblies);
                        break;
                    case CompilerChoice.Roslyn:
                        cr = CompileWithRoslyn(filenames, sources, referencedAssemblies);
                        break;
                    case CompilerChoice.Auto:
                        // Use CodeDom whenever possible because Roslyn takes a long time to load the first time.
                        // The load time could be reduced by running NGen on Roslyn (and its dependencies), 
                        // but that is complicated and requires administrator permission.
                        if (runningOnNetCore || runningOnMono)
                            cr = CompileWithRoslyn(filenames, sources, referencedAssemblies);
                        else
                            cr = CompileWithCodeDom(filenames, sources, referencedAssemblies);
                        break;
                    default:
                        throw new NotSupportedException($"Compiler choice {compilerChoice} is not supported.");
                }
            }
            catch (PlatformNotSupportedException e)
            {
                throw new PlatformNotSupportedException($"Current platform is not supported by the current compiler choice {compilerChoice}. Try a different one.", e);
            }

            if (showProgress)
            {
                watch.Stop();
                Console.WriteLine("({0}ms)", watch.ElapsedMilliseconds);
            }
            return cr;
        }

        private CompilerResults CompileWithCodeDom(List<string> filenames, List<string> sources, ICollection<Assembly> referencedAssemblies)
        {
#if CODEDOM
            CodeDomProvider provider = CodeDomProvider.CreateProvider(language);
            // A special provider is required to compile code using C# 6.  
            // However, it requires a reference to the following NuGet package:
            // https://www.nuget.org/packages/Microsoft.CodeDom.Providers.DotNetCompilerPlatform/
            //CodeDomProvider provider = new Microsoft.CodeDom.Providers.DotNetCompilerPlatform.CSharpCodeProvider();
            CompilerParameters cp = new CompilerParameters();
            cp.IncludeDebugInformation = includeDebugInformation;
            if (optimizeCode)
            {
                cp.CompilerOptions = "/optimize ";
            }

            // If running on the Mono VM, disable all warnings in the generated
            // code because these are treated as errors by the Mono compiler.
            if (runningOnMono)
                cp.CompilerOptions += "/warn:0 ";

            List<string> assemblyNames = new List<string>();
            foreach (Assembly assembly in referencedAssemblies)
            {
                try
                {
                    assemblyNames.Add(assembly.Location);
                }
                catch (NotSupportedException)
                {
                    // do nothing - this assembly does not have a location e.g. it is in memory
                }
                catch (Exception e)
                {
                    Console.WriteLine("Warning, could not add location for assembly: " + assembly);
                    Console.WriteLine(e);
                }
            }
            // we must only add references to assemblies that are actually referenced by the code -- not all the assemblies in memory during model compilation (that could be a much larger set)
            foreach (string s in assemblyNames)
            {
                cp.ReferencedAssemblies.Add(s);
            }
            System.CodeDom.Compiler.CompilerResults cr;
            cp.GenerateInMemory = generateInMemory;
            if (!cp.GenerateInMemory)
            {
                cp.OutputAssembly = Path.ChangeExtension(filenames[0], ".dll");
                try
                {
                    if (File.Exists(cp.OutputAssembly)) File.Delete(cp.OutputAssembly);
                }
                catch
                {
                    for (int i = 0; ; i++)
                    {
                        cp.OutputAssembly = Path.ChangeExtension(Path.ChangeExtension(filenames[0], null) + DateTime.Now.Millisecond, ".dll");
                        try
                        {
                            if (File.Exists(cp.OutputAssembly)) File.Delete(cp.OutputAssembly);
                            break;
                        }
                        catch
                        {
                        }
                    }
                }
            }
            // TODO: allow compiled assemblies to be unloaded, by compiling them as Add-ins (see AddInAttribute)
            if (!writeSourceFiles)
            {
                cr = provider.CompileAssemblyFromSource(cp, sources.ToArray());
            }
            else
            {
                cr = provider.CompileAssemblyFromFile(cp, filenames.ToArray());
            }
            List<string> errors = new List<string>(cr.Errors.Count);
            foreach (var error in cr.Errors)
            {
                errors.Add(error.ToString());
            }
            Assembly compiledAssembly;
            try
            {
                compiledAssembly = cr.CompiledAssembly;
            }
            catch
            {
                compiledAssembly = null;
            }
            return new CompilerResults(compiledAssembly, errors.Count == 0, errors);
#else
            throw new NotSupportedException("This assembly was compiled without CodeDom support.  To use CodeDom, recompile with the CODEDOM compiler flag.");
#endif
        }

        private CompilerResults CompileWithRoslyn(List<string> filenames, List<string> sources, ICollection<Assembly> referencedAssemblies)
        {
#if ROSLYN
            CompilerResults cr;
            CSharpCompilationOptions options = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary);
            EmitOptions emitOptions = new EmitOptions();
            if (includeDebugInformation)
            {
                // Embedded option doesn't seem to work
                // Explanation of options: https://github.com/dotnet/designs/blob/main/accepted/diagnostics/debugging-with-symbols-and-sources.md
                emitOptions = emitOptions.WithDebugInformationFormat(DebugInformationFormat.PortablePdb);
            }
            else
            {
                emitOptions = emitOptions.WithDebugInformationFormat(DebugInformationFormat.Embedded);
            }
            if (optimizeCode)
            {
                options = options.WithOptimizationLevel(OptimizationLevel.Release);
            }

            // If running on the Mono VM, disable all warnings in the generated
            // code because these are treated as errors by the Mono compiler.
            if (runningOnMono)
            {
                options = options.WithWarningLevel(0);
            }

            List<MetadataReference> references = new List<MetadataReference>();
            foreach (Assembly assembly in referencedAssemblies)
            {
                try
                {
                    references.Add(MetadataReference.CreateFromFile(assembly.Location));
                }
                catch (NotSupportedException)
                {
                    // do nothing - this assembly does not have a location e.g. it is in memory
                }
                catch (Exception e)
                {
                    Console.WriteLine("Warning, could not add location for assembly: " + assembly);
                    Console.WriteLine(e);
                }
            }
            // we must only add references to assemblies that are actually referenced by the code -- not all the assemblies in memory during model compilation (that could be a much larger set)

            SyntaxTree[] trees;
            if (!writeSourceFiles)
            {
                // Note: Without source files, the debugger cannot step into generated code, even if a pdb is generated.
                // Theoretically, it should be possible to embed the source into the pdb.
                trees = sources.Select(s => CSharpSyntaxTree.ParseText(s, encoding: Encoding.UTF8)).ToArray();
            }
            else
            {
                trees = filenames.Select(fn => CSharpSyntaxTree.ParseText(File.ReadAllText(fn), null, fn, Encoding.UTF8)).ToArray();
            }

            string targetAssemblyPath = "";
            string pdbPath = "";
            string targetAssemblyName;
            if (!generateInMemory)
            {
                targetAssemblyPath = Path.ChangeExtension(filenames[0], ".dll");
                try
                {
                    if (File.Exists(targetAssemblyPath)) File.Delete(targetAssemblyPath);
                }
                catch
                {
                    for (int i = 0; ; i++)
                    {
                        targetAssemblyPath = Path.ChangeExtension(Path.ChangeExtension(filenames[0], null) + DateTime.Now.Millisecond, ".dll");
                        try
                        {
                            if (File.Exists(targetAssemblyPath)) File.Delete(targetAssemblyPath);
                            break;
                        }
                        catch
                        {
                        }
                    }
                }
                targetAssemblyName = Path.GetFileNameWithoutExtension(targetAssemblyPath);
                pdbPath = Path.ChangeExtension(targetAssemblyPath, ".pdb");
                emitOptions = emitOptions.WithPdbFilePath(pdbPath);
            }
            else
            {
                targetAssemblyName = Guid.NewGuid().ToString(); // Empty names are not allowed
            }

            Compilation compilation = CSharpCompilation.Create(targetAssemblyName, trees, references, options);
            Assembly resultAssembly = null;
            EmitResult result;
            using (Stream assemblyStream = generateInMemory ? (Stream)new MemoryStream() : File.Create(targetAssemblyPath))
            {
                using (Stream pdbStream = generateInMemory ? (Stream)new MemoryStream() : File.Create(pdbPath))
                {
                    if (emitOptions.DebugInformationFormat == DebugInformationFormat.Embedded)
                    {
                        result = compilation.Emit(assemblyStream, options: emitOptions);
                    }
                    else
                    {
                        result = compilation.Emit(assemblyStream, pdbStream, options: emitOptions);
                    }
                    if (result.Success)
                    {
                        // TODO: allow compiled assemblies to be unloaded
                        assemblyStream.Seek(0, SeekOrigin.Begin);
                        var asmBin = new BinaryReader(assemblyStream).ReadBytes((int)assemblyStream.Length);
                        if (emitOptions.DebugInformationFormat == DebugInformationFormat.Embedded)
                        {
                            resultAssembly = Assembly.Load(asmBin);
                        }
                        else
                        {
                            pdbStream.Seek(0, SeekOrigin.Begin);
                            var pdbBin = new BinaryReader(pdbStream).ReadBytes((int)pdbStream.Length);
                            resultAssembly = Assembly.Load(asmBin, pdbBin);
                        }
                    }
                }
            }
            cr = new CompilerResults(resultAssembly, result.Success, result.Diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).Select(d => d.ToString()).ToList());
            return cr;
#else
            throw new NotSupportedException("This assembly was compiled without Roslyn support.  To use Roslyn, recompile with the ROSLYN compiler flag.");
#endif
        }

        public static string ExpressionToString(IExpression expr)
        {
            ILanguageWriter lw = new CSharpWriter() as ILanguageWriter;
            return lw.ExpressionSource(expr);
        }

        internal static string DeclarationToString(ITypeDeclaration declaration)
        {
            try
            {
                StringWriter sw = new StringWriter();
                ILanguageWriter lw = new CSharpWriter() as ILanguageWriter;
                SourceNode sn = lw.GenerateSource(declaration);
                LanguageWriter.WriteSourceNode(sw, sn);
                sw.Close();
                string s = sw.ToString();
                return s;
            }
            catch (Exception ex)
            {
                return "[ERROR WRITING DECLARATION: " + ex.Message + "]"; // +ist;
            }
        }
    }

    public class CompilerResults
    {
        public CompilerResults(Assembly compiledAssembly, bool success, List<string> errors)
        {
            CompiledAssembly = compiledAssembly;
            Success = success;
            Errors = errors;
        }

        public Assembly CompiledAssembly { get; }
        public List<string> Errors { get; }
        public bool Success { get; }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}