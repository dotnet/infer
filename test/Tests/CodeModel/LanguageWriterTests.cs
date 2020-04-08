// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Text;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Xunit;
using System.IO;

namespace Microsoft.ML.Probabilistic.Tests.CodeModel
{
    /// <summary>
    /// Various type declaration tests.
    /// </summary>
    public class TestTypeDeclarator
    {
        /// <summary>
        /// Tests type preservation of literals.
        /// </summary>
        public void TestLiteralTypes()
        {
            // double
            TestTypeDeclaration(0d, typeof (double));
            TestTypeDeclaration(-0d, typeof (double));
            TestTypeDeclaration(0.0, typeof (double));
            TestTypeDeclaration(00d, typeof (double));
            TestTypeDeclaration(-0.000, typeof (double));
            TestTypeDeclaration(-0.000e-3, typeof (double));
            TestTypeDeclaration(42.0000000, typeof (double));
            TestTypeDeclaration(3.14159265, typeof (double));
            TestTypeDeclaration(-3.14159265, typeof (double));
            TestTypeDeclaration(1e6, typeof (double));
            TestTypeDeclaration(1.23456789E6, typeof (double));
            TestTypeDeclaration(1.23456789e+42, typeof (double));
            TestTypeDeclaration(1.23456789e-42, typeof (double));
            TestTypeDeclaration(1.23456789e+42d, typeof (double));
            TestTypeDeclaration(-1e6, typeof (double));
            TestTypeDeclaration(-1.23456789E6, typeof (double));
            TestTypeDeclaration(-1.23456789e+42, typeof (double));
            TestTypeDeclaration(-1.23456789e-42, typeof (double));
            TestTypeDeclaration(-1.23456789e+42d, typeof (double));
            TestTypeDeclaration(double.PositiveInfinity, typeof (double));
            TestTypeDeclaration(double.NegativeInfinity, typeof (double));
            TestTypeDeclaration(double.MaxValue, typeof (double));
            TestTypeDeclaration(double.MinValue, typeof (double));
            TestTypeDeclaration(double.NaN, typeof (double));
            TestTypeDeclaration(double.Epsilon, typeof (double));

            // float
            TestTypeDeclaration(0f, typeof (float));
            TestTypeDeclaration(-0f, typeof (float));
            TestTypeDeclaration(0.0f, typeof (float));
            TestTypeDeclaration(00f, typeof (float));
            TestTypeDeclaration(-0.000f, typeof (float));
            TestTypeDeclaration(-0.000e-3f, typeof (float));
            TestTypeDeclaration(42.0000000f, typeof (float));
            TestTypeDeclaration(3.14159265f, typeof (float));
            TestTypeDeclaration(-3.14159265f, typeof (float));
            TestTypeDeclaration(1e6f, typeof (float));
            TestTypeDeclaration(1.23456789E6f, typeof (float));
            TestTypeDeclaration(1.23456789e+38f, typeof (float));
            TestTypeDeclaration(1.23456789e-342f, typeof (float));
            TestTypeDeclaration(-1e6f, typeof (float));
            TestTypeDeclaration(-1.23456789E6f, typeof (float));
            TestTypeDeclaration(-1.23456789e+38f, typeof (float));
            TestTypeDeclaration(-1.23456789e-342f, typeof (float));
            TestTypeDeclaration(float.PositiveInfinity, typeof (float));
            TestTypeDeclaration(float.NegativeInfinity, typeof (float));
            TestTypeDeclaration(float.MaxValue, typeof (float));
            TestTypeDeclaration(float.MinValue, typeof (float));
            TestTypeDeclaration(float.NaN, typeof (float));
            TestTypeDeclaration(float.Epsilon, typeof (float));

            // decimal
            TestTypeDeclaration(0m, typeof (decimal));
            TestTypeDeclaration(-0m, typeof (decimal));
            TestTypeDeclaration(0.0m, typeof (decimal));
            TestTypeDeclaration(00m, typeof (decimal));
            TestTypeDeclaration(-0.000m, typeof (decimal));
            TestTypeDeclaration(-0.000e-3m, typeof (decimal));
            TestTypeDeclaration(42.0000000m, typeof (decimal));
            TestTypeDeclaration(3.14159265m, typeof (decimal));
            TestTypeDeclaration(-3.14159265m, typeof (decimal));
            TestTypeDeclaration(1e6m, typeof (decimal));
            TestTypeDeclaration(1.23456789E6m, typeof (decimal));
            TestTypeDeclaration(1.23456789e+28m, typeof (decimal));
            TestTypeDeclaration(1.23456789e-342m, typeof (decimal));
            TestTypeDeclaration(-1e6m, typeof (decimal));
            TestTypeDeclaration(-1.23456789E6m, typeof (decimal));
            TestTypeDeclaration(-1.23456789e+28m, typeof (decimal));
            TestTypeDeclaration(-1.23456789e-342m, typeof (decimal));
            TestTypeDeclaration(decimal.Zero, typeof (decimal));
            TestTypeDeclaration(decimal.One, typeof (decimal));
            TestTypeDeclaration(decimal.MinusOne, typeof (decimal));
            TestTypeDeclaration(decimal.MaxValue, typeof (decimal));
            TestTypeDeclaration(decimal.MinValue, typeof (decimal));

            // sbyte
            TestTypeDeclaration((sbyte) 0, typeof (sbyte));
            TestTypeDeclaration((sbyte) 00, typeof (sbyte));
            TestTypeDeclaration((sbyte) -0, typeof (sbyte));
            TestTypeDeclaration((sbyte) 127, typeof (sbyte));
            TestTypeDeclaration((sbyte) -128, typeof (sbyte));
            TestTypeDeclaration((sbyte) 0000123, typeof (sbyte));
            TestTypeDeclaration((sbyte) -0000123, typeof (sbyte));
            TestTypeDeclaration(sbyte.MaxValue, typeof (sbyte));
            TestTypeDeclaration(sbyte.MinValue, typeof (sbyte));

            // byte
            TestTypeDeclaration((byte) 0, typeof (byte));
            TestTypeDeclaration((byte) 000, typeof (byte));
            TestTypeDeclaration((byte) 255, typeof (byte));
            TestTypeDeclaration((byte) 0000123, typeof (byte));
            TestTypeDeclaration(byte.MaxValue, typeof (byte));
            TestTypeDeclaration(byte.MinValue, typeof (byte));

            // short
            TestTypeDeclaration((short) 0, typeof (short));
            TestTypeDeclaration((short) 00, typeof (short));
            TestTypeDeclaration((short) -0, typeof (short));
            TestTypeDeclaration((short) 32767, typeof (short));
            TestTypeDeclaration((short) -32768, typeof (short));
            TestTypeDeclaration((short) 000012345, typeof (short));
            TestTypeDeclaration((short) -000012345, typeof (short));
            TestTypeDeclaration(short.MaxValue, typeof (short));
            TestTypeDeclaration(short.MinValue, typeof (short));

            // ushort
            TestTypeDeclaration((ushort) 0, typeof (ushort));
            TestTypeDeclaration((ushort) 000, typeof (ushort));
            TestTypeDeclaration((ushort) 65535, typeof (ushort));
            TestTypeDeclaration((ushort) 000012345, typeof (ushort));
            TestTypeDeclaration(ushort.MaxValue, typeof (ushort));
            TestTypeDeclaration(ushort.MinValue, typeof (ushort));

            // int
            TestTypeDeclaration(0, typeof (int));
            TestTypeDeclaration(00, typeof (int));
            TestTypeDeclaration(-0, typeof (int));
            TestTypeDeclaration(123456789, typeof (int));
            TestTypeDeclaration(-12340, typeof (int));
            TestTypeDeclaration(0000123456789, typeof (int));
            TestTypeDeclaration(-000012340, typeof (int));
            TestTypeDeclaration(int.MaxValue, typeof (int));
            TestTypeDeclaration(int.MinValue, typeof (int));

            // unsigned int
            TestTypeDeclaration(0U, typeof (uint));
            TestTypeDeclaration(000U, typeof (uint));
            TestTypeDeclaration(123456789U, typeof (uint));
            TestTypeDeclaration(0000123456789U, typeof (uint));
            TestTypeDeclaration(uint.MaxValue, typeof (uint));
            TestTypeDeclaration(uint.MinValue, typeof (uint));

            // long
            TestTypeDeclaration(0L, typeof (long));
            TestTypeDeclaration(-0L, typeof (long));
            TestTypeDeclaration(123456789L, typeof (long));
            TestTypeDeclaration(0000123456789L, typeof (long));
            TestTypeDeclaration(-12340L, typeof (long));
            TestTypeDeclaration(-000012340L, typeof (long));
            TestTypeDeclaration(long.MaxValue, typeof (long));
            TestTypeDeclaration(long.MinValue, typeof (long));

            // unsigned long
            TestTypeDeclaration(0UL, typeof (ulong));
            TestTypeDeclaration(000UL, typeof (ulong));
            TestTypeDeclaration(123456789UL, typeof (ulong));
            TestTypeDeclaration(0000123456789UL, typeof (ulong));
            TestTypeDeclaration(ulong.MaxValue, typeof (ulong));
            TestTypeDeclaration(ulong.MinValue, typeof (ulong));
        }

        private void TestTypeDeclaration(Object obj, Type type)
        {
            // Check type equality
            Type objType = obj.GetType();
            if (!objType.Equals(type))
            {
                throw new ArgumentException("Type of " + obj + " is " + objType + ", but is expected to be " + type);
            }
        }
    }

    /// <summary>
    /// Summary description for LanguageWriterTests
    /// </summary>
    public class LanguageWriterTests
    {
        private CodeBuilder builder = CodeBuilder.Instance;

        // TODO: more exhaustive tests
        [Fact]
        [Trait("Category", "CsoftModel")] // needed for thread safety
        public void TestTypeDeclarationRoslyn()
        {
            // Create the specified type declaration using roslyn
            ITypeDeclaration td = CreateTypeDeclaration(typeof (TestTypeDeclarator), "MyTypeDeclarationTest");

            // Compile the type and get generated code
            dynamic gen = CompileTypeDeclaration(td);

            // Test the generated code (run the type tests)
            gen.TestLiteralTypes();
        }

        [Fact]
        [Trait("Category", "CsoftModel")] // needed for thread safety
        public void TestTypeDeclaration()
        {
            // Manually create a type
            ITypeDeclaration td = CreateTypeDeclaration("MyTypeTest");

            // Compile the type and get generated code
            dynamic gen = CompileTypeDeclaration(td);

            // Test the generated code
            gen.MyMethod(666);
        }

        [Fact]
        public void TestLiteralExpressions()
        {
            // double
            Assert.Equal("0.0", WriteLiteralExpression(0.0));
            Assert.Equal("(-1.0 * 0.0)", WriteLiteralExpression(-1.0 * 0.0)); // some build configurations result in -0.0 producing a positive zero, hence (-1.0 * 0.0)
            Assert.Equal("42.0", WriteLiteralExpression(42.0000000));
            Assert.Equal("3.1415926500000002", WriteLiteralExpression(3.14159265));
            Assert.Equal("-3.1415926500000002", WriteLiteralExpression(-3.14159265));
            Assert.Equal("1000000.0", WriteLiteralExpression(1e6));
            Assert.Equal("1234567.8899999999", WriteLiteralExpression(1.23456789E6));
            Assert.Equal("1.2345678900000001E+42", WriteLiteralExpression(1.23456789e+42));
            Assert.Equal("1.23456789E-42", WriteLiteralExpression(1.23456789e-42));
            Assert.Equal("-1000000.0", WriteLiteralExpression(-1e6));
            Assert.Equal("-1234567.8899999999", WriteLiteralExpression(-1.23456789E6));
            Assert.Equal("-1.2345678900000001E+42", WriteLiteralExpression(-1.23456789e+42));
            Assert.Equal("-1.23456789E-42", WriteLiteralExpression(-1.23456789e-42));
            Assert.Equal("double.PositiveInfinity", WriteLiteralExpression(double.PositiveInfinity));
            Assert.Equal("double.NegativeInfinity", WriteLiteralExpression(double.NegativeInfinity));
            Assert.Equal("double.MaxValue", WriteLiteralExpression(double.MaxValue));
            Assert.Equal("double.MinValue", WriteLiteralExpression(double.MinValue));
            Assert.Equal("double.NaN", WriteLiteralExpression(double.NaN));

            // float
            Assert.Equal("0F", WriteLiteralExpression(0F));
            Assert.Equal("(-1F * 0F)", WriteLiteralExpression(-1F * 0F)); // some build configurations result in -0.0 producing a positive zero, hence (-1.0 * 0.0)
            Assert.Equal("42F", WriteLiteralExpression(42.0000000F));
            Assert.Equal("3.14159274F", WriteLiteralExpression(3.14159274F)); // Single precision insufficient to represent 3.14159265 as such.
            Assert.Equal("-3.14159274F", WriteLiteralExpression(-3.14159274F)); // Single precision insufficient to represent -3.14159265 as such.
            Assert.Equal("1000000F", WriteLiteralExpression(1e6F));
            Assert.Equal("1234567.88F", WriteLiteralExpression(1.23456788E6F)); // Single precision insufficient to represent 1234567.89 as such.
            Assert.Equal("1.23456786E+38F", WriteLiteralExpression(1.23456786e+38F)); // Single precision insufficient to represent 1.23456789e+38F as such.
            Assert.Equal("1.23456791E-38F", WriteLiteralExpression(1.23456791e-38F)); // Single precision insufficient to represent 1.23456789e-38F as such.
            Assert.Equal("-1000000F", WriteLiteralExpression(-1e6F));
            Assert.Equal("-1234567.88F", WriteLiteralExpression(-1.23456788E6F)); // Single precision insufficient to represent -1234567.89 as such.
            Assert.Equal("-1.23456786E+38F", WriteLiteralExpression(-1.23456786e+38F)); // Single precision insufficient to represent -1.23456789e+38f as such.
            Assert.Equal("-1.23456791E-38F", WriteLiteralExpression(-1.23456791e-38F)); // Single precision insufficient to represent -1.23456789e-38f as such.
            Assert.Equal("float.PositiveInfinity", WriteLiteralExpression(float.PositiveInfinity));
            Assert.Equal("float.NegativeInfinity", WriteLiteralExpression(float.NegativeInfinity));
            Assert.Equal("float.MaxValue", WriteLiteralExpression(float.MaxValue));
            Assert.Equal("float.MinValue", WriteLiteralExpression(float.MinValue));
            Assert.Equal("float.NaN", WriteLiteralExpression(float.NaN));

            // decimal
            Assert.Equal("0M", WriteLiteralExpression(0M));
            Assert.Equal("0M", WriteLiteralExpression(-0M));
            Assert.Equal("42.0000000M", WriteLiteralExpression(42.0000000M));
            Assert.Equal("3.14159265M", WriteLiteralExpression(3.14159265M));
            Assert.Equal("-3.14159265M", WriteLiteralExpression(-3.14159265M));
            Assert.Equal("1000000M", WriteLiteralExpression(1e6M));
            Assert.Equal("1234567.89M", WriteLiteralExpression(1.23456789E6M));
            Assert.Equal("12345678900000000000000000000M", WriteLiteralExpression(1.23456789e+28M));
            Assert.Equal("0.0000000000000000000000000000M", WriteLiteralExpression(1.23456789e-38M));
            Assert.Equal("-1000000M", WriteLiteralExpression(-1e6M));
            Assert.Equal("-1234567.89M", WriteLiteralExpression(-1.23456789E6M));
            Assert.Equal("-12345678900000000000000000000M", WriteLiteralExpression(-1.23456789e+28M));
            Assert.Equal("0.0000000000000000000000000000M", WriteLiteralExpression(-1.23456789e-38M));
            Assert.Equal("0M", WriteLiteralExpression(decimal.Zero));
            Assert.Equal("1M", WriteLiteralExpression(decimal.One));
            Assert.Equal("-1M", WriteLiteralExpression(decimal.MinusOne));
            Assert.Equal("decimal.MaxValue", WriteLiteralExpression(decimal.MaxValue));
            Assert.Equal("decimal.MinValue", WriteLiteralExpression(decimal.MinValue));

            // sbyte
            Assert.Equal("((sbyte)0)", WriteLiteralExpression((sbyte) 0));
            Assert.Equal("((sbyte)0)", WriteLiteralExpression((sbyte) -0));
            Assert.Equal("((sbyte)127)", WriteLiteralExpression((sbyte) 127));
            Assert.Equal("((sbyte)-128)", WriteLiteralExpression((sbyte) -128));
            Assert.Equal("((sbyte)123)", WriteLiteralExpression((sbyte) 000123));
            Assert.Equal("((sbyte)-123)", WriteLiteralExpression((sbyte) -000123));
            Assert.Equal("((sbyte)127)", WriteLiteralExpression(sbyte.MaxValue));
            Assert.Equal("((sbyte)-128)", WriteLiteralExpression(sbyte.MinValue));

            // byte
            Assert.Equal("((byte)0)", WriteLiteralExpression((byte) 0));
            Assert.Equal("((byte)0)", WriteLiteralExpression((byte) -0));
            Assert.Equal("((byte)255)", WriteLiteralExpression((byte) 255));
            Assert.Equal("((byte)123)", WriteLiteralExpression((byte) 000123));
            Assert.Equal("((byte)255)", WriteLiteralExpression(byte.MaxValue));
            Assert.Equal("((byte)0)", WriteLiteralExpression(byte.MinValue));

            // short
            Assert.Equal("((short)0)", WriteLiteralExpression((short) 0));
            Assert.Equal("((short)0)", WriteLiteralExpression((short) -0));
            Assert.Equal("((short)32767)", WriteLiteralExpression((short) 32767));
            Assert.Equal("((short)-32768)", WriteLiteralExpression((short) -32768));
            Assert.Equal("((short)12345)", WriteLiteralExpression((short) 00012345));
            Assert.Equal("((short)-12345)", WriteLiteralExpression((short) -00012345));
            Assert.Equal("((short)32767)", WriteLiteralExpression(short.MaxValue));
            Assert.Equal("((short)-32768)", WriteLiteralExpression(short.MinValue));

            // unsigned short
            Assert.Equal("((ushort)0)", WriteLiteralExpression((ushort) 0));
            Assert.Equal("((ushort)0)", WriteLiteralExpression((ushort) -0));
            Assert.Equal("((ushort)65535)", WriteLiteralExpression((ushort) 65535));
            Assert.Equal("((ushort)12345)", WriteLiteralExpression((ushort) 00012345));
            Assert.Equal("((ushort)65535)", WriteLiteralExpression(ushort.MaxValue));
            Assert.Equal("((ushort)0)", WriteLiteralExpression(ushort.MinValue));

            // int
            Assert.Equal("0", WriteLiteralExpression(0));
            Assert.Equal("0", WriteLiteralExpression(-0));
            Assert.Equal("123456789", WriteLiteralExpression(123456789));
            Assert.Equal("-12340", WriteLiteralExpression(-12340));
            Assert.Equal("int.MaxValue", WriteLiteralExpression(int.MaxValue));
            Assert.Equal("int.MinValue", WriteLiteralExpression(int.MinValue));

            // unsigned int
            Assert.Equal("0U", WriteLiteralExpression(0U));
            Assert.Equal("123456789U", WriteLiteralExpression(123456789U));
            Assert.Equal("uint.MaxValue", WriteLiteralExpression(uint.MaxValue));
            Assert.Equal("0U", WriteLiteralExpression(uint.MinValue));

            // long
            Assert.Equal("0L", WriteLiteralExpression(0L));
            Assert.Equal("0L", WriteLiteralExpression(-0L));
            Assert.Equal("123456789L", WriteLiteralExpression(123456789L));
            Assert.Equal("-12340L", WriteLiteralExpression(-12340L));
            Assert.Equal("long.MaxValue", WriteLiteralExpression(long.MaxValue));
            Assert.Equal("long.MinValue", WriteLiteralExpression(long.MinValue));

            // unsigned long
            Assert.Equal("0UL", WriteLiteralExpression(0UL));
            Assert.Equal("123456789UL", WriteLiteralExpression(123456789UL));
            Assert.Equal("ulong.MaxValue", WriteLiteralExpression(ulong.MaxValue));
            Assert.Equal("0UL", WriteLiteralExpression(ulong.MinValue));
        }

        #region Helper

        /// <summary>
        /// Creates a type declaration from a given type.
        /// </summary>
        /// <param name="aType">A type.</param>
        /// <param name="name">A name which is given to the type declaration.</param>
        /// <returns>The type declaration of the specified type.</returns>
        private ITypeDeclaration CreateTypeDeclaration(Type aType, string name)
        {
            // Get the type declaration using roslyn
            ITypeDeclaration td = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance.GetTypeDeclaration(aType, true);
            td.Visibility = TypeVisibility.Public;
            td.Namespace = "TestNameSpace";
            td.Interface = false;
            td.Name = name;

            return td;
        }

        /// <summary>
        /// Declares a type with a specific name.
        /// </summary>
        /// <param name="name">The name of the declared type.</param>
        /// <returns>A type declaration.</returns>
        private ITypeDeclaration CreateTypeDeclaration(string name)
        {
            ITypeDeclaration td = builder.TypeDecl();
            td.Visibility = TypeVisibility.Public;
            td.Namespace = "TestNameSpace";
            td.Interface = false;
            td.Name = name;

            // Derive from an interface
            td.Interfaces.Add((ITypeReference) builder.TypeRef(typeof (ICloneable)));

            // Add some methods
            IParameterDeclaration parm1 = builder.Param("val", typeof (int));
            IMethodDeclaration meth1 = builder.MethodDecl(MethodVisibility.Public, "MyMethod", typeof (int), td, parm1);
            IMethodDeclaration meth2 = builder.MethodDecl(MethodVisibility.Public, "Clone", typeof (object), td);
            IMethodDeclaration meth3 = builder.MethodDecl(MethodVisibility.Public, "ThrowMethod", typeof (void), td);
            IBlockStatement im1bs = builder.BlockStmt();
            IBlockStatement im2bs = builder.BlockStmt();
            IBlockStatement im3bs = builder.BlockStmt();
            im1bs.Statements.Add(builder.Return(builder.ParamRef(parm1)));
            meth1.Body = im1bs;
            im2bs.Statements.Add(builder.Return(builder.LiteralExpr(null)));
            meth2.Body = im2bs;
            im3bs.Statements.Add(builder.ThrowStmt(builder.NewObject(typeof(InferCompilerException))));
            meth3.Body = im3bs;
            td.Methods.Add(meth1);
            td.Methods.Add(meth2);
            td.Methods.Add(meth3);

            IGenericParameter gp1 = builder.GenericTypeParam("T1");
            IGenericParameter gp2 = builder.GenericTypeParam("T2");
            IList<IGenericParameter> gps = new List<IGenericParameter>();
            gps.Add(gp1);
            gps.Add(gp2);
            IParameterDeclaration gparm1 = builder.Param("parm1", gp1);
            IParameterDeclaration gparm2 = builder.Param("parm2", gp2);
            IMethodDeclaration meth4 = builder.GenericMethodDecl(
                MethodVisibility.Public, "MyGenericMethod", gp1, td, gps, gparm1, gparm2);
            IBlockStatement im4bs = builder.BlockStmt();
            im4bs.Statements.Add(builder.Return(builder.ParamRef(gparm1)));
            meth4.Body = im4bs;
            td.Methods.Add(meth4);

            // Add some properties
            IPropertyDeclaration prop1 = builder.PropDecl("MyIntProperty", typeof (int), td, MethodVisibility.Public);
            IPropertyDeclaration prop2 = builder.PropDecl("MyObjProperty", typeof (object), td, MethodVisibility.Public);
            IBlockStatement ip1bs = builder.BlockStmt();
            IBlockStatement ip2bs = builder.BlockStmt();
            ip1bs.Statements.Add(builder.Return(builder.LiteralExpr(1)));
            ((IMethodDeclaration) prop1.GetMethod).Body = ip1bs;
            ip2bs.Statements.Add(builder.Return(builder.LiteralExpr(null)));
            ((IMethodDeclaration) prop2.GetMethod).Body = ip2bs;
            td.Properties.Add(prop1);
            td.Properties.Add(prop2);

            // Add an event
            IEventDeclaration event1 = builder.EventDecl("MyEvent", (ITypeReference) builder.TypeRef(typeof (Action<object>)), td);
            td.Events.Add(event1);

            // Build a wrapper function that allows clients to fire the event
            IMethodDeclaration fireEventMethod = builder.FireEventDecl(MethodVisibility.Public, "MyEventWrapper", event1);
            td.Methods.Add(fireEventMethod);

            return td;
        }

        /// <summary>
        /// Compiles a type declaration and returns its generated code.
        /// </summary>
        /// <param name="td">A type declaration.</param>
        /// <returns>The code generated for the specified type declaration.</returns>
        private dynamic CompileTypeDeclaration(ITypeDeclaration td)
        {
            // Get the source code
            StringWriter sw = new StringWriter();
            ILanguageWriter lw = new CSharpWriter() as ILanguageWriter;
            SourceNode sn = lw.GenerateSource(td);
            LanguageWriter.WriteSourceNode(sw, sn);
            String sourceCode = sw.ToString();

            // Compile the code

            // Delete duplicates
            var allAssemblies = AppDomain.CurrentDomain.GetAssemblies().GroupBy(x => x.FullName).Select(x => x.First());

            var cc = new CodeCompiler();
            cc.includeDebugInformation = true;
            cc.generateInMemory = true;
            cc.writeSourceFiles = false;
            cc.compilerChoice = CompilerChoice.Auto;
            CompilerResults cr = cc.Compile(null, new List<string>() { sourceCode }, allAssemblies.ToArray());
            foreach (string err in cr.Errors) Console.WriteLine(err);

            // There should be no errors
            Assert.Empty(cr.Errors);

            // Now get the generated code
            Type inferenceType = CodeCompiler.GetCompiledType(cr, td);
            dynamic gen = Activator.CreateInstance(inferenceType);

            return gen;
        }

        /// <summary>
        /// Returns a literal expression as a string.
        /// </summary>
        /// <param name="value">The literal expression.</param>
        /// <returns>The string of the specified literal expression.</returns>
        private string WriteLiteralExpression(object value)
        {
            MyCSharpWriter writer = new MyCSharpWriter();
            return writer.AppendLiteralExpression(builder.LiteralExpr(value));
        }

        /// <summary>
        /// Wraps CSharpWriter.
        /// </summary>
        private class MyCSharpWriter : CSharpWriter
        {
            public string AppendLiteralExpression(ILiteralExpression ile)
            {
                StringBuilder sb = new StringBuilder();
                this.AppendLiteralExpression(sb, ile);
                return sb.ToString();
            }
        }

        #endregion
    }
}