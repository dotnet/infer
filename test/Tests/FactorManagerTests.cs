// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Utilities;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Microsoft.ML.Probabilistic.Models;
    using Assert = Xunit.Assert;
    using Microsoft.ML.Probabilistic.Compiler.Reflection;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Compiler;

    public class FactorManagerTests
    {
        private FactorManager factorManager = new FactorManager();

        //[Fact]
        // should take 140ms
        // turn on PrintStatistics in Binding.cs
        internal void SpeedTest()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Factor), "ReplicateWithMarginal<>", typeof(bool[])).GetMethodInfo());
            IDictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            Type ba = typeof(DistributionStructArray<Bernoulli, bool>);
            Type baa = typeof(DistributionRefArray<DistributionStructArray<Bernoulli, bool>, bool[]>);
            parameterTypes["Uses"] = baa;
            parameterTypes["Def"] = ba;
            parameterTypes["Marginal"] = ba;
            parameterTypes["result"] = baa;
            MessageFcnInfo fcninfo;
            fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Uses", parameterTypes);

            parameterTypes["result"] = ba;
            parameterTypes["resultIndex"] = typeof(int);
            fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Uses", parameterTypes);
        }

        /// <summary>
        /// Test that we can reflect on method parameter names.
        /// </summary>
        [Fact]
        public void ParameterNameTest()
        {
            MethodInfo method = typeof(Gaussian).GetMethod("GetMeanAndVariance");
            ParameterInfo[] parameters = method.GetParameters();
            Assert.Equal("mean", parameters[0].Name);
            Assert.Equal("variance", parameters[1].Name);
        }

        [Fact]
        public void MethodReferenceTest()
        {
            MethodInfo method;
            try
            {
                method = (new MethodReference(typeof(Array), "Copy")).GetMethodInfo();
                Assert.True(false, "AmbiguousMatchException not thrown");
            }
            catch (AmbiguousMatchException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
                Console.WriteLine();
            }
            method = (new MethodReference(typeof(Array), "Copy", null, null, typeof(Int32))).GetMethodInfo();
            try
            {
                method = (new MethodReference(typeof(Array), "Find")).GetMethodInfo();
                Assert.True(false, "MissingMethodException not thrown");
            }
            catch (MissingMethodException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
                Console.WriteLine();
            }
            method = (new MethodReference(typeof(Array), "Find<>")).GetMethodInfo();
            method = (new MethodReference(typeof(Array), "Find<>", typeof(int))).GetMethodInfo();
            try
            {
                method = (new MethodReference(typeof(Array), "FindIndex<>")).GetMethodInfo();
                Assert.True(false, "AmbiguousMatchException not thrown");
            }
            catch (AmbiguousMatchException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
                Console.WriteLine();
            }
            method = (new MethodReference(typeof(Array), "FindIndex<>", null, null, null)).GetMethodInfo();
            // check that type parameter constraints are enforced
            try
            {
                method = (new MethodReference(typeof(TypeInferenceTests), "ConstrainClass<>", typeof(double), typeof(double))).GetMethodInfo();
                Assert.True(false, "ArgumentException not thrown");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
                Console.WriteLine();
            }
        }

        [Fact]
        public void GetFactorInfos()
        {
            foreach (FactorManager.FactorInfo info in FactorManager.GetFactorInfos())
            {
                Console.WriteLine(info.Method);
            }
        }

        [Fact]
        public void NotFactorInfo()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Factor), "Not").GetMethodInfo());
            Console.WriteLine(info);
            Assert.True(info.IsDeterministicFactor);
            IDictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            parameterTypes["Not"] = typeof(Bernoulli);
            parameterTypes["B"] = typeof(Bernoulli);
            parameterTypes["result"] = typeof(Bernoulli);
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Not", parameterTypes);
            Assert.False(fcninfo.PassResult);
            Assert.False(fcninfo.PassResultIndex);
            Assert.Equal(1, fcninfo.Dependencies.Count);
            Assert.Equal(1, fcninfo.Requirements.Count);
            Console.WriteLine(fcninfo);
        }

        [Fact]
        public void UnaryFactorInfo()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Factor), "Random<>", typeof(bool), typeof(Bernoulli)).GetMethodInfo());
            Console.WriteLine(info);
            Assert.False(info.IsDeterministicFactor);
            IDictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            parameterTypes["Random"] = typeof(Bernoulli);
            parameterTypes["dist"] = typeof(Bernoulli);
            parameterTypes["result"] = typeof(Bernoulli);
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Random", parameterTypes);
            Console.WriteLine(fcninfo);
            Assert.False(fcninfo.PassResult);
            Assert.False(fcninfo.PassResultIndex);
            Assert.Equal(1, fcninfo.Dependencies.Count);
        }

        [Fact]
        public void ConstrainEqualRandomFactorInfo()
        {
            FactorManager.FactorInfo info =
                FactorManager.GetFactorInfo(new MethodReference(typeof(Constrain), "EqualRandom<,>", typeof(bool), typeof(Bernoulli)).GetMethodInfo());
            Assert.Equal(2, info.ParameterNames.Count);
            Console.WriteLine(info);
            IDictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            parameterTypes["value"] = typeof(Bernoulli);
            parameterTypes["dist"] = typeof(Bernoulli);
            parameterTypes["result"] = typeof(Bernoulli);
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "value", parameterTypes);
            Console.WriteLine(fcninfo);
            Assert.False(fcninfo.PassResult);
            Assert.False(fcninfo.PassResultIndex);
            Assert.Equal(1, fcninfo.Dependencies.Count);
            Assert.Equal(1, fcninfo.Requirements.Count);

            Console.WriteLine();
            Console.WriteLine("All MessageFcnInfos:");
            int count = 0;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos())
            {
                Console.WriteLine(fcninfo2);
                CheckMessageFcnInfo(fcninfo2, info);
                count++;
            }
            //Assert.Equal(4, count);
        }

        [Fact]
        public void ConstrainEqualRandomOpenFactorInfo()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Constrain), "EqualRandom<,>").GetMethodInfo());
            Console.WriteLine(info);

            Console.WriteLine();
            Console.WriteLine("All AverageConditionals:");
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos("AverageConditional", null, null))
            {
                Console.WriteLine(fcninfo2);
                CheckMessageFcnInfo(fcninfo2, info);
            }
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void ReplicateFactorInfo()
        {
            DependencyInformation depInfo;
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Factor), "Replicate<>", typeof(bool)).GetMethodInfo());
            Console.WriteLine(info);
            Assert.True(info.IsDeterministicFactor);
            IDictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            parameterTypes["Uses"] = typeof(Bernoulli[]);
            parameterTypes["Def"] = typeof(Bernoulli);
            parameterTypes["resultIndex"] = typeof(int);
            parameterTypes["result"] = typeof(Bernoulli);
            for (int i = 0; i < 3; i++)
            {
                MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Uses", parameterTypes);
                Console.WriteLine(fcninfo);
                Assert.True(fcninfo.PassResult);
                Assert.True(fcninfo.PassResultIndex);
                Assert.Equal(2, fcninfo.Dependencies.Count);
                Assert.Equal(1, fcninfo.Requirements.Count);
                Assert.True(fcninfo.SkipIfAllUniform);

                if (i == 0)
                {
                    depInfo = FactorManager.GetDependencyInfo(fcninfo.Method);
                    Console.WriteLine("Dependencies:");
                    Console.WriteLine(StringUtil.ToString(depInfo.Dependencies));
                    Console.WriteLine("Requirements:");
                    Console.WriteLine(StringUtil.ToString(depInfo.Requirements));
                    Console.WriteLine("Triggers:");
                    Console.WriteLine(StringUtil.ToString(depInfo.Triggers));
                }
            }

            parameterTypes.Remove("resultIndex");
            MessageFcnInfo fcninfo3 = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Def", parameterTypes);
            Assert.True(fcninfo3.SkipIfAllUniform);

            Console.WriteLine();
            Console.WriteLine("All AverageConditional MessageFcnInfos:");
            int count = 0;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos("AverageConditional", null, null))
            {
                Console.WriteLine(fcninfo2);
                CheckMessageFcnInfo(fcninfo2, info);
                count++;
            }
            //Assert.Equal(4, count);
        }

        [Fact]
        public void ReplicateMessageFcnInfo()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Factor), "Replicate<>", typeof(bool)).GetMethodInfo());
            IDictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            parameterTypes["Uses"] = typeof(DistributionArray<Bernoulli>);
            parameterTypes["Def"] = typeof(Bernoulli);
            parameterTypes["result"] = typeof(Bernoulli); //typeof(DistributionArray<Bernoulli>);
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageLogarithm", "Def", parameterTypes);
            Console.WriteLine(fcninfo);
            CheckMessageFcnInfo(fcninfo, info);

            parameterTypes["Uses"] = typeof(Bernoulli[]);
            fcninfo = info.GetMessageFcnInfo(factorManager, "AverageLogarithm", "Def", parameterTypes);
            Console.WriteLine(fcninfo);
            CheckMessageFcnInfo(fcninfo, info);

            DependencyInformation depInfo;
            depInfo = FactorManager.GetDependencyInfo(fcninfo.Method);
            Console.WriteLine("Dependencies:");
            Console.WriteLine(StringUtil.ToString(depInfo.Dependencies));
            Console.WriteLine("Requirements:");
            Console.WriteLine(StringUtil.ToString(depInfo.Requirements));
            Console.WriteLine("Triggers:");
            Console.WriteLine(StringUtil.ToString(depInfo.Triggers));
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void UsesEqualDefFactorInfo()
        {
            DependencyInformation depInfo;
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Factor), "UsesEqualDef<>", typeof(bool)).GetMethodInfo());
            Assert.True(!info.IsDeterministicFactor);
            IDictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            parameterTypes["Uses"] = typeof(Bernoulli[]);
            parameterTypes["Def"] = typeof(Bernoulli);
            parameterTypes["resultIndex"] = typeof(int);
            parameterTypes["result"] = typeof(Bernoulli);
            for (int i = 0; i < 3; i++)
            {
                MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageLogarithm", "Uses", parameterTypes);
                Assert.True(fcninfo.PassResult);
                Assert.True(fcninfo.PassResultIndex);
                Assert.Equal(2, fcninfo.Dependencies.Count);
                Assert.Equal(1, fcninfo.Requirements.Count);
                Assert.True(fcninfo.SkipIfAllUniform);
                Assert.Equal(1, fcninfo.Triggers.Count);

                if (i == 0)
                {
                    depInfo = FactorManager.GetDependencyInfo(fcninfo.Method);
                    Console.WriteLine("Dependencies:");
                    Console.WriteLine(StringUtil.ToString(depInfo.Dependencies));
                    Console.WriteLine("Requirements:");
                    Console.WriteLine(StringUtil.ToString(depInfo.Requirements));
                    Console.WriteLine("Triggers:");
                    Console.WriteLine(StringUtil.ToString(depInfo.Triggers));
                }
            }
            parameterTypes.Remove("resultIndex");
            MessageFcnInfo fcninfo2 = info.GetMessageFcnInfo(factorManager, "AverageLogarithm", "Def", parameterTypes);
            Assert.True(fcninfo2.SkipIfAllUniform);
            Assert.Equal(1, fcninfo2.Triggers.Count);

            depInfo = FactorManager.GetDependencyInfo(fcninfo2.Method);
            Console.WriteLine("Dependencies:");
            Console.WriteLine(StringUtil.ToString(depInfo.Dependencies));
            Console.WriteLine("Requirements:");
            Console.WriteLine(StringUtil.ToString(depInfo.Requirements));
            Console.WriteLine("Triggers:");
            Console.WriteLine(StringUtil.ToString(depInfo.Triggers));
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void DifferenceFactorInfo()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Factor), "Difference").GetMethodInfo());
            Console.WriteLine(info);
            Assert.True(info.IsDeterministicFactor);
            IDictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            parameterTypes["Difference"] = typeof(Gaussian);
            parameterTypes["A"] = typeof(Gaussian);
            parameterTypes["B"] = typeof(Gaussian);
            //parameterTypes["result"] = typeof(Gaussian);
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Difference", parameterTypes);
            Assert.False(fcninfo.PassResult);
            Assert.False(fcninfo.PassResultIndex);
            Assert.Equal(2, fcninfo.Dependencies.Count);
            Assert.Equal(2, fcninfo.Requirements.Count);
            Console.WriteLine(fcninfo);

            Console.WriteLine("Parameter types:");
            Console.WriteLine(StringUtil.CollectionToString(fcninfo.GetParameterTypes(), ","));

            Console.WriteLine();
            try
            {
                fcninfo = info.GetMessageFcnInfo(factorManager, "Rubbish", "Difference", parameterTypes);
                Assert.True(false, "Did not throw an exception");
            }
            catch (ArgumentException ex)
            {
                if (!ex.Message.Contains("MissingMethodException"))
                    Assert.True(false, "Correctly threw exception, but with wrong message");
                Console.WriteLine("Different exception: " + ex);
            }

            Console.WriteLine();
            try
            {
                parameterTypes["result"] = typeof(double);
                fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Difference", parameterTypes);
                Assert.True(false, "Did not throw an exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }

            Console.WriteLine();
            Console.WriteLine("All messages to A:");
            int count = 0;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos(null, "A", null))
            {
                Console.WriteLine(fcninfo2);
                CheckMessageFcnInfo(fcninfo2, info);
                count++;
            }
            //Assert.Equal(8, count);

            Console.WriteLine();
            Console.WriteLine("All messages to Difference:");
            count = 0;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos(null, "Difference", null))
            {
                Console.WriteLine(fcninfo2);
                CheckMessageFcnInfo(fcninfo2, info);
                count++;
            }
            Assert.Equal(8, count);

            Console.WriteLine();
            Console.WriteLine("All AverageConditionals:");
            count = 0;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos("AverageConditional", null, null))
            {
                Console.WriteLine(fcninfo2);
                CheckMessageFcnInfo(fcninfo2, info);
                count++;
            }
            Assert.Equal(12, count);

            Console.WriteLine();
            Console.WriteLine("All MessageFcnInfos:");
            count = 0;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos())
            {
                Console.WriteLine(fcninfo2);
                CheckMessageFcnInfo(fcninfo2, info);
                count++;
            }
            //Assert.Equal(26, count);
        }

#if false
        [Test, TestMethod]
        public void ProductFactorInfo()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new FactorMethod<double,double,double>(Factor.Product));
            Console.WriteLine(info);
            Assert.True(info.IsDeterministicFactor);
            IDictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            parameterTypes["product"] = typeof(Gaussian);
            parameterTypes["a"] = typeof(Gaussian);
            parameterTypes["b"] = typeof(Gaussian);
            //parameterTypes["result"] = typeof(Gaussian);
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo("AverageConditional", "product", parameterTypes);
            CheckMessageFcnInfo(fcninfo,info);
        }
#endif

        [Fact]
        [Trait("Category", "OpenBug")]
        public void IsPositiveFactorInfo()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new Func<double, bool>(Factor.IsPositive));
            Console.WriteLine(info);
            Assert.True(info.IsDeterministicFactor);
            IDictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            parameterTypes["isPositive"] = typeof(bool);
            parameterTypes["x"] = typeof(Gaussian);
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "x", parameterTypes);
            Assert.True(fcninfo.NotSupportedMessage == null);
            CheckMessageFcnInfo(fcninfo, info);

            DependencyInformation depInfo = FactorManager.GetDependencyInfo(fcninfo.Method);
            Console.WriteLine("Dependencies:");
            Console.WriteLine(StringUtil.ToString(depInfo.Dependencies));
            Console.WriteLine("Requirements:");
            Console.WriteLine(StringUtil.ToString(depInfo.Requirements));

            try
            {
                fcninfo = info.GetMessageFcnInfo(factorManager, "AverageLogarithm", "x", parameterTypes);
                Assert.True(false, "did not throw exception");
            }
            catch (NotSupportedException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            bool found = false;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos("AverageLogarithm", null, null))
            {
                CheckMessageFcnInfo(fcninfo2, info);
                if (fcninfo2.TargetParameter.Equals("x"))
                {
                    Assert.True(fcninfo2.NotSupportedMessage != null);
                    found = true;
                }
            }
            Assert.True(found);
            fcninfo = info.GetMessageFcnInfo(factorManager, "LogAverageFactor", "", parameterTypes);
            CheckMessageFcnInfo(fcninfo, info);
        }

        internal void ToMessageTest()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new Func<double, bool>(Factor.IsPositive));
            IDictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            parameterTypes["isPositive"] = typeof(bool);
            parameterTypes["x"] = typeof(Gaussian);
            MessageFcnInfo fcninfo;
            fcninfo = info.GetMessageFcnInfo(factorManager, "LogEvidenceRatio2", "", parameterTypes);
            CheckMessageFcnInfo(fcninfo, info);
            DependencyInformation depInfo = FactorManager.GetDependencyInfo(fcninfo.Method);
            Console.WriteLine(depInfo);
        }

        internal void CheckMessageFcnInfo(MessageFcnInfo fcninfo, FactorManager.FactorInfo info)
        {
            Assert.True(fcninfo.Suffix != null);
            Assert.True(fcninfo.TargetParameter != null);
            Dictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            foreach (KeyValuePair<string, Type> parameter in fcninfo.GetParameterTypes())
            {
                parameterTypes[parameter.Key] = parameter.Value;
            }
            try
            {
                MessageFcnInfo fcninfo2 = info.GetMessageFcnInfo(factorManager, fcninfo.Suffix, fcninfo.TargetParameter, parameterTypes);
                Assert.Equal(fcninfo2.Method, fcninfo.Method);
            }
            catch (NotSupportedException)
            {
                Assert.True(fcninfo.NotSupportedMessage != null);
            }
        }

        [Fact]
        public void TypeConstraintFailureFactorInfo()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(ShiftAlpha), "ToFactor<>").GetMethodInfo());
            Console.WriteLine(info);
            Dictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            parameterTypes["factor"] = typeof(double);
            parameterTypes["result"] = typeof(double);
            try
            {
                MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Variable", parameterTypes);
                Assert.True(false, "Did not throw an exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
        }

        [Fact]
        public void MissingMethodFailure()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new Func<double, double, double>(Factor.Gaussian));
            Dictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            parameterTypes["sample"] = typeof(Gaussian);
            parameterTypes["mean"] = typeof(Gaussian);
            parameterTypes["precision"] = typeof(Gaussian);
            try
            {
                MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Sample", parameterTypes);
                Assert.True(false, "Did not throw an exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
        }

        /// <summary>
        /// Tests if the compiler correctly detects missing buffer initialization and update methods.
        /// </summary>
        /// <remarks>
        /// When this test was created, <see cref="NullReferenceException"/> was thrown instead of <see cref="CompilationFailedException"/>.
        /// </remarks>
        [Fact]
        public void MissingBufferMethodsFailure()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                var engine = new InferenceEngine();
                engine.Compiler.RequiredQuality = QualityBand.Unknown;
                engine.Infer(Variable<bool>.Factor(MissingBufferMethodsFailureFactorAndOp.Factor, Variable.Bernoulli(0.5)));
            });
        }

        /// <summary>
        /// Factor definition and message operators for <see cref="MissingBufferMethodsFailure"/> test.
        /// </summary>
        [FactorMethod(typeof(MissingBufferMethodsFailureFactorAndOp), "Factor")]
        [Buffers("y")]
        public class MissingBufferMethodsFailureFactorAndOp
        {
            /// <summary>
            /// Factor definition.
            /// </summary>
            /// <param name="x">Factor argument.</param>
            /// <returns>Nothing since an exception is always thrown.</returns>
            [Hidden]
            public static bool Factor(bool x)
            {
                throw new NotImplementedException();
            }

            /// <summary>
            /// The EP message to 'factor'.
            /// </summary>
            /// <param name="x">The message from variable 'x'.</param>
            /// <param name="y">The value of buffer 'y'.</param>
            /// <returns>Nothing since an exception is always thrown.</returns>
            public static Bernoulli FactorAverageConditional(Bernoulli x, int y)
            {
                throw new NotImplementedException();
            }

            /// <summary>
            /// The EP message to 'x'.
            /// </summary>
            /// <param name="factor">The message from variable 'factor'.</param>
            /// <param name="y">The value of buffer 'y'.</param>
            /// <returns>Nothing since an exception is always thrown.</returns>
            public static Bernoulli XAverageConditional(Bernoulli factor, int y)
            {
                throw new NotImplementedException();
            }
        }
    }
}