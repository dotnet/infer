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
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Clone), "ReplicateWithMarginal<>", typeof(bool[])).GetMethodInfo());
            var parameterTypes = new Dictionary<string, Type>();
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
            Assert.Throws<AmbiguousMatchException>(() =>
            {
                method = (new MethodReference(typeof(Array), "Copy")).GetMethodInfo();
            });
            method = (new MethodReference(typeof(Array), "Copy", null, null, typeof(Int32))).GetMethodInfo();
            Assert.Throws<MissingMethodException>(() =>
            {
                method = (new MethodReference(typeof(Array), "Find")).GetMethodInfo();
            });
            method = (new MethodReference(typeof(Array), "Find<>")).GetMethodInfo();
            method = (new MethodReference(typeof(Array), "Find<>", typeof(int))).GetMethodInfo();
            Assert.Throws<AmbiguousMatchException>(() =>
            {
                method = (new MethodReference(typeof(Array), "FindIndex<>")).GetMethodInfo();
            });
            method = (new MethodReference(typeof(Array), "FindIndex<>", null, null, null)).GetMethodInfo();
            // check that type parameter constraints are enforced
            Assert.Throws<ArgumentException>(() =>
            {
                method = (new MethodReference(typeof(TypeInferenceTests), "ConstrainClass<>", typeof(double), typeof(double))).GetMethodInfo();
            });
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
            var parameterTypes = new Dictionary<string, Type>
            {
                ["Not"] = typeof(Bernoulli),
                ["B"] = typeof(Bernoulli),
                ["result"] = typeof(Bernoulli)
            };
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
            Assert.False(info.IsDeterministicFactor);
            var parameterTypes = new Dictionary<string, Type>
            {
                ["Random"] = typeof(Bernoulli),
                ["dist"] = typeof(Bernoulli),
                ["result"] = typeof(Bernoulli)
            };
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Random", parameterTypes);
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
            var parameterTypes = new Dictionary<string, Type>
            {
                ["value"] = typeof(Bernoulli),
                ["dist"] = typeof(Bernoulli),
                ["result"] = typeof(Bernoulli)
            };
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "value", parameterTypes);
            Assert.False(fcninfo.PassResult);
            Assert.False(fcninfo.PassResultIndex);
            Assert.Equal(1, fcninfo.Dependencies.Count);
            Assert.Equal(1, fcninfo.Requirements.Count);

            bool verbose = false;
            if (verbose) Console.WriteLine("All MessageFcnInfos:");
            int count = 0;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos())
            {
                if (verbose) Console.WriteLine(fcninfo2);
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
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Clone), "Replicate<>", typeof(bool)).GetMethodInfo());
            Assert.True(info.IsDeterministicFactor);
            var parameterTypes = new Dictionary<string, Type>
            {
                ["Uses"] = typeof(Bernoulli[]),
                ["Def"] = typeof(Bernoulli),
                ["resultIndex"] = typeof(int),
                ["result"] = typeof(Bernoulli)
            };
            for (int i = 0; i < 3; i++)
            {
                MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Uses", parameterTypes);
                Assert.True(fcninfo.PassResult);
                Assert.True(fcninfo.PassResultIndex);
                Assert.Equal(2, fcninfo.Dependencies.Count);
                Assert.Equal(1, fcninfo.Requirements.Count);
                Assert.True(fcninfo.SkipIfAllUniform);

                if (i == 0)
                {
                    depInfo = FactorManager.GetDependencyInfo(fcninfo.Method);
                }
            }

            parameterTypes.Remove("resultIndex");
            MessageFcnInfo fcninfo3 = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Def", parameterTypes);
            Assert.True(fcninfo3.SkipIfAllUniform);

            bool verbose = false;
            if (verbose) Console.WriteLine("All AverageConditional MessageFcnInfos:");
            int count = 0;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos("AverageConditional", null, null))
            {
                if (verbose) Console.WriteLine(fcninfo2);
                CheckMessageFcnInfo(fcninfo2, info);
                count++;
            }
            //Assert.Equal(4, count);
        }

        [Fact]
        public void ReplicateMessageFcnInfo()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Clone), "Replicate<>", typeof(bool)).GetMethodInfo());
            var parameterTypes = new Dictionary<string, Type>
            {
                ["Uses"] = typeof(DistributionArray<Bernoulli>),
                ["Def"] = typeof(Bernoulli),
                ["result"] = typeof(Bernoulli) //typeof(DistributionArray<Bernoulli>);
            };
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageLogarithm", "Def", parameterTypes);
            CheckMessageFcnInfo(fcninfo, info);

            parameterTypes["Uses"] = typeof(Bernoulli[]);
            fcninfo = info.GetMessageFcnInfo(factorManager, "AverageLogarithm", "Def", parameterTypes);
            CheckMessageFcnInfo(fcninfo, info);

            DependencyInformation depInfo;
            depInfo = FactorManager.GetDependencyInfo(fcninfo.Method);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void UsesEqualDefFactorInfo()
        {
            DependencyInformation depInfo;
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Clone), "UsesEqualDef<>", typeof(bool)).GetMethodInfo());
            Assert.True(!info.IsDeterministicFactor);
            var parameterTypes = new Dictionary<string, Type>
            {
                ["Uses"] = typeof(Bernoulli[]),
                ["Def"] = typeof(Bernoulli),
                ["resultIndex"] = typeof(int),
                ["result"] = typeof(Bernoulli)
            };
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
                }
            }
            parameterTypes.Remove("resultIndex");
            MessageFcnInfo fcninfo2 = info.GetMessageFcnInfo(factorManager, "AverageLogarithm", "Def", parameterTypes);
            Assert.True(fcninfo2.SkipIfAllUniform);
            Assert.Equal(1, fcninfo2.Triggers.Count);

            depInfo = FactorManager.GetDependencyInfo(fcninfo2.Method);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void DifferenceFactorInfo()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new MethodReference(typeof(Factor), "Difference").GetMethodInfo());
            Assert.True(info.IsDeterministicFactor);
            var parameterTypes = new Dictionary<string, Type>
            {
                ["Difference"] = typeof(Gaussian),
                ["A"] = typeof(Gaussian),
                ["B"] = typeof(Gaussian)
            };
            //parameterTypes["result"] = typeof(Gaussian);
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Difference", parameterTypes);
            Assert.False(fcninfo.PassResult);
            Assert.False(fcninfo.PassResultIndex);
            Assert.Equal(2, fcninfo.Dependencies.Count);
            Assert.Equal(2, fcninfo.Requirements.Count);

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

            Assert.Throws<ArgumentException>(() =>
            {
                parameterTypes["result"] = typeof(double);
                fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Difference", parameterTypes);
            });

            bool verbose = false;
            if (verbose) Console.WriteLine("All messages to A:");
            int count = 0;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos(null, "A", null))
            {
                if (verbose) Console.WriteLine(fcninfo2);
                CheckMessageFcnInfo(fcninfo2, info);
                count++;
            }
            //Assert.Equal(8, count);

            if (verbose) Console.WriteLine("All messages to Difference:");
            count = 0;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos(null, "Difference", null))
            {
                CheckMessageFcnInfo(fcninfo2, info);
                count++;
            }
            Assert.Equal(8, count);

            if (verbose) Console.WriteLine("All AverageConditionals:");
            count = 0;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos("AverageConditional", null, null))
            {
                if (verbose) Console.WriteLine(fcninfo2);
                CheckMessageFcnInfo(fcninfo2, info);
                count++;
            }
            Assert.Equal(12, count);

            if (verbose) Console.WriteLine("All MessageFcnInfos:");
            count = 0;
            foreach (MessageFcnInfo fcninfo2 in info.GetMessageFcnInfos())
            {
                if (verbose) Console.WriteLine(fcninfo2);
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
            Assert.True(info.IsDeterministicFactor);
            var parameterTypes = new Dictionary<string, Type>
            {
                ["isPositive"] = typeof(bool),
                ["x"] = typeof(Gaussian)
            };
            MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "x", parameterTypes);
            Assert.True(fcninfo.NotSupportedMessage == null);
            CheckMessageFcnInfo(fcninfo, info);

            DependencyInformation depInfo = FactorManager.GetDependencyInfo(fcninfo.Method);

            Assert.Throws<NotSupportedException>(() =>
            {
                fcninfo = info.GetMessageFcnInfo(factorManager, "AverageLogarithm", "x", parameterTypes);
            });
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
            var parameterTypes = new Dictionary<string, Type>
            {
                ["isPositive"] = typeof(bool),
                ["x"] = typeof(Gaussian)
            };
            MessageFcnInfo fcninfo;
            fcninfo = info.GetMessageFcnInfo(factorManager, "LogEvidenceRatio2", "", parameterTypes);
            CheckMessageFcnInfo(fcninfo, info);
            DependencyInformation depInfo = FactorManager.GetDependencyInfo(fcninfo.Method);
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
            Dictionary<string, Type> parameterTypes = new Dictionary<string, Type>
            {
                ["factor"] = typeof(double),
                ["result"] = typeof(double)
            };
            Assert.Throws<ArgumentException>(() =>
            {
                MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Variable", parameterTypes);
            });
        }

        [Fact]
        public void MissingMethodFailure()
        {
            FactorManager.FactorInfo info = FactorManager.GetFactorInfo(new Func<double, double, double>(Factor.Gaussian));
            Dictionary<string, Type> parameterTypes = new Dictionary<string, Type>
            {
                ["sample"] = typeof(Gaussian),
                ["mean"] = typeof(Gaussian),
                ["precision"] = typeof(Gaussian)
            };
            Assert.Throws<ArgumentException>(() =>
            {
                MessageFcnInfo fcninfo = info.GetMessageFcnInfo(factorManager, "AverageConditional", "Sample", parameterTypes);
            });
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