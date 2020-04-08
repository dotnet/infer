// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Generic;
    using Xunit;
    using Microsoft.ML.Probabilistic;
    using Microsoft.ML.Probabilistic.Compiler.Attributes;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Compiler;
    using Microsoft.ML.Probabilistic.Algorithms;
    using Microsoft.ML.Probabilistic.Models.Attributes;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    public class TransformTests
    {
        [Fact]
        public void NameGeneratorTest()
        {
            var ng = new NameGenerator();
            HashSet<string> names = new HashSet<string>();
            string name2 = ng.GenerateName("car7");
            names.Add(name2);
            for (int i = 0; i < 100; i++)
            {
                string name = ng.GenerateName("car");
                Assert.DoesNotContain(name, names);
                names.Add(name);
            }

            name2 = ng.GenerateName("car9");
            Assert.DoesNotContain(name2, names);
        }

        [Fact]
        public void LanguageWriterTypeSourceTest()
        {
            ILanguageWriter writer = new CSharpWriter();
            string s;
            Type type;
            type = typeof(double[][,]);
            s = writer.TypeSource(CodeBuilder.Instance.TypeRef(type));
            Assert.Equal("double[][,]", s);

            type = typeof(KeyValuePair<double[,][], double[][,]>[][,]);
            s = writer.TypeSource(CodeBuilder.Instance.TypeRef(type));
            Assert.Equal("KeyValuePair<double[,][],double[][,]>[][,]", s);
        }

        [Fact]
        public void ConstantPropagationTest()
        {
            var a = Variable.Bernoulli(0.5).Named("a");
            var b = Variable.Bernoulli(0.5).Named("b");
            var c = Variable.Bernoulli(0.5).Named("c");
            var d = Variable.Bernoulli(0.5).Named("d");
            using (Variable.If(c))
            {
                Variable.ConstrainTrue(d);
            }
            Variable.ConstrainEqual(b, c);
            using (Variable.If(a))
            {
                Variable.ConstrainTrue(b);
            }
            Variable.ConstrainTrue(a);
            InferenceEngine engine = new InferenceEngine();
            Bernoulli dActual = engine.Infer<Bernoulli>(d);
            Bernoulli dExpected = Bernoulli.PointMass(true);
            Assert.Equal(dActual, dExpected);
        }

        [Fact]
        public void ConstantPropagationTest2()
        {
            var a = Variable.Bernoulli(0.5).Named("a");
            var b = Variable.Bernoulli(0.5).Named("b");
            var c = Variable.Bernoulli(0.5).Named("c");
            var d = Variable.Bernoulli(0.5).Named("d");
            using (Variable.If(c))
            {
                Variable.ConstrainTrue(d);
            }
            Variable.ConstrainEqual(b, c);
            using (Variable.IfNot(a))
            {
                Variable.ConstrainTrue(b);
            }
            Variable.ConstrainTrue(a);
            InferenceEngine engine = new InferenceEngine();
            Bernoulli dActual = engine.Infer<Bernoulli>(d);
            Bernoulli dExpected = new Bernoulli(2.0/3);
            Assert.Equal(dActual, dExpected);
        }

        /// <summary>
        /// This test has been put in place due to a bug in the DependencyAnalysis transform
        /// which (as of Feb 9th 2010) marks the following statement as uniform, and so the scheduler removes it.
        /// Dirichlet1 = new Dirichlet(SparseVector.FromSparseValues(2, 0.5, new List&lt;SparseElement&gt;(new SparseElement[0] {})));
        /// </summary>
        [Fact]
        public void PruningTest()
        {
            int numT = 2;
            int numD = 2;
            var D = new Range(numD);
            var T = new Range(numT);
            double alpha = 0.5;
            Vector v = Vector.Constant(numT, alpha, Sparsity.Sparse);

            // Using an observed prior:
            var prior = Variable.New<Dirichlet>();
            prior.ObservedValue = new Dirichlet(v);
            var theta1 = Variable.Array<Vector>(D).Named("theta1");
            theta1[D] = Variable.Random<Vector, Dirichlet>(prior).ForEach(D);

            // Using a constant vector
            var theta2 = Variable.Array<Vector>(D).Named("theta2");
            theta2[D] = Variable.Dirichlet(T, Vector.Constant(numT, alpha, Sparsity.Sparse)).ForEach(D);

            // These should give the same result.
            var engine = new InferenceEngine(new VariationalMessagePassing());
            var postTheta1 = engine.Infer<Dirichlet[]>(theta1);
            var postTheta2 = engine.Infer<Dirichlet[]>(theta2);

            for (int i = 0; i < numD; i++)
            {
                Assert.Equal(postTheta2[i], postTheta1[i]);
            }
        }

        /// <summary>
        /// Tests the ListenToMessages attribute.
        /// </summary>
        [Fact]
        public void ListenToMessagesTest()
        {
            // Ground truth messages for the three coins models below.
            var truth = new Dictionary<string, Bernoulli> 
            {
                { "bothHeads_use_B", new Bernoulli(1.0 / 3.0) },
                { "bothHeads_B", new Bernoulli(1.0 / 3.0) },
                { "bothHeads_F", new Bernoulli(1.0 / 4.0) },
                { "bothHeads_use_F", new Bernoulli(1.0 / 4.0) },
                { "bothHeads_marginal_F", new Bernoulli(1.0 / 7.0) },
            };

            // All messages
            this.ListenToMessagesWithFilter(string.Empty, truth);

            // Forward messages only
            this.ListenToMessagesWithFilter("_B", truth);

            // Backward messages only
            this.ListenToMessagesWithFilter("_F", truth);

            // Use messages only
            this.ListenToMessagesWithFilter("_use", truth);
        }

        /// <summary>
        /// Tests listening to messages with various message filters.
        /// </summary>
        /// <param name="filter">The filter to use</param>
        /// <param name="correctMessages">The correct messages (unfiltered)</param>
        private void ListenToMessagesWithFilter(string filter, Dictionary<string, Bernoulli> correctMessages)
        {
            // Three coins model (non-iterative)
            var firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            var secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            var bothHeads = (firstCoin & secondCoin).Named("bothHeads");

            var thirdCoin = Variable.Bernoulli(0.5).Named("thirdCoin");
            var allHeads = bothHeads & thirdCoin;

            // Observation
            allHeads.ObservedValue = false;

            // Add attribute to listen to messages for 'bothHeads'
            bothHeads.AddAttribute(new ListenToMessages { Containing = filter });
            
            // Create inference engine
            var engine = new InferenceEngine();

            // Add event listener for messages which adds them to a dictionary
            var actualMessages = new Dictionary<string, Bernoulli>();
            engine.MessageUpdated += (alg, ev) => { actualMessages.Add(ev.MessageId, (Bernoulli)ev.Message); };

            // Run inference
            var firstCoinDist = engine.Infer(firstCoin);
            
            // Compare dictionary to truth
            int count = 0;
            foreach (var kvp in correctMessages)
            {
                // Filter the correct messages
                if (!kvp.Key.Contains(filter))
                {
                    continue;
                }

                count++;

                // Check dictionary contains the required message
                Assert.True(actualMessages.ContainsKey(kvp.Key));
                Console.WriteLine(kvp.Key + ": " + actualMessages[kvp.Key]);

                // Check that the message is correct
                Assert.True(actualMessages[kvp.Key].MaxDiff(kvp.Value) < 1e-10);
            }

            Console.WriteLine();

            // Check that the right number of messages were sent.
            Assert.True(count == actualMessages.Count);
        }
    }
}