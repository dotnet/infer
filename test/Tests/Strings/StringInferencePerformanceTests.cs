// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Diagnostics;
    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Distributions.Automata;
    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    /// <summary>
    /// These tests don't check for correctness, just print timings to the console.
    /// They have a timeout which is about 10x the expected time, so really signficant
    /// performance hits will throw an exception.
    /// </summary>
    public class StringInferencePerformanceTests
    {
        private void AssertTimeout(Action action, int timeout)
        {
            // Don't impose a time limit since runtimes are very inconsistent on Azure. 
            action();
        }

        /// <summary>
        /// Measures automaton normalization performance.
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        [Trait("Category", "StringInference")]
        public void AutomatonNormalizationPerformance1()
        {
            AssertTimeout(() =>
            {
                var builder = new StringAutomaton.Builder();
                var nextState = builder.Start.AddTransitionsForSequence("abc");
                nextState.AddSelfTransition('d', Weight.FromValue(0.1));
                nextState.AddTransitionsForSequence("efg").SetEndWeight(Weight.One);
                nextState.AddTransitionsForSequence("hejfhoenmf").SetEndWeight(Weight.One);

                var automaton = builder.GetAutomaton();

                ProfileAction(() => automaton.GetLogNormalizer(), 100000);
            }, 10000);
        }

        /// <summary>
        /// Measures automaton normalization performance.
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        [Trait("Category", "StringInference")]
        public void AutomatonNormalizationPerformance2()
        {
            AssertTimeout(() =>
            {
                var builder = new StringAutomaton.Builder();
                var nextState = builder.Start.AddTransitionsForSequence("abc");
                nextState.SetEndWeight(Weight.One);
                nextState.AddSelfTransition('d', Weight.FromValue(0.1));
                nextState = nextState.AddTransitionsForSequence("efg");
                nextState.SetEndWeight(Weight.One);
                nextState.AddSelfTransition('h', Weight.FromValue(0.2));
                nextState = nextState.AddTransitionsForSequence("grlkhgn;lk3rng");
                nextState.SetEndWeight(Weight.One);
                nextState.AddSelfTransition('h', Weight.FromValue(0.3));

                var automaton = builder.GetAutomaton();

                ProfileAction(() => automaton.GetLogNormalizer(), 100000);
            }, 20000);
        }

        /// <summary>
        /// Measures automaton normalization performance.
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        [Trait("Category", "StringInference")]
        public void AutomatonNormalizationPerformance3()
        {
            AssertTimeout(() =>
            {
                var builder = new StringAutomaton.Builder();
                builder.Start.AddSelfTransition('a', Weight.FromValue(0.5));
                builder.Start.SetEndWeight(Weight.One);
                var nextState = builder.Start.AddTransitionsForSequence("aa");
                nextState.AddSelfTransition('a', Weight.FromValue(0.5));
                nextState.SetEndWeight(Weight.One);

                var automaton = builder.GetAutomaton();
                for (int i = 0; i < 3; ++i)
                {
                    automaton = automaton.Product(automaton);
                }

                ProfileAction(() => automaton.GetLogNormalizer(), 100);
            }, 120000);
        }
        
        /// <summary>
        /// Measures regular expression building performance.
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        [Trait("Category", "StringInference")]
        public void RegexpBuildingPerformanceTest1()
        {
            AssertTimeout(() =>
            {
                StringDistribution dist =
                    StringDistribution.OneOf(
                        StringDistribution.Lower() + StringDistribution.Upper() + StringDistribution.Optional(StringDistribution.Upper(2)),
                        StringDistribution.Digits(3) + StringDistribution.String("XXX") + StringDistribution.Letters(3, 5));

                Console.WriteLine(dist.ToString());

                ProfileAction(() => RegexpTreeBuilder.BuildRegexp(dist.ToAutomaton()), 6000);
            }, 10000);
        }

        /// <summary>
        /// Measures regular expression building performance.
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        [Trait("Category", "StringInference")]
        public void RegexpBuildingPerformanceTest2()
        {
            AssertTimeout(() =>
            {
                StringDistribution dist = StringDistribution.OneOf(StringDistribution.Lower(), StringDistribution.Upper());
                for (int i = 0; i < 3; ++i)
                {
                    dist = StringDistribution.OneOf(dist, dist);
                }

                Console.WriteLine(dist.ToString());

                ProfileAction(() => RegexpTreeBuilder.BuildRegexp(dist.ToAutomaton()), 6000);
            }, 20000);
        }

        /// <summary>
        /// Measures regular expression building performance.
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        [Trait("Category", "StringInference")]
        public void RegexpBuildingPerformanceTest3()
        {
            AssertTimeout(() =>
            {
                StringDistribution dist = StringFormatOp_RequireEveryPlaceholder_NoArgumentNames.FormatAverageConditional(
                    StringDistribution.String("aaaaaaaaaaa"),
                    new[] { StringDistribution.PointMass("a"), StringDistribution.PointMass("aa") });

                Console.WriteLine(dist.ToString());

                ProfileAction(() => RegexpTreeBuilder.BuildRegexp(dist.ToAutomaton()), 1000);
            }, 20000);
        }

        /// <summary>
        /// Measures the performance of inference in a model with <see cref="Factor.StringFormat"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        [Trait("Category", "StringInference")]
        public void StringFormatPerformanceTest1()
        {
            AssertTimeout(() =>
            {
                Rand.Restart(777);

                Variable<string> template = Variable.Constant("{0} {1}").Named("template");
                Variable<string> arg1 = Variable.Random(StringDistribution.Any(minLength: 1, maxLength: 15)).Named("arg1");
                Variable<string> arg2 = Variable.Random(StringDistribution.Any(minLength: 1)).Named("arg2");
                Variable<string> s = Variable.StringFormat(template, arg1, arg2).Named("s");

                var engine = new InferenceEngine();
                engine.Compiler.RecommendedQuality = QualityBand.Experimental;
                engine.ShowProgress = false;

                Action action = () =>
                {
                    // Generate random observed string
                    string observedPattern = string.Empty;
                    for (int j = 0; j < 5; ++j)
                    {
                        for (int k = 0; k < 5; ++k)
                        {
                            observedPattern += (char)Rand.Int('a', 'z' + 1);
                        }

                        observedPattern += ' ';
                    }

                    // Run inference
                    s.ObservedValue = observedPattern;
                    engine.Infer<StringDistribution>(arg1);
                    engine.Infer<StringDistribution>(arg2);
                };

                action(); // To exclude the compilation time from the profile
                ProfileAction(action, 1000);
            }, 10000);
        }

        /// <summary>
        /// Measures the performance of inference in a model with <see cref="Factor.StringFormat"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        [Trait("Category", "StringInference")]
        public void StringFormatPerformanceTest2()
        {
            AssertTimeout(() =>
            {
                Rand.Restart(777);

                Variable<string> template = Variable.Constant("{0} {1}").Named("template");
                Variable<string> arg1 = Variable.Random(StringDistribution.Any(minLength: 1, maxLength: 15)).Named("arg1");
                Variable<string> arg2 = Variable.Random(StringDistribution.Any(minLength: 1)).Named("arg2");
                Variable<string> text = Variable.StringFormat(template, arg1, arg2).Named("text");
                Variable<string> fullTextFormat = Variable.Random(StringDistribution.Any()).Named("fullTextFormat");
                Variable<string> fullText = Variable.StringFormat(fullTextFormat, text).Named("fullText");

                var engine = new InferenceEngine();
                engine.Compiler.RecommendedQuality = QualityBand.Experimental;
                engine.ShowProgress = false;

                Action action = () =>
                {
                    // Generate random observed string
                    string observedPattern = string.Empty;
                    for (int j = 0; j < 5; ++j)
                    {
                        for (int k = 0; k < 5; ++k)
                        {
                            observedPattern += (char)Rand.Int('a', 'z' + 1);
                        }

                        observedPattern += ' ';
                    }

                    // Run inference
                    fullText.ObservedValue = observedPattern;
                    engine.Infer<StringDistribution>(arg1);
                    engine.Infer<StringDistribution>(arg2);
                    engine.Infer<StringDistribution>(text);
                    engine.Infer<StringDistribution>(fullTextFormat);
                };

                action(); // To exclude the compilation time from the profile
                ProfileAction(action, 100);
            }, 10000);
        }

        /// <summary>
        /// Measures the performance of inference in a model with <see cref="Factor.StringFormat"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        [Trait("Category", "StringInference")]
        [Trait("Category", "OpenBug")] // Test failing with AutomatonTooLarge due to determinization added to SetToProduct in change 47614.  Increasing max states to 1M does not fix the issue
        public void PropertyInferencePerformanceTest()
        {
            Rand.Restart(777);

            var namesData = new[] { "Alice", "Bob", "Charlie", "Eve", "Boris", "John" };
            var valueData = new[] { "sender", "receiver", "attacker", "eavesdropper", "developer", "researcher" };
            var templatesData = new[] { "{0} is {1}", "{0} is known as {1}", "{1} is a role of {0}", "{0} -- {1}", "{0} aka {1}" };

            var textsData = new string[10];
            for (int i = 0; i < textsData.Length; ++i)
            {
                int entityIndex = Rand.Int(namesData.Length);
                int templateIndex = Rand.Int(templatesData.Length);
                textsData[i] = string.Format(templatesData[templateIndex], namesData[entityIndex], valueData[entityIndex]);
            }

            var entity = new Range(namesData.Length).Named("entity");
            var template = new Range(templatesData.Length).Named("template");
            var text = new Range(textsData.Length).Named("text");

            var entityNames = Variable.Array<string>(entity).Named("entityNames");
            entityNames[entity] = Variable.Random(StringDistribution.Capitalized()).ForEach(entity);
            var entityValues = Variable.Array<string>(entity).Named("entityValues");
            entityValues[entity] = Variable.Random(StringDistribution.Lower()).ForEach(entity);

            StringDistribution templatePriorMiddle = StringDistribution.ZeroOrMore(ImmutableDiscreteChar.OneOf('{', '}').Complement());
            StringDistribution templatePrior =
                StringDistribution.OneOf(
                    StringDistribution.String("{0} ") + templatePriorMiddle + StringDistribution.String(" {1}"),
                    StringDistribution.String("{1} ") + templatePriorMiddle + StringDistribution.String(" {0}"));
            var templates = Variable.Array<string>(template).Named("templates");
            templates[template] = Variable.Random(templatePrior).ForEach(template);

            var texts = Variable.Array<string>(text).Named("texts");
            using (Variable.ForEach(text))
            {
                var entityIndex = Variable.DiscreteUniform(entity).Named("entityIndex");
                var templateIndex = Variable.DiscreteUniform(template).Named("templateIndex");
                using (Variable.Switch(entityIndex))
                using (Variable.Switch(templateIndex))
                {
                    texts[text] = Variable.StringFormat(templates[templateIndex], entityNames[entityIndex], entityValues[entityIndex]);
                }
            }

            texts.ObservedValue = textsData;

            var engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.OptimiseForVariables = new[] { entityNames, entityValues };
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            // TODO: get this test to work with parallel for loops.
            engine.Compiler.UseParallelForLoops = false;
            engine.NumberOfIterations = 1;

            ProfileAction(
                () =>
                    {
                        Console.WriteLine(engine.Infer<StringDistribution[]>(entityNames)[0]);
                        Console.WriteLine(engine.Infer<StringDistribution[]>(entityValues)[0]);
                    },
                1);
        }

        /// <summary>
        /// Measures the amount of time needed to perform a specified action a given number of times and prints it to console.
        /// </summary>
        /// <param name="action">The action.</param>
        /// <param name="timesToRun">The number of times to run the action.</param>
        /// <param name="caption">The caption.</param>
        private static void ProfileAction(Action action, int timesToRun, string caption = "Elapsed time")
        {
            Stopwatch stopwatch = Stopwatch.StartNew();
            for (int i = 0; i < timesToRun; ++i)
            {
                action();
            }
            
            Console.WriteLine("{0}: {1}", caption, stopwatch.Elapsed);
        }
    }
}
