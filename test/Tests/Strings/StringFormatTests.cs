// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Models;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    public class StringFormatTests
    {
        [Fact]
        [Trait("Category", "OpenBug")]
        public void HelloWorld()
        {
            var a = Variable.Random(WordString()).Named("a");
            var b = Variable.Random(WordString()).Named("b");
            var template = Variable.Random(TemplateArgString() + StringDistribution.Char(' ') + TemplateArgString()).Named("template");
            var c = Variable.StringFormat(template, a, b).Named("c");
            var c2 = Variable.StringFormat(template, a, b).Named("c2");

            var engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.GivePriorityTo(typeof(ReplicateOp_NoDivide));
            //engine.Compiler.BrowserMode = BrowserMode.Always;
            engine.NumberOfIterations = 5;
            //Console.WriteLine("c="+engine.Infer(c)); // expect Uniform(*)

            c.ObservedValue = "Hello World";
            Test("a", engine.Infer<IDistribution<string>>(a), "Hello", "World");
            Test("b", engine.Infer<IDistribution<string>>(b), "Hello", "World");
            Test("t", engine.Infer<IDistribution<string>>(template), "{0} {1}", "{1} {0}");
            Test("c2", engine.Infer<IDistribution<string>>(c2), "Hello World", "Hello Hello", "World Hello", "World World");

            b.ObservedValue = "World";
            TestUnif("a", engine.Infer<IDistribution<string>>(a), "Hello");
            TestUnif("t", engine.Infer<IDistribution<string>>(template), "{0} {1}");
            TestUnif("c2", engine.Infer<IDistribution<string>>(c2), "Hello World");
            b.ClearObservedValue();
            a.ObservedValue = "Hello";
            TestUnif("b", engine.Infer<IDistribution<string>>(b), "World");
            TestUnif("t", engine.Infer<IDistribution<string>>(template), "{0} {1}");
            TestUnif("c2", engine.Infer<IDistribution<string>>(c2), "Hello World");
            b.ObservedValue = "World";
            TestUnif("t", engine.Infer<IDistribution<string>>(template), "{0} {1}");
            TestUnif("c2", engine.Infer<IDistribution<string>>(c2), "Hello World");

            template.ObservedValue = "{0} {1}";
            a.ClearObservedValue();
            b.ClearObservedValue();
            Test("a", engine.Infer<IDistribution<string>>(a), "Hello");
            Test("b", engine.Infer<IDistribution<string>>(b), "World");
            Test("c2", engine.Infer<IDistribution<string>>(c2), "Hello World");

            b.ObservedValue = "World";
            Test("a", engine.Infer<IDistribution<string>>(a), "Hello");
            Test("c2", engine.Infer<IDistribution<string>>(c2), "Hello World");

            b.ClearObservedValue();
            a.ObservedValue = "Hello";
            Test("b", engine.Infer<IDistribution<string>>(b), "World");
            Test("c2", engine.Infer<IDistribution<string>>(c2), "Hello World");
        }

        private static void Test(string name, IDistribution<string> dist, params string[] vals)
        {
            var sa = (StringDistribution)dist;
            Console.Write(name + "=" + sa);
            double sum = 0;
            foreach (var s in vals)
            {
                var logProb = dist.GetLogProb(s);
                sum += Math.Exp(logProb);
            }

            var ok = Math.Abs(sum - 1.0) < 1E-8;
            var valstr = string.Join("|", vals.Select(s => s + "$").ToArray());
            Assert.True(ok, $"Result was {sa} should be ({valstr})");
        }

        private static void TestUnif(string name, IDistribution<string> dist, params string[] vals)
        {
            var sa = (StringDistribution)dist;
            Console.WriteLine(name + "=" + sa);
            var unifLogProb = -Math.Log(vals.Length);
            foreach (var s in vals)
            {
                StringInferenceTestUtilities.TestLogProbability(sa, unifLogProb, s);
            }
        }

        [Fact]
        public void StringFormatTest1()
        {
            var a = Variable.Random(StringDistribution.OneOf("World", "Universe"));
            var b = Variable.StringFormat("Hello {0}!!", a);

            var engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            Test("a", engine.Infer<IDistribution<string>>(a), "World", "Universe");
            Test("b", engine.Infer<IDistribution<string>>(b), "Hello World!!", "Hello Universe!!");
            a.ObservedValue = "World";
            Test("b", engine.Infer<IDistribution<string>>(b), "Hello World!!");
            a.ClearObservedValue();
            b.ObservedValue = "Hello World!!";
            Test("a", engine.Infer<IDistribution<string>>(a), "World");

            var a2 = Variable.Random(WordString());
            var template = Variable.Random(WordStrings.WordPrefix() + StringDistribution.String("{0}") + WordStrings.WordSuffix());
            var b2 = Variable.StringFormat(template, a2);
            b2.ObservedValue = "Hello World!!";
            Test("t", engine.Infer<IDistribution<string>>(template), "Hello {0}!!", "{0}!!", "{0} World!!");
            Test("a2", engine.Infer<IDistribution<string>>(a2), "Hello", "World", "Hello World");
            a2.ObservedValue = "World";
            Test("t", engine.Infer<IDistribution<string>>(template), "Hello {0}!!");
        }

        [Fact]
        public void StringFormatTest2()
        {
            var templates = Variable.Observed(new string[] { "My name is {0}.", "I'm {0}." });
            templates.Name = nameof(templates);
            var a = Variable.Random(StringDistribution.OneOf("John", "Tom"));
            a.Name = nameof(a);
            var templateNumber = Variable.DiscreteUniform(templates.Range);
            templateNumber.Name = nameof(templateNumber);
            var b = Variable.New<string>().Named("b");
            using (Variable.Switch(templateNumber))
            {
                b.SetTo(Variable.StringFormat(templates[templateNumber], a));
            }

            var engine = new InferenceEngine();
            engine.Compiler.GivePriorityTo(typeof(ReplicateOp_NoDivide));
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            Test("a", engine.Infer<IDistribution<string>>(a), "John", "Tom");
            Test("b", engine.Infer<IDistribution<string>>(b), "My name is John.", "My name is Tom.", "I'm John.", "I'm Tom.");
            b.ObservedValue = "My name is John.";
            var tempNum = engine.Infer<Discrete>(templateNumber);
            Assert.Equal(new Discrete(1.0, 0.0), tempNum);
            Test("a", engine.Infer<IDistribution<string>>(a), "John");
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void StringFormatTest3()
        {
            var T = new Range(2).Named("T");
            var templates = Variable.Array<string>(T).Named("templates");
            //templates[T] = Variable.Random(WordStrings.WordPrefix() + StringAutomaton.String("{0}") + WordStrings.WordSuffix()).ForEach(T);
            templates[T] = Variable.Random(StringDistribution.Any()).ForEach(T);
            var a = Variable.Random(StringDistribution.OneOf("John", "Tom")).Named("a");
            var N = new Range(2).Named("N");
            var templateNumber = Variable.Array<int>(N).Named("templateNumber");
            var templateNumberPrior = Variable.Observed(new Vector[] { Vector.FromArray(0.4, 0.6), Vector.FromArray(0.6, 0.4) }, N);
            var b = Variable.Array<string>(N).Named("b");
            using (Variable.ForEach(N))
            {
                templateNumber[N] = Variable.Discrete(T, templateNumberPrior[N]);
                using (Variable.Switch(templateNumber[N]))
                {
                    b[N] = Variable.StringFormat(templates[templateNumber[N]], a);
                }
            }

            Variable.ConstrainEqual(templateNumber[0], 0);

            var engine = new InferenceEngine();
            engine.NumberOfIterations = 5;
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.GivePriorityTo(typeof(ReplicateOp_NoDivide));
            engine.Compiler.UnrollLoops = true;

            Test("a", engine.Infer<IDistribution<string>>(a), "John", "Tom");
            Console.WriteLine(engine.Infer(b));
            b.ObservedValue = new string[] { "My name is John.", "I'm John." };
            Console.WriteLine(engine.Infer(templateNumber));
            Console.WriteLine(engine.Infer(templates));
            Test("a", engine.Infer<IDistribution<string>>(a), "John");
        }

        // not really a test, but an interesting experiment
        [Fact]
        public void StringFormatSimpleTest()
        {
            // number of objects
            var J = new Range(2).Named("J");
            var names = Variable.Array<string>(J).Named("names");
            names[J] = Variable.Random(WordString()).ForEach(J);

            Variable.ConstrainEqual(names[0], "John");
            var i = Variable.DiscreteUniform(J);
            var template = Variable.Random(StringDistribution.Any());
            var s = Variable.New<string>();
            using (Variable.Switch(i))
            {
                s.SetTo(Variable.StringFormat(template, names[i]));
            }

            s.ObservedValue = "John was here";

            var engine = new InferenceEngine { NumberOfIterations = 10 };
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.GivePriorityTo(typeof(ReplicateOp_NoDivide));
            Console.WriteLine("posterior=" + engine.Infer(i));
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void StringFormatTest4()
        {
            // number of templates
            var T = new Range(2).Named("T");
            var templates = Variable.Array<string>(T).Named("templates");
            templates[T] = Variable.Random(StringDistribution.Any()).ForEach(T);

            // number of objects
            var J = new Range(2).Named("J");
            var names = Variable.Array<string>(J).Named("names");
            names[J] = Variable.Random(WordString()).ForEach(J);

            // number of strings
            var N = new Range(3).Named("N");
            var objectNumber = Variable.Array<int>(N).Named("objectNumber");
            var templateNumber = Variable.Array<int>(N).Named("templateNumber");
            var texts = Variable.Array<string>(N).Named("b");

            using (Variable.ForEach(N))
            {
                objectNumber[N] = Variable.DiscreteUniform(J);
                var name = Variable.New<string>().Named("name");
                using (Variable.Switch(objectNumber[N]))
                {
                    name.SetTo(Variable.Copy(names[objectNumber[N]]));
                }

                templateNumber[N] = Variable.DiscreteUniform(T);
                using (Variable.Switch(templateNumber[N]))
                {
                    texts[N] = Variable.StringFormat(templates[templateNumber[N]], name);
                }
            }

            Variable.ConstrainEqual(names[0], "John");
            Variable.ConstrainEqual(templateNumber[0], 0); // break symmetry

            // Initialise templateNumber to break symmetry
            Rand.Restart(0);
            var tempNumInit = new Discrete[N.SizeAsInt];
            for (var i = 0; i < tempNumInit.Length; i++)
            {
                tempNumInit[i] = Discrete.PointMass(Rand.Int(T.SizeAsInt), T.SizeAsInt);
            }

            templateNumber.InitialiseTo(Distribution<int>.Array(tempNumInit));

            var engine = new InferenceEngine();
            engine.NumberOfIterations = 15;
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.GivePriorityTo(typeof(ReplicateOp_NoDivide));
            engine.Compiler.UnrollLoops = true;

            texts.ObservedValue = new string[] { "My name is John", "I'm John", "I'm Tom" };
            Console.WriteLine("templateNumber: \n" + engine.Infer(templateNumber));
            Console.WriteLine("objectNumber: \n" + engine.Infer(objectNumber));
            Console.WriteLine("templates: \n" + engine.Infer(templates));
            Console.WriteLine("names: \n" + engine.Infer(names));
        }

        [Fact]
        public void SemanticWebTest1()
        {
            var prop0 = "Tony Blair";
            var prop1 = "6 May 1953";
            var template = Variable.Random(StringDistribution.Any());
            var text = Variable.StringFormat(template, prop0, prop1);

            var engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.NumberOfIterations = 1;

            var textDist = engine.Infer<StringDistribution>(text);
            Console.WriteLine("textDist={0}", textDist);

            Assert.False(double.IsNegativeInfinity(textDist.GetLogProb("6 May 1953 is the date of birth of Tony Blair.")));
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void SemanticWebTest2()
        {
            var prop0Dist = StringDistribution.OneOf("Anthony Blair", "Tony Blair");
            var prop1Dist = StringDistribution.OneOf("6 May 1953", "May 6, 1953");

            var prop0 = Variable.Random(prop0Dist);
            var prop1 = Variable.Random(prop1Dist);
            var template = Variable.Random(StringDistribution.Any());
            var text = Variable.StringFormat(template, prop0, prop1);

            var engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.NumberOfIterations = 1;

            var textDist = engine.Infer<StringDistribution>(text);
            Console.WriteLine("textDist={0}", textDist);

            Assert.False(double.IsNegativeInfinity(textDist.GetLogProb("6 May 1953 is the date of birth of Tony Blair.")));
            Assert.False(double.IsNegativeInfinity(textDist.GetLogProb("6 May 1953 is the date of birth of Anthony Blair.")));
            Assert.False(double.IsNegativeInfinity(textDist.GetLogProb("Mr. Tony Blair was born on May 6, 1953.")));
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void SemanticWebTest2b()
        {
            var prop0Dist = StringDistribution.OneOf("Anthony Blair", "Tony Blair");
            var prop0 = Variable.Random(prop0Dist);

            var dateStrings = Variable.Observed(new[] { "6 May 1953", "May 6, 1953" });
            var dateFormat = Variable.DiscreteUniform(dateStrings.Range);
            var prop1 = ArrayIndex(dateStrings, dateFormat);

            var template = Variable.Random(StringDistribution.Any());
            var text = Variable.StringFormat(template, prop0, prop1);

            var engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.NumberOfIterations = 1;

            var textDist = engine.Infer<StringDistribution>(text);
            Console.WriteLine("textDist={0}", textDist);

            Assert.False(double.IsNegativeInfinity(textDist.GetLogProb("6 May 1953 is the date of birth of Tony Blair.")));
            Assert.False(double.IsNegativeInfinity(textDist.GetLogProb("6 May 1953 is the date of birth of Anthony Blair.")));
            Assert.False(double.IsNegativeInfinity(textDist.GetLogProb("Mr. Tony Blair was born on May 6, 1953.")));
        }

        [Fact]
        public void SemanticWebTest3()
        {
            // Simplified name model 
            var prop0 = Variable.Random(NamePrior()).Named("prop0");
            // Simplified date model any string ending in a digit
            var prop1 = Variable.Random(
                    StringDistribution.Char(DiscreteChar.Digit()) + StringDistribution.Any() + StringDistribution.Char(DiscreteChar.Digit()))
                .Named("prop1");
            var template = Variable.Random(StringDistribution.String("{0}") + WordStrings.WordMiddle() + StringDistribution.String("{1}"))
                .Named("template");
            var text = Variable.StringFormat(template, prop0, prop1).Named("text");
            text.ObservedValue = "Tony Blair was born on 6 May 1953";

            var engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.NumberOfIterations = 1;

            var prop0Dist = engine.Infer<StringDistribution>(prop0);
            var prop1Dist = engine.Infer<StringDistribution>(prop1);
            var templateDist = engine.Infer<StringDistribution>(template);

            StringInferenceTestUtilities.TestProbability(prop0Dist, 1.0, "Tony Blair");
            StringInferenceTestUtilities.TestProbability(prop1Dist, 0.5, "1953", "6 May 1953");
            StringInferenceTestUtilities.TestProbability(templateDist, 0.5, "{0} was born on {1}", "{0} was born on 6 May {1}");
        }

        [Fact]
        public void SemanticWebTest3b()
        {
            var prop0 = Variable.Random(NamePrior());

            var dateStrings = Variable.Observed(new[] { "6 May 1953", "May 6, 1953" });
            var dateFormat = Variable.DiscreteUniform(dateStrings.Range);
            var prop1 = ArrayIndex(dateStrings, dateFormat);

            var template = Variable.Random(StringDistribution.String("{0}") + WordStrings.WordMiddle() + StringDistribution.String("{1}"));
            var text = Variable.StringFormat(template, prop0, prop1);
            text.ObservedValue = "Tony Blair was born on May 6, 1953";

            var engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.NumberOfIterations = 1;

            var prop0Dist = engine.Infer<StringDistribution>(prop0);
            var prop1Dist = engine.Infer<StringDistribution>(prop1);
            var templateDist = engine.Infer<StringDistribution>(template);
            var dateFormatDist = engine.Infer<Discrete>(dateFormat);

            StringInferenceTestUtilities.TestProbability(prop0Dist, 1.0, "Tony Blair");
            StringInferenceTestUtilities.TestProbability(prop1Dist, 1.0, "May 6, 1953");
            StringInferenceTestUtilities.TestProbability(templateDist, 1.0, "{0} was born on {1}");

            //Assert.AreEqual(1.0, dateFormatDist[1]); // TODO: fix me
        }

        [Fact]
        public void SemanticWebTest4()
        {
            var name = Variable.Random(NamePrior());
            name.Name = nameof(name);

            var dateStrings = Variable.Observed(new[] { "6 May 1953", "May 6, 1953" });
            dateStrings.Name = nameof(dateStrings);
            var dateFormat = Variable.DiscreteUniform(dateStrings.Range);
            dateFormat.Name = nameof(dateFormat);
            var dob = ArrayIndex(dateStrings, dateFormat);
            dob.Name = nameof(dob);

            var template = Variable.Random(StringDistribution.String("{0}") + WordStrings.WordMiddle() + StringDistribution.String("{1}"));
            template.Name = nameof(template);
            var text = Variable.StringFormat(template, name, dob);
            text.Name = nameof(text);

            var template2 = Variable.Random(StringDistribution.Any());
            var fulltext = Variable.StringFormat(template2, text);

            fulltext.ObservedValue = "I believe Mr Tony Blair was born on May 6, 1953 in Edinburgh, UK.";

            var engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.NumberOfIterations = 2;

            var nameDist = engine.Infer<StringDistribution>(name);
            var dobDist = engine.Infer<StringDistribution>(dob);
            var templateDist = engine.Infer<StringDistribution>(template);
            var textDist = engine.Infer<StringDistribution>(text);
            var dateFormatDist = engine.Infer<Discrete>(dateFormat);

            StringInferenceTestUtilities.TestProbability(nameDist, 0.5, "Mr Tony", "Tony Blair");
            StringInferenceTestUtilities.TestProbability(dobDist, 1.0, "May 6, 1953");
            StringInferenceTestUtilities.TestProbability(templateDist, 0.5, "{0} was born on {1}", "{0} Blair was born on {1}");
            StringInferenceTestUtilities.TestProbability(
                textDist,
                0.5,
                "Tony Blair was born on May 6, 1953",
                "Mr Tony Blair was born on May 6, 1953");

            Assert.Equal(1.0, dateFormatDist[1]);
        }

        #region Helpers

        private static Variable<string> ArrayIndex(VariableArray<string> array, Variable<int> index)
        {
            var res = Variable.Random(StringDistribution.Any());
            using (Variable.Switch(index))
            {
                Variable.ConstrainEqual(res, array[index]);
            }

            return res;
        }

        private static StringDistribution NamePrior()
        {
            //TODO: make this closer to: 
            // NP([\s\-]NP)*(\s[""\(]NP[""\)])?([\s\-]NP)+
            var result = NamePart();
            result.AppendInPlace(DiscreteChar.OneOf(' ', '-'));
            result.AppendInPlace(NamePart());

            return result;
        }

        private static StringDistribution NamePart() => StringDistribution.Char(DiscreteChar.Upper()) + StringDistribution.Lower(minLength: 0);

        private static StringDistribution WordString() => StringDistribution.OneOrMore(ImmutableDiscreteChar.InRanges("azAZ09  __''\t\r"));

        private static StringDistribution TemplateArgString() =>
            StringDistribution.Char('{') + StringDistribution.Char(DiscreteChar.Digit()) + StringDistribution.Char('}');

        #endregion
    }
}
