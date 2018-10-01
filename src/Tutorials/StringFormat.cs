// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tutorials
{
    using System;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    [Example("String tutorials", "Using StringFormat operation to reason about strings", Prefix = "2.")]
    public class StringFormat
    {
        public void Run()
        {
            InferArgument();
            InferTemplate();
        }

        private static void InferArgument()
        {
            var engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;

            if (engine.Algorithm is Algorithms.ExpectationPropagation)
            {
                Variable<string> name = Variable.StringCapitalized().Named("name");
                Variable<string> text = Variable.StringFormat("My name is {0}.", name).Named("text");

                text.ObservedValue = "My name is John.";
                
                Console.WriteLine("name is '{0}'", engine.Infer(name));
            }
            else
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
            }
        }

        private static void InferTemplate()
        {
            var engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;

            if (engine.Algorithm is Algorithms.ExpectationPropagation)
            {
                Variable<string> name = Variable.StringCapitalized().Named("name");
                Variable<string> template =
                    (Variable.StringUniform() + Variable.CharNonWord() + "{0}" + Variable.CharNonWord() + Variable.StringUniform()).Named("template");
                Variable<string> text = Variable.StringFormat(template, name).Named("text");

                text.ObservedValue = "Hello, mate! I'm Dave.";

                Console.WriteLine("name is '{0}'", engine.Infer(name));
                Console.WriteLine("template is '{0}'", engine.Infer(template));

                text.ObservedValue = "Hi! My name is John.";

                Console.WriteLine("name is '{0}'", engine.Infer(name));
                Console.WriteLine("template is '{0}'", engine.Infer(template));

                Variable<string> name2 = Variable.StringCapitalized().Named("name2");
                Variable<string> text2 = Variable.StringFormat(template, name2).Named("text2");
                
                text2.ObservedValue = "Hi! My name is Tom.";
                
                Console.WriteLine("name is '{0}'", engine.Infer(name));
                Console.WriteLine("name2 is '{0}'", engine.Infer(name2));
                Console.WriteLine("template is '{0}'", engine.Infer(template));

                Variable<string> text3 = Variable.StringFormat(template, "Boris").Named("text3");

                Console.WriteLine("text3 is '{0}'", engine.Infer(text3));
            }
            else
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
            }
        }
    }
}
