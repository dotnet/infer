// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;

    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Utilities;
    using System.Threading.Tasks;

    /// <summary>
    /// Tests for all implementations of StringFormat operation.
    /// </summary>
    public class StringFormatOpTests
    {
        /// <summary>
        /// Tests message operators for messages to <c>str</c> directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessagesToStr()
        {
            TestMessageToStr(
                StringDistribution.String("A template with {0}."),
                new[] { StringDistribution.String("arg0") },
                StringDistribution.String("A template with arg0."));

            TestMessageToStr(
                StringDistribution.String("A template with {0}}."),
                new[] { StringDistribution.String("arg0") },
                StringDistribution.Zero());

            TestMessageToStr(
                StringDistribution.String("A template with {{0}."),
                new[] { StringDistribution.String("arg0") },
                StringDistribution.Zero());

            TestMessageToStr(
                StringDistribution.String("A template with {}."),
                new[] { StringDistribution.String("arg0") },
                StringDistribution.Zero());

            TestMessageToStr(
                StringDistribution.String("A template with {0}{."),
                new[] { StringDistribution.String("arg0") },
                StringDistribution.Zero());

            TestMessageToStr(
                StringDistribution.String("A template with {0}{}."),
                new[] { StringDistribution.String("arg0") },
                StringDistribution.Zero());

            TestMessageToStr(
                StringDistribution.OneOf("A template with {0{}.", "{0}}", "{{0}", "{0", "0}"),
                new[] { StringDistribution.String("arg0") },
                StringDistribution.Zero());

            TestMessageToStr(
                StringDistribution.String("A template with {0} and {1}."),
                new[] { StringDistribution.String("arg0"), StringDistribution.String("arg1") },
                StringDistribution.String("A template with arg0 and arg1."));

            TestMessageToStr(
                StringDistribution.String("A template with {1} and {0}."),
                new[] { StringDistribution.String("arg0"), StringDistribution.String("arg1") },
                StringDistribution.String("A template with arg1 and arg0."));

            TestMessageToStr(
                StringDistribution.String("A template with {1} and {0}."),
                new[] { StringDistribution.String("{0}"), StringDistribution.String("{1}") },
                StringDistribution.String("A template with {1} and {0}."));

            TestMessageToStr(
                StringDistribution.String("A template with {0}, {1} and {0}."),
                new[] { StringDistribution.String("arg0"), StringDistribution.String("arg1") },
                StringDistribution.Zero());

            TestMessageToStr(
                StringDistribution.String("A template with {0}, {1} and {2}."),
                new[] { StringDistribution.String("arg0"), StringDistribution.String("arg1") },
                StringDistribution.Zero());

            TestMessageToStr(
                StringDistribution.String("A template with {2}, {0} and {1}."),
                new[] { StringDistribution.String("arg0"), StringDistribution.String("arg1"), StringDistribution.String("arg2") },
                StringDistribution.String("A template with arg2, arg0 and arg1."));

            TestMessageToStr(
               StringDistribution.String("A template with {2}, {0} and {1}."),
               new[] { StringDistribution.String("{0}"), StringDistribution.String("arg1"), StringDistribution.String("arg2") },
                StringDistribution.String("A template with arg2, {0} and arg1."));

            TestMessageToStr(
                StringDistribution.OneOf("A template with {0}.", "A template with {0}{1}."),
                new[] { StringDistribution.OneOf("a", "aa"), StringDistribution.OneOf(string.Empty, "a") },
                StringDistribution.OneOf(
                    new Dictionary<string, double> { { "A template with a.", 1.0 }, { "A template with aa.", 2.0 }, { "A template with aaa.", 1.0 } }),
                StringDistribution.OneOf(
                    new Dictionary<string, double> { { "A template with a.", 2.0 }, { "A template with aa.", 3.0 }, { "A template with aaa.", 1.0 } }));

            TestMessageToStr(
                StringDistribution.String("A template with {1}."),
                new[] { StringDistribution.String("arg0"), StringDistribution.String("arg1") },
                StringDistribution.Zero(),
                StringDistribution.String("A template with arg1."));
        }

        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageToStr_Parallel()
        {
            var options = new ParallelOptions();
            options.MaxDegreeOfParallelism = 10;
            Parallel.For(0, 10, options, (k) =>
            {
                TestMessageToFormat(
                StringDistribution.String("A template with arg and arg."),
                new[] { StringDistribution.String("arg"), StringDistribution.String("arg") },
                StringDistribution.OneOf("A template with {0} and {1}.", "A template with {1} and {0}."),
                StringDistribution.OneOf(
                    "A template with {0} and {1}.",
                    "A template with {1} and {0}.",
                    "A template with {0} and arg.",
                    "A template with {1} and arg.",
                    "A template with arg and {0}.",
                    "A template with arg and {1}.",
                    "A template with arg and arg."));
            });
        }

        /// <summary>
        /// Tests message operators for messages to <c>format</c> directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessagesToFormat()
        {
            TestMessageToFormat(
                StringDistribution.String("A template with arg0 and arg1."),
                new[] { StringDistribution.String("arg0"), StringDistribution.String("arg1") },
                StringDistribution.String("A template with {0} and {1}."),
                StringDistribution.OneOf("A template with {0} and {1}.", "A template with arg0 and {1}.", "A template with {0} and arg1.", "A template with arg0 and arg1."));

            TestMessageToFormat(
                StringDistribution.String("A template with arg and arg."),
                new[] { StringDistribution.String("arg"), StringDistribution.String("arg") },
                StringDistribution.OneOf("A template with {0} and {1}.", "A template with {1} and {0}."),
                StringDistribution.OneOf(
                    "A template with {0} and {1}.",
                    "A template with {1} and {0}.",
                    "A template with {0} and arg.",
                    "A template with {1} and arg.",
                    "A template with arg and {0}.",
                    "A template with arg and {1}.",
                    "A template with arg and arg."));

            TestMessageToFormat(
                StringDistribution.String("A template with arg and arg."),
                new[] { StringDistribution.String("arg") },
                StringDistribution.OneOf("A template with {0} and arg.", "A template with arg and {0}."),
                StringDistribution.OneOf(
                    "A template with {0} and arg.",
                    "A template with arg and {0}.",
                    "A template with arg and arg."));

            TestMessageToFormat(
                StringDistribution.String("A template with {0} and arg."),
                new[] { StringDistribution.Any(minLength: 1, maxLength: 4) },
                StringDistribution.OneOf("A template with {0} and arg.", "A template with {0}and arg.", "A template with{0} and arg."));

            TestMessageToFormat(
                StringDistribution.String("Abba."),
                new[] { StringDistribution.String("A"), StringDistribution.String("ba") },
                StringDistribution.String("{0}b{1}."),
                StringDistribution.OneOf("Abba.", "{0}b{1}.", "Ab{1}.", "{0}bba."));

            TestMessageToFormat(
                StringDistribution.String("A template with arg2."),
                new[] { StringDistribution.String("arg1"), StringDistribution.String("arg2") },
                StringDistribution.Zero(),
                StringDistribution.OneOf("A template with arg2.", "A template with {1}."));

            TestMessageToFormat(
                StringDistribution.String("aaa."),
                new[] { StringDistribution.String("aa") },
                StringDistribution.OneOf("{0}a.", "a{0}."),
                StringDistribution.OneOf("{0}a.", "a{0}.", "aaa."));

            TestMessageToFormat(
                StringDistribution.OneOf("A template with arg.", "A template with arg and arg."),
                new[] { StringDistribution.String("arg") },
                StringDistribution.OneOf(   
                    "A template with {0}.",
                    "A template with {0} and arg.",
                    "A template with arg and {0}."),
                StringDistribution.OneOf(   
                    "A template with {0}.",
                    "A template with arg.",
                    "A template with {0} and arg.",
                    "A template with arg and {0}.",
                    "A template with arg and arg."));
        }

        /// <summary>
        /// Tests message operators for messages to <c>args</c> directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageToArgs1()
        {
            TestMessageToArgs(
                StringDistribution.String("A template with arg0 and arg1."),
                StringDistribution.String("A template with {0} and {1}."),
                Util.ArrayInit(2, i => StringDistribution.Any()),
                new[] { StringDistribution.String("arg0"), StringDistribution.String("arg1") });
        }

        /// <summary>
        /// Tests message operators for messages to <c>args</c> directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageToArgs2()
        {
            TestMessageToArgs(
                StringDistribution.String("A template with arg0 and arg1."),
                StringDistribution.String("A template with {0} and {1}."),
                Util.ArrayInit(2, i => StringDistribution.Any()),
                new[] { StringDistribution.String("arg0"), StringDistribution.String("arg1") });
        }

        /// <summary>
        /// Tests message operators for messages to <c>args</c> directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageToArgs3()
        {
            TestMessageToArgs(
                StringDistribution.String("A template with ab."),
                StringDistribution.String("A template with {0}{1}."),
                Util.ArrayInit(2, i => StringDistribution.Any()),
                new[] { StringDistribution.OneOf("","a","ab"), StringDistribution.OneOf("","ab","b") });
        }

        /// <summary>
        /// Tests message operators for messages to <c>args</c> directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageToArgs4()
        {
            TestMessageToArgs(
                StringDistribution.String("A template with ab."),
                StringDistribution.String("A template with {1}."),
                Util.ArrayInit(2, i => StringDistribution.Any()),
                new[] { StringDistribution.Zero(), StringDistribution.Zero() },
                new[] { StringDistribution.Any(), StringDistribution.String("ab") });
        }

        /// <summary>
        /// Tests message operators for messages to <c>args</c> directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        [Trait("Category", "OpenBug")]
        public void MessageToArgs5()
        {
            TestMessageToArgs(
                StringDistribution.String("A and ."),
                StringDistribution.String("{0} and {1}."),
                Util.ArrayInit(2, i => StringDistribution.Any()),
                new[] { StringDistribution.String("A"), StringDistribution.String("") });
        }

        /// <summary>
        /// Tests message operators for messages to <c>args</c> directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageToArgs6()
        {
            TestMessageToArgs(
                StringDistribution.String("X"),
                StringDistribution.String("{0}{1}"),
                Util.ArrayInit(2, i => StringDistribution.Any()),
                new[] { StringDistribution.OneOf("", "X"), StringDistribution.OneOf("", "X") });
        }

        /// <summary>
        /// Tests message operators for messages to <c>args</c> directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageToArgs7()
        {
            TestMessageToArgs(
                StringDistribution.String("XXX"),
                StringDistribution.String("{0}{1}"),
                Util.ArrayInit(2, i => StringDistribution.Any()),
                Util.ArrayInit(2, i => StringDistribution.OneOf("","X", "XX","XXX")));
        }

        /// <summary>
        /// Tests message operators for messages to <c>args</c> directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageToArgs8()
        {
            TestMessageToArgs(
                StringDistribution.OneOf("xxxYYY"),
                StringDistribution.String("{0}{1}"),
                new[] { StringDistribution.Lower(minLength: 3, maxLength: 3), StringDistribution.Upper(minLength: 3, maxLength: 3) },
                new[] { StringDistribution.String("xxx"), StringDistribution.String("YYY") });
        }

        /// <summary>
        /// Tests evidence message computation.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void EvidenceMessages()
        {
            TestEvidence(StringDistribution.OneOf("a", "b"));
            
            TestEvidence("a!", StringDistribution.OneOf("{0}", "{0}!"), new[] { StringDistribution.OneOf("a", "a!") }, Math.Log(0.5));
            
            TestEvidence("b!", StringDistribution.OneOf("{0}", "{0}!"), new[] { StringDistribution.OneOf("a", "a!") }, double.NegativeInfinity);
        }

        /// <summary>
        /// The number of states in the message to <c>str</c> grows exponentially as the <c>ArgCount</c> grows.
        /// The test will fail for <c>ArgCount > 6</c>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void InefficientPermutationRepresentation()
        {
            StringDistribution placeholder = StringDistribution.Char('{') + StringDistribution.Char(DiscreteChar.Digit()) + StringDistribution.Char('}');
            StringDistribution formatPrior = placeholder + StringDistribution.String(" ");
            formatPrior = StringDistribution.ZeroOrMore(formatPrior) + placeholder;
            
            const int ArgCount = 6;
            const int ArgLength = 3;
            StringDistribution[] args = Util.ArrayInit(ArgCount, i => StringDistribution.String(new string((char)('0' + i), ArgLength)));
            
            StringDistribution str = StringFormatOp_AllowMissingPlaceholders_NoArgumentNames.StrAverageConditional(formatPrior, args);
            
            Console.WriteLine(str.GetWorkspaceOrPoint().States.Count);

            StringInferenceTestUtilities.TestIfIncludes(str, "111 333 222", "111 222", "222 111", "111", "222", "333");
            StringInferenceTestUtilities.TestIfExcludes(str, string.Empty, "111 111", "222 222 111", "333 111 222 333", "112", "1");
        }

        /// <summary>
        /// A helper function for testing messages to <c>str</c>.
        /// </summary>
        /// <param name="format">The message from <c>format</c>.</param>
        /// <param name="args">The message from <c>args</c>.</param>
        /// <param name="expectedStrRequireEveryPlaceholder">
        /// The expected message to <c>str</c> if the format string is required to contain placeholders for all arguments.
        /// </param>
        /// <param name="expectedStrAllowMissingPlaceholders">
        /// The expected message to <c>str</c> if the format string may not contain placeholders for some arguments.
        /// </param>
        private static void TestMessageToStr(
            StringDistribution format,
            StringDistribution[] args,
            StringDistribution expectedStrRequireEveryPlaceholder,
            StringDistribution expectedStrAllowMissingPlaceholders)
        {
            string[] argNames = GetDefaultArgumentNames(args.Length);
            Assert.Equal(
                expectedStrRequireEveryPlaceholder,
                StringFormatOp_RequireEveryPlaceholder_NoArgumentNames.StrAverageConditional(format, args));
            Assert.Equal(
                expectedStrRequireEveryPlaceholder,
                StringFormatOp_RequireEveryPlaceholder.StrAverageConditional(format, args, argNames));
            Assert.Equal(
                expectedStrAllowMissingPlaceholders,
                StringFormatOp_AllowMissingPlaceholders_NoArgumentNames.StrAverageConditional(format, args));
            Assert.Equal(
                expectedStrAllowMissingPlaceholders,
                StringFormatOp_AllowMissingPlaceholders.StrAverageConditional(format, args, argNames));
        }

        /// <summary>
        /// A helper function for testing messages to <c>str</c>.
        /// </summary>
        /// <param name="format">The message from <c>format</c>.</param>
        /// <param name="args">The message from <c>args</c>.</param>
        /// <param name="expectedStr">The expected message to <c>str</c>.</param>
        private static void TestMessageToStr(
            StringDistribution format,
            StringDistribution[] args,
            StringDistribution expectedStr)
        {
            TestMessageToStr(format, args, expectedStr, expectedStr);
        }

        /// <summary>
        /// A helper function for testing messages to <c>format</c>.
        /// </summary>
        /// <param name="str">The message from <c>str</c>.</param>
        /// <param name="args">The message from <c>args</c>.</param>
        /// <param name="expectedFormatRequireEveryPlaceholder">
        /// The expected message to <c>format</c> if the format string is required to contain placeholders for all arguments.
        /// </param>
        /// <param name="expectedFormatAllowMissingPlaceholders">
        /// The expected message to <c>format</c> if the format string may not contain placeholders for some arguments.
        /// </param>
        private static void TestMessageToFormat(
            StringDistribution str,
            StringDistribution[] args,
            StringDistribution expectedFormatRequireEveryPlaceholder,
            StringDistribution expectedFormatAllowMissingPlaceholders)
        {
            string[] argNames = GetDefaultArgumentNames(args.Length);
            Assert.Equal(
                expectedFormatRequireEveryPlaceholder,
                StringFormatOp_RequireEveryPlaceholder_NoArgumentNames.FormatAverageConditional(str, args));
            Assert.Equal(
                expectedFormatRequireEveryPlaceholder,
                StringFormatOp_RequireEveryPlaceholder.FormatAverageConditional(str, args, argNames));
            Assert.Equal(
                expectedFormatAllowMissingPlaceholders,
                StringFormatOp_AllowMissingPlaceholders_NoArgumentNames.FormatAverageConditional(str, args));
            Assert.Equal(
                expectedFormatAllowMissingPlaceholders,
                StringFormatOp_AllowMissingPlaceholders.FormatAverageConditional(str, args, argNames));
        }

        /// <summary>
        /// A helper function for testing messages to <c>format</c>.
        /// </summary>
        /// <param name="str">The message from <c>str</c>.</param>
        /// <param name="args">The message from <c>args</c>.</param>
        /// <param name="expectedFormat">The expected message to <c>format</c>.</param>
        private static void TestMessageToFormat(
            StringDistribution str,
            StringDistribution[] args,
            StringDistribution expectedFormat)
        {
            TestMessageToFormat(str, args, expectedFormat, expectedFormat);
        }

        /// <summary>
        /// A helper function for testing messages to <c>args</c>.
        /// </summary>
        /// <param name="str">The message from <c>str</c>.</param>
        /// <param name="format">The message from <c>format</c>.</param>
        /// <param name="args">The message from <c>args</c>.</param>
        /// <param name="expectedArgsRequireEveryPlaceholder">
        /// The expected message to <c>args</c> if the format string is required to contain placeholders for all arguments.
        /// </param>
        /// <param name="expectedArgsAllowMissingPlaceholders">
        /// The expected message to <c>args</c> if the format string may not contain placeholders for some arguments.
        /// </param>
        private static void TestMessageToArgs(
            StringDistribution str,
            StringDistribution format,
            StringDistribution[] args,
            StringDistribution[] expectedArgsRequireEveryPlaceholder,
            StringDistribution[] expectedArgsAllowMissingPlaceholders)
        {
            string[] argNames = GetDefaultArgumentNames(args.Length);
            var result = new StringDistribution[args.Length];
            Assert.Equal(
                expectedArgsRequireEveryPlaceholder,
                StringFormatOp_RequireEveryPlaceholder_NoArgumentNames.ArgsAverageConditional(str, format, args, result));
            Assert.Equal(
                expectedArgsRequireEveryPlaceholder,
                StringFormatOp_RequireEveryPlaceholder.ArgsAverageConditional(str, format, args, argNames, result));
            Assert.Equal(
                expectedArgsAllowMissingPlaceholders,
                StringFormatOp_AllowMissingPlaceholders_NoArgumentNames.ArgsAverageConditional(str, format, args, result));
            var actualArgs = StringFormatOp_AllowMissingPlaceholders.ArgsAverageConditional(str, format, args, argNames, result);
            Assert.Equal(expectedArgsAllowMissingPlaceholders, actualArgs);
        }

        /// <summary>
        /// A helper function for testing messages to <c>args</c>.
        /// </summary>
        /// <param name="str">The message from <c>str</c>.</param>
        /// <param name="format">The message from <c>format</c>.</param>
        /// <param name="args">The message from <c>args</c>.</param>
        /// <param name="expectedArgs">The expected message to <c>args</c>.</param>
        private static void TestMessageToArgs(
            StringDistribution str,
            StringDistribution format,
            StringDistribution[] args,
            StringDistribution[] expectedArgs)
        {
            TestMessageToArgs(str, format, args, expectedArgs, expectedArgs);
        }

        /// <summary>
        /// A helper function for evidence messages when the output of the factor is not observed.
        /// </summary>
        /// <param name="str">The message to <c>str</c>.</param>.
        private static void TestEvidence(StringDistribution str)
        {
            Assert.Equal(0, StringFormatOp_RequireEveryPlaceholder_NoArgumentNames.LogEvidenceRatio(str));
            Assert.Equal(0, StringFormatOp_RequireEveryPlaceholder.LogEvidenceRatio(str));
            Assert.Equal(0, StringFormatOp_AllowMissingPlaceholders_NoArgumentNames.LogEvidenceRatio(str));
            Assert.Equal(0, StringFormatOp_AllowMissingPlaceholders.LogEvidenceRatio(str));
        }

        /// <summary>
        /// A helper function for evidence messages when the output of the factor is observed.
        /// </summary>
        /// <param name="str">The observed value of <c>str</c>.</param>
        /// <param name="format">The message from <c>format</c>.</param>
        /// <param name="args">The message from <c>args</c>.</param>
        /// <param name="expectedLogEvidence">
        /// The expected log-evidence if the format string is required to contain placeholders for all arguments.
        /// </param>
        /// <param name="expectedLogEvidenceWithMissingPlaceholders">
        /// The expected log-evidence if the format string may not contain placeholders for some arguments.
        /// </param>
        private static void TestEvidence(
            string str,
            StringDistribution format,
            StringDistribution[] args,
            double expectedLogEvidence,
            double expectedLogEvidenceWithMissingPlaceholders)
        {
            const double LogEvidenceEps = 1e-6;
            string[] argNames = GetDefaultArgumentNames(args.Length);

            Assert.Equal(
                expectedLogEvidence,
                StringFormatOp_RequireEveryPlaceholder_NoArgumentNames.LogEvidenceRatio(str, format, args),
                LogEvidenceEps);
            Assert.Equal(
                expectedLogEvidence,
                StringFormatOp_RequireEveryPlaceholder.LogEvidenceRatio(str, format, args, argNames),
                LogEvidenceEps);
            Assert.Equal(
                expectedLogEvidenceWithMissingPlaceholders,
                StringFormatOp_AllowMissingPlaceholders_NoArgumentNames.LogEvidenceRatio(str, format, args),
                LogEvidenceEps);
            Assert.Equal(
                expectedLogEvidenceWithMissingPlaceholders,
                StringFormatOp_AllowMissingPlaceholders.LogEvidenceRatio(str, format, args, argNames),
                LogEvidenceEps);
        }

        /// <summary>
        /// A helper function for evidence messages when the output of the factor is observed.
        /// </summary>
        /// <param name="str">The observed value of <c>str</c>.</param>
        /// <param name="format">The message from <c>format</c>.</param>
        /// <param name="args">The message from <c>args</c>.</param>
        /// <param name="expectedLogEvidence">The expected log-evidence.</param>
        private static void TestEvidence(
            string str,
            StringDistribution format,
            StringDistribution[] args,
            double expectedLogEvidence)
        {
            TestEvidence(str, format, args, expectedLogEvidence, expectedLogEvidence);
        }

        /// <summary>
        /// Creates an array of default argument names ("0", "1" and so on) for a given number of arguments.
        /// </summary>
        /// <param name="argumentCount">The number of arguments.</param>
        /// <returns>
        /// The created array of default argument names.
        /// </returns>
        private static string[] GetDefaultArgumentNames(int argumentCount)
        {
            return Util.ArrayInit(argumentCount, i => i.ToString(CultureInfo.InvariantCulture));
        }
    }
}
