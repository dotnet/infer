// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;

    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;

    using Microsoft.ML.Probabilistic.Distributions;

    /// <summary>
    /// Tests for factory methods defined in the <see cref="WordStrings" /> class.
    /// </summary>
    public class WordStringsTests
    {
        [Fact]
        public void WordPrefix()
        {
            var wordPrefix = WordStrings.WordPrefix();
            StringInferenceTestUtilities.TestIfIncludes(wordPrefix, "hello ", string.Empty, "abc.");
            StringInferenceTestUtilities.TestIfExcludes(wordPrefix, "hello", "hello world", "1 2");
            Assert.Equal(wordPrefix.GetLogProb(string.Empty), wordPrefix.GetLogProb("hello "));
            Assert.Equal(wordPrefix.GetLogProb(string.Empty), wordPrefix.GetLogProb("hello world and people in it."));
        }

        [Fact]
        public void WordSuffix()
        {
            var wordSuffix = WordStrings.WordSuffix();
            StringInferenceTestUtilities.TestIfIncludes(wordSuffix, " hello", string.Empty, ".abc");
            StringInferenceTestUtilities.TestIfExcludes(wordSuffix, "hello", "hello world", "1 2");
            Assert.Equal(wordSuffix.GetLogProb(string.Empty), wordSuffix.GetLogProb(" world"));
            Assert.Equal(wordSuffix.GetLogProb(string.Empty), wordSuffix.GetLogProb(". and that's it"));
        }

        [Fact]
        public void WordMiddle()
        {
            var wordMiddle = WordStrings.WordMiddle();
            StringInferenceTestUtilities.TestIfIncludes(wordMiddle, " hello world ", ". abc.", ". abc. def gh. ", " ", "..");
            StringInferenceTestUtilities.TestIfExcludes(wordMiddle, "hello", " world", "hi ", "1 2", string.Empty);
            Assert.Equal(wordMiddle.GetLogProb(" "), wordMiddle.GetLogProb(" world "));
        }

        [Fact]
        public void WordMiddleLimitedLength()
        {
            const double Eps = 1e-6;

            var wordMiddleLimitedLength1 = WordStrings.WordMiddle(3, 5);
            StringInferenceTestUtilities.TestIfIncludes(wordMiddleLimitedLength1, " h w ", ". ab.", "...");
            StringInferenceTestUtilities.TestIfExcludes(
                wordMiddleLimitedLength1,
                "hello",
                " w",
                "hi ",
                "1 2",
                ".",
                "..",
                " hello world ",
                string.Empty);
            Assert.Equal(wordMiddleLimitedLength1.GetLogProb(" h w "), wordMiddleLimitedLength1.GetLogProb(" h "), Eps);

            var wordMiddleLimitedLength2 = WordStrings.WordMiddle(1, 5);
            StringInferenceTestUtilities.TestIfIncludes(wordMiddleLimitedLength2, " ", " h w ", ". ab.", "...");
            StringInferenceTestUtilities.TestIfExcludes(wordMiddleLimitedLength2, "hello", " w", "hi ", "1 2", " hello world ", string.Empty);
            Assert.Equal(wordMiddleLimitedLength2.GetLogProb("."), wordMiddleLimitedLength2.GetLogProb(" h "), Eps);

            var wordMiddleLimitedLength3 = WordStrings.WordMiddle(1, 2);
            StringInferenceTestUtilities.TestIfIncludes(wordMiddleLimitedLength3, " ", " .");
            StringInferenceTestUtilities.TestIfExcludes(wordMiddleLimitedLength3, "hw", " w", "h ", "12", " hello world ", string.Empty);
            Assert.Equal(wordMiddleLimitedLength3.GetLogProb("."), wordMiddleLimitedLength3.GetLogProb(".."), Eps);
        }

        [Fact]
        public void OneOfTest()
        {
            var abc = StringDistribution.OneOf("a", "b", "c");
            var s = abc.ToString();
            Console.WriteLine("a or b or c:" + s);
            Assert.Equal("[a|b|c]", s);
        }
    }
}