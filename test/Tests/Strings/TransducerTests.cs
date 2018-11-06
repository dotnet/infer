// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Linq;
    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Distributions.Automata;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Tests for transducers.
    /// </summary>
    public class TransducerTests
    {
        /// <summary>
        /// Tests whether it's possible to create a transducer with a large number of states.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void LargeTransducer()
        {
            StringAutomaton.MaxStateCount = 1200000; // Something big
            StringAutomaton bigAutomaton = StringAutomaton.Zero();
            bigAutomaton.AddStates(StringAutomaton.MaxStateCount - bigAutomaton.States.Count);
            Func<DiscreteChar, Weight, Tuple<Option<PairDistribution<char, DiscreteChar>>, Weight>> transitionConverter =
                (dist, weight) => Tuple.Create(Option.Some(PairDistribution<char, DiscreteChar>.FromFirstSecond(dist, dist)), weight);
            
            Assert.Throws<AutomatonTooLargeException>(() => StringTransducer.FromAutomaton(bigAutomaton, transitionConverter));

            // Shouldn't throw if the maximum number of states is increased
            int prevMaxStateCount = StringTransducer.MaxStateCount;
            try
            {
                StringTransducer.MaxStateCount = StringAutomaton.MaxStateCount;
                StringTransducer.FromAutomaton(bigAutomaton, transitionConverter);
            }
            finally
            {
                StringTransducer.MaxStateCount = prevMaxStateCount;
            }
        }
        
        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.Consume(TSrcAutomaton)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConsumeAutomaton()
        {
            StringAutomaton automaton = StringAutomaton.Constant(2.0, DiscreteChar.Lower());
            automaton = automaton.Sum(StringAutomaton.ConstantOnElement(3.0, 'a'));
            StringTransducer consume = StringTransducer.Consume(automaton);
            
            StringInferenceTestUtilities.TestTransducerValue(consume, "aaa", string.Empty, 2.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, "bb", string.Empty, 2.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, "a", string.Empty, 5.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, string.Empty, string.Empty, 2.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, "bb", "aaa", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, "bb", "bb", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, string.Empty, "bb", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, string.Empty, "a", 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.Consume(TSrcSequence)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConsumeSequence()
        {
            StringTransducer consume = StringTransducer.Consume("abc");

            StringInferenceTestUtilities.TestTransducerValue(consume, "abc", string.Empty, 1.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, "ab", string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, "abcd", string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, string.Empty, string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, string.Empty, "abc", 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.ConsumeElement(TSrcElement)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConsumeElement()
        {
            StringTransducer consume = StringTransducer.ConsumeElement('x');

            StringInferenceTestUtilities.TestTransducerValue(consume, "x", string.Empty, 1.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, "xx", string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, string.Empty, string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, string.Empty, "x", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(consume, string.Empty, "xx", 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.Produce(TDestAutomaton)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ProduceAutomaton()
        {
            StringAutomaton automaton = StringAutomaton.Constant(2.0, DiscreteChar.Lower());
            automaton = automaton.Sum(StringAutomaton.ConstantOnElement(3.0, 'a'));
            StringTransducer produce = StringTransducer.Produce(automaton);
            
            StringInferenceTestUtilities.TestTransducerValue(produce, string.Empty, "aaa", 2.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, string.Empty, "bb", 2.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, string.Empty, "a", 5.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, string.Empty, string.Empty, 2.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, "aaa", "bb", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, "bb", "bb", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, "bb", string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, "a", string.Empty, 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.Produce(TDestSequence)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ProduceSequence()
        {
            StringTransducer produce = StringTransducer.Produce("abc");

            StringInferenceTestUtilities.TestTransducerValue(produce, string.Empty, "abc", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, string.Empty, "ab", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, string.Empty, "abcd", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, string.Empty, string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, "abc", string.Empty, 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.ProduceElement(TDestElement)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ProduceElement()
        {
            StringTransducer produce = StringTransducer.ProduceElement('x');

            StringInferenceTestUtilities.TestTransducerValue(produce, string.Empty, "x", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, string.Empty, "xx", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, string.Empty, string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, "x", string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(produce, "xx", string.Empty, 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.Replace(TSrcAutomaton, TDestAutomaton)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ReplaceAutomaton()
        {
            StringAutomaton automaton1 = StringAutomaton.Constant(2.0, DiscreteChar.Lower());
            automaton1 = automaton1.Sum(StringAutomaton.ConstantOnElement(3.0, 'a'));
            StringAutomaton automaton2 = StringAutomaton.Constant(0.5, DiscreteChar.Digit());
            automaton2 = automaton2.Sum(StringAutomaton.Constant(2.5, DiscreteChar.LetterOrDigit()));
            StringTransducer replace = StringTransducer.Replace(automaton1, automaton2);

            StringInferenceTestUtilities.TestTransducerValue(replace, string.Empty, "123", 6.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "a", "123", 15.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "ax", "AbC", 5.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "a", "a", 12.5);
            StringInferenceTestUtilities.TestTransducerValue(replace, string.Empty, string.Empty, 6.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "123", string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "AbC", "ax", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "1", "1", 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.Replace(TSrcSequence, TDestSequence)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ReplaceSequence()
        {
            StringTransducer replace = StringTransducer.Replace("hello", "worlds");

            StringInferenceTestUtilities.TestTransducerValue(replace, "hello", "worlds", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "worlds", "hello", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "worlds", "worlds", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "hello", "hello", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "hello", string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, string.Empty, "worlds", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, string.Empty, string.Empty, 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.Replace(TSrcElementDistribution, TDestElementDistribution)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ReplaceElements()
        {
            StringTransducer replace = StringTransducer.Replace(DiscreteChar.Lower(), DiscreteChar.Digit());

            StringInferenceTestUtilities.TestTransducerValue(replace, "hello", "123", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "w", "1337", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "w", string.Empty, 1.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, string.Empty, "17", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, string.Empty, string.Empty, 1.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "123", "worlds", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "123", "123", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "1", string.Empty, 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.Sum(IEnumerable{TThis})"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Sum()
        {
            StringTransducer replace = StringTransducer.Sum(
                StringTransducer.Replace(DiscreteChar.Lower(), DiscreteChar.Digit()),
                StringTransducer.Replace(DiscreteChar.Lower(), DiscreteChar.LetterOrDigit()));

            StringInferenceTestUtilities.TestTransducerValue(replace, "hello", "123", 2.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "w", "1337", 2.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "w", string.Empty, 2.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, string.Empty, "17", 2.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, string.Empty, string.Empty, 2.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "hello", "worlds", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "hello", "WORLDS111", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "123", "worlds", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "123", "123", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(replace, "1", string.Empty, 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.Repeat(TThis, int, int?)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat1()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a");
            StringTransducer repeat = StringTransducer.Repeat(StringTransducer.Consume(automaton), 1, 3);

            StringInferenceTestUtilities.TestTransducerValue(repeat, "a", string.Empty, 2.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, "aa", string.Empty, 4.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, "aaa", string.Empty, 8.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, "aaaa", string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, string.Empty, string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, string.Empty, "aaa", 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.Repeat(TThis, int, int?)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat2()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a", string.Empty);
            StringTransducer repeat = StringTransducer.Repeat(StringTransducer.Copy(automaton), 1, 2);

            StringInferenceTestUtilities.TestTransducerValue(repeat, "a", "a", 10.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, "aa", "aa", 4.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, string.Empty, string.Empty, 6.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, "aaa", "aaa", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, "aa", "a", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, string.Empty, "a", 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.Repeat(TThis, int, int?)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat3()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a", "aa");
            StringTransducer repeat = StringTransducer.Repeat(StringTransducer.Replace(automaton, automaton), minTimes: 0);

            StringInferenceTestUtilities.TestTransducerValue(repeat, string.Empty, string.Empty, 1.0); // TODO: it's not clear from the definition that this should hold
            StringInferenceTestUtilities.TestTransducerValue(repeat, "a", "a", 4.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, "a", "aa", 4.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, "aa", "aa", 20.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, "aaa", "aa", 32.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, "aa", "aaa", 32.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, "a", "aaa", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, string.Empty, "a", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(repeat, "b", "b", 0.0);
        }

        /// <summary>
        /// Tests <see cref="TransducerBase{TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis}.Optional(TThis)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Optional()
        {
            StringAutomaton automaton = StringAutomaton.Constant(1.0, DiscreteChar.Lower());
            StringTransducer copy = StringTransducer.Copy(automaton);
            StringTransducer copyOptional = StringTransducer.Optional(copy);

            StringInferenceTestUtilities.TestTransducerValue(copy, "abc", "abc", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copyOptional, "abc", "abc", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, string.Empty, string.Empty, 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copyOptional, string.Empty, string.Empty, 2.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "abc", "ab", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copyOptional, "abc", "ab", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "abc", string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copyOptional, "abc", string.Empty, 0.0);
        }

        /// <summary>
        /// Tests <see cref="Transducer{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}.Copy()"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Copy()
        {
            StringTransducer copy = StringTransducer.Copy();

            StringInferenceTestUtilities.TestTransducerValue(copy, "important", "important", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "important", "i", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "important", "imp", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "important", "t", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "important", "mpo", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, string.Empty, string.Empty, 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "important", string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, string.Empty, "important", 0.0);
            
            //// Test that projection on Copy() doesn't change the automaton
            
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a", "ab", "ac");
            automaton = automaton.Sum(StringAutomaton.ConstantOn(1.0, "a"));
            automaton = automaton.Sum(StringAutomaton.Constant(2.0));
            automaton = automaton.Product(StringAutomaton.Constant(3.0));

            StringAutomaton automatonClone = copy.ProjectSource(automaton);
            Assert.Equal(automaton, automatonClone);
        }

        /// <summary>
        /// Tests <see cref="Transducer{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}.Copy(TElementDistribution)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void CopyElements()
        {
            StringTransducer copy = StringTransducer.Copy(DiscreteChar.OneOf('a', 'b'));

            StringInferenceTestUtilities.TestTransducerValue(copy, string.Empty, string.Empty, 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "a", "a", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "bb", "bb", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "bab", "bab", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "bab", "ba", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "a", "b", 0.0);
            
            //// Tests that projection on Copy(elements) shrinks the support

            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a", "ab", "ac");
            automaton = automaton.Sum(StringAutomaton.ConstantOn(1.0, "a"));
            automaton = automaton.Sum(StringAutomaton.Constant(2.0));
            automaton = automaton.Product(StringAutomaton.Constant(3.0));

            for (int i = 0; i < 2; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 15, "a");
                StringInferenceTestUtilities.TestValue(automaton, 12, "ab");
                StringInferenceTestUtilities.TestValue(automaton, 6.0, "b", string.Empty);
                StringInferenceTestUtilities.TestValue(automaton, i == 0 ? 12.0 : 0.0, "ac");

                automaton = copy.ProjectSource(automaton);
            }
        }

        /// <summary>
        /// Tests <see cref="Transducer{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}.Copy(TSequence)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void CopySequence1()
        {
            StringTransducer copySequence = StringTransducer.Copy("important");
            StringInferenceTestUtilities.TestTransducerValue(copySequence, "important", "important", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copySequence, "important", "imp", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copySequence, "important", "ortant", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copySequence, "important", string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerProjection(copySequence, StringAutomaton.ConstantOn(3.0, "important"), "important", 3.0);
            StringInferenceTestUtilities.TestTransducerProjection(copySequence, StringAutomaton.ConstantOn(1.5, "important", "not important"), "important", 1.5);
            StringInferenceTestUtilities.TestTransducerProjection(copySequence, StringAutomaton.Constant(2.0), "important", 2.0);
            StringInferenceTestUtilities.TestTransducerProjection(
                copySequence,
                StringAutomaton.Constant(2.0).Append(StringAutomaton.ConstantOn(3.0, "nt")),
                "important",
                6.0);
            StringInferenceTestUtilities.TestIfTransducerRejects(copySequence, string.Empty, "nothing is important", "importance", "imp", "ortant", "a");
        }

        /// <summary>
        /// Tests <see cref="Transducer{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}.Copy(TSequence)"/>.
        /// Also checks that projecting a sequence on a transducer has polynomial complexity.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void CopySequence2()
        {
            const int CopyCount = 10;
            const int LetterCount = 20;
            
            StringTransducer copy = StringTransducer.Copy();
            for (int i = 0; i < CopyCount - 1; ++i)
            {
                copy.AppendInPlace(StringTransducer.Copy());
            }

            var sequence = new string(Enumerable.Repeat('a', LetterCount).ToArray());
            StringAutomaton result = copy.ProjectSource(sequence);
            var expectedLogValue = Math.Log(StringInferenceTestUtilities.Partitions(LetterCount, CopyCount));
            Assert.Equal(expectedLogValue, result.GetLogValue(sequence), 1e-8);
        }

        /// <summary>
        /// Tests <see cref="Transducer{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}.Copy(TAutomaton, byte)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void CopyAutomaton()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(1.0, "prefix1", "prefix2");
            automaton.AppendInPlace(StringAutomaton.Constant(1.0, DiscreteChar.Lower()));
            automaton.AppendInPlace(StringAutomaton.Constant(1.0, DiscreteChar.Upper()));
            automaton.AppendInPlace("!");

            StringTransducer copy = StringTransducer.Copy(automaton);
            StringInferenceTestUtilities.TestTransducerValue(copy, "prefix1!", "prefix1!", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "prefix2!", "prefix2!", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "prefix1lower!", "prefix1lower!", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "prefix2UPPER!", "prefix2UPPER!", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "prefix1lowerUPPER!", "prefix1lowerUPPER!", 1.0);
            StringInferenceTestUtilities.TestIfTransducerRejects(copy, "prefix1lower", "prefix2UPPER", "!", "prefix1lowerUPPER");

            StringInferenceTestUtilities.TestTransducerProjection(copy, automaton, "prefix1!", 1.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, automaton, "prefix2!", 1.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, automaton, "prefix1lower!", 1.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, automaton, "prefix2UPPER!", 1.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, automaton, "prefix1lowerUPPER!", 1.0);
            
            StringAutomaton subsetAutomaton = StringAutomaton.ConstantOn(2.0, "prefix1");
            subsetAutomaton.AppendInPlace(StringAutomaton.ConstantOn(3.0, "lll", "mmm"));
            subsetAutomaton.AppendInPlace(StringAutomaton.ConstantOn(1.5, "!", "U!"));
            StringInferenceTestUtilities.TestTransducerProjection(copy, subsetAutomaton, "prefix1lll!", 9.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, subsetAutomaton, "prefix1mmmU!", 9.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, subsetAutomaton, "prefix1!", 0.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, subsetAutomaton, "prefix2lower!", 0.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, subsetAutomaton, "prefix2U!", 0.0);

            StringAutomaton supersetAutomaton = StringAutomaton.ConstantOn(1.5, "pr");
            supersetAutomaton.AppendInPlace(StringAutomaton.Constant(2.0));
            StringInferenceTestUtilities.TestTransducerProjection(copy, supersetAutomaton, "prefix1!", 3.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, supersetAutomaton, "prefix2!", 3.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, supersetAutomaton, "prefix1lower!", 3.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, supersetAutomaton, "prefix2UPPER!", 3.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, supersetAutomaton, "prefix1lowerUPPER!", 3.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, supersetAutomaton, "prefix11!", 0.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, supersetAutomaton, "prefix1lowerUPPERlower!", 0.0);
            StringInferenceTestUtilities.TestTransducerProjection(copy, supersetAutomaton, "prrrrr!", 0.0);
        }

        /// <summary>
        /// Tests <see cref="Transducer{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}.CopyElement(TElementDistribution)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void CopyElement()
        {
            StringTransducer copy = StringTransducer.CopyElement(DiscreteChar.OneOf('a', 'b'));

            StringInferenceTestUtilities.TestTransducerValue(copy, "a", "a", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "b", "b", 1.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "a", "b", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "b", "a", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, string.Empty, string.Empty, 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "bb", "bb", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "bab", "bab", 0.0);
            StringInferenceTestUtilities.TestTransducerValue(copy, "bab", "ba", 0.0);
            
            //// Tests that projection on CopyElement(elements) shrinks the support

            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a", "ab", "ac");
            automaton = automaton.Sum(StringAutomaton.ConstantOn(1.0, "a"));
            automaton = automaton.Sum(StringAutomaton.Constant(2.0));
            automaton = automaton.Product(StringAutomaton.Constant(3.0));

            for (int i = 0; i < 2; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 15, "a");
                StringInferenceTestUtilities.TestValue(automaton, 6.0, "b");
                StringInferenceTestUtilities.TestValue(automaton, i == 0 ? 6.0 : 0.0, string.Empty);
                StringInferenceTestUtilities.TestValue(automaton, i == 0 ? 12.0 : 0.0, "ac", "ab");

                automaton = copy.ProjectSource(automaton);
            }
        }

        /// <summary>
        /// Tests <see cref="Transducer{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}.Transpose(TThis)"/> and
        /// <see cref="Transducer{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}.TransposeInPlace()"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Transpose()
        {
            StringTransducer transducer = StringTransducer.Consume(StringAutomaton.Constant(3.0));
            StringTransducer transpose1 = StringTransducer.Transpose(transducer);
            StringTransducer transpose2 = transducer.Clone();
            transpose2.TransposeInPlace();

            var pairs = new[]
            {
                new[] { "a", string.Empty },
                new[] { string.Empty, string.Empty },
                new[] { "a", "bc" },
                new[] { "a", "a" }
            };
            
            foreach (string[] valuePair in pairs)
            {
                double referenceValue1 = transducer.GetValue(valuePair[0], valuePair[1]);
                Assert.Equal(referenceValue1, transpose1.GetValue(valuePair[1], valuePair[0]));
                Assert.Equal(referenceValue1, transpose2.GetValue(valuePair[1], valuePair[0]));

                double referenceValue2 = transducer.GetValue(valuePair[1], valuePair[0]);
                Assert.Equal(referenceValue2, transpose1.GetValue(valuePair[0], valuePair[1]));
                Assert.Equal(referenceValue2, transpose2.GetValue(valuePair[0], valuePair[1]));
            }
        }
    }
}
