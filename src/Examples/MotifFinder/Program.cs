// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
namespace MotifFinder
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Utilities;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    /// <summary>
    /// The motif finder program.
    /// </summary>
    public class Program
    {
        /// <summary>
        /// The entry point of the motif finder.
        /// </summary>
        public static void Main()
        {
            Rand.Restart(1337);

            const int SequenceCount = 70;
            const int SequenceLength = 25;
            // To get the first result in the user guide, use MotifPresenceProbability = 1
            const double MotifPresenceProbability = 0.8;

            //// Sample some data

            var trueMotifNucleobaseDist = new[]
            {
                NucleobaseDist(a: 0.8, c: 0.1, g: 0.05, t: 0.05),
                NucleobaseDist(a: 0.0, c: 0.9, g: 0.05, t: 0.05),
                NucleobaseDist(a: 0.0, c: 0.0, g: 0.5, t: 0.5),
                NucleobaseDist(a: 0.25, c: 0.25, g: 0.25, t: 0.25),
                NucleobaseDist(a: 0.1, c: 0.1, g: 0.1, t: 0.7),
                NucleobaseDist(a: 0.0, c: 0.0, g: 0.9, t: 0.1),
                NucleobaseDist(a: 0.9, c: 0.05, g: 0.0, t: 0.05),
                NucleobaseDist(a: 0.5, c: 0.5, g: 0.0, t: 0.0),
            };

            int motifLength = trueMotifNucleobaseDist.Length;
            var backgroundNucleobaseDist = NucleobaseDist(a: 0.25, c: 0.25, g: 0.25, t: 0.25);

            string[] sequenceData;
            int[] motifPositionData;
            SampleMotifData(
                SequenceCount,
                SequenceLength,
                MotifPresenceProbability,
                trueMotifNucleobaseDist,
                backgroundNucleobaseDist,
                out sequenceData,
                out motifPositionData);

            //// Specify the model

            Vector motifNucleobasePseudoCounts = PiecewiseVector.Constant(char.MaxValue + 1, 1e-6);
            motifNucleobasePseudoCounts['A'] = motifNucleobasePseudoCounts['C'] = motifNucleobasePseudoCounts['G'] = motifNucleobasePseudoCounts['T'] = 1.0;

            Range motifCharsRange = new Range(motifLength);
            VariableArray<Vector> motifNucleobaseProbs = Variable.Array<Vector>(motifCharsRange);
            motifNucleobaseProbs[motifCharsRange] = Variable.Dirichlet(motifNucleobasePseudoCounts).ForEach(motifCharsRange);

            var sequenceRange = new Range(SequenceCount);
            VariableArray<string> sequences = Variable.Array<string>(sequenceRange);

            VariableArray<int> motifPositions = Variable.Array<int>(sequenceRange);
            motifPositions[sequenceRange] = Variable.DiscreteUniform(SequenceLength - motifLength + 1).ForEach(sequenceRange);

            VariableArray<bool> motifPresence = Variable.Array<bool>(sequenceRange);
            motifPresence[sequenceRange] = Variable.Bernoulli(MotifPresenceProbability).ForEach(sequenceRange);

            using (Variable.ForEach(sequenceRange))
            {
                using (Variable.If(motifPresence[sequenceRange]))
                {
                    var motifChars = Variable.Array<char>(motifCharsRange);
                    motifChars[motifCharsRange] = Variable.Char(motifNucleobaseProbs[motifCharsRange]);
                    var motif = Variable.StringFromArray(motifChars);

                    var backgroundLengthRight = SequenceLength - motifLength - motifPositions[sequenceRange];
                    var backgroundLeft = Variable.StringOfLength(motifPositions[sequenceRange], backgroundNucleobaseDist);
                    var backgroundRight = Variable.StringOfLength(backgroundLengthRight, backgroundNucleobaseDist);

                    sequences[sequenceRange] = backgroundLeft + motif + backgroundRight;
                }

                using (Variable.IfNot(motifPresence[sequenceRange]))
                {
                    sequences[sequenceRange] = Variable.StringOfLength(SequenceLength, backgroundNucleobaseDist);
                }
            }

            //// Infer the motif from sampled data

            sequences.ObservedValue = sequenceData;

            var engine = new InferenceEngine();
            engine.NumberOfIterations = 30;
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;

            var motifNucleobaseProbsPosterior = engine.Infer<IList<Dirichlet>>(motifNucleobaseProbs);
            var motifPresencePosterior = engine.Infer<IList<Bernoulli>>(motifPresence);
            var motifPositionPosterior = engine.Infer<IList<Discrete>>(motifPositions);

            //// Output inference results

            PrintMotifInferenceResults(
                sequenceData,
                motifPositionData,
                trueMotifNucleobaseDist,
                motifNucleobaseProbsPosterior,
                motifPresencePosterior,
                motifPositionPosterior);

            //// Keep the application alive until the user enters a keystroke
            Console.WriteLine("Done.  Press enter to exit.");
            Console.ReadKey();
        }

        /// <summary>
        /// Samples the data from the model.
        /// </summary>
        /// <param name="sequenceCount">The number of sequences to sample.</param>
        /// <param name="sequenceLength">The length of a sequence.</param>
        /// <param name="motifPresenceProbability">The probability that a sequence will contain the motif.</param>
        /// <param name="motif">The position frequency matrix defining the motif.</param>
        /// <param name="backgroundDist">The background nucleobase distribution.</param>
        /// <param name="sequenceData">The sampled sequences.</param>
        /// <param name="motifPositionData">
        /// The motif positions in the sampled sequences.
        /// If the sequence doesn't contain the motif, the position is set to -1.
        /// </param>
        private static void SampleMotifData(
            int sequenceCount,
            int sequenceLength,
            double motifPresenceProbability,
            DiscreteChar[] motif,
            DiscreteChar backgroundDist,
            out string[] sequenceData,
            out int[] motifPositionData)
        {
            sequenceData = new string[sequenceCount];
            motifPositionData = new int[sequenceCount];
            for (int i = 0; i < sequenceCount; ++i)
            {
                if (Rand.Double() <= motifPresenceProbability)
                {
                    motifPositionData[i] = Rand.Int(sequenceLength - motif.Length + 1);
                    var backgroundBeforeChars = Util.ArrayInit(motifPositionData[i], j => backgroundDist.Sample());
                    var backgroundAfterChars = Util.ArrayInit(sequenceLength - motif.Length - motifPositionData[i], j => backgroundDist.Sample());
                    var sampledMotifChars = Util.ArrayInit(motif.Length, j => motif[j].Sample());
                    sequenceData[i] = new string(backgroundBeforeChars) + new string(sampledMotifChars) + new string(backgroundAfterChars);
                }
                else
                {
                    motifPositionData[i] = -1;
                    var background = Util.ArrayInit(sequenceLength, j => backgroundDist.Sample());
                    sequenceData[i] = new string(background);
                }
            }
        }

        /// <summary>
        /// Prints a position frequency matrix to console.
        /// </summary>
        /// <typeparam name="T">The type of a matrix column.</typeparam>
        /// <param name="caption">The caption.</param>
        /// <param name="positionWeights">A list of matrix columns.</param>
        /// <param name="weightExtractor">A function for extracting probability of a particular nucleobase from a column.</param>
        private static void PrintPositionFrequencyMatrix<T>(
            string caption, IList<T> positionWeights, Func<T, char, double> weightExtractor)
        {
            Console.WriteLine("{0}:", caption);
            foreach (char nucleobase in new[] { 'A', 'C', 'T', 'G' })
            {
                Console.Write("{0}:", nucleobase);
                for (int i = 0; i < positionWeights.Count; ++i)
                {
                    Console.Write("   {0:0.00}", weightExtractor(positionWeights[i], nucleobase));
                }

                Console.WriteLine();
            }
        }

        /// <summary>
        /// Prints the motif inference results.
        /// </summary>
        /// <param name="sequenceData">The observed sequences.</param>
        /// <param name="motifPositionData">The true motif position data.</param>
        /// <param name="trueMotifNucleobaseDist">The true position frequency matrix.</param>
        /// <param name="motifNucleobaseProbsPosterior">The inferred position frequency matrix.</param>
        /// <param name="motifPresencePosterior">The inferred motif presence.</param>
        /// <param name="motifPositionPosterior">The inferred motif positions.</param>
        private static void PrintMotifInferenceResults(
            string[] sequenceData,
            int[] motifPositionData,
            IList<DiscreteChar> trueMotifNucleobaseDist,
            IList<Dirichlet> motifNucleobaseProbsPosterior,
            IList<Bernoulli> motifPresencePosterior,
            IList<Discrete> motifPositionPosterior)
        {
            int sequenceCount = sequenceData.Length;
            int sequenceLength = sequenceData[0].Length;
            int motifLength = motifNucleobaseProbsPosterior.Count;

            Console.WriteLine();

            PrintPositionFrequencyMatrix(
                "True position frequency matrix",
                trueMotifNucleobaseDist,
                (dist, c) => dist[c]);

            Console.WriteLine();

            PrintPositionFrequencyMatrix(
                "Inferred position frequency matrix mean",
                motifNucleobaseProbsPosterior,
                (dist, c) => dist.GetMean()[c]);

            Console.WriteLine();

            const ConsoleColor PredictionColor = ConsoleColor.Yellow;
            const ConsoleColor GroundTruthColor = ConsoleColor.Red;
            const ConsoleColor OverlapColor = ConsoleColor.Green;

            WriteInColor(PredictionColor, "PREDICTION   ");
            WriteInColor(GroundTruthColor, "GROUND TRUTH   ");
            WriteLineInColor(OverlapColor, "OVERLAP");
            Console.WriteLine();

            for (int i = 0; i < Math.Min(sequenceCount, 30); ++i)
            {
                int motifPos = motifPresencePosterior[i].GetProbTrue() > 0.5 ? motifPositionPosterior[i].GetMode() : -1;

                // Print the sequence
                bool inPrediction = false, inGroundTruth = false;
                for (int j = 0; j < sequenceLength; ++j)
                {
                    if (j == motifPos)
                    {
                        inPrediction = true;
                    }
                    else if (j == motifPos + motifLength)
                    {
                        inPrediction = false;
                    }

                    if (j == motifPositionData[i])
                    {
                        inGroundTruth = true;
                    }
                    else if (j == motifPositionData[i] + motifLength)
                    {
                        inGroundTruth = false;
                    }

                    ConsoleColor color = Console.ForegroundColor;
                    if (inPrediction && inGroundTruth)
                    {
                        color = OverlapColor;
                    }
                    else if (inPrediction)
                    {
                        color = PredictionColor;
                    }
                    else if (inGroundTruth)
                    {
                        color = GroundTruthColor;
                    }

                    WriteInColor(color, "{0}", sequenceData[i][j]);
                }

                // Print prediction confidence
                Console.Write("    P(has motif) = {0:0.00}", motifPresencePosterior[i].GetProbTrue());
                if (motifPos != -1)
                {
                    Console.Write("   P(pos={0}) = {1:0.00}", motifPos, motifPositionPosterior[i][motifPos]);
                }

                Console.WriteLine();
            }
        }

        /// <summary>
        /// Writes the text to console using a given color.
        /// </summary>
        /// <param name="color">The color.</param>
        /// <param name="format">A format string.</param>
        /// <param name="args">An array of format string arguments.</param>
        private static void WriteInColor(ConsoleColor color, string format, params object[] args)
        {
            Console.ForegroundColor = color;
            Console.Write(format, args);
            Console.ResetColor();
        }

        /// <summary>
        /// Writes the text to console using a given color and starts a new line.
        /// </summary>
        /// <param name="color">The color.</param>
        /// <param name="format">A format string.</param>
        /// <param name="args">An array of format string arguments.</param>
        private static void WriteLineInColor(ConsoleColor color, string format, params object[] args)
        {
            WriteInColor(color, format, args);
            Console.WriteLine();
        }

        /// <summary>
        /// Creates a distribution over characters that correspond to nucleobases.
        /// </summary>
        /// <param name="a">The probability of adenine.</param>
        /// <param name="c">The probability of cytosine.</param>
        /// <param name="g">The probability of guanine.</param>
        /// <param name="t">The probability of thymine.</param>
        /// <returns>The created distribution.</returns>
        private static DiscreteChar NucleobaseDist(double a, double c, double g, double t)
        {
            Vector probs = PiecewiseVector.Zero(char.MaxValue + 1);
            probs['A'] = a;
            probs['C'] = c;
            probs['G'] = g;
            probs['T'] = t;

            return DiscreteChar.FromVector(probs);
        }
    }
}
