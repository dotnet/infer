// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    /// <summary>
    /// Example of learning causal relationships from data using gates in Infer.NET.
    /// 
    /// In this example, we consider two Boolean variables, A and B, and attempt to 
    /// distinguish whether A causes B or vice versa, through the use of data 
    /// with or without interventions on B.
    /// </summary>
    [Example("Applications", "Learning causal relationships")]
    public class CausalityExample
    {
        public void Run()
        {
            // ***************** Experiment configuration ******************

            // Number of data points
            int numberOfDataPoints = 20;

            // Noise parameter - defines the true strength of the association between A and B
            // This ranges from 0.0 (meaning that A and B are equal) to 0.5 
            // (meaning that A and B are uncorrelated).
            double q = 0.1;

            // How we choose to set B in an intervention e.g. 0.5 is by a coin flip, 
            // This is a chosen parameter of our randomized study.
            double probBIntervention = 0.5;


            // ********************** Model definition *********************

            // Now we write the Infer.NET model to compare between A causing B and B causing A
            // - in this example we only consider these two possibilities.
            //
            // Gates are used to select between the two possibilities and to represent
            // perfect interventions. In Infer.NET gates are represented as stochastic if 
            // statements created using Variable.If() and Variable.IfNot().

            // Uniform prior over our two hypotheses 
            // (True = A causes B, False = B causes A)
            var AcausesB = Variable.Bernoulli(0.5);

            // Range across the data
            var N = new Range(numberOfDataPoints);

            // Set up array variables for the data
            var A = Variable.Array<bool>(N).Named("A");
            var B = Variable.Array<bool>(N).Named("B");
            var doB = Variable.Array<bool>(N).Named("doB");

            // Loop over the data points
            using (Variable.ForEach(N))
            {
                // Intervention case - this is the same for either model
                // defined once here.
                using (Variable.If(doB[N]))
                {
                    // Given intervention B is selected at random 
                    // using a known parameter e.g. 0.5.
                    B[N] = Variable.Bernoulli(probBIntervention);
                }
            }

            // *** First model: that A causes B ***
            using (Variable.If(AcausesB))
            {
                // Loop over the data points
                using (Variable.ForEach(N))
                {
                    // Draw A from uniform prior
                    A[N] = Variable.Bernoulli(0.5);
                    // No intervention case for the A causes B model
                    using (Variable.IfNot(doB[N]))
                    {
                        // Set B to a noisy version of A
                        B[N] = A[N] != (Variable.Bernoulli(q));
                    }
                }
            }

            // *** Second model: that B causes A ***
            using (Variable.IfNot(AcausesB))
            {
                // Loop over the data points
                using (Variable.ForEach(N))
                {
                    // No intervention case for the B causes A model
                    using (Variable.IfNot(doB[N]))
                    {
                        // Draw B from uniform prior
                        B[N] = Variable.Bernoulli(0.5);
                    }
                    // Set A to a noisy version of B
                    A[N] = B[N] != (Variable.Bernoulli(q));
                }
            }

            // ************************* Inference *************************

            // Create an Infer.NET inference engine
            var engine = new InferenceEngine();
            Console.WriteLine("Causal inference using gates in Infer.NET");
            Console.WriteLine("=========================================\r\n");
            Console.WriteLine("Data set of " + numberOfDataPoints + " data points with noise " + q + "\r\n");


            // *** Data without interventions ***
            // Generate data set
            var dataWithoutInterventions = GenerateFromTrueModel(numberOfDataPoints, q, false, probBIntervention);

            // Attach the data without interventions
            A.ObservedValue = dataWithoutInterventions.A;
            B.ObservedValue = dataWithoutInterventions.B;
            doB.ObservedValue = dataWithoutInterventions.doB;

            // Infer probability that A causes B (rather than B causes A)
            Bernoulli AcausesBdist = engine.Infer<Bernoulli>(AcausesB);
            Console.WriteLine("P(A causes B), without interventions=" + AcausesBdist.GetProbTrue() + "\r\n");

            // *** Data WITH interventions ***
            // Number of inference runs to average over (each with a different generated data set)
            int numberOfRuns = 10;

            Console.WriteLine("Executing " + numberOfRuns + " runs with interventions:");
            double tot = 0;
            for (int i = 0; i < numberOfRuns; i++)
            {
                // Generate data with interventions
                var dataWithInterventions = GenerateFromTrueModel(numberOfDataPoints, q, true, probBIntervention);

                // Attach the data with interventions (this replaces any previously attached data)
                A.ObservedValue = dataWithInterventions.A;
                B.ObservedValue = dataWithInterventions.B;
                doB.ObservedValue = dataWithInterventions.doB;

                // Infer probability that A causes B (rather than B causes A)
                Bernoulli AcausesBdist2 = engine.Infer<Bernoulli>(AcausesB);
                tot += AcausesBdist2.GetProbTrue();
                Console.WriteLine("{0,4}. P(A causes B)={1}", i + 1, (float)AcausesBdist2.GetProbTrue());
            }
            Console.WriteLine("Average P(A causes B), with interventions=" + (float)(tot / numberOfRuns));

        }

        /// <summary>
        /// Generates data from the true model: A cause B
        /// </summary>
        /// <param name="N">Number of data points to generate</param>
        /// <param name="q">Noise (flip) probability</param>
        /// <param name="doB">Whether to intervene or not</param>
        /// <param name="probBIntervention">Prob of choosing B=true when intervening</param>
        /// <returns></returns>
        private static Data GenerateFromTrueModel(int N, double q, bool doB, double probBIntervention)
        {
            // Create data object to fill with data.
            Data d = new Data { A = new bool[N], B = new bool[N], doB = new bool[N] };

            // Uniform prior on A
            var Aprior = new Bernoulli(0.5);
            // Noise distribution
            var flipDist = new Bernoulli(q);
            // Distribution over the values of B when we intervene 
            var interventionDist = new Bernoulli(probBIntervention);

            // Loop over data
            for (int i = 0; i < N; i++)
            {
                // Draw A from prior
                d.A[i] = Aprior.Sample();

                // Whether we intervened on B 
                // This is currently the same for all data points - but could easily be modified.
                d.doB[i] = doB;

                if (!d.doB[i])
                {
                    // We are not intervening so use the causal model i.e.
                    // make B a noisy version of A - flipping it with probability q
                    d.B[i] = d.A[i] != flipDist.Sample();
                }
                else
                {
                    // We are intervening - setting B according to a coin flip
                    d.B[i] = interventionDist.Sample();
                }
            }
            return d;
        }
    }


    /// <summary>
    /// Class to store the data
    /// </summary>
    class Data
    {
        public bool[] A; // observations of A
        public bool[] B; // observations of B
        public bool[] doB; // whether we intervened to set B
    }
}
