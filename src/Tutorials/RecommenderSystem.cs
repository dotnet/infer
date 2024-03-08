// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Factors;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Applications", "A general-purpose recommender system")]
    public class RecommenderSystem
    {
        public bool largeData;

        public void Run()
        {
            // This example requires EP
            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is Algorithms.ExpectationPropagation))
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
                return;
            }

            // Define counts
            int numUsers = largeData ? 400 : 50;
            int numItems = largeData ? 400 : 10;
            Variable<int> numObservations = Variable.Observed(largeData ? numUsers * numItems / 2 : 100).Named("numObservations");
            int numTraits = 2;
            int numLevels = 2;

            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var block = Variable.If(evidence);

            // Define ranges
            Range user = new Range(numUsers).Named("user");
            Range item = new Range(numItems).Named("item");
            Range trait = new Range(numTraits).Named("trait");
            Range observation = new Range(numObservations).Named("observation");
            Range level = new Range(numLevels).Named("level");

            // Define latent variables
            var userTraits = Variable.Array(Variable.Array<double>(trait), user).Named("userTraits");
            var itemTraits = Variable.Array(Variable.Array<double>(trait), item).Named("itemTraits");
            var userBias = Variable.Array<double>(user).Named("userBias");
            var itemBias = Variable.Array<double>(item).Named("itemBias");
            var userThresholds = Variable.Array(Variable.Array<double>(level), user).Named("userThresholds");

            // Define priors
            var userTraitsPrior = Variable.Array(Variable.Array<Gaussian>(trait), user).Named("userTraitsPrior");
            var itemTraitsPrior = Variable.Array(Variable.Array<Gaussian>(trait), item).Named("itemTraitsPrior");
            var userBiasPrior = Variable.Array<Gaussian>(user).Named("userBiasPrior");
            var itemBiasPrior = Variable.Array<Gaussian>(item).Named("itemBiasPrior");
            var userThresholdsPrior = Variable.Array(Variable.Array<Gaussian>(level), user).Named("userThresholdsPrior");

            // Define latent variables statistically
            userTraits[user][trait] = Variable<double>.Random(userTraitsPrior[user][trait]);
            itemTraits[item][trait] = Variable<double>.Random(itemTraitsPrior[item][trait]);
            userBias[user] = Variable<double>.Random(userBiasPrior[user]);
            itemBias[item] = Variable<double>.Random(itemBiasPrior[item]);
            userThresholds[user][level] = Variable<double>.Random(userThresholdsPrior[user][level]);

            // Initialise priors
            Gaussian traitPrior = Gaussian.FromMeanAndVariance(0.0, 1.0);
            Gaussian biasPrior = Gaussian.FromMeanAndVariance(0.0, 1.0);

            userTraitsPrior.ObservedValue = Util.ArrayInit(numUsers, u => Util.ArrayInit(numTraits, t => traitPrior));
            itemTraitsPrior.ObservedValue = Util.ArrayInit(numItems, i => Util.ArrayInit(numTraits, t => traitPrior));
            userBiasPrior.ObservedValue = Util.ArrayInit(numUsers, u => biasPrior);
            itemBiasPrior.ObservedValue = Util.ArrayInit(numItems, i => biasPrior);
            userThresholdsPrior.ObservedValue = Util.ArrayInit(numUsers, u => Util.ArrayInit(numLevels, l => Gaussian.FromMeanAndVariance(l - numLevels / 2.0 + 0.5, 1.0)));

            // Break symmetry and remove ambiguity in the traits
            for (int i = 0; i < numTraits; i++)
            {
                // Assume that numTraits < numItems
                for (int j = 0; j < numTraits; j++)
                {
                    itemTraitsPrior.ObservedValue[i][j] = Gaussian.PointMass(0);
                }

                itemTraitsPrior.ObservedValue[i][i] = Gaussian.PointMass(1);
            }

            // Declare training data variables
            var userData = Variable.Array<int>(observation).Named("userData");
            var itemData = Variable.Array<int>(observation).Named("itemData");
            var ratingData = Variable.Array(Variable.Array<bool>(level), observation).Named("ratingData");

            // Set model noises explicitly
            Variable<double> affinityNoiseVariance = Variable.Observed(0.1).Named("affinityNoiseVariance");
            Variable<double> thresholdsNoiseVariance = Variable.Observed(0.1).Named("thresholdsNoiseVariance");

            // Model
            using (Variable.ForEach(observation))
            {
                VariableArray<double> products = Variable.Array<double>(trait).Named("products");
                var rawProduct = userTraits[userData[observation]][trait] * itemTraits[itemData[observation]][trait];
                products[trait] = Variable<double>.Factor(Damp.Backward<double>, rawProduct, numTraits > 2 ? 0.5 : 1.0);

                Variable<double> bias = (userBias[userData[observation]] + itemBias[itemData[observation]]).Named("bias");
                Variable<double> affinity = (bias + Variable.Sum(products).Named("productSum")).Named("affinity");
                Variable<double> noisyAffinity = Variable.GaussianFromMeanAndVariance(affinity, affinityNoiseVariance).Named("noisyAffinity");

                VariableArray<double> noisyThresholds = Variable.Array<double>(level).Named("noisyThresholds");
                noisyThresholds[level] = Variable.GaussianFromMeanAndVariance(userThresholds[userData[observation]][level], thresholdsNoiseVariance);
                ratingData[observation][level] = noisyAffinity > noisyThresholds[level];
            }

            // Observe training data
            var expected = GenerateData(
                numUsers,
                numItems,
                numTraits,
                numObservations.ObservedValue,
                numLevels,
                userData,
                itemData,
                ratingData,
                userTraitsPrior.ObservedValue,
                itemTraitsPrior.ObservedValue,
                userBiasPrior.ObservedValue,
                itemBiasPrior.ObservedValue,
                userThresholdsPrior.ObservedValue,
                affinityNoiseVariance.ObservedValue,
                thresholdsNoiseVariance.ObservedValue);

            // Allow EP to process the product factor as if running VMP
            // as in Stern, Herbrich, Graepel paper.
            engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_SHG09));
            engine.ShowProgress = largeData;
            engine.Compiler.UseParallelForLoops = true;

            block.CloseBlock();
            var ev = engine.Infer<Bernoulli>(evidence).LogOdds / numObservations.ObservedValue;
            Console.WriteLine("evidence per observation = {0}", ev);
            if (largeData && ev < -1)
            {
                throw new Exception("evidence is too low");
            }

            // Run inference
            var userTraitsPosterior = engine.Infer<Gaussian[][]>(userTraits);
            var itemTraitsPosterior = engine.Infer<Gaussian[][]>(itemTraits);
            var userBiasPosterior = engine.Infer<Gaussian[]>(userBias);
            var itemBiasPosterior = engine.Infer<Gaussian[]>(itemBias);
            var userThresholdsPosterior = engine.Infer<Gaussian[][]>(userThresholds);

            Console.WriteLine("| learned parameters     |");
            Console.WriteLine("| ---------------------- |");
            for (int i = 0; i < System.Math.Min(5, numItems); i++)
            {
                Console.WriteLine("| {0,5}    {1,5} | {2,5} |", itemTraitsPosterior[i][0].GetMean().ToString("F"), itemTraitsPosterior[i][1].GetMean().ToString("F"), itemBiasPosterior[i].GetMean().ToString("F"));
                //Console.WriteLine("| {0,5}    {1,5} | {2,5} |", itemTraitsPosterior[i][0], itemTraitsPosterior[i][1], itemBiasPosterior[i]);
                if (largeData)
                {
                    for (int t = 0; t < numTraits; t++)
                    {
                        if (MMath.AbsDiff(itemTraitsPosterior[i][t].GetMean(), expected.itemTraits[i][t]) > 0.4)
                        {
                            throw new Exception("bad itemTraits");
                        }
                    }
                }
            }

            // Feed in the inferred posteriors as the new priors
            userTraitsPrior.ObservedValue = userTraitsPosterior;
            itemTraitsPrior.ObservedValue = itemTraitsPosterior;
            userBiasPrior.ObservedValue = userBiasPosterior;
            itemBiasPrior.ObservedValue = itemBiasPosterior;
            userThresholdsPrior.ObservedValue = userThresholdsPosterior;

            // Make a prediction
            numObservations.ObservedValue = 1;
            userData.ObservedValue = new int[] { 5 };
            itemData.ObservedValue = new int[] { 6 };
            ratingData.ClearObservedValue();

            Bernoulli[] predictedRating = engine.Infer<Bernoulli[][]>(ratingData)[0];
            Console.WriteLine("Predicted rating:");
            foreach (var rating in predictedRating)
            {
                Console.WriteLine(rating);
            }
        }

        // Generates data from the model
        (double[][] userTraits, double[][] itemTraits, double[] userBias, double[] itemBias, double[][] userThresholds) GenerateData(
            int numUsers,
            int numItems,
            int numTraits,
            int numObservations,
            int numLevels,
            VariableArray<int> userData,
            VariableArray<int> itemData,
            VariableArray<VariableArray<bool>, bool[][]> ratingData,
            Gaussian[][] userTraitsPrior,
            Gaussian[][] itemTraitsPrior,
            Gaussian[] userBiasPrior,
            Gaussian[] itemBiasPrior,
            Gaussian[][] userThresholdsPrior,
            double affinityNoiseVariance,
            double thresholdsNoiseVariance)
        {
            if (numObservations > numUsers * numItems)
            {
                throw new ArgumentException("numObservations must be less than or equal to numUsers * numItems");
            }

            int[] generatedUserData = new int[numObservations];
            int[] generatedItemData = new int[numObservations];
            bool[][] generatedRatingData = new bool[numObservations][];

            // Sample model parameters from the priors
            Rand.Restart(12347);
            double[][] userTraits = Util.ArrayInit(numUsers, u => Util.ArrayInit(numTraits, t => userTraitsPrior[u][t].Sample()));
            double[][] itemTraits = Util.ArrayInit(numItems, i => Util.ArrayInit(numTraits, t => itemTraitsPrior[i][t].Sample()));
            double[] userBias = Util.ArrayInit(numUsers, u => userBiasPrior[u].Sample());
            double[] itemBias = Util.ArrayInit(numItems, i => itemBiasPrior[i].Sample());
            double[][] userThresholds = Util.ArrayInit(numUsers, u => Util.ArrayInit(numLevels, l => userThresholdsPrior[u][l].Sample()));

            Console.WriteLine("| true parameters        |");
            Console.WriteLine("| ---------------------- |");
            for (int i = 0; i < System.Math.Min(5, numItems); i++)
            {
                Console.WriteLine("| {0,5}    {1,5} | {2,5} |", itemTraits[i][0].ToString("F"), itemTraits[i][1].ToString("F"), itemBias[i].ToString("F"));
            }

            // Repeat the model with fixed parameters
            HashSet<int> visited = new HashSet<int>();
            for (int observation = 0; observation < numObservations; observation++)
            {
                int user = Rand.Int(numUsers);
                int item = Rand.Int(numItems);

                int userItemPairID = user * numItems + item; // pair encoding
                if (visited.Contains(userItemPairID)) // duplicate generated
                {
                    observation--; // reject pair
                    continue;
                }

                visited.Add(userItemPairID);

                double[] products = Util.ArrayInit(numTraits, t => userTraits[user][t] * itemTraits[item][t]);
                double bias = userBias[user] + itemBias[item];
                double affinity = bias + products.Sum();
                double noisyAffinity = new Gaussian(affinity, affinityNoiseVariance).Sample();
                double[] noisyThresholds = Util.ArrayInit(numLevels, l => new Gaussian(userThresholds[user][l], thresholdsNoiseVariance).Sample());

                generatedUserData[observation] = user;
                generatedItemData[observation] = item;
                generatedRatingData[observation] = Util.ArrayInit(numLevels, l => noisyAffinity > noisyThresholds[l]);
            }

            userData.ObservedValue = generatedUserData;
            itemData.ObservedValue = generatedItemData;
            ratingData.ObservedValue = generatedRatingData;
            return (userTraits, itemTraits, userBias, itemBias, userThresholds);
        }
    }
}