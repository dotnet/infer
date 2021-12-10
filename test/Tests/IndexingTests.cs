// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    using Assert = Xunit.Assert;
    using BernoulliArray = DistributionStructArray<Bernoulli, bool>;
    using BernoulliArrayArray = DistributionRefArray<DistributionStructArray<Bernoulli, bool>, bool[]>;
    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using GaussianArray2D = DistributionStructArray2D<Gaussian, double>;
    using GaussianArrayArray = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;
    using DirichletArray = DistributionRefArray<Dirichlet, Vector>;

    
    public class IndexingTests
    {
        [Fact]
        public void IListFromArrayTest()
        {
            Range item = new Range(3);
            var array = Variable.Array<int>(item);
            array[item] = Variable.DiscreteUniform(10).ForEach(item);
            var list = Variable.IList<int>(item);
            list[item] = Variable.Copy(array[item]);

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(list));
        }

        [Fact]
        public void NoisyCopyTest()
        {
            Range r = new Range(3);
            var x = Variable.Discrete(r, 0, 0, 1);
            var y = NoisyCopy(x, 0.1);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(y));
        }

        public static Variable<int> NoisyCopy(Variable<int> x, double noise)
        {
            var range = x.GetValueRange();
            var rangeCopy = range.Clone();
            var probs = Variable.Array<Vector>(rangeCopy);
            var u = noise / Variable.Double(range.Size);
            using (var fb = Variable.ForEach(rangeCopy))
            {
                var i = fb.Index;
                var array = Variable.Array<double>(range);
                using (var fb2 = Variable.ForEach(range))
                {
                    var j = fb2.Index;
                    var eq = (i == j);
                    using (Variable.If(eq))
                    {
                        array[j] = 1 - noise + u;
                    }
                    using (Variable.IfNot(eq))
                    {
                        array[j] = u;
                    }
                }
                probs[i] = Variable.Vector(array);
            }
            Variable<int> xCopy = Variable.New<int>();
            using (Variable.Switch(x))
            {
                xCopy.SetTo(Variable.Discrete(probs[x]));
            }
            return xCopy;
        }

        [Fact]
        public void SubarrayOfJaggedTest2()
        {
            var trait = new Range(1).Named("trait");
            var user = new Range(1).Named("user");
            var observation = new Range(1).Named("observation");
            var group = new Range(1).Named("group");
            var groupIdsPerObservation = Variable.Observed<int>(new int[] { 0 }, observation).Named("groupIdsPerObservation");
            var groupSizes = Variable.Observed(new int[] { 1 }, group).Named("groupSizes");
            var groupSizesRange = new Range(groupSizes[group]).Named("groupSizesRanges");
            var userIdsForGroups = Variable.Observed(new int[][] { new int[] { 0 } }, group, groupSizesRange).Named("userIdsForGroups");

            var individualUserTraits = Variable.Array(Variable.Array<double>(trait), user).Named("individualUserTraits");
            individualUserTraits[user][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(user, trait);

            using (Variable.ForEach(observation))
            {
                var groupId = groupIdsPerObservation[observation];
                var userIdsForGroup = userIdsForGroups[groupId];
                var groupMember = userIdsForGroup.Range;
                var groupMemberUserTraits = Variable.Subarray(individualUserTraits, userIdsForGroup).Named("groupMemberUserTraits");
                Variable.ConstrainPositive(groupMemberUserTraits[groupMember][trait]);
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(individualUserTraits));
        }

        [Fact]
        public void SubarrayOfJaggedTest()
        {
            var observedInstances = new[] {0, 1, 0};
            var observedOutcomes = new[] {0.0, 2.0, 3.0};
            var observedFeatures = new[] {new[] {1.0}, new[] {1.0, 2.0}};
            var observedFeatureIndices = new[] {new[] {0}, new[] {0, 1}};
            var observedTotalFeatureCount = 2;

            var observationRange = new Range(observedInstances.Length).Named("observationRange");
            var instanceRange = new Range(observedFeatures.Length).Named("instanceRange");
            var totalFeatureRange = new Range(observedTotalFeatureCount).Named("totalFeatureRange");

            var featureCounts = Variable.Array<int>(instanceRange).Named("featureCounts");
            var featureCountRange = new Range(featureCounts[instanceRange]).Named("dependentFeatureCountRange");

            var features = Variable.Array(Variable.Array<double>(featureCountRange), instanceRange).Named("features");
            var featureIndices = Variable.Array(Variable.Array<int>(featureCountRange), instanceRange).Named("featureIndices");
            featureIndices[instanceRange].SetValueRange(totalFeatureRange);

            var instances = Variable.Array<int>(observationRange).Named("instances");
            instances[observationRange] = Variable.DiscreteUniform(instanceRange).ForEach(observationRange);
            var outcomes = Variable.Array<double>(observationRange).Named("outcomes");

            var weights = Variable.Array<double>(totalFeatureRange).Named("weights");
            weights[totalFeatureRange] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(totalFeatureRange);

            using (Variable.ForEach(observationRange))
            using (Variable.Switch(instances[observationRange]))
            {
                VariableArray<double> subweights = Variable.Subarray(weights, featureIndices[instances[observationRange]]).Named("subweights");
                Range subweightRange = subweights.Range;
                VariableArray<double> products = Variable.Array<double>(subweightRange).Named("products");
                products[subweightRange] = subweights[subweightRange]*features[instances[observationRange]][subweightRange];

                outcomes[observationRange] = Variable.Sum(products) + Variable.GaussianFromMeanAndVariance(0, 1);
            }

            featureCounts.ObservedValue = Util.ArrayInit(observedFeatures.Length, i => observedFeatures[i].Length);
            features.ObservedValue = observedFeatures;
            featureIndices.ObservedValue = observedFeatureIndices;
            //instances.ObservedValue = observedInstances;
            outcomes.ObservedValue = observedOutcomes;

            var engine = new InferenceEngine();
            engine.Infer(weights);
        }

        // This test should throw a meaningful exception
        [Fact]
        [Trait("Category", "OpenBug")]
        public void SubarrayOfJaggedError()
        {
            var observedInstances = new[] {0, 1, 0};
            var observedOutcomes = new[] {0.0, 2.0, 3.0};
            var observedFeatures = new[] {new[] {1.0}, new[] {1.0, 2.0}};
            var observedFeatureIndices = new[] {new[] {0}, new[] {0, 1}};
            var observedTotalFeatureCount = 2;

            var observationRange = new Range(observedInstances.Length).Named("observationRange");
            var instanceRange = new Range(observedFeatures.Length).Named("instanceRange");
            var totalFeatureRange = new Range(observedTotalFeatureCount).Named("totalFeatureRange");

            var featureCounts = Variable.Array<int>(instanceRange).Named("featureCounts");
            var featureCountRange = new Range(featureCounts[instanceRange]).Named("dependentFeatureCountRange");

            var features = Variable.Array(Variable.Array<double>(featureCountRange), instanceRange).Named("features");
            var featureIndices = Variable.Array(Variable.Array<int>(featureCountRange), instanceRange).Named("featureIndices");
            featureIndices[instanceRange].SetValueRange(totalFeatureRange);

            var instances = Variable.Array<int>(observationRange).Named("instances");
            instances[observationRange] = Variable.DiscreteUniform(instanceRange).ForEach(observationRange);
            var outcomes = Variable.Array<double>(observationRange).Named("outcomes");

            var weights = Variable.Array<double>(totalFeatureRange).Named("weights");
            weights[totalFeatureRange] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(totalFeatureRange);

            using (Variable.ForEach(observationRange))
            using (Variable.Switch(instances[observationRange]))
            {
                VariableArray<double> subweights = Variable.Subarray(weights, featureIndices[instances[observationRange]]).Named("subweights");
                // featureCountRange cannot be used in this context
                VariableArray<double> products = Variable.Array<double>(featureCountRange).Named("products");
                products[featureCountRange] = subweights[featureCountRange]*features[instances[observationRange]][featureCountRange];

                outcomes[observationRange] = Variable.Sum(products) + Variable.GaussianFromMeanAndVariance(0, 1);
            }

            featureCounts.ObservedValue = Util.ArrayInit(observedFeatures.Length, i => observedFeatures[i].Length);
            features.ObservedValue = observedFeatures;
            featureIndices.ObservedValue = observedFeatureIndices;
            instances.ObservedValue = observedInstances;
            outcomes.ObservedValue = observedOutcomes;

            var engine = new InferenceEngine();
            engine.Infer(weights);
        }

        [Fact]
        public void GetItemTest()
        {
            Range item = new Range(4).Named("item");
            VariableArray<double> array = Variable.Array<double>(item);
            array[item] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(item);
            Variable<int> index = Variable.New<int>().Named("index");
            Variable<double> x = Variable<double>.Factor(Collection.GetItem, array, index);
            Variable.ConstrainEqual(x, 7.0);
            //Variable<double> y = Variable<double>.Factor(Factor.GetItem,array,index);
            //Variable.ConstrainEqual(x,7.0);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            index.ObservedValue = 2;
            Gaussian xPost = engine.Infer<Gaussian>(x);
            DistributionArray<Gaussian> arrayPost = engine.Infer<DistributionArray<Gaussian>>(array);
            Console.WriteLine("x:");
            Console.WriteLine(xPost);
            Console.WriteLine("array:");
            Console.WriteLine(arrayPost);
            Assert.True(xPost.MaxDiff(arrayPost[index.ObservedValue]) < 1e-10);
        }

        [Fact]
        public void GetItemFromElementwiseArrayDefinition()
        {
            var r1 = new Range(3).Named("r1");
            var r2 = new Range(4).Named("r2");

            var va1 = Variable.Array<bool>(r1).Named("va1");

            va1[0] = Variable.Bernoulli(0.1);
            va1[1] = Variable.Bernoulli(0.5);
            va1[2] = Variable.Bernoulli(0.3);
            // must define the array this way to avoid the error:
            //va1[r1] = Variable.Bernoulli(0.1).ForEach(r1);

            var vaIndex = Variable.Constant(0).Named("vaIndex");
            vaIndex.SetValueRange(r1);
            var va2 = Variable.Array<bool>(r2).Named("va2");

            using (Variable.ForEach(r2))
            {
                //va2[r2] = va1[vaIndex[r2]];
                va2[r2] = Variable.Copy(va1[vaIndex]);
            }

            var ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(va2));
        }

        [Fact]
        public void GetItemsFromElementwiseArrayDefinition()
        {
            var r1 = new Range(3).Named("r1");
            var r2 = new Range(4).Named("r2");

            var va1 = Variable.Array<bool>(r1).Named("va1");

            va1[0] = Variable.Bernoulli(0.1);
            va1[1] = Variable.Bernoulli(0.5);
            va1[2] = Variable.Bernoulli(0.3);
            // must define the array this way to avoid the error:
            //va1[r1] = Variable.Bernoulli(0.1).ForEach(r1);

            var vaIndex = Variable.Constant(new int[] {0, 2, 1, 2}, r2).Named("vaIndex");
            vaIndex.SetValueRange(r1);
            var va2 = Variable.Array<bool>(r2).Named("va2");

            using (Variable.ForEach(r2))
            {
                //va2[r2] = va1[vaIndex[r2]];
                va2[r2] = Variable.Copy(va1[vaIndex[r2]]);
            }

            var ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(va2));
        }

        [Fact]
        public void GetItemGaussianMeansTest()
        {
            GetItemGaussianMeans(new ExpectationPropagation());
        }

        [Fact]
        public void GibbsGetItemGaussianMeans()
        {
            GetItemGaussianMeans(new GibbsSampling());
        }

        private void GetItemGaussianMeans(IAlgorithm algorithm)
        {
            int nComponents = 1;
            Range component = new Range(nComponents).Named("component");
            Variable<int> nItemsObs = Variable.New<int>().Named("nItem");
            Range itemObs = new Range(nItemsObs).Named("item");
            VariableArray<double> means = Variable.Array<double>(component).Named("means");
            means[component] = Variable.GaussianFromMeanAndVariance(0, 10).ForEach(component);

            VariableArray<double> x = Variable.Array<double>(itemObs).Named("x");

            using (Variable.ForEach(itemObs))
            {
                Variable<int> xLabel = Variable.New<int>().Named("xLabel");
                xLabel.ObservedValue = 0;
                // we put this line inside the ForEach to make the compiler sweat:
                x[itemObs] = Variable.GaussianFromMeanAndPrecision(means[xLabel], 1);
            }

            x.ObservedValue = new double[] {4};
            nItemsObs.ObservedValue = x.ObservedValue.Length;

            InferenceEngine ie = new InferenceEngine(algorithm);
            int n = x.ObservedValue.Length;
            Gaussian[] meansExpectedArray = new Gaussian[n];
            Gaussian prior = new Gaussian(0, 10);
            for (int i = 0; i < meansExpectedArray.Length; i++)
            {
                meansExpectedArray[i] = prior*(new Gaussian(x.ObservedValue[i], 1));
            }
            IDistribution<double[]> meansExpected = Distribution<double>.Array(meansExpectedArray);
            object meansActual = ie.Infer(means);
            Console.WriteLine(StringUtil.JoinColumns("means = ", meansActual, " should be ", meansExpected));
            Assert.True(meansExpected.MaxDiff(meansActual) < 1e-10);
        }

        [Fact]
        public void GetItemsGaussianMeansTest()
        {
            GetItemsGaussianMeans(new ExpectationPropagation());
        }

        [Fact]
        public void GibbsGetItemsGaussianMeansTest()
        {
            GetItemsGaussianMeans(new GibbsSampling());
        }

        private void GetItemsGaussianMeans(IAlgorithm algorithm)
        {
            int nComponents = 2;
            Range component = new Range(nComponents).Named("component");
            Variable<int> nItemsObs = Variable.New<int>().Named("nItem");
            Range itemObs = new Range(nItemsObs).Named("item");
            VariableArray<double> means = Variable.Array<double>(component).Named("means");
            means[component] = Variable.GaussianFromMeanAndVariance(0, 10).ForEach(component);

            VariableArray<double> x = Variable.Array<double>(itemObs).Named("x");
            VariableArray<int> xLabel = Variable.Array<int>(itemObs).Named("xLabel");

            using (Variable.ForEach(itemObs))
            {
                xLabel[itemObs] = Variable.Discrete(component, new double[] {.5, .5});
                // we put this line inside the ForEach to make the compiler sweat:
                x[itemObs] = Variable.GaussianFromMeanAndPrecision(means[xLabel[itemObs]], 1);
            }

            x.ObservedValue = new double[] {-1, 4};
            xLabel.ObservedValue = new int[] {0, 1};
            nItemsObs.ObservedValue = x.ObservedValue.Length;

            InferenceEngine ie = new InferenceEngine(algorithm);
            Gaussian[] meansExpectedArray = new Gaussian[2];
            Gaussian prior = new Gaussian(0, 10);
            for (int i = 0; i < 2; i++)
            {
                meansExpectedArray[i] = prior*(new Gaussian(x.ObservedValue[i], 1));
            }
            IDistribution<double[]> meansExpected = Distribution<double>.Array(meansExpectedArray);
            object meansActual = ie.Infer(means);
            Console.WriteLine(StringUtil.JoinColumns("means = ", meansActual, " should be ", meansExpected));
            Assert.True(meansExpected.MaxDiff(meansActual) < 1e-10);
        }

        internal void GetItemsTest_anitha()
        {
            bool flag = false;
            int termCount = 2;
            Range allTerm = new Range(termCount).Named("allTerm");

            int[] termsInDoc = new int[] {2, 2};
            int docCount = termsInDoc.Length;
            Range doc = new Range(docCount).Named("doc");
            VariableArray<int> termSizes = Variable.Constant(termsInDoc, doc).Named("termSizes");
            Range term = new Range(termSizes[doc]).Named("item");

            VariableArray<double> p = Variable.Array<double>(allTerm);
            p[allTerm] = Variable.Beta(1, 1).ForEach(allTerm).Named("p");


            var indexIntoDoc = Variable.Array(Variable.Array<int>(term), doc).Named("indexIntoDoc");

            var docTermPair = Variable.Array(Variable.Array<bool>(term), doc).Named("docTermPair");
            using (Variable.ForEach(doc))
            {
                VariableArray<double> pCur = Variable.GetItems(p, indexIntoDoc[doc]).Named("pCur");
                if (flag)
                {
                    docTermPair[doc][term] = Variable.Bernoulli(pCur[term]);
                }
            }

            //inference
            InferenceEngine ie = new InferenceEngine();

            double[] pDocRel = new double[] {.8, .45, .1};
            IList<int>[] docTermPairIndex = new List<int>[docCount];
            bool[][] docTermPairValue = new bool[docCount][];
            for (int d = 0; d < docCount; d++)
            {
                docTermPairIndex[d] = new List<int>();
                docTermPairValue[d] = new bool[termsInDoc[d]];
                for (int t = 0; t < termsInDoc[d]; t++)
                {
                    docTermPairIndex[d].Add(t);
                    docTermPairValue[d][t] = Bernoulli.Sample(pDocRel[t]);
                }
            }

            indexIntoDoc.ObservedValue = Array.ConvertAll(docTermPairIndex, list => list.ToArray());
            if (flag)
            {
                var docTermPairData = Variable.Constant<bool>(docTermPairValue, doc, term).Named("docTermPairData");
                Variable.ConstrainEqual(docTermPair[doc][term], docTermPairData[doc][term]);
            }
            Console.WriteLine(ie.Infer(p));
        }

        [Fact]
        public void VaryingSizeVectorArrayGetItemsTest()
        {
            VaryingSizeVectorArrayGetItems(new ExpectationPropagation());
            //varyingSizeVectorArrayGetItemsTest(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsVaryingSizeVectorArrayGetItemsTest()
        {
            VaryingSizeVectorArrayGetItems(new GibbsSampling());
        }

        private void VaryingSizeVectorArrayGetItems(IAlgorithm algorithm)
        {
            Range item = new Range(3).Named("item");
            VariableArray<Dirichlet> priors = Variable.Array<Dirichlet>(item).Named("priors");
            VariableArray<Vector> probs = Variable.Array<Vector>(item).Named("probs");
            probs[item] = Variable<Vector>.Random(priors[item]);
            VariableArray<int> x = Variable.Array<int>(item).Named("x");
            x[item] = Variable.Discrete(probs[item]);
            var indices = Variable.Observed(new int[] {1, 2}).Named("indices");
            Range yitem = indices.Range.Named("yitem");
            //VariableArray<int> y = Variable.Array<int>(yitem).Named("y");
            Variable<int> y = x[indices[yitem]].Named("y");
            VariableArray<int> yCopy = Variable.Array<int>(yitem).Named("yCopy");
            //yCopy[yitem] = Variable.Copy(x[indices[yitem]]);
            yCopy[yitem] = Variable.Copy(y);

            InferenceEngine engine = new InferenceEngine(algorithm);
            if (algorithm is GibbsSampling)
            {
                Rand.Restart(12347);
                engine.NumberOfIterations = 100000;
                engine.ShowProgress = false;
            }
            Dirichlet[] ppriors = new Dirichlet[]
                {
                    new Dirichlet(0.4, 0.6),
                    new Dirichlet(0.1, 0.2, 0.7),
                    new Dirichlet(0.1, 0.3, 0.4, 0.2)
                };
            priors.ObservedValue = ppriors;
            DistributionArray<Discrete> yActual = engine.Infer<DistributionArray<Discrete>>(yCopy);
            Discrete[] yExpected = new Discrete[indices.ObservedValue.Length];
            for (int i = 0; i < yExpected.Length; i++)
            {
                yExpected[i] = new Discrete(ppriors[indices.ObservedValue[i]].GetMean());
            }
            Console.WriteLine(StringUtil.JoinColumns("y = ", yActual));
            Assert.True(Distribution<int>.Array(yExpected).MaxDiff(yActual) < 1e-2);
        }

        [Fact]
        public void VaryingSizeVectorArrayGetItemTest()
        {
            VaryingSizeVectorArrayGetItem(new ExpectationPropagation());
            //varyingSizeVectorArrayGetItemTest(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsVaryingSizeVectorArrayGetItemTest()
        {
            VaryingSizeVectorArrayGetItem(new GibbsSampling());
        }

        private void VaryingSizeVectorArrayGetItem(IAlgorithm algorithm)
        {
            Range item = new Range(3).Named("item");
            VariableArray<Dirichlet> priors = Variable.Array<Dirichlet>(item).Named("priors");
            VariableArray<Vector> probs = Variable.Array<Vector>(item).Named("probs");
            probs[item] = Variable<Vector>.Random(priors[item]);
            VariableArray<int> x = Variable.Array<int>(item).Named("x");
            x[item] = Variable.Discrete(probs[item]);
            Variable<int> index = Variable.Observed(1).Named("index");
            Variable<int> y = Variable.Copy(x[index]).Named("y");

            InferenceEngine engine = new InferenceEngine(algorithm);
            if (algorithm is GibbsSampling)
            {
                Rand.Restart(12347);
                engine.NumberOfIterations = 100000;
                engine.ShowProgress = false;
            }

            Dirichlet[] ppriors = new Dirichlet[]
                {
                    new Dirichlet(0.4, 0.6),
                    new Dirichlet(0.1, 0.2, 0.7),
                    new Dirichlet(0.1, 0.3, 0.4, 0.2)
                };
            priors.ObservedValue = ppriors;
            Discrete yActual = engine.Infer<Discrete>(y);
            Discrete yExpected = new Discrete(ppriors[index.ObservedValue].GetMean());
            Console.WriteLine(StringUtil.JoinColumns("y = ", yActual, " should be ", yExpected));
            Assert.True(yExpected.MaxDiff(yActual) < 1e-2);
        }

        [Fact]
        public void GetItemInMultipleGates()
        {
            Range item = new Range(2).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(0.3).ForEach(item);
            Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
            Variable<int> index = Variable.Observed(0).Named("index");
            using (Variable.If(c))
            {
                Variable.ConstrainTrue(bools[index]);
            }
            using (Variable.IfNot(c))
            {
                Variable.ConstrainFalse(bools[index]);
            }

            InferenceEngine engine = new InferenceEngine();
            Bernoulli[] boolsExpectedArray = new Bernoulli[item.SizeAsInt];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli(0.3);
            }
            boolsExpectedArray[index.ObservedValue] = new Bernoulli(0.1*0.3/(0.1*0.3 + (1 - 0.1)*(1 - 0.3)));
            IDistribution<bool[]> boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void GetItem2DTest()
        {
            GetItem2D(new ExpectationPropagation());
            GetItem2D(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsGetItem2DTest()
        {
            GetItem2D(new GibbsSampling());
        }

        private void GetItem2D(IAlgorithm algorithm)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling)) block = Variable.If(evidence);
            Range item = new Range(2).Named("item");
            Range item2 = new Range(3).Named("item2");
            VariableArray2D<bool> array = Variable.Array<bool>(item, item2).Named("array");
            array[item, item2] = Variable.Bernoulli(0.3).ForEach(item, item2);
            Variable<int> index = Variable.New<int>().Named("index");
            Variable<int> index2 = Variable.New<int>().Named("index2");
            Variable.ConstrainEqualRandom(array[index, index2], new Bernoulli(0.2));
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(algorithm);
            index.ObservedValue = 1;
            index2.ObservedValue = 1;
            //Bernoulli xActual = engine.Infer<Bernoulli>(array[index, index2]);
            double z = 0.2*0.3 + (1 - 0.2)*(1 - 0.3);
            Bernoulli xExpected = new Bernoulli(0.2*0.3/z);
            DistributionArray2D<Bernoulli> arrayPost = engine.Infer<DistributionArray2D<Bernoulli>>(array);
            //Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
            Console.WriteLine("array:");
            Console.WriteLine(arrayPost);
            //Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            Assert.True(xExpected.MaxDiff(arrayPost[index.ObservedValue, index2.ObservedValue]) < 1e-10);
            if (!(algorithm is GibbsSampling))
            {
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                double evExpected = System.Math.Log(z);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-10);
            }
        }

        [Fact]
        public void GetItemAtObservedIndexTest()
        {
            GetItemAtObservedIndex(new ExpectationPropagation());
            GetItemAtObservedIndex(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsGetItemAtObservedIndexTest()
        {
            GetItemAtObservedIndex(new GibbsSampling());
        }

        private void GetItemAtObservedIndex(IAlgorithm algorithm)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling)) block = Variable.If(evidence);
            Range item = new Range(4).Named("item");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            array[item] = Variable.Bernoulli(0.3).ForEach(item);
            Variable<int> index = Variable.New<int>().Named("index");
            Variable.ConstrainEqualRandom(array[index], new Bernoulli(0.2));
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(algorithm);
            index.ObservedValue = 2;
            //Bernoulli xActual = engine.Infer<Bernoulli>(array[index]);
            double z = 0.2*0.3 + (1 - 0.2)*(1 - 0.3);
            Bernoulli xExpected = new Bernoulli(0.2*0.3/z);
            DistributionArray<Bernoulli> arrayPost = engine.Infer<DistributionArray<Bernoulli>>(array);
            //Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
            Console.WriteLine("array:");
            Console.WriteLine(arrayPost);
            //Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            Assert.True(xExpected.MaxDiff(arrayPost[index.ObservedValue]) < 1e-10);
            if (!(algorithm is GibbsSampling))
            {
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                double evExpected = System.Math.Log(z);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-10);
            }
        }

        [Fact]
        public void GetItemAtConstantIndexTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range item = new Range(4).Named("item");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            array[item] = Variable.Bernoulli(0.3).ForEach(item);
            Variable<int> index = Variable.Constant(2).Named("index");
            Variable.ConstrainEqualRandom(array[index], new Bernoulli(0.2));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 1) engine.Algorithm = new VariationalMessagePassing();
                //Bernoulli xActual = engine.Infer<Bernoulli>(array[index]);
                double z = 0.2*0.3 + (1 - 0.2)*(1 - 0.3);
                Bernoulli xExpected = new Bernoulli(0.2*0.3/z);
                DistributionArray<Bernoulli> arrayPost = engine.Infer<DistributionArray<Bernoulli>>(array);
                //Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Console.WriteLine("array:");
                Console.WriteLine(arrayPost);
                //Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
                Assert.True(xExpected.MaxDiff(arrayPost[index.ObservedValue]) < 1e-10);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                double evExpected = System.Math.Log(z);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-10);
            }
        }

        [Fact]
        public void GetItemManyUsesTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range item = new Range(4).Named("item");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            double prior = 0.3;
            array[item] = Variable.Bernoulli(prior).ForEach(item);
            Variable<int> index = Variable.Constant(2).Named("index");
            double data = 0.2;
            double data2 = 0.4;
            Variable.ConstrainEqualRandom(array[index], new Bernoulli(data));
            Variable.ConstrainEqualRandom(array[index], new Bernoulli(data2));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 1) engine.Algorithm = new VariationalMessagePassing();
                //Bernoulli xActual = engine.Infer<Bernoulli>(array[index]);
                double z = prior*data*data2 + (1 - prior)*(1 - data)*(1 - data2);
                Bernoulli xExpected = new Bernoulli(prior*data*data2/z);
                DistributionArray<Bernoulli> arrayPost = engine.Infer<DistributionArray<Bernoulli>>(array);
                //Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Console.WriteLine("array:");
                Console.WriteLine(arrayPost);
                //Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
                Assert.True(xExpected.MaxDiff(arrayPost[index.ObservedValue]) < 1e-10);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                double evExpected = System.Math.Log(z);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-10);
            }
        }


        [Fact]
        public void GetItemObservedTest()
        {
            GetItemObserved(new ExpectationPropagation());
            GetItemObserved(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsGetItemObservedTest()
        {
            GetItemObserved(new GibbsSampling());
        }

        private void GetItemObserved(IAlgorithm algorithm)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling)) block = Variable.If(evidence);
            Range item = new Range(4).Named("item");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            array[item] = Variable.Bernoulli(0.3).ForEach(item);
            Variable<int> index = Variable.New<int>().Named("index");
            Variable<bool> x = Variable<bool>.Factor(Collection.GetItem, array, index).Named("x");
            // this doesn't work:
            //Variable<bool> x = array[index];
            x.ObservedValue = true;
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(algorithm);
            index.ObservedValue = 2;
            IDistribution<bool[]> arrayExpected = Distribution<bool>.Array(4,
                                                                           delegate(int i)
                                                                               {
                                                                                   if (i == index.ObservedValue) return Bernoulli.PointMass(x.ObservedValue);
                                                                                   else return new Bernoulli(0.3);
                                                                               });
            DistributionArray<Bernoulli> arrayActual = engine.Infer<DistributionArray<Bernoulli>>(array);
            Console.WriteLine(StringUtil.JoinColumns("array = ", arrayActual, " should be ", arrayExpected));
            Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-10);
            if (!(algorithm is GibbsSampling))
            {
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                double z = 0.3;
                double evExpected = System.Math.Log(z);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-10);
            }
        }

        [Fact]
        public void GetItemsModelTest()
        {
            GetItemsModel(new ExpectationPropagation());
            GetItemsModel(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsGetItemsModelTest()
        {
            GetItemsModel(new GibbsSampling());
        }

        private void GetItemsModel(IAlgorithm algorithm)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling)) block = Variable.If(evidence);
            Range item = new Range(4).Named("item");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            VariableArray<double> priors = Variable.Constant(new double[] {0.1, 0.2, 0.3, 0.4}, item).Named("priors");
            array[item] = Variable.Bernoulli(priors[item]);
            Variable<int> indicesLength = Variable.New<int>().Named("indicesLength");
            Range xitem = new Range(indicesLength).Named("xitem");
            VariableArray<int> indices = Variable.Array<int>(xitem).Named("indices");
            Variable<bool> x = array[indices[xitem]].Named("x");
            VariableArray<Bernoulli> xLike = Variable.Array<Bernoulli>(xitem).Named("xLike");
            double[] likes = {0.6, 0.7};
            xLike.ObservedValue = Util.ArrayInit(likes.Length, i => new Bernoulli(likes[i]));
            Variable.ConstrainEqualRandom(x, xLike[xitem]);
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(algorithm);
            for (int trial = 0; trial < 2; trial++)
            {
                Bernoulli[] arrayExpectedArray = new Bernoulli[item.SizeAsInt];
                for (int i = 0; i < arrayExpectedArray.Length; i++)
                {
                    arrayExpectedArray[i] = new Bernoulli(priors.ObservedValue[i]);
                }
                double z;
                if (trial == 0)
                {
                    indices.ObservedValue = new int[] {0, 3};
                    z = 1;
                    for (int i = 0; i < indices.ObservedValue.Length; i++)
                    {
                        double pi = priors.ObservedValue[indices.ObservedValue[i]];
                        double zi = likes[i]*pi + (1 - likes[i])*(1 - pi);
                        arrayExpectedArray[indices.ObservedValue[i]] = new Bernoulli(likes[i]*pi/zi);
                        z *= zi;
                    }
                }
                else
                {
                    indices.ObservedValue = new int[] {2, 2};
                    double p = priors.ObservedValue[indices.ObservedValue[0]];
                    z = likes[0]*likes[1]*p + (1 - likes[0])*(1 - likes[1])*(1 - p);
                    for (int i = 0; i < indices.ObservedValue.Length; i++)
                    {
                        arrayExpectedArray[indices.ObservedValue[i]] = new Bernoulli(likes[0]*likes[1]*p/z);
                    }
                }
                indicesLength.ObservedValue = indices.ObservedValue.Length;
                IDistribution<bool[]> arrayExpected = Distribution<bool>.Array(arrayExpectedArray);
                object arrayActual = engine.Infer(array);
                Console.WriteLine(StringUtil.JoinColumns("array = ", arrayActual, " should be ", arrayExpected));
                Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-10);
                if (!(algorithm is GibbsSampling))
                {
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    double evExpected = System.Math.Log(z);
                    Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                    Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-10);
                }
            }
        }

        [Fact]
        public void GetItemsJaggedTest()
        {
            GetItemsJagged(new ExpectationPropagation());
            GetItemsJagged(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsGetItemsJaggedTest()
        {
            GetItemsJagged(new GibbsSampling());
        }

        public static Bernoulli BernoulliFromProbTrue(double probTrue)
        {
            return new Bernoulli(probTrue);
        }

        private void GetItemsJagged(IAlgorithm algorithm)
        {
            foreach(bool easyCase in new[] { false, true })
            {
                GetItemsJagged(algorithm, easyCase);
            }
        }

        private void GetItemsJagged(IAlgorithm algorithm, bool easyCase)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling)) block = Variable.If(evidence);
            Range outer = new Range(4).Named("outer");
            Range middle = new Range(3).Named("middle");
            Range inner = new Range(2).Named("inner");
            var array = Variable.Array(Variable.Array(Variable.Array<bool>(inner), middle), outer).Named("array");
            var priors = Variable.Array(Variable.Array(Variable.Array<double>(inner), middle), outer).Named("priors");
            double[] probTrueArray = new double[] {0.1, 0.2, 0.3, 0.4};
            priors.ObservedValue = Util.ArrayInit(outer.SizeAsInt, i =>
                                                                   Util.ArrayInit(middle.SizeAsInt, j =>
                                                                                                    Util.ArrayInit(inner.SizeAsInt, k => 1.0/(1 + i + j + k))));
            if (easyCase)
            {
                array[outer][middle][inner] = Variable.Bernoulli(priors[outer][middle][inner]);
            }
            else
            {
                // this definition creates a complicated MarginalPrototype
                var priors2 = Variable.Array(Variable.Array(Variable.Array<Bernoulli>(inner), middle), outer).Named("priors2");
                priors2[outer][middle][inner] = Variable<Bernoulli>.Factor(BernoulliFromProbTrue, priors[outer][middle][inner]);
                array[outer][middle][inner] = Variable<bool>.Random(priors2[outer][middle][inner]);
            }
            Variable<int> indicesLength = Variable.New<int>().Named("indicesLength");
            Range xitem = new Range(indicesLength).Named("xitem");
            VariableArray<int> indices = Variable.Array<int>(xitem).Named("indices");
            Variable<bool> x;
            if (easyCase)
            {
                x = array[indices[xitem]][middle][inner];
            }
            else
            {
                // this requires propagating the MarginalPrototype from array to array2
                var array2 = Variable.Array(Variable.Array(Variable.Array<bool>(inner), middle), xitem).Named("array2");
                array2.SetTo(Variable.GetItems(array, indices));
                x = array2[xitem][middle][inner];
            }
            double xLike = 0.6;
            Variable.ConstrainEqualRandom(x, new Bernoulli(xLike));
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(algorithm);
            // Gibbs cannot achieve the desired accuracy when loops are unrolled.
            if (algorithm is GibbsSampling) engine.Compiler.UnrollLoops = false;
            for (int trial = 0; trial < 2; trial++)
            {
                Bernoulli[][][] arrayExpectedArray = new Bernoulli[outer.SizeAsInt][][];
                for (int i = 0; i < arrayExpectedArray.Length; i++)
                {
                    arrayExpectedArray[i] = new Bernoulli[middle.SizeAsInt][];
                    for (int j = 0; j < arrayExpectedArray[i].Length; j++)
                    {
                        arrayExpectedArray[i][j] = new Bernoulli[inner.SizeAsInt];
                        for (int k = 0; k < arrayExpectedArray[i][j].Length; k++)
                        {
                            arrayExpectedArray[i][j][k] = new Bernoulli(priors.ObservedValue[i][j][k]);
                        }
                    }
                }
                double z = 1;
                if (trial == 0)
                {
                    indices.ObservedValue = new int[] {0, 3};
                    for (int i = 0; i < indices.ObservedValue.Length; i++)
                    {
                        for (int j = 0; j < arrayExpectedArray[i].Length; j++)
                        {
                            for (int k = 0; k < arrayExpectedArray[i][j].Length; k++)
                            {
                                double pi = priors.ObservedValue[indices.ObservedValue[i]][j][k];
                                double zi = xLike*pi + (1 - xLike)*(1 - pi);
                                arrayExpectedArray[indices.ObservedValue[i]][j][k] = new Bernoulli(xLike*pi/zi);
                                z *= zi;
                            }
                        }
                    }
                }
                else
                {
                    indices.ObservedValue = new int[] {2, 2};
                    for (int i = 0; i < indices.ObservedValue.Length; i++)
                    {
                        for (int j = 0; j < arrayExpectedArray[i].Length; j++)
                        {
                            for (int k = 0; k < arrayExpectedArray[i][j].Length; k++)
                            {
                                double p = priors.ObservedValue[indices.ObservedValue[i]][j][k];
                                double zi = xLike*xLike*p + (1 - xLike)*(1 - xLike)*(1 - p);
                                arrayExpectedArray[indices.ObservedValue[i]][j][k] = new Bernoulli(xLike*xLike*p/zi);
                                z *= zi;
                            }
                        }
                    }
                    z = System.Math.Sqrt(z);
                }
                indicesLength.ObservedValue = indices.ObservedValue.Length;
                IDistribution<bool[][][]> arrayExpected = Distribution<bool>.Array(arrayExpectedArray);
                object arrayActual = engine.Infer(array);
                Console.WriteLine(StringUtil.JoinColumns("array = ", arrayActual, " should be ", arrayExpected));
                Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-10);
                if (!(algorithm is GibbsSampling))
                {
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    double evExpected = System.Math.Log(z);
                    Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                    Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-10);
                }
            }
        }

        [Fact]
        public void GetItemsObservedTest()
        {
            GetItemsObserved(new ExpectationPropagation());
            GetItemsObserved(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsGetItemsObservedTest()
        {
            GetItemsObserved(new GibbsSampling());
        }

        private void GetItemsObserved(IAlgorithm algorithm)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling)) block = Variable.If(evidence);
            Range item = new Range(4).Named("item");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            VariableArray<double> priors = Variable.Constant(new double[] {0.1, 0.2, 0.3, 0.4}, item).Named("priors");
            array[item] = Variable.Bernoulli(priors[item]);
            Variable<int> indicesLength = Variable.New<int>().Named("indicesLength");
            Range xitem = new Range(indicesLength).Named("xitem");
            VariableArray<int> indices = Variable.Array<int>(xitem).Named("indices");
            VariableArray<bool> x = Variable.GetItems(array, indices).Named("x");
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(algorithm);
            for (int trial = 0; trial < 2; trial++)
            {
                indicesLength.ObservedValue = 2;
                if (trial == 0)
                {
                    indices.ObservedValue = new int[] {0, 3};
                    x.ObservedValue = new bool[] {true, false};
                }
                else
                {
                    indices.ObservedValue = new int[] {2, 2};
                    x.ObservedValue = new bool[] {true, true};
                }
                Bernoulli[] arrayExpectedArray = new Bernoulli[4];
                double z = 1;
                for (int i = 0; i < arrayExpectedArray.Length; i++)
                {
                    arrayExpectedArray[i] = new Bernoulli(priors.ObservedValue[i]);
                }
                for (int i = 0; i < indices.ObservedValue.Length; i++)
                {
                    z *= System.Math.Exp(arrayExpectedArray[indices.ObservedValue[i]].GetLogProb(x.ObservedValue[i]));
                    arrayExpectedArray[indices.ObservedValue[i]] = Bernoulli.PointMass(x.ObservedValue[i]);
                    if (trial == 1) break;
                }
                object arrayActual = engine.Infer<object>(array);
                IDistribution<bool[]> arrayExpected = Distribution<bool>.Array(arrayExpectedArray);
                Console.WriteLine(StringUtil.JoinColumns("array = ", arrayActual, " should be ", arrayExpected));
                Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-10);
                if (!(algorithm is GibbsSampling))
                {
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    double evExpected = System.Math.Log(z);
                    Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                    Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-10);
                }
            }
        }

        [Fact]
        public void GetItemsModel2Test()
        {
            Range item = new Range(1).Named("item");
            Variable<int> n = Variable.Observed<int>(1).Named("n");
            Range xitem = new Range(n).Named("xitem");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            array[item] = Variable.Bernoulli(0.1).ForEach(item);
            VariableArray<int> indices = Variable.Constant(new int[] {0}, xitem).Named("indices");
            //VariableArray<bool> x = Variable.GetItems(array, indices).Named("x");
            Variable<bool> xi = array[indices[xitem]];
            IVariableArray x = xi.ArrayVariable;
            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Bernoulli> xActual = engine.Infer<DistributionArray<Bernoulli>>(x);
            IDistribution<bool[]> xExpected = Distribution<bool>.Array(n.ObservedValue,
                                                                       delegate(int i) { return new Bernoulli(0.1); });
            Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void SubarrayTest()
        {
            Subarray(new ExpectationPropagation());
            Subarray(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsSubarrayTest()
        {
            Subarray(new GibbsSampling());
        }

        private void Subarray(IAlgorithm algorithm)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling)) block = Variable.If(evidence);
            Range item = new Range(4).Named("item");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            VariableArray<double> priors = Variable.Constant(new double[] {0.1, 0.2, 0.3, 0.4}, item).Named("priors");
            array[item] = Variable.Bernoulli(priors[item]);
            Variable<int> indicesLength = Variable.New<int>().Named("indicesLength");
            Range xitem = new Range(indicesLength).Named("xitem");
            VariableArray<int> indices = Variable.Array<int>(xitem).Named("indices");
            VariableArray<bool> xa = Variable.Subarray(array, indices).Named("x");
            Variable<bool> x = xa[xitem];
            Variable.ConstrainEqualRandom(x, new Bernoulli(0.6));
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            indices.ObservedValue = new int[] {0, 3};
            indicesLength.ObservedValue = indices.ObservedValue.Length;
            Bernoulli[] xExpected = new Bernoulli[indicesLength.ObservedValue];
            double z = 1;
            for (int i = 0; i < xExpected.Length; i++)
            {
                double pi = priors.ObservedValue[indices.ObservedValue[i]];
                double zi = 0.6*pi + (1 - 0.6)*(1 - pi);
                xExpected[i] = new Bernoulli(0.6*pi/zi);
                z *= zi;
            }
            InferenceEngine engine = new InferenceEngine(algorithm);
            double tolerance = 1e-10;
            if (algorithm is GibbsSampling)
            {
                Rand.Restart(12347);
                engine.NumberOfIterations = 20000;
                engine.ShowProgress = false;
                tolerance = 0.05;
            }
            DistributionArray<Bernoulli> xActual = engine.Infer<DistributionArray<Bernoulli>>(x.ArrayVariable);
            DistributionArray<Bernoulli> arrayPost = engine.Infer<DistributionArray<Bernoulli>>(array);
            Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", Distribution<bool>.Array(xExpected)));
            Console.WriteLine("array:");
            Console.WriteLine(arrayPost);
            for (int i = 0; i < indices.ObservedValue.Length; i++)
            {
                Assert.True(xExpected[i].MaxDiff(xActual[i]) < tolerance);
                Assert.True(xExpected[i].MaxDiff(arrayPost[indices.ObservedValue[i]]) < tolerance);
            }
            if (!(algorithm is GibbsSampling))
            {
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                double evExpected = System.Math.Log(z);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-10);
            }
        }

        [Fact]
        public void SubarrayObservedTest()
        {
            SubarrayObserved(new ExpectationPropagation());
            SubarrayObserved(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsSubarrayObservedTest()
        {
            SubarrayObserved(new GibbsSampling());
        }

        private void SubarrayObserved(IAlgorithm algorithm)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling)) block = Variable.If(evidence);
            Range item = new Range(4).Named("item");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            VariableArray<double> c = Variable.Constant(new double[] {0.1, 0.2, 0.3, 0.4}, item).Named("c");
            array[item] = Variable.Bernoulli(c[item]);
            Variable<int> indicesLength = Variable.New<int>().Named("indicesLength");
            Range xitem = new Range(indicesLength).Named("xitem");
            VariableArray<int> indices = Variable.Array<int>(xitem).Named("indices");
            VariableArray<bool> x = Variable.Subarray(array, indices).Named("x");
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            indicesLength.ObservedValue = 2;
            indices.ObservedValue = new int[] {0, 3};
            x.ObservedValue = new bool[] {true, false};
            Bernoulli[] arrayExpectedArray = new Bernoulli[4];
            double z = 1;
            for (int i = 0; i < arrayExpectedArray.Length; i++)
            {
                arrayExpectedArray[i] = new Bernoulli(c.ObservedValue[i]);
            }
            for (int i = 0; i < indices.ObservedValue.Length; i++)
            {
                z *= System.Math.Exp(arrayExpectedArray[indices.ObservedValue[i]].GetLogProb(x.ObservedValue[i]));
                arrayExpectedArray[indices.ObservedValue[i]] = Bernoulli.PointMass(x.ObservedValue[i]);
            }
            InferenceEngine engine = new InferenceEngine(algorithm);
            object arrayActual = engine.Infer(array);
            IDistribution<bool[]> arrayExpected = Distribution<bool>.Array(arrayExpectedArray);
            Console.WriteLine(StringUtil.JoinColumns("array = ", arrayActual, " should be ", arrayExpected));
            Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-10);
            if (!(algorithm is GibbsSampling))
            {
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                double evExpected = System.Math.Log(z);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-10);
            }
        }

        [Fact]
        public void GetItemsTest()
        {
            GetItems(new ExpectationPropagation());
            GetItems(new VariationalMessagePassing());
            GetItems(new GibbsSampling());
        }

        private void GetItems(IAlgorithm algorithm)
        {
            for (int observeArray = 0; observeArray < 2; observeArray++)
            {
                for (int observeItems = 0; observeItems < 2; observeItems++)
                {
                    List<IDistribution<bool[]>> posteriors = new List<IDistribution<bool[]>>();
                    List<double> evidences = new List<double>();
                    for (int modelType = 0; modelType < 3; modelType++)
                    {
                        double evidence;
                        var posterior = GetItemsModel(algorithm, modelType, observeArray > 0, observeItems > 0, out evidence);
                        posteriors.Add(posterior);
                        evidences.Add(evidence);
                        Console.WriteLine(posterior);
                        Console.WriteLine("log evidence = {0}", evidence);
                        if (posteriors.Count > 1)
                        {
                            Assert.True(posteriors[modelType - 1].MaxDiff(posterior) < 1e-10);
                            Assert.True(MMath.AbsDiff(evidences[modelType - 1], evidence) < 1e-10);
                        }
                    }
                }
            }
        }

        public IDistribution<bool[]> GetItemsModel(IAlgorithm algorithm, int modelType, bool observeArray, bool observeItems, out double modelEvidence)
        {
            Range F = new Range(3).Named("F");
            Range C = new Range(5).Named("C");

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling)) block = Variable.If(evidence);

            var bools = Variable.Array<bool>(C).Named("bools");
            var priors = Variable.Array<Bernoulli>(C).Named("priors");
            priors.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => new Bernoulli(0.1*(j + 1)));
            bools[C] = Variable<bool>.Random(priors[C]);
            if (observeArray) bools.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => true);

            Bernoulli like = new Bernoulli(0.2);
            var indices = Variable.Observed(new int[] {1, 3, 1}, F).Named("indices");
            if (modelType == 0)
            {
                for (int i = 0; i < indices.ObservedValue.Length; i++)
                {
                    Variable.ConstrainEqualRandom(bools[indices.ObservedValue[i]], like);
                    if (observeItems) Variable.ConstrainTrue(bools[indices.ObservedValue[i]]);
                }
            }
            else if (modelType == 1)
            {
                var items = Variable.GetItems(bools, indices).Named("items");
                Variable.ConstrainEqualRandom(items[F], like);
                if (observeItems) items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => true);
            }
            else
            {
                var items = Variable.Array<bool>(F).Named("items");
                items[F] = Variable.Copy(bools[indices[F]]);
                Variable.ConstrainEqualRandom(items[F], like);
                if (observeItems) items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => true);
            }
            F.AddAttribute(new Sequential());
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            var engine = new InferenceEngine();
            engine.Algorithm = algorithm;
            if (algorithm is GibbsSampling) modelEvidence = 0;
            else modelEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
            IDistribution<bool[]> weightsActual = engine.Infer<IDistribution<bool[]>>(bools);
            return weightsActual;
        }

        [Fact]
        public void GetItemsGateTest()
        {
            Range userThreshold = new Range(2).Named("userThreshold");
            Range user = new Range(2).Named("user");
            var userThresholds = Variable.Array(Variable.Array<double>(userThreshold), user);
            userThresholds.Name = nameof(userThresholds);
            userThresholds[user][userThreshold] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(user, userThreshold);
            Range observation = new Range(2);
            observation.Name = nameof(observation);
            var userIds = Variable.Observed(new int[] { 0, 1 }, observation);
            userIds.Name = nameof(userIds);

            var useSharedUserThresholds = Variable.Observed(default(bool)).Named("UseSharedUserThresholds");

            using (Variable.ForEach(observation))
            {
                var userId = userIds[observation];
                var userIdForThresholds = Variable.New<int>().Named("UserIdForThresholds");

                var userThresholdsObs = Variable.Array<double>(userThreshold).Named("userThresholdsObs");

                using (Variable.If(useSharedUserThresholds))
                {
                    userIdForThresholds.SetTo(0);
                    userThresholdsObs.SetTo(Variable.Copy(userThresholds[0]));
                }
                using (Variable.IfNot(useSharedUserThresholds))
                {
                    userIdForThresholds.SetTo(userId);
                    // does not work if index is userIdForThresholds
                    userThresholdsObs.SetTo(Variable.Copy(userThresholds[userId]));
                }
                Variable.ConstrainPositive(userThresholdsObs[userThreshold]);
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(0, e.Warnings.Count);
            };
            engine.Infer(userThresholds);
        }

        [Fact]
        public void GetItemsDeepIndexTest()
        {
            Range C = new Range(5).Named("C");
            var bools = Variable.Array<bool>(C).Named("bools");
            var priors = Variable.Array<Bernoulli>(C).Named("priors");
            priors.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => new Bernoulli(0.1 * (j + 1)));
            bools[C] = Variable<bool>.Random(priors[C]);
            Range F = new Range(3).Named("F");
            var indices = Variable.Observed(new int[] { 1, 3, 1 }, F).Named("indices");
            indices.SetValueRange(C);
            Range G = new Range(3).Named("G");
            var indices2 = Variable.Observed(new int[] { 2, 0, 1 }, G).Named("indices2");
            indices2.SetValueRange(F);
            Bernoulli like = new Bernoulli(0.2);
            Variable.ConstrainEqualRandom(bools[indices[indices2[G]]], like);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(1, e.Warnings.Count);
            };
            var boolsActual = engine.Infer<IList<Bernoulli>>(bools);
            var boolsExpected = new BernoulliArray(C.SizeAsInt, i =>
            {
                if (i == 1)
                    return priors.ObservedValue[i] * like * like;
                else if(i == 3)
                    return priors.ObservedValue[i] * like;
                else
                    return priors.ObservedValue[i];
            });
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void GetItemsDeepIndexTest2()
        {
            Range C = new Range(5).Named("C");
            var bools = Variable.Array<bool>(C).Named("bools");
            var priors = Variable.Array<Bernoulli>(C).Named("priors");
            priors.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => new Bernoulli(0.1 * (j + 1)));
            bools[C] = Variable<bool>.Random(priors[C]);
            Range F = new Range(3).Named("F");
            Range G = new Range(1).Named("G");
            var indices = Variable.Observed(new int[][] { new int[] { 1 }, new int[] { 3 }, new int[] { 1 } }, F, G).Named("indices");
            indices.SetValueRange(C);
            Bernoulli like = new Bernoulli(0.2);
            if (true)
            {
                Variable.ConstrainEqualRandom(bools[indices[F][0]], like);
            }
            else
            {
                var zero = Variable.Observed(0).Named("zero");
                Variable.ConstrainEqualRandom(bools[indices[F][zero]], like);
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(1, e.Warnings.Count);
            };
            var boolsActual = engine.Infer<IList<Bernoulli>>(bools);
            var boolsExpected = new BernoulliArray(C.SizeAsInt, i =>
            {
                if (i == 1)
                    return priors.ObservedValue[i] * like * like;
                else if (i == 3)
                    return priors.ObservedValue[i] * like;
                else
                    return priors.ObservedValue[i];
            });
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void GetItemsDeepIndexTest3()
        {
            Range C = new Range(5).Named("C");
            var bools = Variable.Array<bool>(C).Named("bools");
            var priors = Variable.Array<Bernoulli>(C).Named("priors");
            priors.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => new Bernoulli(0.1 * (j + 1)));
            bools[C] = Variable<bool>.Random(priors[C]);
            Range F = new Range(3).Named("F");
            Range G = new Range(1).Named("G");
            var indices = Variable.Observed(new int[][] { new int[] { 1,3,1 } }, G, F).Named("indices");
            indices.SetValueRange(C);
            var indices2 = Variable.Observed(new int[] { 0,0,0 }, F).Named("indices2");
            Bernoulli like = new Bernoulli(0.2);
            Variable.ConstrainEqualRandom(bools[indices[indices2[F]][F]], like);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(1, e.Warnings.Count);
            };
            var boolsActual = engine.Infer<IList<Bernoulli>>(bools);
            var boolsExpected = new BernoulliArray(C.SizeAsInt, i =>
            {
                if (i == 1)
                    return priors.ObservedValue[i] * like * like;
                else if (i == 3)
                    return priors.ObservedValue[i] * like;
                else
                    return priors.ObservedValue[i];
            });
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void GetItemsDeepIndexTest4()
        {
            Range C = new Range(5).Named("C");
            var bools = Variable.Array<bool>(C).Named("bools");
            var priors = Variable.Array<Bernoulli>(C).Named("priors");
            priors.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => new Bernoulli(0.1 * (j + 1)));
            bools[C] = Variable<bool>.Random(priors[C]);
            Range F = new Range(3).Named("F");
            Range G = new Range(1).Named("G");
            var indices = Variable.Observed(new int[][] { new int[] { 1 }, new int[] { 3 }, new int[] { 1 } }, F, G).Named("indices");
            indices.SetValueRange(C);
            Bernoulli like = new Bernoulli(0.2);
            using (Variable.ForEach(F))
            {
                var indexMin = Variable<int>.Factor(System.Math.Min, indices[F][0], 3).Named("indexMin");
                Variable.ConstrainEqualRandom(bools[indexMin], like);
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(1, e.Warnings.Count);
            };
            var boolsActual = engine.Infer<IList<Bernoulli>>(bools);
            var boolsExpected = new BernoulliArray(C.SizeAsInt, i =>
            {
                if (i == 1)
                    return priors.ObservedValue[i] * like * like;
                else if (i == 3)
                    return priors.ObservedValue[i] * like;
                else
                    return priors.ObservedValue[i];
            });
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        /// <summary>
        /// This test fails with engine.Compiler.UnrollLoops=true
        /// </summary>
        [Fact]
        public void GetDeepJaggedItemsTest()
        {
            GetDeepJaggedItems(new ExpectationPropagation());
            GetDeepJaggedItems(new VariationalMessagePassing());
            GetDeepJaggedItems(new GibbsSampling());
        }

        private void GetDeepJaggedItems(IAlgorithm algorithm)
        {
            for (int observeArray = 0; observeArray < 2; observeArray++)
            {
                for (int observeItems = 0; observeItems < 2; observeItems++)
                {
                    List<IDistribution<bool[]>> posteriors = new List<IDistribution<bool[]>>();
                    List<double> evidences = new List<double>();
                    for (int modelType = 0; modelType < 3; modelType++)
                    {
                        double evidence;
                        var posterior = GetDeepJaggedItemsModel(algorithm, modelType, observeArray > 0, observeItems > 0, out evidence);
                        posteriors.Add(posterior);
                        evidences.Add(evidence);
                        Console.WriteLine(posterior);
                        Console.WriteLine("log evidence = {0}", evidence);
                        if (posteriors.Count > 1)
                        {
                            Assert.True(posteriors[modelType - 1].MaxDiff(posterior) < 1e-10);
                            Assert.True(MMath.AbsDiff(evidences[modelType - 1], evidence) < 1e-10);
                        }
                    }
                }
            }
        }

        public IDistribution<bool[]> GetDeepJaggedItemsModel(IAlgorithm algorithm, int modelType, bool observeArray, bool observeItems, out double modelEvidence)
        {
            Range F = new Range(2).Named("F");
            Range F2 = new Range(2).Named("F2");
            Range F3 = new Range(3).Named("F3");
            Range C = new Range(5).Named("C");

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling))
                block = Variable.If(evidence);

            var bools = Variable.Array<bool>(C).Named("bools");
            var priors = Variable.Array<Bernoulli>(C).Named("priors");
            priors.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => new Bernoulli(0.1 * (j + 1)));
            bools[C] = Variable<bool>.Random(priors[C]);
            if (observeArray)
                bools.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => true);

            Bernoulli like = new Bernoulli(0.2);
            var indices = Variable.Observed(new int[][][] {
                new int[][] { new int[] { 1, 3, 1 }, new int[] { 2, 1, 2 } },
                new int[][] { new int[] { 1, 3, 1 }, new int[] { 2, 1, 2 } },
            }, F, F2, F3).Named("indices");
            if (modelType == 0)
            {
                for (int i = 0; i < indices.ObservedValue.Length; i++)
                {
                    for (int j = 0; j < indices.ObservedValue[i].Length; j++)
                    {
                        for (int k = 0; k < indices.ObservedValue[i][j].Length; k++)
                        {
                            Variable.ConstrainEqualRandom(bools[indices.ObservedValue[i][j][k]], like);
                            if (observeItems)
                                Variable.ConstrainTrue(bools[indices.ObservedValue[i][j][k]]);
                        }
                    }
                }
            }
            else if (modelType == 1)
            {
                var items = GetDeepJaggedItems(bools, indices).Named("items");
                Variable.ConstrainEqualRandom(items[F][F2][F3], like);
                if (observeItems)
                    items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => Util.ArrayInit(F2.SizeAsInt, j => Util.ArrayInit(F3.SizeAsInt, k => true)));
            }
            else
            {
                var items = Variable.Array(Variable.Array(Variable.Array<bool>(F3),F2),F).Named("items");
                items[F][F2][F3] = Variable.Copy(bools[indices[F][F2][F3]]);
                Variable.ConstrainEqualRandom(items[F][F2][F3], like);
                if (observeItems)
                    items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => Util.ArrayInit(F2.SizeAsInt, j => Util.ArrayInit(F3.SizeAsInt, k => true)));
            }
            F.AddAttribute(new Sequential());
            if (!(algorithm is GibbsSampling))
                block.CloseBlock();

            var engine = new InferenceEngine();
            engine.Algorithm = algorithm;
            if (algorithm is GibbsSampling)
                modelEvidence = 0;
            else
                modelEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
            IDistribution<bool[]> post = engine.Infer<IDistribution<bool[]>>(bools);
            return post;
        }

        public static VariableArray<VariableArray<T>, T[][]> GetJaggedItems<T>(VariableArray<T> array, VariableArray<VariableArray<int>, int[][]> indices)
        {
            var range1 = indices.Range;
            var range2 = indices[range1].Range;
            var result = Variable.Array(Variable.Array<T>(range2), range1);
            result.SetTo(Collection.GetJaggedItems, array, indices);
            return result;
        }

        [Fact]
        public void GetJaggedItemsTest()
        {
            GetJaggedItems(new ExpectationPropagation());
            GetJaggedItems(new VariationalMessagePassing());
            GetJaggedItems(new GibbsSampling());
        }

        private void GetJaggedItems(IAlgorithm algorithm)
        {
            for (int observeArray = 0; observeArray < 2; observeArray++)
            {
                for (int observeItems = 0; observeItems < 2; observeItems++)
                {
                    List<IDistribution<bool[]>> posteriors = new List<IDistribution<bool[]>>();
                    List<double> evidences = new List<double>();
                    for (int modelType = 0; modelType < 3; modelType++)
                    {
                        double evidence;
                        var posterior = GetJaggedItemsModel(algorithm, modelType, observeArray > 0, observeItems > 0, out evidence);
                        posteriors.Add(posterior);
                        evidences.Add(evidence);
                        Console.WriteLine(posterior);
                        Console.WriteLine("log evidence = {0}", evidence);
                        if (posteriors.Count > 1)
                        {
                            Assert.True(posteriors[modelType - 1].MaxDiff(posterior) < 1e-10);
                            Assert.True(MMath.AbsDiff(evidences[modelType - 1], evidence) < 1e-10);
                        }
                    }
                }
            }
        }

        public IDistribution<bool[]> GetJaggedItemsModel(IAlgorithm algorithm, int modelType, bool observeArray, bool observeItems, out double modelEvidence)
        {
            Range F = new Range(2).Named("F");
            Range F2 = new Range(3).Named("F2");
            Range C = new Range(5).Named("C");

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling))
                block = Variable.If(evidence);

            var bools = Variable.Array<bool>(C).Named("bools");
            var priors = Variable.Array<Bernoulli>(C).Named("priors");
            priors.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => new Bernoulli(0.1 * (j + 1)));
            bools[C] = Variable<bool>.Random(priors[C]);
            if (observeArray)
                bools.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => true);

            Bernoulli like = new Bernoulli(0.2);
            var indices = Variable.Observed(new int[][] { new int[] { 1, 3, 1 }, new int[] { 2, 1, 2 } }, F, F2).Named("indices");
            if (modelType == 0)
            {
                for (int i = 0; i < indices.ObservedValue.Length; i++)
                {
                    for (int j = 0; j < indices.ObservedValue[i].Length; j++)
                    {
                        Variable.ConstrainEqualRandom(bools[indices.ObservedValue[i][j]], like);
                        if (observeItems)
                            Variable.ConstrainTrue(bools[indices.ObservedValue[i][j]]);
                    }
                }
            }
            else if (modelType == 1)
            {
                var items = GetJaggedItems(bools, indices).Named("items");
                Variable.ConstrainEqualRandom(items[F][F2], like);
                if (observeItems)
                    items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => Util.ArrayInit(F2.SizeAsInt, j => true));
            }
            else
            {
                var items = Variable.Array(Variable.Array<bool>(F2), F).Named("items");
                items[F][F2] = Variable.Copy(bools[indices[F][F2]]);
                Variable.ConstrainEqualRandom(items[F][F2], like);
                if (observeItems)
                    items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => Util.ArrayInit(F2.SizeAsInt, j => true));
            }
            F.AddAttribute(new Sequential());
            if (!(algorithm is GibbsSampling))
                block.CloseBlock();

            var engine = new InferenceEngine();
            engine.Algorithm = algorithm;
            if (algorithm is GibbsSampling)
                modelEvidence = 0;
            else
                modelEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
            IDistribution<bool[]> post = engine.Infer<IDistribution<bool[]>>(bools);
            return post;
        }

        [Fact]
        public void GetJaggedItemsWarning()
        {
            Range F = new Range(2).Named("F");
            Range F2 = new Range(3).Named("F2");
            Range C = new Range(5).Named("C");

            var bools = Variable.Array<bool>(C).Named("bools");
            var priors = Variable.Array<Bernoulli>(C).Named("priors");
            priors.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => new Bernoulli(0.1 * (j + 1)));
            bools[C] = Variable<bool>.Random(priors[C]);

            Bernoulli like = new Bernoulli(0.2);
            var indices = Variable.Observed(new int[][] { new int[] { 1, 3, 1 }, new int[] { 2, 1, 2 } }, F, F2).Named("indices");
            var items = Variable.Array(Variable.Array<bool>(F2), F).Named("items");
            using (Variable.ForEach(F))
            {
                var indicesLocal = Variable.Array<int>(F2).Named("indicesLocal");
                indicesLocal[F2] = Variable.Copy(indices[F][F2]);
                items[F][F2] = Variable.Copy(bools[indicesLocal[F2]]);
            }
            Variable.ConstrainEqualRandom(items[F][F2], like);

            var engine = new InferenceEngine();
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(1, e.Warnings.Count);
            };
            IDistribution<bool[]> post = engine.Infer<IDistribution<bool[]>>(bools);            
        }

        public static VariableArray<VariableArray<VariableArray<T>, T[][]>, T[][][]> GetDeepJaggedItems<T>(VariableArray<T> array, VariableArray<VariableArray<VariableArray<int>, int[][]>, int[][][]> indices)
        {
            var range1 = indices.Range;
            var range2 = indices[range1].Range;
            var range3 = indices[range1][range2].Range;
            var result = Variable.Array(Variable.Array(Variable.Array<T>(range3), range2), range1);
            result.SetTo(Collection.GetDeepJaggedItems, array, indices);
            return result;
        }

        [Fact]
        public void GetItemsFromJaggedTest()
        {
            GetItemsFromJagged(new ExpectationPropagation());
            GetItemsFromJagged(new VariationalMessagePassing());
            GetItemsFromJagged(new GibbsSampling());
        }

        private void GetItemsFromJagged(IAlgorithm algorithm)
        {
            for (int observeArray = 0; observeArray < 2; observeArray++)
            {
                for (int observeItems = 0; observeItems < 2; observeItems++)
                {
                    List<Diffable> posteriors = new List<Diffable>();
                    List<double> evidences = new List<double>();
                    for (int modelType = 0; modelType < 3; modelType++)
                    {
                        double evidence;
                        var posterior = GetItemsFromJaggedModel(algorithm, modelType, observeArray > 0, observeItems > 0, out evidence);
                        posteriors.Add(posterior);
                        evidences.Add(evidence);
                        Console.WriteLine(posterior);
                        Console.WriteLine("log evidence = {0}", evidence);
                        if (posteriors.Count > 1)
                        {
                            Assert.True(posteriors[modelType - 1].MaxDiff(posterior) < 1e-10);
                            Assert.True(MMath.AbsDiff(evidences[modelType - 1], evidence) < 1e-10);
                        }
                    }
                }
            }
        }

        public IDistribution<bool[][]> GetItemsFromJaggedModel(IAlgorithm algorithm, int modelType, bool observeArray, bool observeItems, out double modelEvidence)
        {
            Range F = new Range(3).Named("F");
            Range outer = new Range(5).Named("outer");
            Range inner = new Range(5).Named("inner");

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling))
                block = Variable.If(evidence);

            var bools = Variable.Array(Variable.Array<bool>(inner), outer).Named("bools");
            var priorArray = Util.ArrayInit(outer.SizeAsInt, i => 
                Util.ArrayInit(inner.SizeAsInt, j => 
                    new Bernoulli((i + j + 1)/10.0)));
            var priors = Variable.Observed(priorArray, outer, inner).Named("priors");
            bools[outer][inner] = Variable<bool>.Random(priors[outer][inner]);
            if (observeArray)
                bools.ObservedValue = Util.ArrayInit(outer.SizeAsInt, i => 
                    Util.ArrayInit(inner.SizeAsInt, j => true));

            Bernoulli like = new Bernoulli(0.2);
            var indices = Variable.Observed(new int[] { 1, 3, 1 }, F).Named("indices");
            var indices2 = Variable.Observed(new int[] { 2, 1, 2 }, F).Named("indices2");
            if (modelType == 0)
            {
                for (int i = 0; i < indices.ObservedValue.Length; i++)
                {
                    Variable.ConstrainEqualRandom(bools[indices.ObservedValue[i]][indices2.ObservedValue[i]], like);
                    if (observeItems)
                        Variable.ConstrainTrue(bools[indices.ObservedValue[i]][indices2.ObservedValue[i]]);
                }
            }
            else if (modelType == 1)
            {
                var items = GetItemsFromJagged(bools, indices, indices2).Named("items");
                Variable.ConstrainEqualRandom(items[F], like);
                if (observeItems)
                    items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => true);
            }
            else
            {
                var items = Variable.Array<bool>(F).Named("items");
                items[F] = Variable.Copy(bools[indices[F]][indices2[F]]);
                Variable.ConstrainEqualRandom(items[F], like);
                if (observeItems)
                    items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => true);
            }
            F.AddAttribute(new Sequential());
            if (!(algorithm is GibbsSampling))
                block.CloseBlock();

            var engine = new InferenceEngine();
            engine.Algorithm = algorithm;
            if (algorithm is GibbsSampling)
                modelEvidence = 0;
            else
                modelEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
            IDistribution<bool[][]> weightsActual = engine.Infer<IDistribution<bool[][]>>(bools);
            return weightsActual;
        }

        [Fact]
        public void GetItemsFromJaggedWarning()
        {
            Range F = new Range(3).Named("F");
            Range F2 = new Range(3).Named("F2");
            Range outer = new Range(5).Named("outer");
            Range inner = new Range(5).Named("inner");

            var bools = Variable.Array(Variable.Array<bool>(inner), outer).Named("bools");
            var priorArray = Util.ArrayInit(outer.SizeAsInt, i =>
                Util.ArrayInit(inner.SizeAsInt, j =>
                    new Bernoulli((i + j + 1) / 10.0)));
            var priors = Variable.Observed(priorArray, outer, inner).Named("priors");
            bools[outer][inner] = Variable<bool>.Random(priors[outer][inner]);

            Bernoulli like = new Bernoulli(0.2);
            var indices = Variable.Observed(new int[] { 1, 3, 1 }, F).Named("indices");
            var indices2 = Variable.Observed(new int[][] { new int[] { 2, 1, 2 }, new int[] { 2, 1, 2 }, new int[] { 2, 1, 2 } }, F, F2).Named("indices2");
            var items = Variable.Array(Variable.Array<bool>(F2), F).Named("items");
            items[F][F2] = Variable.Copy(bools[indices[F]][indices2[F][F2]]);
            Variable.ConstrainEqualRandom(items[F][F2], like);

            var engine = new InferenceEngine();
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(1, e.Warnings.Count);
            };
            IDistribution<bool[][]> weightsActual = engine.Infer<IDistribution<bool[][]>>(bools);
        }

        /// <summary>
        /// Gets a variable array containing (possibly duplicated) items of a source array
        /// </summary>
        /// <typeparam name="T">The domain type of array elements</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="indices">Variable array containing the depth 1 indices of the elements to get.  Indices may be duplicated.</param>
        /// <param name="indices2">Variable array containing the depth 2 indices of the elements to get.  Indices may be duplicated.</param>
        /// <returns>variable array with the specified items</returns>
        /// <remarks>
        /// If the indices are known to be all different, use <see cref="Subarray"/> for greater efficiency.
        /// </remarks>
        public static VariableArray<T> GetItemsFromJagged<T>(VariableArray<VariableArray<T>, T[][]> array, VariableArray<int> indices, VariableArray<int> indices2)
        {
            VariableArray<T> result = new VariableArray<T>(indices.Range);
            result.SetTo(Collection.GetItemsFromJagged, array, indices, indices2);
            return result;
        }

        [Fact]
        public void GetItemsFromDeepJaggedTest()
        {
            GetItemsFromDeepJagged(new ExpectationPropagation());
            GetItemsFromJagged(new VariationalMessagePassing());
            GetItemsFromJagged(new GibbsSampling());
        }

        private void GetItemsFromDeepJagged(IAlgorithm algorithm)
        {
            for (int observeArray = 0; observeArray < 2; observeArray++)
            {
                for (int observeItems = 0; observeItems < 2; observeItems++)
                {
                    List<Diffable> posteriors = new List<Diffable>();
                    List<double> evidences = new List<double>();
                    for (int modelType = 0; modelType < 3; modelType++)
                    {
                        double evidence;
                        var posterior = GetItemsFromDeepJaggedModel(algorithm, modelType, observeArray > 0, observeItems > 0, out evidence);
                        posteriors.Add(posterior);
                        evidences.Add(evidence);
                        Console.WriteLine(posterior);
                        Console.WriteLine("log evidence = {0}", evidence);
                        if (posteriors.Count > 1)
                        {
                            Assert.True(posteriors[modelType - 1].MaxDiff(posterior) < 1e-10);
                            Assert.True(MMath.AbsDiff(evidences[modelType - 1], evidence) < 1e-10);
                        }
                    }
                }
            }
        }

        public IDistribution<bool[][][]> GetItemsFromDeepJaggedModel(IAlgorithm algorithm, int modelType, bool observeArray, bool observeItems, out double modelEvidence)
        {
            Range F = new Range(3).Named("F");
            Range outer = new Range(5).Named("outer");
            Range middle = new Range(5).Named("middle");
            Range inner = new Range(5).Named("inner");

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling))
                block = Variable.If(evidence);

            var bools = Variable.Array(Variable.Array(Variable.Array<bool>(inner), middle), outer).Named("bools");
            var priorArray = Util.ArrayInit(outer.SizeAsInt, i =>
                Util.ArrayInit(middle.SizeAsInt, j =>
                Util.ArrayInit(inner.SizeAsInt, k =>
                    new Bernoulli((i + j + k + 1)/15.0))));
            var priors = Variable.Observed(priorArray, outer, middle, inner).Named("priors");
            bools[outer][middle][inner] = Variable<bool>.Random(priors[outer][middle][inner]);
            if (observeArray)
                bools.ObservedValue = Util.ArrayInit(outer.SizeAsInt, i =>
                    Util.ArrayInit(middle.SizeAsInt, j =>
                    Util.ArrayInit(inner.SizeAsInt, k => true)));

            Bernoulli like = new Bernoulli(0.2);
            var indices = Variable.Observed(new int[] { 1, 3, 1 }, F).Named("indices");
            var indices2 = Variable.Observed(new int[] { 2, 1, 2 }, F).Named("indices2");
            var indices3 = Variable.Observed(new int[] { 3, 2, 3 }, F).Named("indices3");
            if (modelType == 0)
            {
                for (int i = 0; i < indices.ObservedValue.Length; i++)
                {
                    Variable.ConstrainEqualRandom(bools[indices.ObservedValue[i]][indices2.ObservedValue[i]][indices3.ObservedValue[i]], like);
                    if (observeItems)
                        Variable.ConstrainTrue(bools[indices.ObservedValue[i]][indices2.ObservedValue[i]][indices3.ObservedValue[i]]);
                }
            }
            else if (modelType == 1)
            {
                var items = GetItemsFromDeepJagged(bools, indices, indices2, indices3).Named("items");
                Variable.ConstrainEqualRandom(items[F], like);
                if (observeItems)
                    items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => true);
            }
            else
            {
                var items = Variable.Array<bool>(F).Named("items");
                items[F] = Variable.Copy(bools[indices[F]][indices2[F]][indices3[F]]);
                Variable.ConstrainEqualRandom(items[F], like);
                if (observeItems)
                    items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => true);
            }
            F.AddAttribute(new Sequential());
            if (!(algorithm is GibbsSampling))
                block.CloseBlock();

            var engine = new InferenceEngine();
            engine.Algorithm = algorithm;
            if (algorithm is GibbsSampling)
                modelEvidence = 0;
            else
                modelEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
            IDistribution<bool[][][]> weightsActual = engine.Infer<IDistribution<bool[][][]>>(bools);
            return weightsActual;
        }

        /// <summary>
        /// Gets a variable array containing (possibly duplicated) items of a source array
        /// </summary>
        /// <typeparam name="T">The domain type of array elements</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="indices">Variable array containing the depth 1 indices of the elements to get.  Indices may be duplicated.</param>
        /// <param name="indices2">Variable array containing the depth 2 indices of the elements to get.  Indices may be duplicated.</param>
        /// <param name="indices3">Variable array containing the depth 3 indices of the elements to get.  Indices may be duplicated.</param>
        /// <returns>variable array with the specified items</returns>
        /// <remarks>
        /// If the indices are known to be all different, use <see cref="Subarray"/> for greater efficiency.
        /// </remarks>
        public static VariableArray<T> GetItemsFromDeepJagged<T>(VariableArray<VariableArray<VariableArray<T>, T[][]>, T[][][]> array, VariableArray<int> indices, VariableArray<int> indices2, VariableArray<int> indices3)
        {
            VariableArray<T> result = new VariableArray<T>(indices.Range);
            result.SetTo(Collection.GetItemsFromDeepJagged, array, indices, indices2, indices3);
            return result;
        }

        [Fact]
        public void GetJaggedItemsFromJaggedTest()
        {
            GetJaggedItemsFromJagged(new ExpectationPropagation());
            GetJaggedItemsFromJagged(new VariationalMessagePassing());
            GetJaggedItemsFromJagged(new GibbsSampling());
        }

        private void GetJaggedItemsFromJagged(IAlgorithm algorithm)
        {
            for (int observeArray = 0; observeArray < 2; observeArray++)
            {
                for (int observeItems = 0; observeItems < 2; observeItems++)
                {
                    List<Diffable> posteriors = new List<Diffable>();
                    List<double> evidences = new List<double>();
                    for (int modelType = 2; modelType < 3; modelType++)
                    {
                        double evidence;
                        var posterior = GetJaggedItemsFromJaggedModel(algorithm, modelType, observeArray > 0, observeItems > 0, out evidence);
                        posteriors.Add(posterior);
                        evidences.Add(evidence);
                        Console.WriteLine(posterior);
                        Console.WriteLine("log evidence = {0}", evidence);
                        if (posteriors.Count > 1)
                        {
                            Assert.True(posteriors[modelType - 1].MaxDiff(posterior) < 1e-10);
                            Assert.True(MMath.AbsDiff(evidences[modelType - 1], evidence) < 1e-10);
                        }
                    }
                }
            }
        }

        public IDistribution<bool[][]> GetJaggedItemsFromJaggedModel(IAlgorithm algorithm, int modelType, bool observeArray, bool observeItems, out double modelEvidence)
        {
            Range F = new Range(2).Named("F");
            Range F2 = new Range(3).Named("F2");
            Range outer = new Range(5).Named("outer");
            Range inner = new Range(5).Named("inner");

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling))
                block = Variable.If(evidence);

            var bools = Variable.Array(Variable.Array<bool>(inner), outer).Named("bools");
            var priorArray = Util.ArrayInit(outer.SizeAsInt, i =>
                Util.ArrayInit(inner.SizeAsInt, j =>
                    new Bernoulli((i + j + 1) / 10.0)));
            var priors = Variable.Observed(priorArray, outer, inner).Named("priors");
            bools[outer][inner] = Variable<bool>.Random(priors[outer][inner]);
            if (observeArray)
                bools.ObservedValue = Util.ArrayInit(outer.SizeAsInt, i =>
                    Util.ArrayInit(inner.SizeAsInt, j => true));

            Bernoulli like = new Bernoulli(0.2);
            var indices = Variable.Observed(new int[][] { new int[] { 1, 3, 1 }, new int[] { 2, 1, 2 } }, F, F2).Named("indices");
            var indices2 = Variable.Observed(new int[][] { new int[] { 2, 1, 2 }, new int[] { 1, 3, 1 } }, F, F2).Named("indices2");
            if (modelType == 0)
            {
                for (int i = 0; i < indices.ObservedValue.Length; i++)
                {
                    for (int j = 0; j < indices.ObservedValue[i].Length; j++)
                    {
                        Variable.ConstrainEqualRandom(bools[indices.ObservedValue[i][j]][indices2.ObservedValue[i][j]], like);
                        if (observeItems)
                            Variable.ConstrainTrue(bools[indices.ObservedValue[i][j]][indices2.ObservedValue[i][j]]);
                    }
                }
            }
            else if (modelType == 1)
            {
                var items = GetJaggedItemsFromJagged(bools, indices, indices2).Named("items");
                Variable.ConstrainEqualRandom(items[F][F2], like);
                if (observeItems)
                    items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => Util.ArrayInit(F2.SizeAsInt, j => true));
            }
            else
            {
                var items = Variable.Array(Variable.Array<bool>(F2),F).Named("items");
                items[F][F2] = Variable.Copy(bools[indices[F][F2]][indices2[F][F2]]);
                Variable.ConstrainEqualRandom(items[F][F2], like);
                if (observeItems)
                    items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => Util.ArrayInit(F2.SizeAsInt, j => true));
            }
            F.AddAttribute(new Sequential());
            if (!(algorithm is GibbsSampling))
                block.CloseBlock();

            var engine = new InferenceEngine();
            engine.Algorithm = algorithm;
            if (algorithm is GibbsSampling)
                modelEvidence = 0;
            else
                modelEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
            IDistribution<bool[][]> weightsActual = engine.Infer<IDistribution<bool[][]>>(bools);
            return weightsActual;
        }

        /// <summary>
        /// Gets a variable array containing (possibly duplicated) items of a source array
        /// </summary>
        /// <typeparam name="T">The domain type of array elements</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="indices">Variable array containing the depth 1 indices of the elements to get.  Indices may be duplicated.</param>
        /// <param name="indices2">Variable array containing the depth 2 indices of the elements to get.  Indices may be duplicated.</param>
        /// <returns>variable array with the specified items</returns>
        /// <remarks>
        /// If the indices are known to be all different, use <see cref="Subarray"/> for greater efficiency.
        /// </remarks>
        public static VariableArray<VariableArray<T>, T[][]> GetJaggedItemsFromJagged<T>(VariableArray<VariableArray<T>, T[][]> array, VariableArray<VariableArray<int>,int[][]> indices, VariableArray<VariableArray<int>, int[][]> indices2)
        {
            var range1 = indices.Range;
            var range2 = indices[range1].Range;
            var result = Variable.Array(Variable.Array<T>(range2), range1);
            result.SetTo(Collection.GetJaggedItemsFromJagged, array, indices, indices2);
            return result;
        }

        internal void JaggedSubarrayRoundoffTest()
        {
            Range item = new Range(1);
            Gaussian arrayPrior = Gaussian.FromMeanAndPrecision(1, 1e-8);
            var array = Variable.Array<double>(item).Named("array");
            array[item] = Variable.Random(arrayPrior).ForEach(item);
            Range outer = new Range(2);
            Range inner = new Range(1);
            var indices = Variable.Observed(new int[][] { new int[] { 0 }, new int[] { 0 } }, outer, inner);
            var items = Variable.JaggedSubarray(array, indices);
            var meanPrior = Gaussian.FromMeanAndPrecision(0, 1e-6);
            var means = Variable.Array<double>(outer);
            means[outer] = Variable.Random(meanPrior).ForEach(outer);
            means.AddAttribute(new PointEstimate());
            var precs = Variable.Observed(new double[] { 1e20, 1e-0 }, outer);
            Variable.ConstrainEqual(items[outer][0], Variable.GaussianFromMeanAndPrecision(means[outer], precs[outer]));

            Gaussian messageFromMean0 = GaussianOp.SampleAverageConditional(meanPrior, precs.ObservedValue[0]);
            Gaussian messageFromMean1 = GaussianOp.SampleAverageConditional(meanPrior, precs.ObservedValue[1]);
            Gaussian arrayExpected = arrayPrior * messageFromMean0 * messageFromMean1;
            Gaussian messageToMean0 = GaussianOp.MeanAverageConditional(arrayPrior * messageFromMean1, precs.ObservedValue[0]);
            Gaussian messageToMean1 = GaussianOp.MeanAverageConditional(arrayPrior * messageFromMean0, precs.ObservedValue[1]);
            Gaussian[] meansExpectedArray = new Gaussian[] { messageToMean0 * meanPrior, messageToMean1 * meanPrior };
            var meansExpected = Distribution<double>.Array(meansExpectedArray);

            InferenceEngine engine = new InferenceEngine();
            //engine.Compiler.GivePriorityTo(typeof(JaggedSubarrayOp_NoDivide<>));
            var arrayActual = engine.Infer<IList<Gaussian>>(array);
            Console.WriteLine("array[0] = {0} should be {1}", arrayActual[0], arrayExpected);
            var meansActual = engine.Infer<IList<Gaussian>>(means);
            Console.WriteLine(StringUtil.JoinColumns("means = ", meansActual, " should be ", meansExpected));
            Assert.True(arrayExpected.MaxDiff(arrayActual[0]) < 1e-10);
            Assert.True(meansExpected.MaxDiff(meansActual) < 1e-10);
        }

        [Fact]
        public void JaggedSubarrayTest()
        {
            JaggedSubarray(new ExpectationPropagation());
            JaggedSubarray(new VariationalMessagePassing());
            JaggedSubarray(new GibbsSampling());
        }

        private void JaggedSubarray(IAlgorithm algorithm)
        {
            for (int observeArray = 0; observeArray < 2; observeArray++)
            {
                for (int observeItems = 0; observeItems < 2; observeItems++)
                {
                    for (int indicesReadOnly = 0; indicesReadOnly < 2; indicesReadOnly++)
                    {
                        List<IDistribution<bool[]>> posteriors = new List<IDistribution<bool[]>>();
                        List<double> evidences = new List<double>();
                        for (int modelType = 0; modelType < 3; modelType++)
                        {
                            double evidence;
                            var posterior = JaggedSubarrayModel(algorithm, modelType, observeArray > 0, observeItems > 0, indicesReadOnly > 0, out evidence);
                            posteriors.Add(posterior);
                            evidences.Add(evidence);
                            Console.WriteLine(posterior);
                            Console.WriteLine("log evidence = {0}", evidence);
                            if (posteriors.Count > 1)
                            {
                                // Compare to the results on the previous modelType
                                Assert.True(posteriors[modelType - 1].MaxDiff(posterior) < 1e-10);
                                Assert.True(MMath.AbsDiff(evidences[modelType - 1], evidence) < 1e-10);
                            }
                        }
                    }
                }
            }
        }

        public IDistribution<bool[]> JaggedSubarrayModel(IAlgorithm algorithm, int modelType, bool observeArray, bool observeItems, bool indicesReadOnly,
                                                         out double modelEvidence)
        {
            Range N = new Range(2).Named("N");
            Range F = new Range(3).Named("F");
            Range C = new Range(5).Named("C");

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling)) block = Variable.If(evidence);

            var bools = Variable.Array<bool>(C).Named("bools");
            var priors = Variable.Array<Bernoulli>(C).Named("priors");
            priors.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => new Bernoulli(0.1*(j + 1)));
            bools[C] = Variable<bool>.Random(priors[C]);
            if (observeArray) bools.ObservedValue = Util.ArrayInit(C.SizeAsInt, j => true);

            Bernoulli like = new Bernoulli(0.2);
            var indices = Variable.Observed(new int[2][] {new int[] {1, 2, 3}, new int[] {1, 3, 4}}, N, F).Named("indices");
            if (indicesReadOnly) indices.IsReadOnly = true;
            if (modelType == 0)
            {
                for (int i = 0; i < indices.ObservedValue.Length; i++)
                {
                    for (int j = 0; j < indices.ObservedValue[i].Length; j++)
                    {
                        Variable.ConstrainEqualRandom(bools[indices.ObservedValue[i][j]], like);
                        if (observeItems) Variable.ConstrainTrue(bools[indices.ObservedValue[i][j]]);
                    }
                }
            }
            else if (modelType == 1)
            {
                var items = Variable.JaggedSubarray(bools, indices).Named("items");
                using (Variable.ForEach(N))
                {
                    Variable.ConstrainEqualRandom(items[N][F], like);
                }
                if (observeItems) items.ObservedValue = Util.ArrayInit(N.SizeAsInt, n => Util.ArrayInit(F.SizeAsInt, i => true));
            }
            else
            {
                using (Variable.ForEach(N))
                {
                    var items = Variable.Subarray(bools, indices[N]).Named("items");
                    Variable.ConstrainEqualRandom(items[F], like);
                    if (observeItems) items.ObservedValue = Util.ArrayInit(F.SizeAsInt, i => true);
                }
            }
            N.AddAttribute(new Sequential());
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            var engine = new InferenceEngine();
            engine.Algorithm = algorithm;
            if (algorithm is GibbsSampling) modelEvidence = 0;
            else modelEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
            IDistribution<bool[]> weightsActual = engine.Infer<IDistribution<bool[]>>(bools);
            return weightsActual;
        }

        [Fact]
        public void ArrayGetItemTest()
        {
            Range N = new Range(2).Named("N");
            Range M = new Range(3).Named("M");
            var priors = Variable.Array(Variable.Array<Gaussian>(M), N).Named("priors");
            priors.ObservedValue = Util.ArrayInit(N.SizeAsInt, i => Util.ArrayInit(M.SizeAsInt, j => new Gaussian()));
            var array = Variable.Array(Variable.Array<double>(M), N).Named("array");
            array[N][M] = Variable<double>.Random(priors[N][M]);
            var index = Variable.Observed(2).Named("index");
            using (Variable.ForEach(N))
            {
                var x = Variable<double>.Factor(Collection.GetItem, array[N], index).Named("x");
                Variable.ConstrainPositive(x);
            }
            var engine = new InferenceEngine();
            engine.Infer(array);
        }


        [Fact]
        public void GetItemsRepeatedRangeTest()
        {
            GetItemsRepeatedRange(new ExpectationPropagation());
            GetItemsRepeatedRange(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsGetItemsRepeatedRangeTest()
        {
            GetItemsRepeatedRange(new GibbsSampling());
        }

        private void GetItemsRepeatedRange(IAlgorithm algorithm)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling)) block = Variable.If(evidence);
            Range item = new Range(2).Named("item");
            Range item2 = new Range(2).Named("item2");
            var array = Variable.Array(Variable.Array<bool>(item2), item).Named("array");
            double[][] priorsArray = new double[][]
                {
                    new double[] {0.1, 0.2},
                    new double[] {0.3, 0.4}
                };
            var priors = Variable.Constant(priorsArray, item, item2).Named("priors");
            array[item][item2] = Variable.Bernoulli(priors[item][item2]);
            Variable<int> indicesLength = Variable.New<int>().Named("indicesLength");
            Range xitem = new Range(indicesLength).Named("xitem");
            VariableArray<int> indices1 = Variable.Array<int>(xitem).Named("indices1");
            VariableArray<int> indices2 = Variable.Array<int>(xitem).Named("indices2");
            double xLike = 0.6;
            if (true)
            {
                Variable<bool> x = array[indices1[xitem]][indices2[xitem]];
                Variable.ConstrainEqualRandom(x, new Bernoulli(xLike));
            }
            else
            {
                // this provides a useful test of IndexingTransform
                using (Variable.ForEach(xitem))
                {
                    Variable<int> temp = Variable.Copy(indices1[xitem]).Named("temp");
                    Variable<bool> x = array[temp][indices2[xitem]];
                    Variable.ConstrainEqualRandom(x, new Bernoulli(xLike));
                }
            }
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(algorithm);
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                // TODO: determine if this is worth checking
                //Assert.True(e.Warnings.Count == 1);
            };
            // Gibbs cannot achieve the desired accuracy when loops are unrolled.
            if (algorithm is GibbsSampling) engine.Compiler.UnrollLoops = false;
            for (int trial = 0; trial < 1; trial++)
            {
                Bernoulli[][] arrayExpectedArray = new Bernoulli[item.SizeAsInt][];
                for (int i = 0; i < arrayExpectedArray.Length; i++)
                {
                    arrayExpectedArray[i] = new Bernoulli[item2.SizeAsInt];
                    for (int j = 0; j < arrayExpectedArray[i].Length; j++)
                    {
                        arrayExpectedArray[i][j] = new Bernoulli(priors.ObservedValue[i][j]);
                    }
                }
                double z;
                if (trial == 0)
                {
                    indices1.ObservedValue = new int[] {0, 1};
                    indices2.ObservedValue = new int[] {1, 0};
                    indicesLength.ObservedValue = indices1.ObservedValue.Length;
                    z = 1;
                    for (int i = 0; i < indicesLength.ObservedValue; i++)
                    {
                        double pi = priors.ObservedValue[indices1.ObservedValue[i]][indices2.ObservedValue[i]];
                        double zi = xLike*pi + (1 - xLike)*(1 - pi);
                        arrayExpectedArray[indices1.ObservedValue[i]][indices2.ObservedValue[i]] = new Bernoulli(xLike*pi/zi);
                        z *= zi;
                    }
                }
                else
                {
                    indices1.ObservedValue = new int[] {1, 1};
                    indices2.ObservedValue = new int[] {0, 0};
                    indicesLength.ObservedValue = indices1.ObservedValue.Length;
                    double p = priors.ObservedValue[indices1.ObservedValue[0]][indices2.ObservedValue[0]];
                    z = xLike*xLike*p + (1 - xLike)*(1 - xLike)*(1 - p);
                    for (int i = 0; i < indicesLength.ObservedValue; i++)
                    {
                        arrayExpectedArray[indices1.ObservedValue[i]][indices2.ObservedValue[i]] = new Bernoulli(xLike*xLike*p/z);
                    }
                }
                object arrayActual = engine.Infer(array);
                IDistribution<bool[][]> arrayExpected = Distribution<bool>.Array(arrayExpectedArray);
                Console.WriteLine(StringUtil.JoinColumns("array = ", arrayActual, " should be ", arrayExpected));
                Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-10);
                if (!(algorithm is GibbsSampling))
                {
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    double evExpected = System.Math.Log(z);
                    Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                    Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-10);
                }
            }
        }

        [Fact]
        public void GetItemsSwitchedArrayTest()
        {
            GetItemsSwitchedArray(new ExpectationPropagation());
            //getItemsSwitchedArrayTest(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsGetItemsSwitchedArrayTest()
        {
            GetItemsSwitchedArray(new GibbsSampling());
        }

        private void GetItemsSwitchedArray(IAlgorithm algorithm)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            if (algorithm is GibbsSampling) block.CloseBlock();
            Range item = new Range(2).Named("item");
            Range item2 = new Range(2).Named("item2");
            var array = Variable.Array(Variable.Array<bool>(item2), item).Named("array");
            double[][] priorsArray = new double[][]
                {
                    new double[] {0.1, 0.2},
                    new double[] {0.3, 0.4}
                };
            var priors = Variable.Constant(priorsArray, item, item2).Named("priors");
            array[item][item2] = Variable.Bernoulli(priors[item][item2]);
            Variable<int> indicesLength = Variable.New<int>().Named("indicesLength");
            Range xitem = new Range(indicesLength).Named("xitem");
            VariableArray<int> indices = Variable.Array<int>(xitem).Named("indices");
            Variable<int> switchVar = Variable.DiscreteUniform(item).Named("switchVar");
            double xLike = 0.6;
            using (Variable.Switch(switchVar))
            {
                Variable<bool> x = array[switchVar][indices[xitem]].Named("x");
                Variable.ConstrainEqualRandom(x, new Bernoulli(xLike));
            }
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(algorithm);
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(0, e.Warnings.Count);
            };
            indices.ObservedValue = new int[] {0};
            indicesLength.ObservedValue = indices.ObservedValue.Length;
            Bernoulli[][] arrayExpectedArray = new Bernoulli[item.SizeAsInt][];
            double[][] arrayProbs = new double[item.SizeAsInt][];
            double z = 0;
            Discrete switchVarExpected = Discrete.Uniform(item.SizeAsInt);
            Vector switchProbs = switchVarExpected.GetWorkspace();
            for (int s = 0; s < switchProbs.Count; s++)
            {
                double zs = switchVarExpected[s];
                for (int i = 0; i < indices.ObservedValue.Length; i++)
                {
                    double pi = priors.ObservedValue[s][indices.ObservedValue[i]];
                    double zi = xLike*pi + (1 - xLike)*(1 - pi);
                    zs *= zi;
                }
                switchProbs[s] = zs;
                z += zs;
            }
            switchVarExpected.SetProbs(switchProbs);
            for (int i = 0; i < arrayProbs.Length; i++)
            {
                arrayProbs[i] = new double[item2.SizeAsInt];
                for (int j = 0; j < arrayProbs[i].Length; j++)
                {
                    arrayProbs[i][j] = priors.ObservedValue[i][j];
                }
            }
            for (int s = 0; s < switchProbs.Count; s++)
            {
                for (int i = 0; i < indices.ObservedValue.Length; i++)
                {
                    int ind = indices.ObservedValue[i];
                    double pi = priors.ObservedValue[s][ind];
                    double zi = xLike*pi + (1 - xLike)*(1 - pi);
                    arrayProbs[s][ind] += switchVarExpected[s]*(xLike*pi/zi - pi);
                }
            }
            for (int i = 0; i < arrayProbs.Length; i++)
            {
                arrayExpectedArray[i] = new Bernoulli[arrayProbs[i].Length];
                for (int j = 0; j < arrayProbs[i].Length; j++)
                {
                    arrayExpectedArray[i][j] = new Bernoulli(arrayProbs[i][j]);
                }
            }
            object arrayPost = engine.Infer(array);
            IDistribution<bool[][]> arrayExpected = Distribution<bool>.Array(arrayExpectedArray);
            Console.WriteLine(StringUtil.JoinColumns("array = ", arrayPost, " should be ", arrayExpected));
            double tolerance = 1e-10;
            if (algorithm is GibbsSampling) tolerance = 1e-1;
            Assert.True(arrayExpected.MaxDiff(arrayPost) < tolerance);
            if (!(algorithm is GibbsSampling))
            {
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                double evExpected = System.Math.Log(z);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < tolerance);
            }
        }

        [Fact]
        public void GetItemsSwitchedIndexTest()
        {
            GetItemsSwitchedIndex(new ExpectationPropagation());
            //GetItemsSwitchedIndexTest(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsGetItemsSwitchedIndexTest()
        {
            GetItemsSwitchedIndex(new GibbsSampling());
        }

        private void GetItemsSwitchedIndex(IAlgorithm algorithm)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            if (algorithm is GibbsSampling) block.CloseBlock();
            Range item = new Range(4).Named("item");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            VariableArray<double> priors = Variable.Constant(new double[] {0.1, 0.2, 0.3, 0.4}, item).Named("priors");
            array[item] = Variable.Bernoulli(priors[item]);
            Range switchItem = new Range(2).Named("switchItem");
            Variable<int> switchVar = Variable.DiscreteUniform(switchItem).Named("switchVar");
            VariableArray<int> indicesLength = Variable.Array<int>(switchItem).Named("indicesLength");
            Range xitem = new Range(indicesLength[switchItem]).Named("xitem");
            var indices = Variable.Array(Variable.Array<int>(xitem), switchItem).Named("indices");
            double xLike = 0.6;
            using (Variable.Switch(switchVar))
            {
                Variable<bool> x = array[indices[switchVar][xitem]];
                Variable.ConstrainEqualRandom(x, new Bernoulli(xLike));
            }
            if (!(algorithm is GibbsSampling)) block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(algorithm);
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(0, e.Warnings.Count);
            };
            indices.ObservedValue = new int[][] {new int[] {0}, new int[] {3}};
            indicesLength.ObservedValue = new int[] {indices.ObservedValue[0].Length, indices.ObservedValue[1].Length};
            Bernoulli[] arrayExpectedArray = new Bernoulli[item.SizeAsInt];
            double[] arrayProbs = new double[item.SizeAsInt];
            double z = 0;
            Discrete switchVarExpected = Discrete.Uniform(switchItem.SizeAsInt);
            Vector switchProbs = switchVarExpected.GetWorkspace();
            for (int s = 0; s < indices.ObservedValue.Length; s++)
            {
                double zs = switchVarExpected[s];
                for (int i = 0; i < indices.ObservedValue[s].Length; i++)
                {
                    double pi = priors.ObservedValue[indices.ObservedValue[s][i]];
                    double zi = xLike*pi + (1 - xLike)*(1 - pi);
                    zs *= zi;
                }
                switchProbs[s] = zs;
                z += zs;
            }
            switchVarExpected.SetProbs(switchProbs);
            for (int i = 0; i < arrayProbs.Length; i++)
            {
                arrayProbs[i] = priors.ObservedValue[i];
            }
            for (int s = 0; s < indices.ObservedValue.Length; s++)
            {
                for (int i = 0; i < indices.ObservedValue[s].Length; i++)
                {
                    int ind = indices.ObservedValue[s][i];
                    double pi = priors.ObservedValue[ind];
                    double zi = xLike*pi + (1 - xLike)*(1 - pi);
                    arrayProbs[ind] += switchVarExpected[s]*(xLike*pi/zi - pi);
                }
            }
            for (int i = 0; i < arrayProbs.Length; i++)
            {
                arrayExpectedArray[i] = new Bernoulli(arrayProbs[i]);
            }
            object arrayPost = engine.Infer(array);
            IDistribution<bool[]> arrayExpected = Distribution<bool>.Array(arrayExpectedArray);
            Console.WriteLine(StringUtil.JoinColumns("array = ", arrayPost, " should be ", arrayExpected));
            double tolerance = 1e-10;
            if (algorithm is GibbsSampling) tolerance = 1e-1;
            Assert.True(arrayExpected.MaxDiff(arrayPost) < tolerance);
            if (!(algorithm is GibbsSampling))
            {
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                double evExpected = System.Math.Log(z);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < tolerance);
            }
        }

        // Tests a model where an array has backward messages only for certain elements.
        [Fact]
        public void MixtureInferMeansPartiallyObsGetItem()
        {
            int T = 3;
            double[] xobs = new double[] {6.5, 9, 2};

            Range item = new Range(T).Named("item");
            VariableArray<double> x = Variable.Constant<double>(xobs, item).Named("x");
            Variable<Vector> D = Variable.Dirichlet(new double[] {1, 1}).Named("D");

            // EP fails with an ImproperMessageException if the prior precisions are too large.
            Variable<double> mean1 = Variable.GaussianFromMeanAndPrecision(6.0, 0.1).Named("mean1");
            Variable<double> mean2 = Variable.GaussianFromMeanAndPrecision(7.0, 0.1).Named("mean2");

            VariableArray<int> c = Variable.Array<int>(item);
            c[item] = Variable.Discrete(D).Named("c").ForEach(item);

            using (Variable.ForEach(item))
            {
                using (Variable.Case(c[item], 0))
                {
                    Variable.ConstrainEqual(x[item], Variable.GaussianFromMeanAndPrecision(mean1, 1.0));
                }
                using (Variable.Case(c[item], 1))
                {
                    Variable.ConstrainEqual(x[item], Variable.GaussianFromMeanAndPrecision(mean2, 1.0));
                }
            }

            Variable<int> index = Variable.New<int>().Named("index");
            Variable<int> h = c[index].Named("h");
            Variable.ConstrainEqual(h, 1);
            index.ObservedValue = 1;

            Variable<int> index2 = Variable.New<int>().Named("index2");
            Variable<int> h2 = c[index2].Named("h2");
            Variable.ConstrainEqual(h2, 0);
            index2.ObservedValue = 2;


            InferenceEngine ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(D));
            Console.WriteLine(ie.Infer(c));
            Console.WriteLine(ie.Infer(mean1));
            Console.WriteLine(ie.Infer(mean2));
            //  Console.ReadKey();
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}