// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Serialization;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class InnerProductArrayTests
    {
        public void RunTest()
        {

            //int numUsers = 2,
            //    numItems = 2,
            //    numTraits = 3;

            //double userPriorMean = 1,
            //       userPriorVariance = 10,
            //       itemPriorMean = -2,
            //       itemPriorVariance = 1;

            // TestPointMass();
            // TestSmallVariance();
            //string data_dir = @"C:\Users\t-botcse\Source\mlp\infernet\Prototypes\SweeplessRecommender\Data\special-case-study\data-check-1.mat";

            TestInnerProductArrayRecomenderCrazier("");
            // TestInnerProductArrayRecomenderCrazyToo("");
            // TestInnerProductArrayRecomenderCrazy("");
            
            //TestMatrixMultiplyRecomender(numUsers, numItems, numTraits, userPriorMean, userPriorVariance, itemPriorMean, itemPriorVariance);
            //TestInnerProductArrayRecomender(numUsers, numItems, numTraits, userPriorMean, userPriorVariance, itemPriorMean, itemPriorVariance);
            //TestInnerProductArray_BPM();
            //Console.ReadKey();

        }

        public void TestPointMass()
        {

            //
            // This should fail after the first cycle when all elements become point masses
            //

            int n = 2;
            Range I = new Range(n).Named("I");

            VariableArray<double> u = Variable.Array<double>(I).Named("u");
            VariableArray<double> v = Variable.Array<double>(I).Named("v");

            u[I] = Variable<double>.GaussianFromMeanAndVariance(1,0.1).ForEach(I);
            v[I] = Variable<double>.GaussianFromMeanAndVariance(2,0.1).ForEach(I);

            var z = Variable<double>.Factor(Factor.InnerProduct, u, v).Named("z");
            z.ObservedValue = 0;

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowFactorGraph = true;

            Gaussian[] uPost, vPost;

            for (int iter = 1; iter < 20; iter++)
            {
                engine.NumberOfIterations = iter;
                uPost = engine.Infer<Gaussian[]>(u);
                vPost = engine.Infer<Gaussian[]>(v);

                Console.WriteLine("Iteration {0}", iter);
                Console.WriteLine("Means:");
                for (int k = 0; k < n; k++)
                {
                    Console.Write("\t {0} {1}\n", uPost[k].GetMean(), vPost[k].GetMean());
                }
                Console.WriteLine("Variances:");
                for (int k = 0; k < n; k++)
                {
                    Console.Write("\t {0} {1}\n", uPost[k].GetVariance(), vPost[k].GetVariance());
                }
                
                Console.ReadKey();
            }

        }

        public void TestSmallVariance()
        {

            //
            // Find a vector v perpendicular to u = (1,1). This should be all right if initialised well. The v = (0,0) initialisation is a killer,
            // so take care when using InnerProductArray for forcing perpendicularity of arrays (vectors). 
            //

            // choose which vector to find and look for spontaneous symmetry breaking, will it converge to v = 0?

            int n = 2;
            Range I = new Range(n).Named("I");

            VariableArray<double> u = Variable.Array<double>(I).Named("u");
            VariableArray<double> v = Variable.Array<double>(I).Named("v");

            
            VariableArray<double> vmean = Variable.Constant(new double[] { 0, 0 }, I);
            // VariableArray<double> vmean = Variable.Constant(new double[] { 0.0, 1.0 }, I);
            // VariableArray<double> vmean = Variable.Constant(new double[] { 0.0, -1.0 }, I);
            // VariableArray<double> vmean = Variable.Constant(new double[] { -1.0, -1.0 }, I);

            u[I] = Variable<double>.GaussianFromMeanAndVariance(1, 1e-4).ForEach(I);
            v[I] = Variable<double>.GaussianFromMeanAndVariance(vmean[I], 1e4);

            var z = Variable<double>.GaussianFromMeanAndVariance(Variable.InnerProduct(u, v), 1e-4).Named("z");
            z.ObservedValue = 0;
 
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowFactorGraph = true;

            Gaussian[] uPost, vPost;

            for (int iter = 1; iter < 20; iter++)
            {
                engine.NumberOfIterations = iter;
                uPost = engine.Infer<Gaussian[]>(u);
                vPost = engine.Infer<Gaussian[]>(v);

                Console.WriteLine("Iteration {0}", iter);
                Console.WriteLine("Means:");
                Console.Write("\t u v\n");
                for (int k = 0; k < n; k++)
                {
                    Console.Write("\t {0} {1}\n", uPost[k].GetMean(), vPost[k].GetMean());
                }
                Console.WriteLine("Variances:");
                for (int k = 0; k < n; k++)
                {
                    Console.Write("\t {0} {1}\n", uPost[k].GetVariance(), vPost[k].GetVariance());
                }

                Console.ReadKey();
            }

        }


        public void TestMatrixMultiplyRecomender(int numUsers, int numItems, int numTraits, double userPriorMean, double userPriorVariance, double itemPriorMean, double itemPriorVariance)
        {

            double noiseVariance = 100;

            //int numUsers = 2,
            //    numItems = 2,
            //    numTraits = 1;

            //double userPriorMean =0, 
            //       userPriorVariance =1,
            //       itemPriorMean =0, 
            //       itemPriorVariance =1;

            // Define ranges
            Range user = new Range(numUsers).Named("user");
            Range item = new Range(numItems).Named("item");
            Range trait = new Range(numTraits).Named("trait");

            // Define latent variables
            VariableArray2D<double> userTraits = Variable.Array<double>(user, trait).Named("userTraits");
            VariableArray2D<double> itemTraits = Variable.Array<double>(trait, item).Named("itemTraits");
            VariableArray2D<double> z = Variable.Array<double>(user, item).Named("z");
            VariableArray2D<double> znoisy = Variable.Array<double>(user, item).Named("znoisy");
            VariableArray2D<bool> y = Variable.Array<bool>(user, item).Named("y");

            userTraits[user, trait] = Variable<double>.GaussianFromMeanAndVariance(userPriorMean, userPriorVariance).ForEach(user, trait);
            itemTraits[trait, item] = Variable<double>.GaussianFromMeanAndVariance(itemPriorMean, itemPriorVariance).ForEach(trait, item);

            z = Variable.MatrixMultiply(userTraits, itemTraits);
            znoisy[user, item] = Variable<double>.GaussianFromMeanAndVariance(z[user, item], noiseVariance);
            y[user, item] = znoisy[user, item] > 0;
            y.ObservedValue = new bool[,] { { true, false }, { false, true } };

            //InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            // engine.ShowFactorGraph = true;
            // engine.Compiler.WriteSourceFiles = true;
            Gaussian[,] userTraitsPosterior;
            Gaussian[,] itemTraitsPosterior;

            Console.WriteLine("MatrixMultiply");
            for (int iter = 1; iter < 20; iter++)
            {
                engine.NumberOfIterations = iter;
                userTraitsPosterior = engine.Infer<Gaussian[,]>(userTraits);
                itemTraitsPosterior = engine.Infer<Gaussian[,]>(itemTraits);

                Console.WriteLine("Iteration {0}", iter);
                Console.WriteLine("userTraitMeanPosterior:");
                for (int k1 = 0; k1 < numUsers; k1++)
                {
                    for (int k2 = 0; k2 < numTraits; k2++)
                    {
                        // Console.Write("\t {0},", userTraitsPosterior[k1, k2].GetMean());
                        Console.Write("\t {0},", userTraitsPosterior[k1, k2].GetVariance());
                    }
                    Console.Write("\n");
                }

                Console.WriteLine("iterTraitMeanPosterior:");
                for (int k1 = 0; k1 < numItems; k1++)
                {
                    for (int k2 = 0; k2 < numTraits; k2++)
                    {
                        // Console.Write("\t {0},", itemTraitsPosterior[k2, k1].GetMean());
                        Console.Write("\t {0},", itemTraitsPosterior[k2, k1].GetVariance());
                    }
                    Console.Write("\n");
                }
                //Console.ReadKey();
            }
        }


        public void TestInnerProductArrayRecomender(int numUsers, int numItems, int numTraits, double userPriorMean, double userPriorVariance, double itemPriorMean, double itemPriorVariance)
        {

            double noiseVariance = 100;

            //int numUsers = 2,
            //    numItems = 2,
            //    numTraits = 1,
            //    numObs = 4;
            int numObs = numUsers * numItems;

            //double userPriorMean = 0,
            //       userPriorVariance = 1,
            //       itemPriorMean = 0,
            //       itemPriorVariance = 1;



            // Define ranges
            Range user = new Range(numUsers).Named("user");
            Range item = new Range(numItems).Named("item");
            Range trait = new Range(numTraits).Named("trait");
            Range obs = new Range(numObs).Named("obs");

            // Define latent variables

            var userTraits = Variable.Array(Variable.Array<double>(trait), user).Named("userTraits");
            var itemTraits = Variable.Array(Variable.Array<double>(trait), item).Named("itemTraits");

            VariableArray<double> z = Variable.Array<double>(obs).Named("z");
            VariableArray<double> znoisy = Variable.Array<double>(obs).Named("znoisy");
            VariableArray<bool> y = Variable.Array<bool>(obs).Named("y");

            VariableArray<int> userData = Variable.Array<int>(obs).Named("userData");
            VariableArray<int> itemData = Variable.Array<int>(obs).Named("itemData");

            userTraits[user][trait] = Variable<double>.GaussianFromMeanAndVariance(userPriorMean, userPriorVariance).ForEach(user, trait);
            itemTraits[item][trait] = Variable<double>.GaussianFromMeanAndVariance(itemPriorMean, itemPriorVariance).ForEach(item, trait);

            z[obs] = Variable<double>.Factor(Factor.InnerProduct, userTraits[userData[obs]], itemTraits[itemData[obs]]);
            znoisy[obs] = Variable.GaussianFromMeanAndVariance(z[obs], noiseVariance);
            y[obs] = znoisy[obs] > 0;

            userData.ObservedValue = new int[] { 0, 0, 1, 1 };
            itemData.ObservedValue = new int[] { 0, 1, 0, 1 };
            y.ObservedValue = new bool[] { true, false, false, true };

            //InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            // engine.ShowFactorGraph = true;
            //engine.Compiler.WriteSourceFiles = true;
            Gaussian[][] userTraitsPosterior;
            Gaussian[][] itemTraitsPosterior;


            Console.WriteLine("ArrayInnerProduct: \n");
            for (int iter = 1; iter < 20; iter++)
            {
                engine.NumberOfIterations = iter;
                userTraitsPosterior = engine.Infer<Gaussian[][]>(userTraits);
                itemTraitsPosterior = engine.Infer<Gaussian[][]>(itemTraits);

                Console.WriteLine("Iteration {0}", iter);
                Console.WriteLine("userTraitMeanPosterior:");
                for (int k1 = 0; k1 < numUsers; k1++)
                {
                    for (int k2 = 0; k2 < numTraits; k2++)
                    {
                        // Console.Write("\t {0},", userTraitsPosterior[k1][k2].GetMean());
                        Console.Write("\t {0},", userTraitsPosterior[k1][k2].GetVariance());
                    }
                    Console.Write("\n");
                }

                Console.WriteLine("iterTraitMeanPosterior:");
                for (int k1 = 0; k1 < numItems; k1++)
                {
                    for (int k2 = 0; k2 < numTraits; k2++)
                    {
                        // Console.Write("\t {0},", itemTraitsPosterior[k1][k2].GetMean());
                        Console.Write("\t {0},", itemTraitsPosterior[k1][k2].GetVariance());
                    }
                    Console.Write("\n");
                }

                // Console.ReadKey();
            }
        }


        public void TestInnerProductArray_BPM()
        {
            // Bayes point machine

            // creating the model and doing inference
            double[] incomes = { 63, 16, 28, 55, 22, 20 };
            double[] ages = { 38, 23, 40, 27, 18, 40 };
            bool[] willBuy = { true, false, true, true, false, false };


            Range j = new Range(3).Named("j");
            Range i = new Range(incomes.Length).Named("i");

            // Create x vector, augmented by 1
            double[][] xdata = new double[incomes.Length][];
            for (int k = 0; k < xdata.Length; k++)
                xdata[k] = new double[3] { incomes[k], ages[k], 1 };

            VariableArray<VariableArray<double>, double[][]> x = Variable.Array(Variable.Array<double>(j), i).Named("x");
            x.ObservedValue = Util.ArrayInit(i.SizeAsInt, ii => Util.ArrayInit(j.SizeAsInt, jj => xdata[ii][jj]));

            // Create target y
            VariableArray<bool> y = Variable.Observed(willBuy, i).Named("y");

            // Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3))).Named("w");

            VariableArray<double> w = Variable.Array<double>(j).Named("w");
            w[j] = Variable.GaussianFromMeanAndPrecision(0, 0.1).ForEach(j);

            VariableArray<double> z = Variable.Array<double>(i).Named("z");
            VariableArray<double> znoisy = Variable.Array<double>(i).Named("znoisy");


            // z[j] = Variable<double>.Factor(Factor.VectorMultiplyOp, w, x[i]).ForEach(j);
            double noise = 1;
            z[i] = Variable<double>.Factor(Factor.InnerProduct, w, x[i]);
            znoisy[i] = Variable.GaussianFromMeanAndVariance(z[i], noise);
            y[i] = znoisy[i] > 0;

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            //InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            // engine.ShowFactorGraph = true;
            // engine.Compiler.WriteSourceFiles = true;

            double m, v;
            Gaussian[] wPost;
            for (int iter = 1; iter < 10; iter++)
            {
                engine.NumberOfIterations = iter;
                wPost = engine.Infer<Gaussian[]>(w);
                for (int k = 0; k < 3; k++)
                {
                    wPost[k].GetMeanAndVariance(out m, out v);
                    Console.WriteLine("iter: {0}, mean[{1}] = {2}, \t var[{1}] = {3}", iter, k, m, v);
                }
            }

            // creating testing data and inferring predictions
            //double[] incomesTest = { 58, 18, 22 };
            //double[] agesTest = { 36, 24, 37 };
            //Range itest = new Range(agesTest.Length);

            //double[][] xdataTest = new double[incomesTest.Length][];
            //for (int k = 0; k < xdataTest.Length; k++)
            //    xdataTest[k] = new double[]{incomesTest[k], agesTest[k], 1};


            //VariableArray<VariableArray<double>, double[][]> xTest = Variable.Array(Variable.Array<double>(j), itest).Named("x");
            //xTest.ObservedValue = Util.ArrayInit(itest.SizeAsInt, ii => Util.ArrayInit(j.SizeAsInt, jj => xdata[ii][jj]));

            //VariableArray<bool> ytest = Variable.Array<bool>(itest);

            //// Bayes Point Machine
            //VariableArray<Gaussian> wPred = Variable.Array<Gaussian>(j);
            //wPred.ObservedValue = Util.ArrayInit(j.SizeAsInt, jj => wPost[jj]);

            //VariableArray<double> ztest = Variable.Array<double>(itest);
            //ztest[itest] = Variable<double>.Factor(Factor.VectorMultiply, wPred, x[itest]);
            //ytest[itest] = Variable.GaussianFromMeanAndVariance(ztest[itest], noise) > 0;
            //Console.WriteLine("output=\n" + engine.Infer(ytest));

            //Console.ReadKey();
        }


        public void TestMatrixMultiplyRecomenderCrazy(int numUsers, int numItems, int numTraits, double userPriorMean, double userPriorVariance, double itemPriorMean, double itemPriorVariance)
        {

            double noiseVariance = 100;

            //int numUsers = 2,
            //    numItems = 2,
            //    numTraits = 1;

            //double userPriorMean =0, 
            //       userPriorVariance =1,
            //       itemPriorMean =0, 
            //       itemPriorVariance =1;

            // Define ranges
            Range user = new Range(numUsers).Named("user");
            Range item = new Range(numItems).Named("item");
            Range trait = new Range(numTraits).Named("trait");

            // Define latent variables

            VariableArray<double> userBias = Variable.Array<double>(user).Named("userBias");
            VariableArray<double> itemBias = Variable.Array<double>(item).Named("itemBias");
            VariableArray2D<double> allBias = Variable.Array<double>(user, item).Named("allBias");

            VariableArray2D<double> userTraits = Variable.Array<double>(user, trait).Named("userTraits");
            VariableArray2D<double> itemTraits = Variable.Array<double>(trait, item).Named("itemTraits");
            VariableArray2D<double> z = Variable.Array<double>(user, item).Named("z");
            VariableArray2D<double> znoisy = Variable.Array<double>(user, item).Named("znoisy");
            VariableArray2D<bool> y = Variable.Array<bool>(user, item).Named("y");
 
            userTraits[user, trait] = Variable<double>.GaussianFromMeanAndVariance(userPriorMean, userPriorVariance).ForEach(user, trait);
            itemTraits[trait, item] = Variable<double>.GaussianFromMeanAndVariance(itemPriorMean, itemPriorVariance).ForEach(trait, item);
                        
            userBias[user] = Variable<double>.GaussianFromMeanAndPrecision(0, 1e-2).ForEach(user);
            itemBias[item] = Variable<double>.GaussianFromMeanAndPrecision(0, 1e-2).ForEach(item);

            z = Variable.MatrixMultiply(userTraits, itemTraits);
            allBias[user, item] = userBias[user] + itemBias[item];
            znoisy[user, item] = Variable<double>.GaussianFromMeanAndVariance(z[user, item] + allBias[user, item], noiseVariance);
            y[user, item] = znoisy[user, item] > 0;
            y.ObservedValue = new bool[,] { { true, false }, { false, true } };

            //InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowFactorGraph = true;
            // engine.Compiler.WriteSourceFiles = true;
            Gaussian[,] userTraitsPosterior;
            Gaussian[,] itemTraitsPosterior;
            Gaussian[] userBiasPosterior;
            Gaussian[] itemBiasPosterior;


            Console.WriteLine("MatrixMultiply");
            for (int iter = 1; iter < 20; iter++)
            {
                engine.NumberOfIterations = iter;
                userTraitsPosterior = engine.Infer<Gaussian[,]>(userTraits);
                itemTraitsPosterior = engine.Infer<Gaussian[,]>(itemTraits);
                userBiasPosterior = engine.Infer<Gaussian[]>(userBias);
                itemBiasPosterior = engine.Infer<Gaussian[]>(itemBias);

                Console.WriteLine("Iteration {0}", iter);
                Console.WriteLine("userTraitMeanPosterior:");
                for (int k1 = 0; k1 < numUsers; k1++)
                {
                    for (int k2 = 0; k2 < numTraits; k2++)
                    {
                        // Console.Write("\t {0},", userTraitsPosterior[k1, k2].GetMean());
                        Console.Write("\t {0},", userTraitsPosterior[k1, k2].GetVariance());
                    }
                    Console.Write("\n");
                }

                Console.WriteLine("iterTraitMeanPosterior:");
                for (int k1 = 0; k1 < numItems; k1++)
                {
                    for (int k2 = 0; k2 < numTraits; k2++)
                    {
                        // Console.Write("\t {0},", itemTraitsPosterior[k2, k1].GetMean());
                        Console.Write("\t {0},", itemTraitsPosterior[k2, k1].GetVariance());
                    }
                    Console.Write("\n");
                }
            }
            Console.ReadKey();
        }


        public void TestInnerProductArrayRecomenderCrazy(string data_dir)
        {
            int numTraits = 2,
                numUsers,
                numItems,
                numObs;

            double userTraitsMean = 1,
                   itemTraitsMean = 1;

            double noiseVariance = 1,
                   userTraitsPrecision = 1e-4,
                   itemTraitsPrecision = 1,
                   userWeightsPrecision = 1e-4,
                   itemWeightsPrecision = 1e-4,
                   userBiasPrecision = 1e-4,
                   itemBiasPrecision = 1e-4;
           
            int[] userDataTrain, itemDataTrain;
            bool[] classDataTrain;
            double[] noisyAffinityDataTrain;

            if (data_dir.Length == 0)
            {
                numUsers = 2;
                numItems = 2;
                numObs = 4;
                userDataTrain = new int[] { 0, 0, 1, 1 };
                itemDataTrain = new int[] { 0, 1, 0, 1 };
                classDataTrain = new bool[] { true, true, false, false };
                // double[] noisyAffinityDataTrain;                
            }
            else
            {
                Dictionary<String, Object> r;
                // string data_dir = @"C:\Users\t-botcse\Source\mlp\infernet\Prototypes\SweeplessRecommender\Data\special-case-study\data-check-1.mat";
                r = MatlabReader.Read(data_dir);
                Console.WriteLine("File loaded...");

                Func<string, int> loadInt = key => (int)((Matrix)r[key])[0, 0];
                Func<string, double> loadDouble = key => (double)((Matrix)r[key])[0, 0];
                Func<string, bool> loadBool = key => ((Matrix)r[key])[0, 0] == 1;

                Func<string, int> loadInt2 = key => ((int[])r[key])[0];

                try
                {
                    numUsers = loadInt("numUsers");
                    numItems = loadInt("numItems");
                    numObs = loadInt("numObservations");

                    userDataTrain = (int[])r["generatedUserData"];
                    itemDataTrain = (int[])r["generatedItemData"];
                    classDataTrain = (bool[])r["generatedClassData"];
                    noisyAffinityDataTrain = ((Matrix)r["generatedNoisyAffinityData"]).SourceArray;
                }
                catch (InvalidCastException e)
                {

                    numUsers = loadInt2("numUsers");
                    numItems = loadInt2("numItems");
                    numObs = loadInt2("numObservations");

                    userDataTrain = getArray(r["generatedUserData"]);
                    itemDataTrain = getArray(r["generatedItemData"]);
                    classDataTrain = getArrayBool(r["generatedClassData"]);

                    Console.WriteLine(e);

                    try
                    {
                        noisyAffinityDataTrain = getArrayFloat(r["generatedNoisyAffinityData"]);
                    }
                    catch (Exception ee)
                    {
                        Console.WriteLine(ee);
                    }

                }
                Console.WriteLine("loaded numUsers " + numUsers + " from dataDir " + data_dir);
            }


            //int numObs = numUsers * numItems;

            Range user = new Range(numUsers).Named("user");
            Range item = new Range(numItems).Named("item");
            Range trait = new Range(numTraits).Named("trait");
            Range obs = new Range(numObs).Named("obs");

            // Define ranges
            double[] ZerosTraits = new double[numTraits];
            for (int k = 0; k < numTraits; k++)
                ZerosTraits[k] = 0.0;

            // Define latent variables

            var userTraits = Variable.Array(Variable.Array<double>(trait), user).Named("userTraits");
            var itemTraits = Variable.Array(Variable.Array<double>(trait), item).Named("itemTraits");

            var userTraitsT = Variable.Array(Variable.Array<double>(user), trait).Named("userTraitsT");
            var itemTraitsT = Variable.Array(Variable.Array<double>(item), trait).Named("itemTraitsT");

            var userTraitsSum = Variable.Array<double>(trait).Named("userTraitsSum");
            var itemTraitsSum = Variable.Array<double>(trait).Named("itemTraitsSum");

            VariableArray<double> userBias = Variable.Array<double>(user).Named("userBias");
            VariableArray<double> itemBias = Variable.Array<double>(item).Named("itemBias");
            VariableArray<double> bias = Variable.Array<double>(obs).Named("bias");

            VariableArray<double> userWeights = Variable.Array<double>(trait).Named("userWeights");
            VariableArray<double> itemWeights = Variable.Array<double>(trait).Named("itemWeights");

            VariableArray<double> userReg = Variable.Array<double>(user).Named("userReg");
            VariableArray<double> itemReg = Variable.Array<double>(item).Named("itemReg");
            VariableArray<double> reg = Variable.Array<double>(obs).Named("reg");

            VariableArray<double> z = Variable.Array<double>(obs).Named("z");
            VariableArray<double> znoisy = Variable.Array<double>(obs).Named("znoisy");
            VariableArray<bool> y = Variable.Array<bool>(obs).Named("y");

            VariableArray<int> userData = Variable.Array<int>(obs).Named("userData");
            VariableArray<int> itemData = Variable.Array<int>(obs).Named("itemData");

            userTraits[user][trait] = Variable<double>.GaussianFromMeanAndPrecision(userTraitsMean, userTraitsPrecision).ForEach(user, trait);
            itemTraits[item][trait] = Variable<double>.GaussianFromMeanAndPrecision(itemTraitsMean, itemTraitsPrecision).ForEach(item, trait);

            userTraitsT[trait][user] = Variable<double>.GaussianFromMeanAndPrecision(userTraits[user][trait], 1e4);
            itemTraitsT[trait][item] = Variable<double>.GaussianFromMeanAndPrecision(itemTraits[item][trait], 1e4);

            userTraitsSum[trait] = Variable<double>.GaussianFromMeanAndPrecision(Variable.Sum(userTraitsT[trait]), 1e4);
            itemTraitsSum[trait] = Variable<double>.GaussianFromMeanAndPrecision(Variable.Sum(itemTraitsT[trait]), 1e4);
            userTraitsSum.ObservedValue = ZerosTraits;
            itemTraitsSum.ObservedValue = ZerosTraits;

            userWeights[trait] = Variable<double>.GaussianFromMeanAndPrecision(0, userWeightsPrecision).ForEach(trait);
            itemWeights[trait] = Variable<double>.GaussianFromMeanAndPrecision(0, itemWeightsPrecision).ForEach(trait);

            userReg[user] = Variable<double>.Factor(Factor.InnerProduct, userTraits[user], userWeights);
            itemReg[item] = Variable<double>.Factor(Factor.InnerProduct, itemTraits[item], itemWeights);

            userBias[user] = Variable<double>.GaussianFromMeanAndPrecision(0, userBiasPrecision).ForEach(user);
            itemBias[item] = Variable<double>.GaussianFromMeanAndPrecision(0, itemBiasPrecision).ForEach(item);

            z[obs] = Variable<double>.Factor(Factor.InnerProduct, userTraits[userData[obs]], itemTraits[itemData[obs]]);
            bias[obs] = userBias[userData[obs]] + itemBias[itemData[obs]];
            reg[obs] = userReg[userData[obs]] + itemReg[itemData[obs]];
            Variable<double> ipWeights = Variable<double>.Factor(Factor.InnerProduct, userWeights, itemWeights);
            znoisy[obs] = Variable.GaussianFromMeanAndVariance(z[obs] + reg[obs] + bias[obs] + ipWeights, noiseVariance);
            y[obs] = znoisy[obs] > 0;

            userData.ObservedValue = userDataTrain;
            itemData.ObservedValue = itemDataTrain;
            y.ObservedValue = classDataTrain;

            //InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowFactorGraph = true;
            //engine.Compiler.WriteSourceFiles = true;
            Gaussian[][] userTraitsPosterior;
            Gaussian[][] itemTraitsPosterior;
            Gaussian[][] userTraitsTPosterior;
            Gaussian[][] itemTraitsTPosterior;
            Gaussian[] userBiasPosterior;
            Gaussian[] itemBiasPosterior;
            Gaussian[] biasPosterior;
            Gaussian[] userTraitsSumPosterior;
            Gaussian[] itemTraitsSumPosterior;
            Gaussian[] userWeightsPosterior;
            Gaussian[] itemWeightsPosterior;
            Gaussian[] znoisyPosterior;

            Console.WriteLine("ArrayInnerProduct: \n");
            for (int iter = 1; iter < 20; iter++)
            {
                engine.NumberOfIterations = iter;
                userTraitsPosterior = engine.Infer<Gaussian[][]>(userTraits);
                itemTraitsPosterior = engine.Infer<Gaussian[][]>(itemTraits);
                userTraitsTPosterior = engine.Infer<Gaussian[][]>(userTraitsT);
                itemTraitsTPosterior = engine.Infer<Gaussian[][]>(itemTraitsT);

                userBiasPosterior = engine.Infer<Gaussian[]>(userBias);
                itemBiasPosterior = engine.Infer<Gaussian[]>(itemBias);
                biasPosterior = engine.Infer<Gaussian[]>(bias);
                userTraitsSumPosterior = engine.Infer<Gaussian[]>(userTraitsSum);
                itemTraitsSumPosterior = engine.Infer<Gaussian[]>(itemTraitsSum);
                userWeightsPosterior = engine.Infer<Gaussian[]>(userWeights);
                itemWeightsPosterior = engine.Infer<Gaussian[]>(itemWeights);
                znoisyPosterior = engine.Infer<Gaussian[]>(znoisy);

                Console.WriteLine("Iteration {0}", iter);
                Console.WriteLine("userTraitMeanPosterior:");
                for (int k1 = 0; k1 < numUsers; k1++)
                {
                    for (int k2 = 0; k2 < numTraits; k2++)
                    {
                        Console.Write("\t {0},", userTraitsPosterior[k1][k2].GetMean());
                        // Console.Write("\t {0},", userTraitsPosterior[k1][k2].GetVariance());
                    }
                    Console.Write("\n");
                }

                Console.WriteLine("iterTraitMeanPosterior:");
                for (int k1 = 0; k1 < numItems; k1++)
                {
                    for (int k2 = 0; k2 < numTraits; k2++)
                    {
                        Console.Write("\t {0},", itemTraitsPosterior[k1][k2].GetMean());
                        // Console.Write("\t {0},", itemTraitsPosterior[k1][k2].GetVariance());
                    }
                    Console.Write("\n");
                }
            }
            Console.ReadKey();
        }


        public void TestInnerProductArrayRecomenderCrazyToo(string data_dir)
        {

            int numTraits = 2,
                numUsers,
                numItems,
                numObs;

            double userTraitsMeanMean = 0,
                   itemTraitsMeanMean = 0,
                   userTraitsMeanPrecision = 1e-4,
                   itemTraitsMeanPrecision = 1e-4;

            double noiseVariance = 1,
                   userTraitsPrecision = 1e-4,
                   itemTraitsPrecision = 1,
                   userBiasPrecision = 1e4,
                   itemBiasPrecision = 1e4;

            int[] userDataTrain, itemDataTrain;
            bool[] classDataTrain;
            double[] noisyAffinityDataTrain;

            if (data_dir.Length == 0)
            {
                numUsers = 2;
                numItems = 2;
                numObs = 4;
                userDataTrain = new int[] { 0, 0, 1, 1 };
                itemDataTrain = new int[] { 0, 1, 0, 1 };
                classDataTrain = new bool[] { true, true, false, false };
                // double[] noisyAffinityDataTrain;                
            }
            else
            {
                Dictionary<String, Object> r;
                // string data_dir = @"C:\Users\t-botcse\Source\mlp\infernet\Prototypes\SweeplessRecommender\Data\special-case-study\data-check-1.mat";
                r = MatlabReader.Read(data_dir);
                Console.WriteLine("File loaded...");

                Func<string, int> loadInt = key => (int)((Matrix)r[key])[0, 0];
                Func<string, double> loadDouble = key => (double)((Matrix)r[key])[0, 0];
                Func<string, bool> loadBool = key => ((Matrix)r[key])[0, 0] == 1;

                Func<string, int> loadInt2 = key => ((int[])r[key])[0];

                try
                {
                    numUsers = loadInt("numUsers");
                    numItems = loadInt("numItems");
                    numObs = loadInt("numObservations");

                    userDataTrain = (int[])r["generatedUserData"];
                    itemDataTrain = (int[])r["generatedItemData"];
                    classDataTrain = (bool[])r["generatedClassData"];
                    noisyAffinityDataTrain = ((Matrix)r["generatedNoisyAffinityData"]).SourceArray;
                }
                catch (InvalidCastException e)
                {

                    numUsers = loadInt2("numUsers");
                    numItems = loadInt2("numItems");
                    numObs = loadInt2("numObservations");

                    userDataTrain = getArray(r["generatedUserData"]);
                    itemDataTrain = getArray(r["generatedItemData"]);
                    classDataTrain = getArrayBool(r["generatedClassData"]);

                    Console.WriteLine(e);

                    try
                    {
                        noisyAffinityDataTrain = getArrayFloat(r["generatedNoisyAffinityData"]);
                    }
                    catch (Exception ee)
                    {
                        Console.WriteLine(ee);
                    }

                }
                Console.WriteLine("loaded numUsers " + numUsers + " from dataDir " + data_dir);
            }



            //int numObs = numUsers * numItems;

            Range user = new Range(numUsers).Named("user");
            Range item = new Range(numItems).Named("item");
            Range trait = new Range(numTraits).Named("trait");
            Range obs = new Range(numObs).Named("obs");

            // Define latent variables

            var userTraits = Variable.Array(Variable.Array<double>(trait), user).Named("userTraits");
            var itemTraits = Variable.Array(Variable.Array<double>(trait), item).Named("itemTraits");

            var userTraitsSum = Variable.Array<double>(trait).Named("userTraitsSum");
            var itemTraitsSum = Variable.Array<double>(trait).Named("itemTraitsSum");

            VariableArray<double> userBias = Variable.Array<double>(user).Named("userBias");
            VariableArray<double> itemBias = Variable.Array<double>(item).Named("itemBias");
            VariableArray<double> bias = Variable.Array<double>(obs).Named("bias");

            VariableArray<double> userTraitsMean = Variable.Array<double>(trait).Named("userTraitsMean");
            VariableArray<double> itemTraitsMean = Variable.Array<double>(trait).Named("itemTraitsMean");

            VariableArray<double> z = Variable.Array<double>(obs).Named("z");
            VariableArray<double> znoisy = Variable.Array<double>(obs).Named("znoisy");
            VariableArray<bool> y = Variable.Array<bool>(obs).Named("y");

            VariableArray<int> userData = Variable.Array<int>(obs).Named("userData");
            VariableArray<int> itemData = Variable.Array<int>(obs).Named("itemData");

            userTraitsMean[trait] = Variable<double>.GaussianFromMeanAndPrecision(userTraitsMeanMean, userTraitsMeanPrecision).ForEach(trait);
            itemTraitsMean[trait] = Variable<double>.GaussianFromMeanAndPrecision(itemTraitsMeanMean, itemTraitsMeanPrecision).ForEach(trait);

            userTraits[user][trait] = Variable<double>.GaussianFromMeanAndPrecision(userTraitsMean[trait], userTraitsPrecision).ForEach(user);
            itemTraits[item][trait] = Variable<double>.GaussianFromMeanAndPrecision(itemTraitsMean[trait], itemTraitsPrecision).ForEach(item);

            userBias[user] = Variable<double>.GaussianFromMeanAndPrecision(0, userBiasPrecision).ForEach(user);
            itemBias[item] = Variable<double>.GaussianFromMeanAndPrecision(0, itemBiasPrecision).ForEach(item);

            bias[obs] = userBias[userData[obs]] + itemBias[itemData[obs]];
            z[obs]      = Variable<double>.Factor(Factor.InnerProduct, userTraits[userData[obs]], itemTraits[itemData[obs]]) + bias[obs];
            znoisy[obs] = Variable.GaussianFromMeanAndVariance(z[obs], noiseVariance);
            y[obs] = znoisy[obs] > 0;

            userData.ObservedValue = userDataTrain;
            itemData.ObservedValue = itemDataTrain;
            y.ObservedValue        = classDataTrain;

            //InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowFactorGraph = true;
            //engine.Compiler.WriteSourceFiles = true;
            Gaussian[][] userTraitsPosterior;
            Gaussian[][] itemTraitsPosterior;
            Gaussian[] userBiasPosterior;
            Gaussian[] itemBiasPosterior;
            Gaussian[] biasPosterior;
            Gaussian[] znoisyPosterior;
            Gaussian[] userTraitsMeanPosterior;
            Gaussian[] itemTraitsMeanPosterior;


            Console.WriteLine("ArrayInnerProduct: \n");
            for (int iter = 1; iter < 20; iter++)
            {
                engine.NumberOfIterations = iter;
                userTraitsPosterior = engine.Infer<Gaussian[][]>(userTraits);
                itemTraitsPosterior = engine.Infer<Gaussian[][]>(itemTraits);
                
                userBiasPosterior = engine.Infer<Gaussian[]>(userBias);
                itemBiasPosterior = engine.Infer<Gaussian[]>(itemBias);
                biasPosterior = engine.Infer<Gaussian[]>(bias);
                znoisyPosterior = engine.Infer<Gaussian[]>(znoisy);
                userTraitsMeanPosterior = engine.Infer<Gaussian[]>(userTraitsMean);
                itemTraitsMeanPosterior = engine.Infer<Gaussian[]>(itemTraitsMean);


                Console.WriteLine("Iteration {0}", iter);
                Console.WriteLine("userTraitsMeanPosterior:");
                for (int k1 = 0; k1 < numUsers; k1++)
                {
                    for (int k2 = 0; k2 < numTraits; k2++)
                    {
                        Console.Write("\t {0},", userTraitsPosterior[k1][k2].GetMean());
                        // Console.Write("\t {0},", userTraitsPosterior[k1][k2].GetVariance());
                    }
                    Console.Write("\n");
                }

                Console.WriteLine("iterTraitsMeanPosterior:");
                for (int k1 = 0; k1 < numItems; k1++)
                {
                    for (int k2 = 0; k2 < numTraits; k2++)
                    {
                        Console.Write("\t {0},", itemTraitsPosterior[k1][k2].GetMean());
                        // Console.Write("\t {0},", itemTraitsPosterior[k1][k2].GetVariance());
                    }
                    Console.Write("\n");
                }
            }
            Console.ReadKey();
        }


        public void TestInnerProductArrayRecomenderCrazier(string data_dir)
        {
            int numTraits = 2,
                numUsers,
                numItems,
                numObs;

            double userTraitsMean = 0,
                   itemTraitsMean = 0;

            double noisePrecision = 100,
                   userTraitsPrecision = 1e-4,
                   itemTraitsPrecision = 1,
                   userBiasPrecision = 1e-4,
                   itemBiasPrecision = 1e-4;

            int[] userDataTrain, itemDataTrain;
            bool[] classDataTrain;
            double[] noisyAffinityDataTrain;

            if (data_dir.Length == 0)
            {
                numUsers = 3;
                numItems = 3;
                numObs = 9;
                userDataTrain = new int[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
                itemDataTrain = new int[] { 0, 1, 2, 0, 1, 2, 0, 1, 2 };
                classDataTrain = new bool[] { true, false, false, false, true, false, false, false, true };
                // double[] noisyAffinityDataTrain;                
            }
            else
            {
                Dictionary<String, Object> r;
                // string data_dir = @"C:\Users\t-botcse\Source\mlp\infernet\Prototypes\SweeplessRecommender\Data\special-case-study\data-check-1.mat";
                r = MatlabReader.Read(data_dir);
                Console.WriteLine("File loaded...");

                Func<string, int> loadInt = key => (int)((Matrix)r[key])[0, 0];
                Func<string, double> loadDouble = key => (double)((Matrix)r[key])[0, 0];
                Func<string, bool> loadBool = key => ((Matrix)r[key])[0, 0] == 1;

                Func<string, int> loadInt2 = key => ((int[])r[key])[0];

                try
                {
                    numUsers = loadInt("numUsers");
                    numItems = loadInt("numItems");
                    numObs = loadInt("numObservations");

                    userDataTrain = (int[])r["generatedUserData"];
                    itemDataTrain = (int[])r["generatedItemData"];
                    classDataTrain = (bool[])r["generatedClassData"];
                    noisyAffinityDataTrain = ((Matrix)r["generatedNoisyAffinityData"]).SourceArray;
                }
                catch (InvalidCastException e)
                {

                    numUsers = loadInt2("numUsers");
                    numItems = loadInt2("numItems");
                    numObs = loadInt2("numObservations");

                    userDataTrain = getArray(r["generatedUserData"]);
                    itemDataTrain = getArray(r["generatedItemData"]);
                    classDataTrain = getArrayBool(r["generatedClassData"]);

                    Console.WriteLine(e);

                    try
                    {
                        noisyAffinityDataTrain = getArrayFloat(r["generatedNoisyAffinityData"]);
                    }
                    catch (Exception ee)
                    {
                        Console.WriteLine(ee);
                    }

                }
                Console.WriteLine("loaded numUsers " + numUsers + " from dataDir " + data_dir);
            }


            //int numObs = numUsers * numItems;

            Range user = new Range(numUsers).Named("user");
            Range item = new Range(numItems).Named("item");
            Range trait = new Range(numTraits).Named("trait");
            Range obs = new Range(numObs).Named("obs");

            // Define ranges
            double[] ZerosTraits = new double[numTraits];
            for (int k = 0; k < numTraits; k++)
                ZerosTraits[k] = 0.0;

            // Define latent variables

            var userTraits = Variable.Array(Variable.Array<double>(trait), user).Named("userTraits");
            var itemTraits = Variable.Array(Variable.Array<double>(trait), item).Named("itemTraits");

            var userTraitsT = Variable.Array(Variable.Array<double>(user), trait).Named("userTraitsT");
            var itemTraitsT = Variable.Array(Variable.Array<double>(item), trait).Named("itemTraitsT");

            var userTraitsBiasCC = Variable.Array<double>(trait).Named("userTraitsSum");
            var itemTraitsBiasCC = Variable.Array<double>(trait).Named("itemTraitsSum");

            VariableArray<double> userBias = Variable.Array<double>(user).Named("userBias");
            VariableArray<double> itemBias = Variable.Array<double>(item).Named("itemBias");
            VariableArray<double> bias = Variable.Array<double>(obs).Named("bias");

            VariableArray<double> z = Variable.Array<double>(obs).Named("z");
            VariableArray<double> znoisy = Variable.Array<double>(obs).Named("znoisy");
            VariableArray<bool> y = Variable.Array<bool>(obs).Named("y");

            VariableArray<int> userData = Variable.Array<int>(obs).Named("userData");
            VariableArray<int> itemData = Variable.Array<int>(obs).Named("itemData");

            userTraits[user][trait] = Variable<double>.GaussianFromMeanAndPrecision(userTraitsMean, userTraitsPrecision).ForEach(user, trait);
            itemTraits[item][trait] = Variable<double>.GaussianFromMeanAndPrecision(itemTraitsMean, itemTraitsPrecision).ForEach(item, trait);

            userTraitsT[trait][user] = Variable<double>.GaussianFromMeanAndPrecision(userTraits[user][trait], 1e4);
            itemTraitsT[trait][item] = Variable<double>.GaussianFromMeanAndPrecision(itemTraits[item][trait], 1e4);

            userBias[user] = Variable<double>.GaussianFromMeanAndPrecision(0, userBiasPrecision).ForEach(user);
            itemBias[item] = Variable<double>.GaussianFromMeanAndPrecision(0, itemBiasPrecision).ForEach(item);

            userTraitsBiasCC[trait] = Variable<double>.GaussianFromMeanAndPrecision(Variable.InnerProduct(userTraitsT[trait], userBias), 1e4);
            itemTraitsBiasCC[trait] = Variable<double>.GaussianFromMeanAndPrecision(Variable.InnerProduct(itemTraitsT[trait], itemBias), 1e4);
            userTraitsBiasCC.ObservedValue = ZerosTraits;
            itemTraitsBiasCC.ObservedValue = ZerosTraits;

            z[obs] = Variable<double>.Factor(Factor.InnerProduct, userTraits[userData[obs]], itemTraits[itemData[obs]]);
            bias[obs] = userBias[userData[obs]] + itemBias[itemData[obs]];
            znoisy[obs] = Variable.GaussianFromMeanAndPrecision(z[obs] + bias[obs], noisePrecision);
            y[obs] = znoisy[obs] > 0;

            userData.ObservedValue = userDataTrain;
            itemData.ObservedValue = itemDataTrain;
            y.ObservedValue = classDataTrain;

            //InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowFactorGraph = true;
            //engine.Compiler.WriteSourceFiles = true;
            Gaussian[][] userTraitsPosterior;
            Gaussian[][] itemTraitsPosterior;
            Gaussian[][] userTraitsTPosterior;
            Gaussian[][] itemTraitsTPosterior;
            Gaussian[] userBiasPosterior;
            Gaussian[] itemBiasPosterior;
            Gaussian[] biasPosterior;
            Gaussian[] userTraitsBiasCCPosterior;
            Gaussian[] itemTraitsBiasCCPosterior;

            Gaussian[] znoisyPosterior;

            Console.WriteLine("ArrayInnerProduct: \n");
            for (int iter = 1; iter < 20; iter++)
            {
                engine.NumberOfIterations = iter;
                userTraitsPosterior = engine.Infer<Gaussian[][]>(userTraits);
                itemTraitsPosterior = engine.Infer<Gaussian[][]>(itemTraits);
                userTraitsTPosterior = engine.Infer<Gaussian[][]>(userTraitsT);
                itemTraitsTPosterior = engine.Infer<Gaussian[][]>(itemTraitsT);

                userBiasPosterior = engine.Infer<Gaussian[]>(userBias);
                itemBiasPosterior = engine.Infer<Gaussian[]>(itemBias);
                biasPosterior = engine.Infer<Gaussian[]>(bias);
                userTraitsBiasCCPosterior = engine.Infer<Gaussian[]>(userTraitsBiasCC);
                itemTraitsBiasCCPosterior = engine.Infer<Gaussian[]>(itemTraitsBiasCC);
                znoisyPosterior = engine.Infer<Gaussian[]>(znoisy);

                Console.WriteLine("Iteration {0}", iter);
                Console.WriteLine("userTraitMeanPosterior:");
                for (int k1 = 0; k1 < numUsers; k1++)
                {
                    for (int k2 = 0; k2 < numTraits; k2++)
                    {
                        Console.Write("\t {0},", userTraitsPosterior[k1][k2].GetMean());
                        // Console.Write("\t {0},", userTraitsPosterior[k1][k2].GetVariance());
                    }
                    Console.Write("\n");
                }

                Console.WriteLine("iterTraitMeanPosterior:");
                for (int k1 = 0; k1 < numItems; k1++)
                {
                    for (int k2 = 0; k2 < numTraits; k2++)
                    {
                        Console.Write("\t {0},", itemTraitsPosterior[k1][k2].GetMean());
                        // Console.Write("\t {0},", itemTraitsPosterior[k1][k2].GetVariance());
                    }
                    Console.Write("\n");
                }
            }
            Console.ReadKey();
        }



        double[][] getJagged(Object o)
        {
            Matrix matrix = (Matrix)o;

            int n = matrix.Rows;
            int m = matrix.Cols;

            double[][] jaggedArray = Util.ArrayInit(n, i => Util.ArrayInit(m, j => matrix[i, j]));

            return jaggedArray;
        }
        int[] getArray(Object o)
        {
            double[,] multiArray = ((Matrix)o).ToArray();

            int n = multiArray.GetLength(1);

            int[] jaggedArray = Util.ArrayInit(n, i => (int)multiArray[0, i]);

            return jaggedArray;
        }

        double[] getArrayFloat(Object o)
        {
            double[,] multiArray = ((Matrix)o).ToArray();

            int n = multiArray.GetLength(1);

            double[] jaggedArray = Util.ArrayInit(n, i => (double)multiArray[0, i]);

            return jaggedArray;
        }

        bool[] getArrayBool(Object o)
        {
            double[,] multiArray = ((Matrix)o).ToArray();

            int n = multiArray.GetLength(1);

            bool[] jaggedArray = Util.ArrayInit(n, i => multiArray[0, i] > 0);

            return jaggedArray;
        }
    }
}