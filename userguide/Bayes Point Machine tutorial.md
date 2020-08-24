---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## Tutorial 4: Bayes Point Machine

This tutorial demonstrates how to use Infer.NET in a standard supervised learning setting. We will define a classifier, train it using a training set and then apply it to a test set. The classifier in question is the [Bayes Point Machine](http://jmlr.org/papers/v1/herbrich01a.html) (R. Herbrich, T. Graepel, and C. Campbell, **Bayes Point Machines**, JMLR, 2001) trained via [Expectation Propagation](http://research.microsoft.com/~minka/papers/ep/) (T. Minka, UAI, 2001).

You can run the code in this tutorial either using the [Examples Browser](The examples browser.md) or by opening the Tutorials solution in Visual Studio and uncommenting the lines to execute [BayesPointMachineExample.cs](https://github.com/dotnet/infer/blob/master/src/Tutorials/BayesPointMachineExample.cs).  Code is also available in [F#](https://github.com/dotnet/infer/blob/master/test/TestFSharp/BayesPointMachine.fs) and [Python](https://github.com/dotnet/infer/blob/master/test/TestPython/test_tutorials.py).

### Will they, won't they?

The problem that we will look at is trying to predict whether someone will buy a particular product, based on some demographic information about them. In particular, for each person, we know their **age** in years and **income** in thousands of pounds. For the training set, we also know whether or not they bought the product. Here is the training data:

```csharp
double[] incomes = { 63, 16, 28, 55, 22, 20 };  
double[] ages = { 38, 23, 40, 27, 18, 40 };  
bool[] willBuy = { true, false, true, true, false, false };
```

We'll use the data to train our Bayes Point Machine. To do this, we need to create two arrays of observations, one called **x** consisting of vectors of the input features (augmented by appending a 1) and one called **y** which just wraps **willBuy**.

```csharp
// Create x vector, augmented by 1
Vector[] xdata = new  Vector[incomes.Length];  
for (int i = 0; i < xdata.Length; i++)  
  xdata[i] = Vector.FromArray(incomes[i], ages[i], 1);  
VariableArray<Vector> x = Variable.Observed(xdata);  

// Create target y  
VariableArray<bool> y = Variable.Observed(willBuy, x.Range);
```

### Defining and training the Bayes Point Machine

A Bayes Point Machine (BPM) classifies an input vector **x** by:

*   finding the inner product of **x** with a weight vector **w** 
*   giving the output **y** as true if the inner product **w.x** (plus some Gaussian noise) is positive and false otherwise. 

When learning a BPM, we aim to learn a posterior distribution over the weight **w**, given some prior and the training data. The uncertainty in **w** will lead to appropriate uncertainty in the predictions given by the BPM on test data. The noise is essential as it allows for the case where there is no setting of **w** that perfectly classifies the training data i.e. where the training data is not linearly separable into the two classes.

We can define a Bayes Point Machine in Infer.NET as follows,

```csharp
Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3)));  
Range j = y.Range;  
double noise = 0.1;
y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]),noise)>0;
```

The first line creates a random variable for **w** with a broad prior. We then create a range **j** which is the size of the data. The last line constrains the target **y\[j\]** to be true if the inner product **w.x** plus noise is positive or false otherwise. Note: here we have picked the noise variance to be 0.1 which sets the 'slack' in the classifier - you should tune this value when using a BPM for your own problem.

We can now train our Bayes Point Machine by using expectation propagation to infer the posterior over **w**,

```csharp
InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());  
VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);  
Console.WriteLine("Dist over w=\n"+wPosterior);
```

When this code is executed, the result is

```
Dist over w =  
VectorGaussian(0.06323 -0.006453 -1.408,  0.001157   -0.0004818  -0.01197 )  
                                         -0.0004818   0.0006099  -0.006733  
                                         -0.01197    -0.006733    0.5215
```

This is the multivariate Gaussian approximation to the posterior over our weight vector. Next, we will look at how to use it to classifiy some test data.

### Testing the Bayes Point Machine

We've done all this work so that we can classify new, test examples. In fact, with a minor modification, we can re-use the existing code to do this. We put the modelling code in a method called **BayesPointMachine()** and pass in the variables **w** and **y** along with the data. The new method looks like this:

```csharp
public void BayesPointMachine (double[] incomes, double[] ages, Variable<Vector> w, VariableArray<bool> y)
{ // Create x vector, augmented by 1 
  Range j = y.Range; Vector[] xdata = new Vector[incomes.Length]; 
  for (int i = 0; i < xdata.Length; i++)  
    xdata[i] = Vector.FromArray(incomes[i], ages[i], 1); 
  VariableArray<Vector> x = Variable.Observed(xdata,j); // Bayes Point Machine double noise = 0.1;  
  y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]),noise)>0;  
}
```

This method can be used for training by passing in **w** and **y** defined above. We can also use it to classify some test data with the following:

```csharp
double[] incomesTest = { 58, 18, 22 };  
double[] agesTest = { 36, 24, 37 };  
VariableArray<bool> ytest = Variable.Array<bool>(new Range(agesTest.Length));  
BayesPointMachine(incomesTest, agesTest, Variable.Random(wPosterior), ytest);  
Console.WriteLine("output=\n" + engine.Infer(ytest));
```

where **wposterior** is the posterior distribution over **w** that we learned during training. This code prints out

```
classifier output=  
[0] Bernoulli(0.9555)  
[1] Bernoulli(0.1565)  
[2] Bernoulli(0.287)
```

so the BPM classifier has predicted that the first person will almost definitely buy and the other two most likely won't.

Please see the [Bayes Point Machine Learner](Learners/Bayes Point Machine classifiers.md) for a comprehensive implementation of a BPM.

I predict you're ready to move on to the [next tutorial](Clinical trial tutorial.md).
