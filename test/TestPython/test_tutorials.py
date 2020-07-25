# For help using pythonnet:
# https://pythonnet.github.io/
# To stop the debugger in .NET code, follow the instructions at:
# https://mail.python.org/archives/list/pythonnet@python.org/message/AZ4BKHN72PIRAG32ERFY3XS52NVNCL47/
import clr
from System import Boolean, Double, Int32, Array

folder = "../Tests/bin/DebugFull/net461/"
clr.AddReference(folder+"Microsoft.ML.Probabilistic")
clr.AddReference(folder+"Microsoft.ML.Probabilistic.Compiler")
from Microsoft.ML.Probabilistic import *
from Microsoft.ML.Probabilistic.Distributions import VectorGaussian, Beta, Bernoulli, Discrete
from Microsoft.ML.Probabilistic.Math import Rand, Vector, DenseVector, PositiveDefiniteMatrix
from Microsoft.ML.Probabilistic.Models import Variable, InferenceEngine, Range, VariableArray
from Microsoft.ML.Probabilistic.Compiler.Reflection import Invoker

def TwoCoins():
    firstCoin = Variable.Bernoulli(0.5)
    secondCoin = Variable.Bernoulli(0.5)
    bothHeads = firstCoin.op_BitwiseAnd(firstCoin, secondCoin)
    engine = InferenceEngine()
    print("Probability both coins are heads: %s" % engine.Infer(bothHeads))
    bothHeads.ObservedValue = False
    print("Probability distribution over firstCoin: %s" % engine.Infer(firstCoin))

def TruncatedGaussianEfficient():
    threshold = Variable.New[Double]()
    x = Variable.GaussianFromMeanAndVariance(0.0, 1.0)
    Variable.ConstrainTrue(x.op_GreaterThan(x,threshold))
    engine = InferenceEngine()
    for thresh in [i*0.1 for i in range(11)]:
        threshold.ObservedValue = thresh
        print("Dist over x given thresh of %s = %s" % (thresh, engine.Infer(x)))

def LearningAGaussian():
    # Restart the infer.NET random number generator
    Rand.Restart(12347)
    data = [Rand.Normal(0.0, 1.0) for i in range(100)]

    mean = Variable.GaussianFromMeanAndVariance(0.0, 1.0).Named("mean")
    precision = Variable.GammaFromShapeAndScale(1.0, 1.0).Named("precision")

    for i in range(len(data)):
        x = Variable.GaussianFromMeanAndPrecision(mean, precision)
        x.ObservedValue = data[i]

    engine = InferenceEngine()
    print("mean=%s" % engine.Infer(mean))
    print("precision=%s" % engine.Infer(precision))

def LearningAGaussianWithRanges():
    # Restart the infer.NET random number generator
    Rand.Restart(12347)
    data = [Rand.Normal(0.0, 1.0) for i in range(100)]

    mean = Variable.GaussianFromMeanAndVariance(0.0, 1.0).Named("mean")
    precision = Variable.GammaFromShapeAndScale(1.0, 1.0).Named("precision")

    data_range = Range(len(data)).Named("n")
    x = Variable.Array[Double](data_range)
    v = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(data_range)
    x.set_Item(data_range, v)
    x.ObservedValue = data

    engine = InferenceEngine()
    print("mean=%s" % engine.Infer(mean))
    print("precision=%s" %engine.Infer(precision))

def BayesPointMachine(incomes, ages, w, y):
    j = y.Range
    xData = [Vector.FromArray(income, age, 1) for income, age in zip(incomes, ages)]
    x = VariableObserved(xData, j)
    # The following does not work in pythonnet:
    #x = Variable.Observed[Vector](Array[Vector](xData))

    noise = 0.1
    ip = Variable.InnerProduct(w, x.get_Item(j))
    v = Variable.GaussianFromMeanAndVariance(ip, noise)
    v = v.op_GreaterThan(v, 0.0)
    y.set_Item(j, v)

def BayesPointMachineExample():
    incomes = [63, 16, 28, 55, 22, 20]
    ages = [38, 23, 40, 27, 18, 40]
    will_buy = [True, False, True, True, False, False]

    # The following does not work in pythonnet:
    #y = Variable.Observed[bool](will_buy)
    #y = Variable.Observed[bool].Overloads[Array[bool]](will_buy)
    y = VariableObserved(will_buy)

    eye = PositiveDefiniteMatrix.Identity(3)
    m = Vector.Zero(3)
    vg = VectorGaussian(m, eye)
    w = Variable.Random[Vector](vg)
    BayesPointMachine(incomes, ages, w, y)

    engine = InferenceEngine()
    wPosterior = engine.Infer(w)
    print("Dist over w=\n%s" % wPosterior)

    incomesTest = [ 58, 18, 22 ]
    agesTest = [ 36, 24, 37 ]
    ytest = Variable.Array[bool](Range(len(agesTest)))
    BayesPointMachine(incomesTest, agesTest, Variable.Random[Vector](wPosterior), ytest)
    print("output=\n%s" % engine.Infer(ytest));

def ClinicalTrial():
    # Data from clinical trial
    control_group_data = [False, False, True, False, False]
    control_group = VariableObserved(control_group_data)

    treated_group_data = [True, False, True, True, True]
    treated_group = VariableObserved(treated_group_data)

    i = control_group.Range
    j = treated_group.Range

    # Prior on being effective treatment
    is_effective = Variable.Bernoulli(0.5)

    # if block
    if_var = Variable.If(is_effective)
    # Model if treatment is effective
    probIfControl = Variable.Beta(1.0, 1.0)
    t = Variable.Bernoulli(probIfControl).ForEach(i)
    control_group.set_Item(i, t)

    probIfTreated = Variable.Beta(1.0, 1.0)
    t = Variable.Bernoulli(probIfTreated).ForEach(j)
    treated_group.set_Item(j, t)
    if_var.Dispose()

    # A bit of background
    if_var = Variable.IfNot(is_effective)
    # Model if treatment is not effective
    prob_all = Variable.Beta(1.0, 1.0)
    control_group.set_Item(i, Variable.Bernoulli(prob_all).ForEach(i))
    treated_group.set_Item(j, Variable.Bernoulli(prob_all).ForEach(j))
    if_var.Dispose()

    # Clinical accuracy
    engine = InferenceEngine()
    print("Probability treatment has an effect = %s" % engine.Infer(is_effective))
    print("Probability of good outcome if given treatment = %s" % engine.Infer[Beta](probIfTreated).GetMean())
    print("Probability of good outcome if control = %s" % engine.Infer[Beta](probIfControl).GetMean())

def MixtureOfGaussians():
    # Define a range for the number of mixture components
    k = Range(2)

    # Mixture component means
    means = Variable.Array[Vector](k)
    means_k = Variable.VectorGaussianFromMeanAndPrecision(Vector.FromArray(0.0, 0.0), PositiveDefiniteMatrix.IdentityScaledBy(2, 0.01)).ForEach(k)
    means.set_Item(k, means_k)

    # Mixture component precisions
    precs = Variable.Array[PositiveDefiniteMatrix](k).Named("precs")
    precs_k = Variable.WishartFromShapeAndScale(100.0, PositiveDefiniteMatrix.IdentityScaledBy(2, 0.01)).ForEach(k)
    precs.set_Item(k, precs_k)
            
    # Mixture weights 
    weights = Variable.Dirichlet(k, [ 1.0, 1.0 ]).Named("weights")

    # Create a variable array which will hold the data
    n = Range(300).Named("n")
    data = Variable.Array[Vector](n).Named("x")

    # Create latent indicator variable for each data point
    z = Variable.Array[Int32](n).Named("z")

    # The mixture of Gaussians model
    forEachBlock = Variable.ForEach(n)
    z.set_Item(n, Variable.Discrete(weights))
    switchBlock = Variable.Switch(z.get_Item(n))
    data.set_Item(n, Variable.VectorGaussianFromMeanAndPrecision(means.get_Item(z.get_Item(n)), precs.get_Item(z.get_Item(n))))
    switchBlock.CloseBlock()
    forEachBlock.CloseBlock()

    # Attach some generated data
    data.ObservedValue = GenerateData(n.SizeAsInt)

    # Initialise messages randomly to break symmetry
    zInit = Variable.Array[Discrete](n).Named("zInit")
    zInit.ObservedValue = [Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt) for i in range(n.SizeAsInt)]
    # The following does not work in pythonnet:
    #z.get_Item(n).InitialiseTo[Discrete].Overloads[Variable[Discrete]](zInit.get_Item(n))
    InitialiseTo(z.get_Item(n), zInit.get_Item(n))

    # The inference
    engine = InferenceEngine();
    print("Dist over pi=%s" % engine.Infer(weights))
    print("Dist over means=\n%s" % engine.Infer(means))
    print("Dist over precs=\n%s" % engine.Infer(precs))

def GenerateData(nData):
    trueM1 = Vector.FromArray(2.0, 3.0)
    trueM2 = Vector.FromArray(7.0, 5.0)
    trueP1 = PositiveDefiniteMatrix(ToClr([ [ 3.0, 0.2 ], [ 0.2, 2.0 ] ]))
    trueP2 = PositiveDefiniteMatrix(ToClr([ [ 2.0, 0.4 ], [ 0.4, 4.0 ] ]))
    trueVG1 = VectorGaussian.FromMeanAndPrecision(trueM1, trueP1)
    trueVG2 = VectorGaussian.FromMeanAndPrecision(trueM2, trueP2)
    truePi = 0.6
    trueB = Bernoulli(truePi)

    # Restart the infer.NET random number generator
    Rand.Restart(12347)
    return [trueVG1.Sample() if trueB.Sample() else trueVG2.Sample() for j in range(nData)]

def VariableObserved(list, range=None):
    t = type(list[0])
    array = Array[t](list)
    if range == None:
        args = [array]
    else:
        args = [array, range]
    return Invoker.InvokeStatic(Variable, "Observed", args)

def InitialiseTo(variable, distribution):
    Invoker.InvokeInstance("InitialiseTo", variable, distribution)

def ToClr(matrix):
    # See https://github.com/pythonnet/pythonnet/blob/ec424bb0a8f0217c496898395880cf9b99d073e6/src/tests/test_array.py#L1112
    nr = len(matrix)
    nc = len(matrix[0]) if nr > 0 else 0
    t = type(matrix[0][0])
    items = Array.CreateInstance(t, nr, nc)
    for i in range(nr):
        for j in range(nc):
            items.SetValue(t(matrix[i][j]), (i, j))
    return items

def __list_methods(obj):
    for m in dir(obj):
        try:
            print(m)
        except:
            pass

def __list_method_overloads(method):
    for m in f'{method.__overloads__}'.split('\n'):
        print(' '.join(m.split(' ')[1:]))

# Using pytest in Visual Studio:
# https://devblogs.microsoft.com/python/whats-new-for-python-in-visual-studio-16-3-preview-2/
# pytest searches for tests in files named test_*.py or *_test.py
# pytest collects functions with names prefixed by "test", either outside a class or in a class with name prefixed by "Test"

def testTutorials():
    TwoCoins()
    TruncatedGaussianEfficient()
    LearningAGaussian()
    LearningAGaussianWithRanges()
    BayesPointMachineExample()
    ClinicalTrial()
    MixtureOfGaussians()

if __name__ == '__main__':
    testTutorials()