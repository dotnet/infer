# For help using pythonnet:
# https://pythonnet.github.io/
# To stop the debugger in .NET code, follow the instructions at:
# https://mail.python.org/archives/list/pythonnet@python.org/message/AZ4BKHN72PIRAG32ERFY3XS52NVNCL47/
import clr
from System import Boolean, Double, Int32, Array, Char
from System.Collections import IList
from System import Console, ConsoleColor

import sys
import os
# Path to references depends on the Solution Configuration
# by default assumed DebugFull configuration
sys.path.append(os.path.join(os.path.dirname(__file__), r'../Tests/bin/DebugFull/net472/'))
clr.AddReference("Microsoft.ML.Probabilistic")
clr.AddReference("Microsoft.ML.Probabilistic.Compiler")
from Microsoft.ML.Probabilistic import *
from Microsoft.ML.Probabilistic.Distributions import VectorGaussian, Beta, Bernoulli, Discrete, DiscreteChar, Dirichlet
from Microsoft.ML.Probabilistic.Math import Rand, Vector, DenseVector, PositiveDefiniteMatrix, PiecewiseVector
from Microsoft.ML.Probabilistic.Models import Variable, InferenceEngine, Range, VariableArray
from Microsoft.ML.Probabilistic.Compiler.Reflection import Invoker
from Microsoft.ML.Probabilistic.Factors.Attributes import QualityBand
from Microsoft.ML.Probabilistic.Utilities import Util


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

def HelloStrings():
    str1 = Variable.StringUniform()
    str2 = Variable.StringUniform()
    # TODO: text = str1 + " " + str2
    text = str1.op_Addition(str1, " ")
    text = text.op_Addition(text, str2)
    text.ObservedValue = "hello uncertain world"
    engine = InferenceEngine()
    print(f"str1: {engine.Infer(str1)}")
    print(f"str2: {engine.Infer(str2)}")
    dist_of_str1 = engine.Infer(str1)
    for s in ["hello", "hello uncertain", "hello uncertain world"]:
        print(f"P(str1 = '{s}') = {dist_of_str1.GetProb(s)}")

def StringFormat():
    # Infer argument
    name = Variable.StringCapitalized()
    text = VariableStringFormat("My name is {0}.", name)
    text.ObservedValue = "My name is John."
    engine = InferenceEngine()
    engine.Compiler.RecommendedQuality = QualityBand.Experimental
    print(f"name is '{engine.Infer(name)}'")

    # Infer template
    name = Variable.StringCapitalized()
    # The following does not work in pythonnet:
    #template = Variable.StringUniform() + Variable.CharNonWord()
    template = Variable.StringUniform().op_Addition(Variable.StringUniform(), Variable.CharNonWord())
    template = template.op_Addition(template, "{0}")
    template = template.op_Addition(template, Variable.CharNonWord())
    template = template.op_Addition(template, Variable.StringUniform())
    text = VariableStringFormat(template, name)

    text.ObservedValue = "Hello, mate! I'm Dave."
    print(f"name is '{engine.Infer(name)}'")
    print(f"template is '{engine.Infer(template)}'")

    # With a slightly different observation.
    text.ObservedValue = "Hi! My name is John."
    print(f"name is '{engine.Infer(name)}'")
    print(f"template is '{engine.Infer(template)}'")

    # Provide more data to reduce ambiguity.
    name2 = Variable.StringCapitalized()
    text2 = VariableStringFormat(template, name2)
    text2.ObservedValue = "Hi! My name is Tom."                
    print(f"name is '{engine.Infer(name)}'")
    print(f"name2 is '{engine.Infer(name2)}'")
    print(f"template is '{engine.Infer(template)}'")

    # Generate text with the learned template.
    text3 = VariableStringFormat(template, "Boris")
    print(f"text3 is '{engine.Infer(text3)}'")

def MotifFinder():
    Rand.Restart(1337)

    SequenceCount = 50
    SequenceLength = 25
    MotifPresenceProbability = 0.8

    trueMotifNucleobaseDist = [
        NucleobaseDist(a=0.8, c=0.1, g=0.05, t=0.05),
        NucleobaseDist(a=0.0, c=0.9, g=0.05, t=0.05),
        NucleobaseDist(a=0.0, c=0.0, g=0.5, t=0.5),
        NucleobaseDist(a=0.25, c=0.25, g=0.25, t=0.25),
        NucleobaseDist(a=0.1, c=0.1, g=0.1, t=0.7),
        NucleobaseDist(a=0.0, c=0.0, g=0.9, t=0.1),
        NucleobaseDist(a=0.9, c=0.05, g=0.0, t=0.05),
        NucleobaseDist(a=0.5, c=0.5, g=0.0, t=0.0),
    ]

    backgroundNucleobaseDist = NucleobaseDist(a=0.25, c=0.25, g=0.25, t=0.25)

    sequenceData, motifPositionData = SampleMotifData(SequenceCount, SequenceLength, 
                                                      MotifPresenceProbability, 
                                                      trueMotifNucleobaseDist, 
                                                      backgroundNucleobaseDist)

    assert(sequenceData[0] == "CTACTTCGAATTTACCCCTATATTT")
    # should be CTACTTCGAATTTACCCCTATATTT
    assert(len(sequenceData) == 50) 
    assert(motifPositionData[:10] ==[2, 15, -1, 0, 14, 5, -1, 5, 1, 9])
    assert(len(motifPositionData) == 50)
    # Char.MaxValue is a string '\uffff', so we convert the hex to decimal.
    motif_nucleobase_pseudo_counts = PiecewiseVector.Constant(int('ffff', 16) + 1, 1e-6)
    # Cannot call managed PiecewiseVector object's indexer with ['A'], i.e. cannot do
    # motif_nucleobase_pseudo_counts['A'] = 2.0
    motif_nucleobase_pseudo_counts[ord('A')] = 2.0
    motif_nucleobase_pseudo_counts[ord('C')] = 2.0
    motif_nucleobase_pseudo_counts[ord('G')] = 2.0
    motif_nucleobase_pseudo_counts[ord('T')] = 2.0
   
    motifLength = len(trueMotifNucleobaseDist)  # Assume we know the true motif length.
    motifCharsRange = Range(motifLength)
    motifNucleobaseProbs = Variable.Array[Vector](motifCharsRange)
    # Cannot do motifNucleobaseProbs[motifCharsRange] = Variable.Dirichlet...
    motifNucleobaseProbs.set_Item(motifCharsRange, Variable.Dirichlet(motif_nucleobase_pseudo_counts).ForEach(motifCharsRange))
    sequenceRange = Range(SequenceCount)
    sequences = Variable.Array[str](sequenceRange)

    motifPositions = Variable.Array[int](sequenceRange)
    motifPositions.set_Item(sequenceRange, Variable.DiscreteUniform(SequenceLength - motifLength + 1).ForEach(sequenceRange))

    motifPresence = Variable.Array[bool](sequenceRange)
    motifPresence.set_Item(sequenceRange, Variable.Bernoulli(MotifPresenceProbability).ForEach(sequenceRange))

    forEachBlock = Variable.ForEach(sequenceRange)
    ifVar = Variable.If(motifPresence.get_Item(sequenceRange))

    motifChars = Variable.Array[Char](motifCharsRange)
    motifChars.set_Item(motifCharsRange, Variable.Char(motifNucleobaseProbs.get_Item(motifCharsRange)))
    motif = Variable.StringFromArray(motifChars)
    motifPos = motifPositions.get_Item(sequenceRange)

    backgroundLengthRight = motifPos.op_Subtraction(SequenceLength - motifLength, motifPositions.get_Item(sequenceRange))
    backgroundLeft = VariableStringOfLength(motifPositions.get_Item(sequenceRange), backgroundNucleobaseDist)
    backgroundRight = VariableStringOfLength(backgroundLengthRight, backgroundNucleobaseDist)
    added_vars = backgroundLeft.op_Addition(backgroundLeft, motif)
    added_vars = added_vars.op_Addition(added_vars, backgroundRight)
    sequences.set_Item(sequenceRange, added_vars)

    ifVar.Dispose()

    ifNotVar = Variable.IfNot(motifPresence.get_Item(sequenceRange))

    sequences.set_Item(sequenceRange, VariableStringOfLength(SequenceLength, backgroundNucleobaseDist))

    ifNotVar.Dispose()
    forEachBlock.CloseBlock()

    sequences.ObservedValue = sequenceData
    engine = InferenceEngine()
    engine.NumberOfIterations = 30  #30
    engine.Compiler.RecommendedQuality = QualityBand.Experimental

    motifNucleobaseProbsPosterior = engine.Infer[Array[Dirichlet]](motifNucleobaseProbs)
    motifPresencePosterior = engine.Infer[Array[Bernoulli]](motifPresence)
    motifPositionPosterior = engine.Infer[Array[Discrete]](motifPositions)

    # PrintMotifInferenceResults
    PrintPositionFrequencyMatrix("\nTrue position frequency matrix:",
                                 trueMotifNucleobaseDist,
                                 lambda dist, c: dist[c])  # Distributions.DiscreteChar indexer is implemented.

    PrintPositionFrequencyMatrix("\nInferred position frequency matrix mean:",
                                 motifNucleobaseProbsPosterior, # Array of Distribtions.Dirichlet; mean of each is a PiecewiseVector
                                 lambda dist, c: dist.GetMean()[ord(c)])  # PiecewiseVector indexer is implemented, but not for strings...
    # TypeError: No method matches given arguments for get_Item: (<class 'str'>) -> need to do ord(c)
    # Tried importing Console and ConsoleColor from System which works in powershell but not in VS console.

    printc("\nPREDICTION   ", ConsoleColor.Yellow)
    printc("GROUND TRUTH    ", ConsoleColor.Red)
    printc("OVERLAP    \n\n", ConsoleColor.Green)
    for i in range(min(SequenceCount, 30)):
        motifPos = motifPositionPosterior[i].GetMode() if motifPresencePosterior[i].GetProbTrue() > 0.5 else -1

        inPrediction, inGroundTruth = False, False
        for j in range(SequenceLength):
            if j == motifPos:
                inPrediction = True
            elif j == motifPos + motifLength:
                inPrediction = False
            if j == motifPositionData[i]:
                inGroundTruth = True
            elif j == motifPositionData[i] + motifLength:
                inGroundTruth = False

            color = Console.ForegroundColor
            if (inPrediction and inGroundTruth):
                color = ConsoleColor.Green
            elif (inPrediction):
                color = ConsoleColor.Yellow
            elif inGroundTruth:
                color = ConsoleColor.Red
            printc(sequenceData[i][j], color)
        print(f"    P(has motif) = {motifPresencePosterior[i].GetProbTrue():.2f}", end="");
        if (motifPos != -1):
            print(f"   P(pos={motifPos}) = {motifPositionPosterior[i][motifPos]:.2f}", end="");
        print()

def PrintPositionFrequencyMatrix(caption, positionWeights, weightExtractor):
    print(caption)
    for nucleobase in ['A', 'C', 'T', 'G']:
        print(f"{nucleobase}:   ", end="")
        freqs = [weightExtractor(positionWeights[i], nucleobase) for i in range(len(positionWeights))]
        print("   ".join([f"{freq:.2f}" for freq in freqs]))


def NucleobaseDist(a=0.0, c=0.0, g=0.0, t=0.0):
    probs = PiecewiseVector.Zero(int('ffff', 16) + 1);
    probs[ord('A')] = a;
    probs[ord('C')] = c;
    probs[ord('G')] = g;
    probs[ord('T')] = t;
    return DiscreteChar.FromVector(probs);

def SampleMotifData(sequenceCount, sequenceLength, motifPresenceProbability, motif, backgroundDist):
    """Samples data from the model.
    Args:
      motif (DiscreteChar[]) : Position frequency matrix defining the motif (true nucleobase distributions).

    Returns:
      sequenceData (str[]) : The sampled sequences
      motifPositionData (int[]) : The motif positions in the sampled sequences.
    """
    sequenceData, motifPositionData = [None] * sequenceCount, [None] * sequenceCount
    motifLength = len(motif)
    for i in range(sequenceCount):
        if (Rand.Double() < motifPresenceProbability):
            motifPositionData[i] = Rand.Int(sequenceLength - motifLength + 1)
            # Converter is prototype of lambda; has to be either lambda or another function.
            # Instead of relying on Util.ArrayInit (following doesn't work)...
            # backgroundBeforeChars = Util.ArrayInit(motifPositionData[i], lambda j :backgroundDist.Sample())
            # We can make the array ourselves
            backgroundBeforeChars = [backgroundDist.Sample() for _ in range(motifPositionData[i])]
            backgroundAfterChars = [backgroundDist.Sample() for _ in range(sequenceLength - motifLength - motifPositionData[i])]
            sampledMotifChars = [motif[j].Sample() for j in range(motifLength)]
            sequenceData[i] = ''.join(backgroundBeforeChars) + ''.join(sampledMotifChars) + ''.join(backgroundAfterChars)
        else:
            motifPositionData[i] = -1
            background = [backgroundDist.Sample() for _ in range(sequenceLength)]
            sequenceData[i] = ''.join(background)
    return sequenceData, motifPositionData

def printc(text, color=ConsoleColor.Yellow):
    Console.ForegroundColor = color
    Console.Write(text)
    Console.ResetColor()

# Python.NET helper functions

def VariableObserved(list, range=None):
    t = type(list[0])
    array = Array[t](list)
    if range == None:
        args = [array]
    else:
        args = [array, range]
    return Invoker.InvokeStatic(Variable, "Observed", args)

def InitialiseTo(variable, distribution):
    # Reflection essentially, public in the Infer.NET API but not in the manual
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

def VariableStringFormat(format, arg):
    return Invoker.InvokeStatic(Variable, "StringFormat", [format, arg])

def VariableStringOfLength(length, string):
    return Invoker.InvokeStatic(Variable, "StringOfLength", [length, string])

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

def test_tutorials():
    TwoCoins()
    TruncatedGaussianEfficient()
    LearningAGaussian()
    LearningAGaussianWithRanges()
    BayesPointMachineExample()
    ClinicalTrial()
    MixtureOfGaussians()
    HelloStrings()
    StringFormat()
    MotifFinder()

if __name__ == '__main__':
    test_tutorials()