// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#light
namespace Microsoft.ML.Probabilistic.FSharp

open Microsoft.FSharp.Quotations.Patterns
open Microsoft.ML.Probabilistic
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Math
open Microsoft.ML.Probabilistic.Models

//************************************************************************************************
//************************************************************************************************
//************************************************************************************************
/// Distributions over a float-based domain
 module FloatDistribution =
    // Distribution types over float-based domains
    type Gaussian = Microsoft.ML.Probabilistic.Distributions.Gaussian
    type VectorGaussian = Microsoft.ML.Probabilistic.Distributions.VectorGaussian //type vector
    type Beta = Microsoft.ML.Probabilistic.Distributions.Beta
    type Dirichlet = Microsoft.ML.Probabilistic.Distributions.Dirichlet // type Vector
    type Gamma = Microsoft.ML.Probabilistic.Distributions.Gamma
    type Wishart = Microsoft.ML.Probabilistic.Distributions.Wishart
    type Vector = Microsoft.ML.Probabilistic.Math.Vector
    type PositiveDefiniteMatrix = Microsoft.ML.Probabilistic.Math.PositiveDefiniteMatrix
     
    //=======================================================
    //===== Gaussian distribution array definitions =========
    //=======================================================
    /// Gaussian distribution array
    type GaussianArray = DistributionStructArray<Gaussian, float>
    /// 2-Dimensional Gaussian distribution array
    type GaussianArray2D = DistributionStructArray2D<Gaussian, float>
    /// Gaussian distribution array of array
    type GaussianArrayOfArray = DistributionRefArray<GaussianArray, float[]>
    /// 2-Dimensional Gaussian distribution array of array
    type GaussianArray2DOfArray = DistributionRefArray2D<GaussianArray, float[]>
    /// Gaussian distribution array of 2-Dimensional array
    type GaussianArrayOfArray2D = DistributionRefArray<GaussianArray2D, float[,]>
    /// Gaussian distribution array of array of array
    type GaussianArrayOfArrayOfArray = DistributionRefArray<GaussianArrayOfArray, float[][]>
      
    /// Variable Gaussian distribution array
    type VariableGaussianArray = VariableArray<Gaussian>
    /// Variable Gaussian distribution array of array
    type VariableGaussianArrayOfArray =  VariableArray<VariableGaussianArray, Gaussian[][]>
    /// Variable 2-Dimensional Gaussian distribution array
    type VariableGaussianArray2D = VariableArray2D<Gaussian>
    /// Variable 2-Dimensional Gaussian distribution array of array
    type VariableGaussianArray2DOfArray = VariableArray2D<VariableGaussianArray, Gaussian[][,]>
    /// Variable Gaussian distribution array of 2-Dimensional array
    type VariableGaussianArrayOfArray2D = VariableArray<VariableGaussianArray2D, Gaussian[,][]>
    /// Variable Gaussian distribution array of array of array
    type VariableGaussianArrayOfArrayOfArray = VariableArray<VariableGaussianArrayOfArray,Gaussian[][][]>
      
    //=======================================================================
    //========= VectorGaussian distribution array definitions ===============
    //=======================================================================
    /// VectorGaussian distribution array
    type VectorGaussianArray = DistributionRefArray<VectorGaussian, Vector>
    /// 2-Dimensional VectorGaussian distribution array
    type VectorGaussianArray2D = DistributionRefArray2D<VectorGaussian, Vector>
    /// VectorGaussian distribution array of array
    type VectorGaussianArrayOfArray = DistributionRefArray<VectorGaussianArray, Vector[]>
    /// 2-Dimensional VectorGaussian distribution array of array
    type VectorGaussianArray2DOfArray = DistributionRefArray2D<VectorGaussianArray, Vector[]>
    ///VectorGaussian distribution array of 2-Dimensional array
    type VectorGaussianArrayOfArray2D = DistributionRefArray<VectorGaussianArray2D, Vector[,]>
    /// VectorGaussian distribution array of array of array
    type VectorGaussianArrayOfArrayOfArray = DistributionRefArray<VectorGaussianArrayOfArray, Vector[][]>
    
    /// Variable VectorGaussian distribution array.
    type VariableVectorGaussianArray = VariableArray<VectorGaussian>
    /// Variable VectorGaussian distribution array of array
    type VariableVectorGaussianArrayOfArray =  VariableArray<VariableVectorGaussianArray, VectorGaussian[][]>
    /// Variable 2-Dimensional VectorGaussian distribution array
    type VariableVectorGaussianArray2D = VariableArray2D<VectorGaussian>
    /// Variable 2-Dimensional VectorGaussian distribution array of array
    type VariableVectorGaussianArray2DOfArray = VariableArray2D<VariableVectorGaussianArray, VectorGaussian[][,]>
    /// Variable VectorGausssian distribution array of 2-Dimensional array
    type VariableVectorGaussianArrayOfArray2D = VariableArray<VariableVectorGaussianArray2D, VectorGaussian[,][]>
    /// Variable VectorGaussian distribution array of array of array
    type VariableVectorGaussianArrayOfArrayOfArray = VariableArray<VariableVectorGaussianArrayOfArray,VectorGaussian[][][]>
     
    //=======================================================================
    //========== Dirichlet distribution array definitions ===================
    //=======================================================================
    /// Dirichlet distribution array
    type DirichletArray = DistributionRefArray<Dirichlet, Vector>
    /// 2-Dimensional Dirichlet distribution array
    type DirichletArray2D = DistributionRefArray2D<Dirichlet, Vector>
    /// Dirichlet distribution array of array
    type DirichletArrayOfArray = DistributionRefArray<DirichletArray, Vector[]>
    /// 2-Dimensional Dirichlet distribution array of array
    type DirichletArray2DOfArray = DistributionRefArray2D<DirichletArray, Vector[]>
    ///Dirichlet distribution array of 2-Dimensional array
    type DirichletArrayOfArray2D = DistributionRefArray<DirichletArray2D, Vector[,]>
    /// Dirichlet distribution array of array of array
    type DirichletArrayOfArrayOfArray = DistributionRefArray<DirichletArrayOfArray, Vector[][]>
      
    /// Variable Dirichlet distribution array
    type VariableDirichletArray = VariableArray<Dirichlet>
    /// Variable Dirichlet distribution array of array
    type VariableDirichletArrayOfArray =  VariableArray<VariableDirichletArray, Dirichlet[][]>
    /// Variable 2-Dimensional Dirichlet distribution array
    type VariableDirichletArray2D = VariableArray2D<Dirichlet>
    /// Variable 2-Dimensional Dirichlet distribution array of array
    type VariableDirichletArray2DOfArray = VariableArray2D<VariableDirichletArray, Dirichlet[][,]>
    /// Variable Dirichlet distribution array of 2-Dimensional array
    type VariableDirichletArrayOfArray2D = VariableArray<VariableDirichletArray2D, Dirichlet[,][]>
    /// Variable Dirichlet distribution array of array of array
    type VariableDirichletArrayOfArrayOfArray = VariableArray<VariableDirichletArrayOfArray,Dirichlet[][][]>
      
    //=======================================================================
    //============= Wishart distribution array definitions ==================
    //=======================================================================
    /// Wishart distribution array
    type WishartArray = DistributionRefArray<Wishart, PositiveDefiniteMatrix>
    /// 2-Dimensional Wishart distribution array
    type WishartArray2D = DistributionRefArray2D<Wishart, PositiveDefiniteMatrix>
    /// Wishart distribution array of array
    type WishartArrayOfArray = DistributionRefArray<WishartArray, PositiveDefiniteMatrix[]>
    /// 2-Dimensional Wishart distribution array of array
    type WishartArray2DOfArray = DistributionRefArray2D<WishartArray, PositiveDefiniteMatrix[]>
    /// Wishart distribution array of 2-Dimensional array
    type WishartArrayOfArray2D = DistributionRefArray<WishartArray2D, PositiveDefiniteMatrix[,]>
    /// Wishart distribution array of array of array
    type WishartArrayOfArrayOfArray = DistributionRefArray<WishartArrayOfArray, PositiveDefiniteMatrix[][]>
      
    /// Variable Wishart distribution array
    type VariableWishartArray = VariableArray<Wishart>
    /// Variable Wishart distribution array of array
    type VariableWishartArrayOfArray =  VariableArray<VariableWishartArray, Wishart[][]>
    /// Variable 2-Dimensional Wishart distribution array
    type VariableWishartArray2D = VariableArray2D<Wishart>
    /// Variable 2-Dimensional Wishart distribution array of array
    type VariableWishartArray2DOfArray = VariableArray2D<VariableWishartArray, Wishart[][,]>
    /// Variable Wishart distribution array of 2-Dimensional array
    type VariableWishartArrayOfArray2D = VariableArray<VariableWishartArray2D, Wishart[,][]>
    /// Variable Wishart distribution array of array of array
    type VariableWishartArrayOfArrayOfArray = VariableArray<VariableWishartArrayOfArray,Wishart[][][]>
      
    //=======================================================================
    //================= Beta Distribution array definitions =================
    //=======================================================================
    /// Beta distribution array
    type BetaArray = DistributionStructArray<Beta, float>
    /// 2-Dimensional Beta distribution array
    type BetaArray2D = DistributionStructArray2D<Beta, float>
    /// Beta distribution array of array
    type BetaArrayOfArray = DistributionRefArray<BetaArray, float[]>
    /// 2-Dimensional Beta distribution array of array
    type BetaArray2DOfArray = DistributionRefArray2D<BetaArray, float[]>
    /// Beta distribution array of 2-Dimensional array
    type BetaArrayOfArray2D = DistributionRefArray<BetaArray2D, float[,]>
    /// Beta distribution array of array of array
    type BetaArrayOfArrayOfArray = DistributionRefArray<BetaArrayOfArray, float[][]>
      
    /// Variable Beta distribution array
    type VariableBetaArray = VariableArray<Beta>
    /// Variable Beta distribution array of array
    type VariableBetaArrayOfArray =  VariableArray<VariableBetaArray, Beta[][]>
    /// Variable 2-Dimensional Beta distribution array
    type VariableBetaArray2D = VariableArray2D<Beta>
    /// Variable 2-Dimensional Beta distribution array of array
    type VariableBetaArray2DOfArray = VariableArray2D<VariableBetaArray, Beta[][,]>
    /// Variable Beta distribution array of 2-Dimensional array
    type VariableBetaArrayOfArray2D = VariableArray<VariableBetaArray2D, Beta[,][]>
    /// Variable Beta distribution array of array of array
    type VariableBetaArrayOfArrayOfArray = VariableArray<VariableBetaArrayOfArray,Beta[][][]>
    
    //=======================================================================
    //================ Gamma Distribution array definitions =================
    //=======================================================================
    /// Gamma distribution array
    type GammaArray = DistributionStructArray<Gamma, float>
    /// 2-Dimensional Gamma distribution array
    type GammaArray2D = DistributionStructArray2D<Gamma, float>
    /// Gamma distribution array of array
    type GammaArrayOfArray = DistributionRefArray<GammaArray, float[]>
    /// 2-Dimensional Gamma distribution array of array
    type GammaArray2DOfArray = DistributionRefArray2D<GammaArray, float[]>
    /// Gamma distribution array of 2-Dimensional array
    type GammaArrayOfArray2D = DistributionRefArray<GammaArray2D, float[,]>
    /// Gamma distribution array of array of array
    type GammaArrayOfArrayOfArray = DistributionRefArray<GammaArrayOfArray, float[][]>
      
    /// Variable Gamma distribution array
    type VariableGammaArray = VariableArray<Gamma>
    /// Variable Gamma distribution array of array
    type VariableGammaArrayOfArray =  VariableArray<VariableGammaArray, Gamma[][]>
    /// Variable 2-Dimensional Gamma distribution array
    type VariableGammaArray2D = VariableArray2D<Gamma>
    /// Variable 2-Dimensional Gamma distribution array of array
    type VariableGammaArray2DOfArray = VariableArray2D<VariableGammaArray, Gamma[][,]>
    /// Variable Gamma distribution array of 2-Dimensional array
    type VariableGammaArrayOfArray2D = VariableArray<VariableGammaArray2D, Gamma[,][]>
    /// Variable Gamma distribution array of array of array
    type VariableGammaArrayOfArrayOfArray = VariableArray<VariableGammaArrayOfArray,Gamma[][][]>
    
    //========================================================================
    //====================== Float variable arrays ===========================
    //========================================================================
    /// Variable array of float
    type ArrayOfDouble = VariableArray<float>
    /// Variable array of array of float
    type ArrayOfArrayOfDouble = VariableArray<ArrayOfDouble, float[][]>
    /// Variable array of 2-Dimensional arrays of float
    type ArrayOfArray2DOfDouble = VariableArray<ArrayOfDouble, float[,][]>
    /// 2-Dimensional Variable array of array of float
    type Array2DOfArrayOfDouble = VariableArray<ArrayOfDouble, float[][,]>
    /// Variable array of array of array of float
    type ArrayOfArrayOfArrayOfDouble = VariableArray<ArrayOfArrayOfDouble, float[][][]>
    
     
//************************************************************************************************
//************************************************************************************************
//************************************************************************************************
/// Distributions over an integer domain     
module IntDistribution =
    // Distribution types over an integer domain
    type Discrete =  Distributions.Discrete
    type Poisson = Microsoft.ML.Probabilistic.Distributions.Poisson
     
    //=======================================================================
    //============= Discrete Distribution array definitions =================
    //=======================================================================
    /// Discrete distribution array
    type DiscreteArray =  DistributionRefArray<Discrete,int>
    /// 2-Dimensional Discrete distribution array
    type DiscreteArray2D = DistributionRefArray2D<Discrete, int>
    /// Discrete distribution array of array
    type DiscreteArrayOfArray = DistributionRefArray<DiscreteArray, int[]>
    /// Discrete distribution array of 2-Dimensional array
    type DiscreteArrayOfArray2D = DistributionRefArray<DiscreteArray2D, int[,]>
    /// 2-Dimensional Discrete distribution array of array
    type DiscreteArray2DOfArray = DistributionRefArray2D<DiscreteArray, int[]>
    /// Discrete distribution array of array of array
    type DiscreteArrayOfArrayOfArray = DistributionRefArray<DiscreteArrayOfArray, int[][]>
      
    /// Variable Discrete distribution array
    type VariableDiscreteArray = VariableArray<Discrete>
    /// Variable Discrete distribution array of array
    type VariableDiscreteArrayOfArray =  VariableArray<VariableDiscreteArray, Discrete[][]>
    /// Variable 2-Dimensional Discrete distribution array
    type VariableDiscreteArray2D = VariableArray2D<Discrete>
    /// Variable 2-Dimensional Discrete distribution array of array
    type VariableDiscreteArray2DOfArray = VariableArray2D<VariableDiscreteArray, Discrete[][,]>
    /// Variable Discrete distribution array of 2-Dimensional array
    type VariableDiscreteArrayOfArray2D = VariableArray<VariableDiscreteArray2D, Discrete[,][]>
    /// Variable Discrete distribution array of array of array
    type VariableDiscreteArrayOfArrayOfArray = VariableArray<VariableDiscreteArrayOfArray,Discrete[][][]>
       
    //=======================================================================
    //=============== Poisson distribution array definitions ================
    //=======================================================================
    /// Poisson distribution array
    type PoissonArray = DistributionStructArray<Poisson, int>
    /// 2-Dimensional Poisson distribution array
    type PoissonArray2D = DistributionStructArray2D<Poisson, int>
    /// Poisson distribution array of array
    type PoissonArrayOfArray = DistributionRefArray<PoissonArray, int[]>
    /// 2-Dimensional Poisson distribution array of array
    type PoissonArray2DOfArray = DistributionRefArray2D<PoissonArray, int[]>
    ///Poisson distribution array of 2-Dimensional array
    type PoissonArrayOfArray2D = DistributionRefArray<PoissonArray2D, int[,]>
    /// Poisson distribution array of array of array
    type PoissonArrayOfArrayOfArray = DistributionRefArray<PoissonArrayOfArray, int[][]>
        
    /// Variable Poisson distribution array
    type VariablePoissonArray = VariableArray<Poisson>
    /// Variable Poisson distribution array of array
    type VariablePoissonArrayOfArray =  VariableArray<VariablePoissonArray, Poisson[][]>
    /// Variable 2-Dimensional Poisson distribution array
    type VariablePoissonArray2D = VariableArray2D<Poisson>
    /// Variable 2-Dimensional Poisson distribution array of array
    type VariablePoissonArray2DOfArray = VariableArray2D<VariablePoissonArray, Poisson[][,]>
    /// Variable Poisson distribution array of 2-Dimensional array
    type VariablePoissonArrayOfArray2D = VariableArray<VariablePoissonArray2D, Poisson[,][]>
    /// Variable Poisson distribution array of array of array
    type VariablePoissonArrayOfArrayOfArray = VariableArray<VariablePoissonArrayOfArray,Poisson[][][]>
       
    //========================================================================
    //====================== Integer variable arrays =========================
    //========================================================================
    /// Variable array of integer
    type ArrayOfInt = VariableArray<int>
    /// Variable array of array of integer
    type ArrayOfArrayOfInt = VariableArray<ArrayOfInt, int[][]>
    /// 2-Dimensional variable array of integer
    type Array2DOfInt = VariableArray2D<int>
    /// 2-Dimensional variable array of array of integer
    type Array2DOfArrayOfInt = VariableArray2D<ArrayOfInt, int[][,]>
    /// Variable array of 2-Dimensional array of integer
    type ArrayOfArray2DOfInt = VariableArray<Array2DOfInt, int[,][]>
    /// Variable array of array of array of integer
    type ArrayOfArrayOfArrayOfInt = VariableArray<ArrayOfArrayOfInt, int[][][]>

//************************************************************************************************
//************************************************************************************************
//************************************************************************************************
     
/// Distributions over a boolean domain    
module BoolDistribution =
    // Distribution types over a boolean domain
    type Bernoulli = Microsoft.ML.Probabilistic.Distributions.Bernoulli
      
    //=======================================================================
    //=============== Bernoulli distribution array definitions ================
    //=======================================================================
    /// Bernoulli distribution array
    type BernoulliArray = DistributionStructArray<Bernoulli, bool>
    /// 2-Dimensional Bernoulli distribution array
    type BernoulliArray2D = DistributionStructArray2D<Bernoulli, bool>
    /// Bernoulli distribution array of array
    type BernoulliArrayOfArray = DistributionRefArray<BernoulliArray, bool[]>
    /// Bernoulli distribution array of 2-Dimensional array
    type BernoulliArrayOfArray2D = DistributionRefArray<BernoulliArray2D, bool[,]>
    /// 2-Dimensional Bernoulli distribution array of array
    type BernoulliArray2DOfArray = DistributionRefArray2D<BernoulliArray, bool[]>
    /// Bernoulli distribution array of array of array
    type BernoulliArrayOfArrayOfArray = DistributionRefArray<BernoulliArrayOfArray, bool[][]>
    
    /// Variable Bernoulli distribution array
    type VariableBernoulliArray = VariableArray<Bernoulli>
    /// Variable Bernoulli distribution array of array
    type VariableBernoulliArrayOfArray =  VariableArray<VariableBernoulliArray, Bernoulli[][]>
    /// Variable 2-Dimensional Bernoulli distribution array
    type VariableBernoulliArray2D = VariableArray2D<Bernoulli>
    /// Variable 2-Dimensional Bernoulli distribution array of array
    type VariableBernoulliArray2DOfArray = VariableArray2D<VariableBernoulliArray, Bernoulli[][,]>
    /// Variable Bernoulli distribution array of 2-Dimensional array
    type VariableBernoulliArrayOfArray2D = VariableArray<VariableBernoulliArray2D, Bernoulli[,][]>
    /// Variable Bernoulli distribution array of array of array
    type VariableBernoulliArrayOfArrayOfArray =  VariableArray<VariableBernoulliArrayOfArray, Bernoulli[][][]>
    
    //========================================================================
    //====================== Boolean variable arrays =========================
    //========================================================================
    /// Variable array of bool
    type ArrayOfBool = VariableArray<bool>
    /// Variable array of array of bool
    type ArrayOfArrayOfBool = VariableArray<ArrayOfBool, bool[][]>
    /// 2-Dimensional variable array of bool
    type Array2DOfBool = VariableArray2D<bool>
    /// 2-Dimensional variable array of array of bool
    type Array2DOfArrayOfBool = VariableArray2D<ArrayOfBool, bool[][,]>
    /// Variable array of 2-Dimensional array of bool
    type ArrayOfArray2DOfBool = VariableArray<Array2DOfBool, bool[,][]>
    /// Variable array of array of array of bool
    type ArrayOfArrayOfArrayOfBool = VariableArray<ArrayOfArrayOfBool, bool[][][]>
        

//************************************************************************************************
//************************************************************************************************
//************************************************************************************************
/// Provides methods for use with 2-Dimensional .NET arrays        
module Array2D =
    /// Creates a 2-Dimensional array from a list of lists of values
    let create2D (xss:_ list list) =
        let xss = Seq.map Seq.toArray xss |> Seq.toArray
        let n,m = xss.Length,xss.[0].Length
        Array2D.init n m (fun i j -> xss.[i].[j])

//************************************************************************************************
//************************************************************************************************
//************************************************************************************************
/// Provides methods for use with Variable objects
module Variable =
    //========================================================================
    //================================ Blocks ================================
    //========================================================================
    /// Apply the function body for each element in Range r
    let ForeachBlock (r:Range) (body:Range -> unit) =
        let block = Variable.ForEach(r)
        body r
        block.Dispose()

    /// Switch on Variable r and apply the function body
    let SwitchBlock (r:Variable<int>) (body:Variable<int> -> unit) =
        let block = Variable.Switch(r)
        body r
        block.Dispose()

    /// Switch on Variable r and apply the function body, returning the result.
    let SwitchExpr (r:Variable<int>) (body:Variable<int> -> Variable<'a>) =                
        let block = Variable.Switch(r)
        let tmp : Variable<'a> = body r 
        block.Dispose()
        tmp

    /// Calls function f1 if the argument Variable<bool> is true, and function f2 if argument Variable<bool> is false
    let IfBlock  (test:Variable<bool>)
        (f1IfTest: Variable<bool> -> unit) (f2NotTest: Variable<bool> -> unit) =
        do using (Variable.If(test))  (fun _ -> f1IfTest test )
        do using (Variable.IfNot(test)) (fun _ -> f2NotTest test)
       
    //========================================================================
    //=============== Variable array creation and assignment =================
    //========================================================================
    /// Creates a Variable Array of type 'a given the dimension and a generator function to compute the elements
    let ArrayInit<'a> (n:Range) (f : Range -> Variable<'a>) =
        let z = Variable.Array<'a>(n)
        ForeachBlock n (fun i ->
        z.[i] <- f i)
        z
                    
    /// Apply the generator function f to a VariableArray of type 'a and Range r   
    let AssignVariableArray (v:VariableArray<'a>) (r:Range) (f : Range -> Variable<'a>) =  
        let _ = ForeachBlock r (fun i ->v.[i] <- f i)
        v

    /// Apply the generator function f to a 2-Dimensional VariableArray of type 'a and Ranges r,s   
    let AssignVariableArray2D  (v:VariableArray2D<'a>) (r:Range) (s:Range) (f : Range->Range-> Variable<'a>) =  
        v.[r,s]<- f r s
        v
        
    /// Apply the generator function f to a Jagged VariableArray of type 'a and Ranges r and s        
    let AssignJagged1DVariableArray (v: VariableArray<VariableArray<'a>, 'a[][]>) (r:Range) (s:Range) (f : Range->Range-> Variable<'a>) =  
        v.[r].[s]<- f r s
        v                      
                        
//************************************************************************************************
//************************************************************************************************
//************************************************************************************************
/// Provides operator overloads for comparison type Variable<'a> with Variable<'a> or type 'a returning type Variable<bool>
#nowarn "0077"
[<AutoOpen>]  
module Operators =
    /// Strictly Less Than Operator
    let inline (<<) (x:^a) (y:^b) : ^c = ( ^a : (static member (<) : ^a * ^b -> ^c) (x,y))
    /// Strictly Greater Than Operator
    let inline (>>) (x:^a) (y:^b) : ^c = ( ^a : (static member (>) : ^a * ^b -> ^c) (x,y))
    /// Equality Operator
    let inline (==) (x:^a) (y:^b) : ^c = ( ^a : (static member (=) : ^a * ^b -> ^c) (x,y))
    /// Equality Operator
    let inline (<<>>) (x:^a) (y:^b) : ^c = ( ^a : (static member (<>) : ^a * ^b -> ^c) (x,y))
    /// Less Than Operator
    let inline (<<==) (x:^a) (y:^b) : ^c = ( ^a : (static member (<=) : ^a * ^b -> ^c) (x,y))
    /// Greater Than Operator
    let inline (>>==) (x:^a) (y:^b) : ^c = ( ^a : (static member (>=) : ^a * ^b -> ^c) (x,y))   
            
//************************************************************************************************
//************************************************************************************************
//************************************************************************************************
/// Provides methods for Inferring .NET arrays of Distributions from a VariableArray<T> of a given Distribution
module Inference = 
    //========================================================================
    //=============== Inference for float-based arrays =======================
    //========================================================================
    ///Infer a VariableArray<float> as a .NET array of Gaussian distributions
    let InferGaussianArray (ie: InferenceEngine, m: VariableArray<float>) =
        ie.Infer<Gaussian[]>(m)
       
    ///Infer a VariableArray<Vector> as a.NET array of VectorGaussian distributions
    let InferVectorGaussianArray (ie: InferenceEngine, m: VariableArray<Vector>) =
        ie.Infer<VectorGaussian[]>(m)

    ///Infer a VariableArray<Vector> as a.NET array of Dirichlet distributions  
    let InferDirichletArray (ie: InferenceEngine, m: VariableArray<Vector>) =
        ie.Infer<Dirichlet[]>(m)

    ///Infer a VariableArray<PositiveDefiniteMatrix> as a.NET array of Wishart distributions  
    let InferWishartArray (ie: InferenceEngine, m: VariableArray<PositiveDefiniteMatrix>) =
        ie.Infer<Wishart[]>(m)
            
    ///Infer a VariableArray<float> as a.NET array of Beta distributions  
    let InferBetaArray (ie: InferenceEngine, m: VariableArray<float>) =
        ie.Infer<Beta[]>(m)

    ///Infer a VariableArray<float> as a.NET array of Gamma distributions  
    let InferGammaArray (ie: InferenceEngine, m: VariableArray<float>) =
        ie.Infer<Gamma[]>(m)

    //========================================================================
    //=================== Inference for int arrays ===========================
    //========================================================================
    ///Infer a VariableArray<int> as a.NET array of Discrete distributions  
    let InferDiscreteArray (ie: InferenceEngine, m: VariableArray<'a>) =
        ie.Infer<Discrete[]>(m)

    ///Infer a VariableArray<int> as a.NET array of Poisson distributions  
    let InferPoissonArray (ie: InferenceEngine, m: VariableArray<'a>) =
        ie.Infer<Poisson[]>(m)
            
    //========================================================================
    //=================== Inference for bool arrays ==========================
    //========================================================================
    ///Infer a VariableArray<bool> as a.NET array of Bernoulli distributions  
    let InferBernoulliArray (ie: InferenceEngine, m: VariableArray<bool>) =
        ie.Infer<Bernoulli[]>(m)
            
//************************************************************************************************
//************************************************************************************************
//************************************************************************************************
/// Provides method creating a delegate for declaring and registering a new Factor       
module Factors = 
    /// Create a delegate for declaring and registering a new Factor.
    /// Takes an quotation of the factor function as an argument giving it type signature Expr->System.Delegate
    let createDelegate x =
        let info = 
            match x with
                | Call (target,info,args) -> info
                | _ -> failwith "Not a method"

        let returnType = info.ReturnType
        let argInfo = info.GetParameters()
        let argTypes = argInfo |> Array.map (fun pi ->pi.ParameterType)
        let a = Array.append argTypes [|returnType|] 
        let methodString = @"System.Func`" + (string a.Length)
        let assembly = (typeof<System.Func<_>>).Assembly
        let methodTyp = assembly.GetType(methodString).MakeGenericType(a)
        let d = System.Delegate.CreateDelegate(methodTyp,info)
        d
   