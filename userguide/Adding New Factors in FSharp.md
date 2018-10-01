---
layout: default 
--- 
[Infer.NET user guide](index.md) : [FSharp Wrapper](FSharp Wrapper.md)

## Adding new Factors in F\#

How to add a new factor to Infer.NET is documented in detail for [C#](How to add a new factor and message operators.md). This section highlights differences if you decide to implement your new factor in F\#.

1.  Attention must be paid to the signature for the factor method. For example if declaring a Factor called "Greater Than" to compare two integers, the signature must be declared in the following way:
    
    ```fsharp
    static member GreaterThan(x, y) =  x > (y:int)  
      
    //the equivalent C# definition is :  
    public static bool GreaterThan(int x, int y);
    ```
    
    If however the function is declared in the following way (i.e. without the parentheses), the equivalent C# definition is a FastFunc which is the wrong kind of signature :
    
    ```fsharp
    static member GreaterThan x y  =  x > (y:int)  
      
    //the equivalent C# definition is :  
    public static FastFunc<int,bool> GreaterThan(int x);
    ```
    
2.  A delegate must be created to provide a FastFunc wrapper of the original factor method. The module Factors provides a method for creating a delegate which directly references the factor method. Factors.createDelegate a utility function that will do this. Its argument is a quotation of the factor method and the function has type signature: _Expr -> System.Delegate_.  
    
    ```fsharp
    let gtDelegate = createDelegate <@ MyFactors.GreaterThan(0,0) @>
    ```
    
3.  If you want to use an operator as a syntactical short-cut for the factor, you need to register that fact:  
    
    ```fsharp
    Variable.RegisterOperatorFactor(Operator.GreaterThan, gtDelegate)
    ```
    
4.  You must the create an operator class and message operators as with C#.
5.  The assembly which contains the operator class must be annotated to tell Infer.NET to look for message functions in it. In F#, this can be done as follows:
    
    ```fsharp
    [<assembly: HasMessageFunctions()>]  
    ()
    ```

It should be noted that, the class containing message passing methods must be in a namespace and not in a module. If it is in a module, the generated code sees this as a nested class, and this is not yet supported in the Infer.NET language writer. Also, the message passing methods must use the correct names for arguments i.e. you must use 'result' rather than '_result'.