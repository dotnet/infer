---
layout: default
---
[Infer.NET development](index.md)

## Transform tutorial

Insert your transform into `ModelCompiler.ConstructTransformChain()`. The first example simply renames the variable `firstCoin` to `x`.

```csharp
internal class DummyTransform : CopyTransform
{
  public override string Name { get { return "DummyTransform"; } }

  IVariableDeclaration newDecl;

  protected override IVariableDeclaration ConvertVariableDecl(IVariableDeclaration ivd)
  {
    if (ivd.Name == "firstCoin")
    {
      newDecl = Builder.VarDecl("x", ivd.VariableType);
      return newDecl;
    }
    return base.ConvertVariableDecl(ivd);
  }

  protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
  {
    var ivd = Recognizer.GetVariableDeclaration(ivre);
    if (ivd.Name == "firstCoin")
    {
      var ivre2 = Builder.VarRefExpr(newDecl);
      return ivre2;
    }
           
    return base.ConvertVariableRefExpr(ivre);
  }
}
```
 
The second example inserts comments for the number of operator calls.

```csharp
internal class OperationCountTransform : ShallowCopyTransform
{
  public override string Name { get { return "OperationCountTransform"; } }
  int count;
  int multiplier = 1;

  protected override IStatement ConvertFor(IForStatement ifs)
  {
    IExpression len = Recognizer.LoopSizeExpression(ifs);
    int lenAsInt = 1;
    if (len is ILiteralExpression)
    {
      ILiteralExpression ile = (ILiteralExpression)len;
      lenAsInt = (int)ile.Value;
      multiplier *= lenAsInt;
    }
    var result = base.ConvertFor(ifs);
    multiplier /= lenAsInt;
    return result;
  }

  protected override IStatement DoConvertStatement(IStatement ist)
  {
    if(!context.InputAttributes.Has<OperatorStatement>(ist))
      return base.DoConvertStatement(ist);

    if (ist is IExpressionStatement)
    {
      IExpressionStatement ies = (IExpressionStatement)ist;
      if (ies.Expression is IMethodInvokeExpression)
      {
        if (Recognizer.IsStaticMethod(ies.Expression, typeof(InferNet), "Infer"))
        {
          return base.DoConvertStatement(ist);
        }
      }
      if (ies.Expression is IAssignExpression)
      {
        IAssignExpression iae = (IAssignExpression)ies.Expression;
        //Console.WriteLine(ist);
        count += multiplier;
        ICommentStatement ics = Builder.CommentStmt(String.Format("{0}", multiplier));
        context.OutputAttributes.Set(ics, new DependencyInformation());
        context.AddStatementBeforeCurrent(ics);
        return base.DoConvertStatement(ist);
      }
    }
    return base.DoConvertStatement(ist);
  }

  protected override void DoConvertMethodBody(IStatementCollection outputs, IStatementCollection inputs)
  {
    base.DoConvertMethodBody(outputs, inputs);
    ICommentStatement ics = Builder.CommentStmt(String.Format("Code has {0} operator calls", count));
    context.OutputAttributes.Set(ics, new DependencyInformation());
    outputs.Add(ics);
  }
}
```