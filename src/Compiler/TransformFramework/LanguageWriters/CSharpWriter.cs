// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// C# language writer 

namespace Microsoft.ML.Probabilistic.Compiler
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Text;
    using Microsoft.ML.Probabilistic.Compiler.CodeModel;

    internal class CSharpWriter : LanguageWriter
    {
        public static bool PutOpenBraceOnNewLine;
        public static bool PutDelegateOnNewLine;
        public static readonly bool InsertSpaceAfterComma = true;

        private static readonly string comma = InsertSpaceAfterComma ? ", " : ",";

        private static readonly long NegativeZeroBits = BitConverter.DoubleToInt64Bits(-1.0 * 0.0);

        /// <summary>
        /// static constructor
        /// </summary>
        static CSharpWriter()
        {
            // C# binary operator look-ups
            BinaryOpLookUp.Add(BinaryOperator.Add, "+");
            BinaryOpLookUp.Add(BinaryOperator.Subtract, "-");
            BinaryOpLookUp.Add(BinaryOperator.Multiply, "*");
            BinaryOpLookUp.Add(BinaryOperator.Divide, "/");
            BinaryOpLookUp.Add(BinaryOperator.Modulus, "%");
            BinaryOpLookUp.Add(BinaryOperator.ShiftLeft, "<<");
            BinaryOpLookUp.Add(BinaryOperator.ShiftRight, ">>");
            BinaryOpLookUp.Add(BinaryOperator.ValueEquality, "==");
            BinaryOpLookUp.Add(BinaryOperator.ValueInequality, "!=");
            BinaryOpLookUp.Add(BinaryOperator.BitwiseOr, "|");
            BinaryOpLookUp.Add(BinaryOperator.BitwiseAnd, "&");
            BinaryOpLookUp.Add(BinaryOperator.BitwiseExclusiveOr, "^");
            BinaryOpLookUp.Add(BinaryOperator.BooleanOr, "||");
            BinaryOpLookUp.Add(BinaryOperator.BooleanAnd, "&&");
            BinaryOpLookUp.Add(BinaryOperator.LessThan, "<");
            BinaryOpLookUp.Add(BinaryOperator.LessThanOrEqual, "<=");
            BinaryOpLookUp.Add(BinaryOperator.GreaterThan, ">");
            BinaryOpLookUp.Add(BinaryOperator.GreaterThanOrEqual, ">=");

            // C# unary operator look-ups
            UnaryOpLookUp.Add(UnaryOperator.BitwiseNot, "~");
            UnaryOpLookUp.Add(UnaryOperator.BooleanNot, "!");
            UnaryOpLookUp.Add(UnaryOperator.Negate, "-");
            UnaryOpLookUp.Add(UnaryOperator.PostDecrement, "--");
            UnaryOpLookUp.Add(UnaryOperator.PreDecrement, "--");
            UnaryOpLookUp.Add(UnaryOperator.PostIncrement, "++");
            UnaryOpLookUp.Add(UnaryOperator.PreIncrement, "++");

            IntrinsicTypeAlias.Add("System.Boolean", "bool");
            IntrinsicTypeAlias.Add("System.SByte", "sbyte");
            IntrinsicTypeAlias.Add("System.Byte", "byte");
            IntrinsicTypeAlias.Add("System.Int16", "short");
            IntrinsicTypeAlias.Add("System.UInt16", "ushort");
            IntrinsicTypeAlias.Add("System.Int32", "int");
            IntrinsicTypeAlias.Add("System.UInt32", "uint");
            IntrinsicTypeAlias.Add("System.Int64", "long");
            IntrinsicTypeAlias.Add("System.UInt64", "ulong");
            IntrinsicTypeAlias.Add("System.Char", "char");
            IntrinsicTypeAlias.Add("System.Single", "float");
            IntrinsicTypeAlias.Add("System.Double", "double");
            IntrinsicTypeAlias.Add("System.Decimal", "decimal");
            IntrinsicTypeAlias.Add("System.String", "string");
            IntrinsicTypeAlias.Add("System.Object", "object");
            IntrinsicTypeAlias.Add("System.Void", "void");

            string[] reservedWords =
                {
                    "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked",
                    "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else",
                    "enum", "event", "explicit", "extern", "false", "finally", "fixed", "float", "for",
                    "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is",
                    "lock", "long", "namespace", "new", "null", "object", "operator", "out", "override",
                    "params", "private", "protected", "public", "readonly", "ref", "return", "sbyte",
                    "sealed", "short", "sizeof", "stackalloc", "static", "string", "struct", "switch",
                    "this", "throw", "true", "try", "typeof", "uint", "ulong", "unchecked", "unsafe",
                    "ushort", "using", "virtual", "volatile", "void", "while"
                };

            foreach (string rw in reservedWords)
            {
                ReservedMap.Add(rw, "@" + rw);
            }
        }

        /// <summary>
        /// Append interfaces from which a class is derived
        /// </summary>
        /// <param name="sb">The string builder to append to</param>
        /// <param name="interfaces">The interfaces</param>
        /// <param name="hasBaseType">Whether the type has a base type</param>
        protected override void AppendInterfaces(StringBuilder sb, List<ITypeReference> interfaces, bool hasBaseType)
        {
            if (interfaces == null || interfaces.Count <= 0)
                return;

            if (hasBaseType)
                sb.AppendLine(", ");
            else
                sb.Append(" : ");

            int i;
            for (i = 0; i < interfaces.Count; i++)
            {
                if (i != 0)
                    sb.Append(", ");
                AppendType(sb, interfaces[i]);
                //sb.Append(ValidIdentifier(interfaces[i].Name));
            }
            return;
        }


        /// <summary>
        /// Append an attribute.
        /// </summary>
        /// <param name="sb">StringBuilder to append to </param>
        /// <param name="attr">The attribute</param>
        /// <remarks>This is not exhaustive</remarks>
        protected override void AppendAttribute(StringBuilder sb, ICustomAttribute attr)
        {
            sb.Append(GetTabString());
            sb.Append("[");
            string ctorName = attr.Constructor.Name;
            if (ctorName == ".ctor") ctorName = attr.Constructor.DeclaringType.DotNetType.Name;
            if (ctorName.EndsWith("Attribute")) ctorName = ctorName.Substring(0, ctorName.Length - "Attribute".Length);
            sb.Append(ValidIdentifier(ctorName));
            if (attr.Arguments != null && attr.Arguments.Count > 0)
            {
                sb.Append("(");
                int i;
                for (i = 0; i < attr.Arguments.Count - 1; i++)
                {
                    AppendExpression(sb, attr.Arguments[i]);
                    sb.Append(", ");
                }
                AppendExpression(sb, attr.Arguments[i]);
                sb.Append(")");
            }
            sb.Append("]");
        }

        /// <summary>
        /// Append generic arguments to a type or method name
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="genericArguments"></param>
        protected override void AppendGenericArguments(StringBuilder sb, IEnumerable<IType> genericArguments)
        {
            using (var argEnumerator = genericArguments.GetEnumerator())
            {
                if (argEnumerator.MoveNext())
                {
                    sb.Append("<");
                    AppendType(sb, argEnumerator.Current);
                    while (argEnumerator.MoveNext())
                    {
                        sb.Append(","); // must be kept consistent with LanguageWriterTypeSourceTest
                        AppendType(sb, argEnumerator.Current);
                    }
                    sb.Append(">");
                }
            }
        }

        /// <summary>
        /// Append an array rankn to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ar">array rank</param>
        protected override void AppendArrayRank(StringBuilder sb, int ar)
        {
            sb.Append("[");
            for (int i = 1; i < ar; i++)
            {
                sb.Append(",");
            }
            sb.Append("]");
        }

        /// <summary>
        /// Append a variable declaration to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ivd">IVariableDeclaration</param>
        protected override void AppendVariableDeclaration(StringBuilder sb, IVariableDeclaration ivd)
        {
            AppendType(sb, ivd.VariableType);
            sb.Append(" ");
            sb.Append(ValidIdentifier(ivd.Name));
            // The following line is useful for tracking variable identities
            //sb.Append("(decl"+System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(ivd)+")");
        }

        /// <summary>
        /// Append address out expression
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="iae">address out expression</param>
        protected override void AppendAddressOutExpression(StringBuilder sb, IAddressOutExpression iae)
        {
            sb.Append("out ");
            AppendExpression(sb, iae.Expression);
        }

        /// <summary>
        /// Append argument reference expression
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="iare"></param>
        protected override void AppendArgumentReferenceExpression(StringBuilder sb, IArgumentReferenceExpression iare)
        {
            sb.Append(ValidIdentifier(iare.Parameter.Name));
        }

        /// <summary>
        /// Append an array type to a string builder
        /// </summary>
        /// <param name="sb"></param>
        /// <param name="iat"></param>
        protected override void AppendArrayType(StringBuilder sb, IArrayType iat)
        {
            IType innermostElementType = iat.ElementType;
            while (innermostElementType is IArrayType)
            {
                innermostElementType = ((IArrayType) innermostElementType).ElementType;
            }
            AppendType(sb, innermostElementType);

            while (true)
            {
                AppendArrayRank(sb, iat.Rank);
                IType elementType = iat.ElementType;
                if (elementType is IArrayType) iat = (IArrayType) elementType;
                else break;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="iae"></param>
        protected override void AppendArrayCreateExpression(StringBuilder sb, IArrayCreateExpression iae)
        {
            sb.Append("new ");
            IType innermostElementType = iae.Type;
            while (innermostElementType is IArrayType)
            {
                innermostElementType = ((IArrayType) innermostElementType).ElementType;
            }
            AppendType(sb, innermostElementType);

            sb.Append("[");
            for (int i = 0; i < iae.Dimensions.Count; i++)
            {
                if (i != 0) sb.Append(comma);
                AppendExpression(sb, iae.Dimensions[i]);
            }
            sb.Append("]");

            // Stick on all the remaining array dimensions
            IType elementType = iae.Type;
            while (elementType is IArrayType iat)
            {
                AppendArrayRank(sb, iat.Rank);
                elementType = iat.ElementType;
            }

            // The initialiser
            if (iae.Initializer != null)
            {
                sb.Append(" ");
                AppendExpression(sb, iae.Initializer);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="iaie"></param>
        protected override void AppendArrayIndexerExpression(StringBuilder sb, IArrayIndexerExpression iaie)
        {
            bool needParens = MayNeedParentheses(iaie.Target);
            if (needParens) sb.Append("(");
            AppendExpression(sb, iaie.Target);
            if (needParens) sb.Append(")");
            sb.Append("[");
            for (int i = 0; i < iaie.Indices.Count; i++)
            {
                if (i != 0) sb.Append(comma);
                AppendExpression(sb, iaie.Indices[i]);
            }
            sb.Append("]");
        }

        /// <summary>
        /// Append an IAssignExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="iae">IAssignExpression</param>
        protected override void AppendAssignExpression(StringBuilder sb, IAssignExpression iae)
        {
            AppendExpression(sb, iae.Target);
            sb.Append(" = ");
            AppendExpression(sb, iae.Expression);
        }

        /// <summary>
        /// Append an IBaseReferenceExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ibre">IBaseReferenceExpression</param>
        protected override void AppendBaseReferenceExpression(StringBuilder sb, IBaseReferenceExpression ibre)
        {
            sb.Append("base");
        }

        /// <summary>
        /// Append an IBinaryExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ibe">IBinaryExpression</param>
        protected override void AppendBinaryExpression(StringBuilder sb, IBinaryExpression ibe)
        {
            BinaryOperator op = ibe.Operator;
            switch (op)
            {
                case BinaryOperator.IdentityEquality:
                case BinaryOperator.IdentityInequality:
                    if (op == BinaryOperator.IdentityInequality)
                        sb.Append("!");
                    sb.Append("Object.ReferenceEquals(");
                    AppendExpression(sb, ibe.Left);
                    sb.Append(comma);
                    AppendExpression(sb, ibe.Right);
                    sb.Append(")");
                    break;
                default:
                    bool needParens = MayNeedParentheses(ibe.Left);
                    if (needParens) sb.Append("(");
                    AppendExpression(sb, ibe.Left);
                    if (needParens) sb.Append(")");
                    sb.Append(BinaryOpLookUp[op]);
                    needParens = MayNeedParentheses(ibe.Right);
                    if (needParens) sb.Append("(");
                    AppendExpression(sb, ibe.Right);
                    if (needParens) sb.Append(")");
                    break;
            }
        }

        /// <summary>
        /// Append an IBlockExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ibe">IBlockExpression</param>
        protected override void AppendBlockExpression(StringBuilder sb, IBlockExpression ibe)
        {
            sb.Append("{ ");
            for (int i = 0; i < ibe.Expressions.Count; i++)
            {
                if (i != 0) sb.Append(comma);
                AppendExpression(sb, ibe.Expressions[i]);
            }
            sb.Append(" }");
        }

        protected override void AppendCanCastExpression(StringBuilder sb, ICanCastExpression icce)
        {
            bool needParens = MayNeedParentheses(icce.Expression);
            if (needParens) sb.Append("(");
            AppendExpression(sb, icce.Expression);
            if (needParens) sb.Append(")");
            sb.Append(" is ");
            AppendType(sb, icce.TargetType);
        }

        /// <summary>
        /// Append an ICastExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ice">ICastExpression</param>
        protected override void AppendCastExpression(StringBuilder sb, ICastExpression ice)
        {
            sb.Append("(");
            AppendType(sb, ice.TargetType);
            sb.Append(")");
            bool needParens = MayNeedParentheses(ice.Expression);
            if (needParens) sb.Append("(");
            AppendExpression(sb, ice.Expression);
            if (needParens) sb.Append(")");
        }

        /// <summary>
        /// Append an ICheckedExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ice">ICheckedExpression</param>
        protected override void AppendCheckedExpression(StringBuilder sb, ICheckedExpression ice)
        {
            sb.Append("checked(");
            AppendExpression(sb, ice.Expression);
            sb.Append(")");
        }

        /// <summary>
        /// Append an IConditionExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ice">IConditionExpression</param>
        protected override void AppendConditionExpression(StringBuilder sb, IConditionExpression ice)
        {
            sb.Append("(");
            AppendExpression(sb, ice.Condition);
            sb.Append(") ? (");
            AppendExpression(sb, ice.Then);
            sb.Append(") : (");
            AppendExpression(sb, ice.Else);
            sb.Append(")");
        }

        /// <summary>
        /// Append an IDelegateCreateExpression expression
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="idce">IDelegateCreateExpression</param>
        protected override void AppendDelegateCreateExpression(StringBuilder sb, IDelegateCreateExpression idce)
        {
            sb.Append("new ");
            AppendType(sb, idce.DelegateType);
            sb.Append("(");
            AppendExpression(sb, idce.Target);
            sb.Append("}");
        }

        /// <summary>
        /// Append an IEventReferenceExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="iere">IEventReferenceExpression</param>
        protected override void AppendEventReferenceExpression(StringBuilder sb, IEventReferenceExpression iere)
        {
            bool needParens = MayNeedParentheses(iere.Target);
            if (needParens) sb.Append("(");
            AppendExpression(sb, iere.Target);
            if (needParens) sb.Append(")");
            sb.Append(".");
            sb.Append(ValidIdentifier(iere.Event.Name));
        }

        /// <summary>
        /// Append an IFieldReferenceExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ifre">IFieldReferenceExpression</param>
        protected override void AppendFieldReferenceExpression(StringBuilder sb, IFieldReferenceExpression ifre)
        {
            bool needParens = MayNeedParentheses(ifre.Target);
            if (needParens) sb.Append("(");
            AppendExpression(sb, ifre.Target);
            if (needParens) sb.Append(")");
            sb.Append(".");
            sb.Append(ValidIdentifier(ifre.Field.Name));
        }

        /// <summary>
        /// Append an ILambdaExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ile">ILambdaExpression</param>
        protected override void AppendLambdaExpression(StringBuilder sb, ILambdaExpression ile)
        {
            bool isFirst = true;
            foreach (IVariableDeclaration ivd in ile.Parameters)
            {
                sb.Append(ivd.Name);
                if (!isFirst) sb.Append(comma);
                isFirst = false;
            }
            sb.Append(" => ");
            AppendExpression(sb, ile.Body);
        }

        /// <summary>
        /// Append an ILiteralExpression to a string builder, maximizing readability.
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ile">ILiteralExpression</param>
        protected override void AppendLiteralExpression(StringBuilder sb, ILiteralExpression ile)
        {
            if (ile.Value == null)
            {
                sb.Append("null");
                return;
            }

            Type t = ile.GetExpressionType();
            if (t == typeof (string))
            {
                sb.Append("\"");
                string val = (string) ile.Value;
                val = val.Replace("\\", @"\\"); // must be done first
                val = val.Replace("\"", "\\\"");
                val = val.Replace("\'", @"\'");
                // Escape all special characters
                // See http://msdn.microsoft.com/en-us/library/h21280bw(VS.80).aspx
                val = val.Replace("\a", @"\a");
                val = val.Replace("\b", @"\b");
                val = val.Replace("\f", @"\f");
                val = val.Replace("\n", @"\n");
                val = val.Replace("\r", @"\r");
                val = val.Replace("\t", @"\t");
                val = val.Replace("\v", @"\v");
                sb.Append(val);
                sb.Append("\"");
            }
            else if (t == typeof (char))
            {
                char ch = (char) ile.Value;
                if (Char.IsLetterOrDigit(ch) || Char.IsSymbol(ch))
                {
                    sb.Append("'" + ch + "'");
                }
                else
                {
                    sb.AppendFormat(@"'\u{0:x4}'", (int) ch);
                }
            }
            else if (t == typeof (bool))
            {
                sb.Append(((bool) ile.Value) ? "true" : "false");
            }
            else if (t == typeof (double))
            {
                double d = (double) ile.Value;
                if (double.IsNegativeInfinity(d))
                {
                    sb.Append("double.NegativeInfinity");
                }
                else if (double.IsPositiveInfinity(d))
                {
                    sb.Append("double.PositiveInfinity");
                }
                else if (double.IsNaN(d))
                {
                    sb.Append("double.NaN");
                }
                else if (d.Equals(double.MaxValue))
                {
                    sb.Append("double.MaxValue");
                }
                else if (d.Equals(double.MinValue))
                {
                    sb.Append("double.MinValue");
                }
                else if (IsNegativeZero(d))
                {
                    // some build configurations result in -0.0 producing a positive zero
                    sb.Append("(-1.0 * 0.0)");
                }
                else
                {
                    string s = d.ToString("G17", CultureInfo.InvariantCulture);
                    sb.Append(s);
                    if (!s.Contains(".") && !s.Contains("E") && !s.Contains("e"))
                        sb.Append(".0");
                }
            }
            else if (t == typeof (float))
            {
                float f = (float) ile.Value;
                if (float.IsNegativeInfinity(f))
                {
                    sb.Append("float.NegativeInfinity");
                }
                else if (float.IsPositiveInfinity(f))
                {
                    sb.Append("float.PositiveInfinity");
                }
                else if (float.IsNaN(f))
                {
                    sb.Append("float.NaN");
                }
                else if (f.Equals(float.MaxValue))
                {
                    sb.Append("float.MaxValue");
                }
                else if (f.Equals(float.MinValue))
                {
                    sb.Append("float.MinValue");
                }
                else if (IsNegativeZero((double) f))
                {
                    // some build configurations result in -0.0 producing a positive zero
                    sb.Append("(-1F * 0F)");
                }
                else
                {
                    sb.Append(f.ToString("G9", CultureInfo.InvariantCulture));
                    sb.Append("F");
                }
            }
            else if (t == typeof (decimal))
            {
                decimal m = (decimal) ile.Value;
                if (m.Equals(decimal.MaxValue))
                {
                    sb.Append("decimal.MaxValue");
                }
                else if (m.Equals(decimal.MinValue))
                {
                    sb.Append("decimal.MinValue");
                }
                else
                {
                    sb.Append(m.ToString(CultureInfo.InvariantCulture));
                    sb.Append("M");
                }
            }
            else if (t == typeof (int))
            {
                int i = (int) ile.Value;
                if (i.Equals(int.MaxValue))
                {
                    sb.Append("int.MaxValue");
                }
                else if (i.Equals(int.MinValue))
                {
                    sb.Append("int.MinValue");
                }
                else
                {
                    sb.Append(i.ToString(CultureInfo.InvariantCulture));
                }
            }
            else if (t == typeof (uint))
            {
                uint u = (uint) ile.Value;
                if (u.Equals(uint.MaxValue))
                {
                    sb.Append("uint.MaxValue");
                }
                else
                {
                    sb.Append(u.ToString(CultureInfo.InvariantCulture));
                    sb.Append("U");
                }
            }
            else if (t == typeof (long))
            {
                long l = (long) ile.Value;
                if (l.Equals(long.MaxValue))
                {
                    sb.Append("long.MaxValue");
                }
                else if (l.Equals(long.MinValue))
                {
                    sb.Append("long.MinValue");
                }
                else
                {
                    sb.Append(l.ToString(CultureInfo.InvariantCulture));
                    sb.Append("L");
                }
            }
            else if (t == typeof (ulong))
            {
                ulong ul = (ulong) ile.Value;
                if (ul.Equals(ulong.MaxValue))
                {
                    sb.Append("ulong.MaxValue");
                }
                else
                {
                    sb.Append(ul.ToString(CultureInfo.InvariantCulture));
                    sb.Append("UL");
                }
            }
            else if (t == typeof (byte))
            {
                sb.Append("((byte)");
                sb.Append(ile.Value.ToString());
                sb.Append(")");
            }
            else if (t == typeof (sbyte))
            {
                sb.Append("((sbyte)");
                sb.Append(ile.Value.ToString());
                sb.Append(")");
            }
            else if (t == typeof (short))
            {
                sb.Append("((short)");
                sb.Append(ile.Value.ToString());
                sb.Append(")");
            }
            else if (t == typeof (ushort))
            {
                sb.Append("((ushort)");
                sb.Append(ile.Value.ToString());
                sb.Append(")");
            }
            else if (typeof (Enum).IsAssignableFrom(t))
            {
                AppendType(sb, t);
                sb.Append(".");
                sb.Append(Enum.GetName(t, ile.Value));
            }
            else
            {
                sb.Append(ile.Value.ToString());
            }
        }

        /// <summary>
        /// Append an IMethodInvokeExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="imie">IMethodInvokeExpression</param>
        protected override void AppendMethodInvokeExpression(StringBuilder sb, IMethodInvokeExpression imie)
        {
            IMethodReferenceExpression imre = imie.Method;
            if (imre.Method.Name == "get_Item" || imre.Method.Name == "set_Item")
            {
                if (imre.Method.Name == "set_Item")
                {
                    bool needParens = MayNeedParentheses(imre.Target);
                    if (needParens) sb.Append("(");
                    AppendExpression(sb, imre.Target);
                    if (needParens) sb.Append(")");
                    if (imie.Arguments.Count > 1)
                    {
                        sb.Append("[");
                        for (int i = 0; i < imie.Arguments.Count - 1; i++)
                        {
                            if (i > 0)
                                sb.Append(comma);
                            AppendExpression(sb, imie.Arguments[i]);
                        }
                        sb.Append("]");
                    }
                    sb.Append(" = ");
                    AppendExpression(sb, imie.Arguments[imie.Arguments.Count - 1]);
                }
                else if (imre.Method.Name == "get_Item")
                {
                    bool needParens = MayNeedParentheses(imre.Target);
                    if (needParens) sb.Append("(");
                    AppendExpression(sb, imre.Target);
                    if (needParens) sb.Append(")");
                    if (imie.Arguments.Count > 0)
                    {
                        sb.Append("[");
                        for (int i = 0; i < imie.Arguments.Count; i++)
                        {
                            if (i > 0) sb.Append(comma);
                            AppendExpression(sb, imie.Arguments[i]);
                        }
                        sb.Append("]");
                    }
                }
                return;
            }
            AppendMethodReferenceExpression(sb, imre);
            sb.Append("(");
            if (imie.Arguments.Count > 0)
            {
                for (int i = 0; i < imie.Arguments.Count; i++)
                {
                    if (i != 0)
                        sb.Append(", ");
                    AppendExpression(sb, imie.Arguments[i]);
                }
            }
            sb.Append(")");
        }

        /// <summary>
        /// True if the expression may need to be parenthesized when used in a larger expression.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        protected virtual bool MayNeedParentheses(IExpression expr)
        {
            if (expr is IVariableReferenceExpression) return false;
            else if (expr is IArgumentReferenceExpression) return false;
            else if (expr is ILiteralExpression) return false;
            else if (expr is IArrayIndexerExpression) return false;
            else if (expr is IAnonymousMethodExpression) return false;
            else if (expr is IBlockExpression) return false;
            else if (expr is IFieldReferenceExpression) return false;
            else if (expr is IEventReferenceExpression) return false;
            else if (expr is IPropertyReferenceExpression) return false;
            else if (expr is IPropertyIndexerExpression) return false;
            else if (expr is ITypeOfExpression) return false;
            else if (expr is ITypeReferenceExpression) return false;
            else if (expr is IThisReferenceExpression) return false;
            else if (expr is IMethodInvokeExpression) return false;
            else return true;
        }

        /// <summary>
        /// Append an IMethodReferenceExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="imre">IMethodReferenceExpression</param>
        protected override void AppendMethodReferenceExpression(StringBuilder sb, IMethodReferenceExpression imre)
        {
            bool needParens = MayNeedParentheses(imre.Target);
            if (needParens) sb.Append("(");
            AppendExpression(sb, imre.Target);
            if (needParens) sb.Append(")");
            sb.Append(".");
            IMethodReference imr = imre.Method;
            sb.Append(ValidIdentifier(imre.Method.Name));
            // The generic argument types
            AppendGenericArguments(sb, imr.GenericArguments);
        }

        /// <summary>
        /// Append an IObjectCreateExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ioce">IObjectCreateExpression</param>
        protected override void AppendObjectCreateExpression(StringBuilder sb, IObjectCreateExpression ioce)
        {
            sb.Append("new ");
            AppendType(sb, ioce.Type);
            sb.Append("(");
            if (ioce.Arguments.Count > 0)
            {
                for (int i = 0; i < ioce.Arguments.Count; i++)
                {
                    if (i != 0)
                        sb.Append(", ");
                    AppendExpression(sb, ioce.Arguments[i]);
                }
            }
            sb.Append(")");
            if (ioce.Initializer != null) AppendBlockExpression(sb, ioce.Initializer);
        }

        /// <summary>
        /// Append an IPropertyIndexerExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ipie">IPropertyIndexerExpression</param>
        protected override void AppendPropertyIndexerExpression(StringBuilder sb, IPropertyIndexerExpression ipie)
        {
            // this.Item[i] should be written as this[i]
            IExpression target =
            (ipie.Target.Property.Name == "Item") ?
                ipie.Target.Target
                :
                ipie.Target;
            bool needParens = MayNeedParentheses(target);
            if (needParens) sb.Append("(");
            AppendExpression(sb, target);
            if (needParens) sb.Append(")");
            sb.Append("[");
            for (int i = 0; i < ipie.Indices.Count; i++)
            {
                if (i != 0)
                    sb.Append(comma);
                AppendExpression(sb, ipie.Indices[i]);
            }
            sb.Append("]");
        }

        /// <summary>
        /// Append an IPropertyReferenceExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ipre">IPropertyReferenceExpression</param>
        protected override void AppendPropertyReferenceExpression(StringBuilder sb, IPropertyReferenceExpression ipre)
        {
            if (ipre.Target != null)
            {
                bool needParens = MayNeedParentheses(ipre.Target);
                if (needParens) sb.Append("(");
                AppendExpression(sb, ipre.Target);
                if (needParens) sb.Append(")");
                sb.Append(".");
            }
            sb.Append(ValidIdentifier(ipre.Property.Name));
        }

        /// <summary>
        /// Append an IThisReferenceExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="itre">IThisReferenceExpression</param>
        protected override void AppendThisReferenceExpression(StringBuilder sb, IThisReferenceExpression itre)
        {
            sb.Append("this");
        }

        /// <summary>
        /// Append an ITypeOfExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="itoe">ITypeOfExpression</param>
        protected override void AppendTypeOfExpression(StringBuilder sb, ITypeOfExpression itoe)
        {
            sb.Append("typeof(");
            AppendType(sb, itoe.Type);
            sb.Append(")");
        }

        /// <summary>
        /// Append an ITypeReferenceExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="itre">ITypeReferenceExpression></param>
        protected override void AppendTypeReferenceExpression(StringBuilder sb, ITypeReferenceExpression itre)
        {
            AppendType(sb, itre.Type);
        }

        /// <summary>
        /// Append an IUnaryExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="iue">IUnaryExpression</param>
        protected override void AppendUnaryExpression(StringBuilder sb, IUnaryExpression iue)
        {
            bool needParens = MayNeedParentheses(iue.Expression);
            UnaryOperator op = iue.Operator;
            switch (op)
            {
                case UnaryOperator.PostIncrement:
                case UnaryOperator.PostDecrement:
                    if (needParens) sb.Append("(");
                    AppendExpression(sb, iue.Expression);
                    if (needParens) sb.Append(")");
                    sb.Append(UnaryOpLookUp[op]);
                    break;
                default:
                    sb.Append(UnaryOpLookUp[op]);
                    if (needParens) sb.Append("(");
                    AppendExpression(sb, iue.Expression);
                    if (needParens) sb.Append(")");
                    break;
            }
        }

        /// <summary>
        /// Append an IVariableDeclarationExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ivde">IVariableDeclarationExpression</param>
        protected override void AppendVariableDeclarationExpression(StringBuilder sb, IVariableDeclarationExpression ivde)
        {
            AppendVariableDeclaration(sb, ivde.Variable);
        }

        /// <summary>
        /// Append an IVariableReferenceExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="ivre">IVariableReferenceExpression</param>
        protected override void AppendVariableReferenceExpression(StringBuilder sb, IVariableReferenceExpression ivre)
        {
            IVariableDeclaration ivd = ivre.Variable.Resolve();
            sb.Append(ValidIdentifier(ivd.Name));
            // The following line is useful for tracking variable identities
            //sb.Append("(ref"+System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(ivd)+")");
        }

        /// <summary>
        /// Append an IAddressReferenceExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="iare">IAddressReferenceExpression</param>
        protected override void AppendAddressReferenceExpression(StringBuilder sb, IAddressReferenceExpression iare)
        {
            sb.Append("ref ");
            AppendExpression(sb, iare.Expression);
        }

        /// <summary>
        /// Append an IAddressDereferenceExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="iade">IAddressDereferenceExpression</param>
        protected override void AppendAddressDereferenceExpression(StringBuilder sb, IAddressDereferenceExpression iade)
        {
            sb.Append("*(");
            AppendExpression(sb, iade.Expression);
            sb.Append(")");
        }

        /// <summary>
        /// Append an IAnonymousMethodExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="iame">IAnonymousMethodExpression</param>
        protected override void AppendAnonymousMethodExpression(StringBuilder sb, IAnonymousMethodExpression iame)
        {
            if (iame.DelegateType is ITypeReference) AddReferencedAssembly(iame.DelegateType as ITypeReference);
            if (PutDelegateOnNewLine)
            {
                sb.AppendLine();
                currTab++;
                sb.Append(GetTabString());
            }
            sb.Append("delegate(");
            AppendParameterDeclarationCollection(sb, iame.Parameters);
            sb.Append(")");
            if (PutOpenBraceOnNewLine)
            {
                sb.AppendLine();
                sb.Append(GetTabString());
            }
            else
            {
                sb.Append(" ");
            }
            sb.Append("{");
            sb.AppendLine();
            SourceNode bodySN = new SourceNode();
            currTab++;
            nodeStack.Push(bodySN);
            AttachStatements(iame.Body.Statements);
            nodeStack.Pop();
            StringWriter sw = new StringWriter();
            WriteSourceNode(sw, bodySN);
            sb.Append(sw.ToString());
            currTab--;
            sb.Append(GetTabString());
            if (PutDelegateOnNewLine)
            {
                currTab--;
            }
            sb.Append("}");
        }

        /// <summary>
        /// Append an IMemberInitializerExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="imie">IMemberInitializerExpression</param>
        protected override void AppendMemberInitializerExpression(StringBuilder sb, IMemberInitializerExpression imie)
        {
            sb.Append(ValidIdentifier(imie.Member.Name));
            sb.Append(" = ");
            AppendExpression(sb, imie.Value);
        }

        /// <summary>
        /// Append an IDelegateInvokeExpression to a string builder
        /// </summary>
        /// <param name="sb">string builder</param>
        /// <param name="idie">IDelegateInvokeExpression</param>
        protected override void AppendDelegateInvokeExpression(StringBuilder sb, IDelegateInvokeExpression idie)
        {
            AppendExpression(sb, idie.Target);
            sb.Append("(");
            if (idie.Arguments.Count > 0)
            {
                for (int i = 0; i < idie.Arguments.Count; i++)
                {
                    if (i != 0)
                        sb.Append(", ");
                    AppendExpression(sb, idie.Arguments[i]);
                }
            }
            sb.Append(")");
        }

        protected override void AppendDefaultExpression(StringBuilder sb, IDefaultExpression ide)
        {
            sb.Append("default(");
            AppendType(sb, ide.Type);
            sb.Append(")");
        }

        /// <summary>
        /// Append parameter declaration to a string builder
        /// </summary>
        /// <param name="sb">String builder</param>
        /// <param name="ipd">Parameter declaration</param>
        protected override void AppendParameterDeclaration(StringBuilder sb, IParameterDeclaration ipd)
        {
            // We will ignore parameter attributes
            AppendType(sb, ipd.ParameterType);
            sb.Append(" ");
            sb.Append(ValidIdentifier(ipd.Name));
            // The following line is useful for tracking variable identities
            //sb.Append("("+System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(ipd)+")");
        }

        /// <summary>
        /// Append parameter declaration collection to a string builder
        /// </summary>
        /// <param name="sb">String builder</param>
        /// <param name="parameters">Parameter collection</param>
        protected override void AppendParameterDeclarationCollection(StringBuilder sb, IList<IParameterDeclaration> parameters)
        {
            for (int i = 0; i < parameters.Count; i++)
            {
                if (i != 0)
                    sb.Append(", ");

                AppendParameterDeclaration(sb, parameters[i]);
            }
        }

        /// <summary>
        /// Attach a property accessor
        /// </summary>
        /// <param name="imr">Method reference</param>
        /// <param name="set">True if set rather than get</param>
        /// <remarks>This is not fully general</remarks>
        protected override void AttachPropertyAccessor(IMethodReference imr, bool set)
        {
            if (imr == null) return;
            IMethodDeclaration imd = imr as IMethodDeclaration;
            if (imd == null) throw new NotImplementedException();

            StringBuilder sb = new StringBuilder();

            sb.Append(GetTabString());

            if (imd.Visibility == MethodVisibility.Private)
                sb.Append("private ");

            string methodName = set ? "set" : "get";

            sb.Append(ValidIdentifier(methodName));
            if (PutOpenBraceOnNewLine)
            {
                sb.AppendLine();
                sb.Append(GetTabString());
            }
            else
            {
                sb.Append(" ");
            }
            sb.Append("{");
            sb.AppendLine();

            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();

            // Create the node...
            SourceNode sn = new SourceNode(sb.ToString(), sbEnd.ToString(), imd);
            AddChild(sn);
            nodeStack.Push(sn);
            currTab++;

            // Add the body
            IBlockStatement body = imd.Body;
            if (body != null)
                AttachStatements(body.Statements);

            currTab--;
            nodeStack.Pop();
        }

        /// <summary>
        /// Generate the source
        /// </summary>
        /// <param name="itd">The type declaration</param>
        public override SourceNode GenerateSource(ITypeDeclaration itd)
        {
            Initialise();

            SourceNode sourceTree = new SourceNode();
            nodeStack.Push(sourceTree);
            bool hasNamespace = itd.Namespace != null && itd.Namespace.Length > 0;
            if(hasNamespace)
                currTab++;

            // Attach the type
            AttachTypeDeclaration(itd);

            if(hasNamespace)
                currTab--;
            nodeStack.Pop();

            // Fill in the using statements
            SortedDictionary<string, int> namespaces = new SortedDictionary<string, int>();
            StringBuilder sb = new StringBuilder();

            // Make StyleCop ignore this code
            sb.AppendLine("// <auto-generated />");

            // Ignore warnings on badly formatted or missing XML comments
            sb.AppendLine("#pragma warning disable 1570, 1591");
            sb.AppendLine();

            sb.AppendLine("using System;");

            int count = 0;
            if(hasNamespace)
                namespaces.Add(itd.Namespace, count++);
            namespaces.Add("System", count++);
            foreach (string ns in typeReferenceMap.Values)
            {
                if (namespaces.ContainsKey(ns))
                    continue;

                namespaces.Add(ns, count++);
                sb.Append("using ");
                sb.Append(ns);
                sb.Append(";");
                sb.AppendLine();
            }
            sb.AppendLine();
            if (hasNamespace)
            {
                sb.Append("namespace ");
                sb.Append(itd.Namespace);
                sb.AppendLine();
                sb.Append("{");
                sb.AppendLine();
                StringBuilder sbEnd = new StringBuilder();
                sbEnd.Append("}");
                sbEnd.AppendLine();
                sourceTree.EndString = sbEnd.ToString();
            }
            sourceTree.StartString = sb.ToString();
            sourceTree.ASTElement = itd;

            return sourceTree;
        }

        public override SourceNode GeneratePartialSource(ITypeDeclaration itd)
        {
            Initialise();
            SourceNode sourceTree = new SourceNode();
            nodeStack.Push(sourceTree);

            // Attach the type
            SourceNode node = AttachTypeDeclaration(itd);

            nodeStack.Pop();
            return node;
        }

        public override SourceNode GeneratePartialSource(IStatement ist)
        {
            Initialise();
            SourceNode sourceTree = new SourceNode();
            nodeStack.Push(sourceTree);

            SourceNode node = AttachStatement(ist);

            nodeStack.Pop();
            return node;
        }

        /// <summary>
        /// Appends documentation.
        /// </summary>
        /// <param name="sb"></param>
        /// <param name="doc"></param>
        protected void AppendDocumentation(StringBuilder sb, string doc)
        {
            if (doc == null) return;
            string[] docs = doc.Split(Environment.NewLine[0]);
            foreach (string docline in docs)
            {
                sb.Append(GetTabString());
                sb.AppendLine("/// " + docline.TrimStart());
            }
        }

        /// <summary>
        /// Attach a type to the source tree
        /// </summary>
        /// <param name="itd">Type declaration</param>
        /// <returns>Source node</returns>
        public override SourceNode AttachTypeDeclaration(ITypeDeclaration itd)
        {
            string typnam = itd.Name;
            string oldNamespace = CurrentNamespace;

            // namespace mapping
            if (itd.Namespace != "")
            {
                typeReferenceMap.Add(typnam, itd.Namespace);
                CurrentNamespace = itd.Namespace;
            }

            StringBuilder sb = new StringBuilder();

            // documentation
            AppendDocumentation(sb, itd.Documentation);

            // Custom attributes
            foreach (ICustomAttribute attr in itd.Attributes)
            {
                AppendAttribute(sb, attr);
                sb.AppendLine();
            }

            sb.Append(GetTabString());

            // Public or private
            switch (itd.Visibility)
            {
                case TypeVisibility.Private:
                case TypeVisibility.NestedPrivate:
                    sb.Append("private ");
                    break;
                case TypeVisibility.Public:
                case TypeVisibility.NestedPublic:
                    sb.Append("public ");
                    break;
                case TypeVisibility.NestedFamily:
                    sb.Append("protected ");
                    break;
                case TypeVisibility.NestedAssembly:
                    sb.Append("internal ");
                    break;
                case TypeVisibility.NestedFamilyOrAssembly:
                    sb.Append("protected internal ");
                    break;
                default:
                    // C# does not allow any other visibility patterns than the 5 above
                    throw new NotSupportedException("C# does not support type visibility " + itd.Visibility);
            }

            if (itd.Partial)
                sb.Append("partial ");

            // Whether this is a class, struct or interface
            //if (itd.ValueType)
            //{
            //    sb.Append("struct ");
            //}
            //else 
            if (itd.Interface)
            {
                sb.Append("interface ");
            }
            else
            {
                if (itd.Sealed)
                    sb.Append("sealed ");
                if (itd.Abstract)
                    sb.Append("abstract ");
                sb.Append("class ");
            }

            sb.Append(typnam);

            // What this type is derived from
            bool hasBaseType = false;
            if (itd.BaseType != null && itd.BaseType.Name != "Object")
            {
                sb.Append(" : ");
                hasBaseType = true;
                AppendType(sb, itd.BaseType);
            }
            AppendInterfaces(sb, itd.Interfaces, hasBaseType);

            sb.AppendLine();
            sb.Append(GetTabString());
            sb.Append("{");
            sb.AppendLine();

            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            sbEnd.AppendLine();

            // Create the node
            SourceNode sn = new SourceNode(sb.ToString(), sbEnd.ToString(), itd);
            AddChild(sn);
            nodeStack.Push(sn);
            currTab++;

            // Generate and attach the source for the fields...
            AttachFieldDeclarationCollection(itd.Fields);

            // ... the properties ...
            AttachPropertyDeclarationCollection(itd.Properties);

            // ...the nested types ...
            AttachTypeDeclarationCollection(itd.NestedTypes);

            // ... and the methods ....
            AttachMethodDeclarationCollection(itd.Methods);

            // ... and the events ....
            AttachEventDeclarationCollection(itd.Events);

            currTab--;
            nodeStack.Pop();
            CurrentNamespace = oldNamespace;

            return sn;
        }

        /// <summary>
        /// IBlockStatement start string
        /// </summary>
        /// <param name="ibs">IBlockStatement</param>
        /// <returns></returns>
        protected override string BlockStatementStart(IBlockStatement ibs)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("{");
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// IBlockStatement end string
        /// </summary>
        /// <param name="ibs">IBlockStatement</param>
        /// <returns></returns>
        protected override string BlockStatementEnd(IBlockStatement ibs)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// IBreakStatement start string
        /// </summary>
        /// <param name="ibs">IBreakStatement</param>
        /// <returns></returns>
        protected override string BreakStatementStart(IBreakStatement ibs)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("break;");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// ICommentStatement start string
        /// </summary>
        /// <param name="ics">ICommentStatement</param>
        /// <returns></returns>
        protected override string CommentStatementStart(ICommentStatement ics)
        {
            StringBuilder sb = new StringBuilder();
            string comment = ics.Comment.Text;
            comment = comment.Replace('\r', ' ');
            comment = comment.Replace('\n', ' ');
            sb.Append(GetTabString());
            sb.Append("// ");
            sb.AppendLine(comment);
            return sb.ToString();
        }

        /// <summary>
        /// IConditionStatement IfThen start string
        /// </summary>
        /// <param name="ics">IConditionStatement</param>
        /// <returns></returns>
        protected override string IfStatementStart(IConditionStatement ics)
        {
            StringBuilder sbIfThen = new StringBuilder();
            sbIfThen.Append(GetTabString());
            sbIfThen.Append("if (");
            AppendExpression(sbIfThen, ics.Condition);
            sbIfThen.Append(")");
            if (PutOpenBraceOnNewLine)
            {
                sbIfThen.AppendLine();
                sbIfThen.Append(GetTabString());
            }
            else
            {
                sbIfThen.Append(" ");
            }
            sbIfThen.Append("{");
            sbIfThen.AppendLine();
            return sbIfThen.ToString();
        }

        /// <summary>
        /// IConditionStatement IfThen end string
        /// </summary>
        /// <param name="ics">IConditionStatement</param>
        /// <returns></returns>
        protected override string IfStatementEnd(IConditionStatement ics)
        {
            StringBuilder sbIfThenEnd = new StringBuilder();
            sbIfThenEnd.Append(GetTabString());
            sbIfThenEnd.Append("}");
            sbIfThenEnd.AppendLine();
            return sbIfThenEnd.ToString();
        }

        /// <summary>
        /// IConditionStatement Else start string
        /// </summary>
        /// <param name="ics">IConditionStatement</param>
        /// <returns></returns>
        protected override string ElseStatementStart(IConditionStatement ics)
        {
            StringBuilder sbElse = new StringBuilder();
            sbElse.Append(GetTabString());
            sbElse.Append("else");
            sbElse.AppendLine();
            sbElse.Append(GetTabString());
            sbElse.Append("{");
            sbElse.AppendLine();
            return sbElse.ToString();
        }

        /// <summary>
        /// IConditionStatement Else end string
        /// </summary>
        /// <param name="ics">IConditionStatement</param>
        /// <returns></returns>
        protected override string ElseStatementEnd(IConditionStatement ics)
        {
            StringBuilder sbElseEnd = new StringBuilder();
            sbElseEnd.Append(GetTabString());
            sbElseEnd.Append("}");
            sbElseEnd.AppendLine();
            return sbElseEnd.ToString();
        }

        /// <summary>
        /// IContinueStatement start string
        /// </summary>
        /// <param name="ics">IContinueStatement</param>
        /// <returns></returns>
        protected override string ContinueStatementStart(IContinueStatement ics)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("continue;");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IExpressionStatement start string
        /// </summary>
        /// <param name="ies">IExpressionStatement</param>
        /// <returns></returns>
        protected override string ExpressionStatementStart(IExpressionStatement ies)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            AppendExpression(sb, ies.Expression);
            sb.Append(";");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IExpressionStatement end string
        /// </summary>
        /// <param name="ies">IExpressionStatement</param>
        /// <returns></returns>
        protected override string ExpressionStatementEnd(IExpressionStatement ies)
        {
            return "";
        }

        /// <summary>
        /// IForEachStatement start string
        /// </summary>
        /// <param name="ifes">IForEachStatement</param>
        /// <returns></returns>
        protected override string ForEachStatementStart(IForEachStatement ifes)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("foreach(");
            AppendType(sb, ifes.Variable.VariableType);
            sb.Append(" ");
            sb.Append(ValidIdentifier(ifes.Variable.Name));
            sb.Append(" in ");
            AppendExpression(sb, ifes.Expression);
            sb.Append(")");
            sb.AppendLine();
            sb.Append(GetTabString());
            sb.Append("{");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IForEachStatement end string
        /// </summary>
        /// <param name="ifes">IForEachStatement</param>
        /// <returns></returns>
        protected override string ForEachStatementEnd(IForEachStatement ifes)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// IForStatement start string
        /// </summary>
        /// <param name="ifs">IForStatement</param>
        /// <returns></returns>
        protected override string ForStatementStart(IForStatement ifs)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("for(");
            if (ifs.Initializer != null)
            {
                IExpressionStatement ies = ifs.Initializer as IExpressionStatement;
                if (ies == null)
                    throw new NotSupportedException("C# Language Writer: For statement initialiser must be an expression statement");
                AppendExpression(sb, ies.Expression);
            }
            sb.Append("; ");
            AppendExpression(sb, ifs.Condition);
            sb.Append("; ");
            if (ifs.Increment != null)
            {
                IExpressionStatement ies = ifs.Increment as IExpressionStatement;
                if (ies == null)
                    throw new NotSupportedException("C# Language Writer: For statement increment must be an expression statement");
                AppendExpression(sb, ies.Expression);
            }
            sb.Append(")");
            if (PutOpenBraceOnNewLine)
            {
                sb.AppendLine();
                sb.Append(GetTabString());
            }
            else
            {
                sb.Append(" ");
            }
            sb.Append("{");
            if (ifs is IBrokenForStatement)
                sb.Append(" // broken");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IForStatement end string
        /// </summary>
        /// <param name="ifs">IForStatement</param>
        /// <returns></returns>
        protected override string ForStatementEnd(IForStatement ifs)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// IRepeatStatement start string
        /// </summary>
        /// <param name="irs">IRepeatStatement</param>
        /// <returns></returns>
        protected override string RepeatStatementStart(IRepeatStatement irs)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("repeat(");
            AppendExpression(sb, irs.Count);
            sb.Append(")");
            if (PutOpenBraceOnNewLine)
            {
                sb.AppendLine();
                sb.Append(GetTabString());
            }
            else
            {
                sb.Append(" ");
            }
            sb.Append("{");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IRepeatStatement end string
        /// </summary>
        /// <param name="irs">IRepeatStatement</param>
        /// <returns></returns>
        protected override string RepeatStatementEnd(IRepeatStatement irs)
        {
            return ForStatementEnd(null);
        }

        /// <summary>
        /// IMethodReturnStatement start string
        /// </summary>
        /// <param name="imrs">IMethodReturnStatement</param>
        /// <returns></returns>
        protected override string MethodReturnStatementStart(IMethodReturnStatement imrs)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("return ");
            AppendExpression(sb, imrs.Expression);
            sb.Append(";");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IMethodReturnStatement end string
        /// </summary>
        /// <param name="imrs">IMethodReturnStatement</param>
        /// <returns></returns>
        protected override string MethodReturnStatementEnd(IMethodReturnStatement imrs)
        {
            return "";
        }

        /// <summary>
        /// ISwitchStatement start string
        /// </summary>
        /// <param name="iss">ISwitchStatement</param>
        /// <returns></returns>
        protected override string SwitchStatementStart(ISwitchStatement iss)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("switch (");
            AppendExpression(sb, iss.Expression);
            sb.Append(")");
            sb.AppendLine();
            sb.Append(GetTabString());
            sb.Append("{");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// ISwitchStatement end string
        /// </summary>
        /// <param name="iss">ISwitchStatement</param>
        /// <returns></returns>
        protected override string SwitchStatementEnd(ISwitchStatement iss)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// ISwitchCase start string
        /// </summary>
        /// <param name="isc">ISwitchCase</param>
        /// <returns></returns>
        protected override string SwitchCaseStart(ISwitchCase isc)
        {
            StringBuilder csb = new StringBuilder();
            csb.Append(GetTabString());
            if (isc is IConditionCase)
            {
                csb.Append("case ");
                AppendExpression(csb, (isc as IConditionCase).Condition);
                csb.Append(":");
            }
            else
            {
                csb.AppendLine("default:");
            }
            csb.AppendLine();
            return csb.ToString();
        }

        /// <summary>
        /// ISwitchCase end string
        /// </summary>
        /// <param name="isc">ISwitchCase</param>
        /// <returns></returns>
        protected override string SwitchCaseEnd(ISwitchCase isc)
        {
            return "";
        }

        /// <summary>
        /// Throw exception start string
        /// </summary>
        /// <param name="ites">IThrowExceptionStatement</param>
        /// <returns></returns>
        protected override string ThrowExceptionStart(IThrowExceptionStatement ites)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("throw ");
            AppendExpression(sb, ites.Expression);
            sb.Append(";");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// Throw exception end string
        /// </summary>
        /// <param name="ites">IThrowExceptionStatement</param>
        /// <returns></returns>
        protected override string ThrowExceptionEnd(IThrowExceptionStatement ites)
        {
            return "";
        }

        /// <summary>
        /// ITryCatchFinally start string for Try block
        /// </summary>
        /// <param name="itcfs">ITryCatchFinally</param>
        /// <returns></returns>
        protected override string TryBlockStart(ITryCatchFinallyStatement itcfs)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("try");
            sb.AppendLine();
            sb.Append(GetTabString());
            sb.Append("{");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// ITryCatchFinally end string for Try block
        /// </summary>
        /// <param name="itcfs">ITryCatchFinally</param>
        /// <returns></returns>
        protected override string TryBlockEnd(ITryCatchFinallyStatement itcfs)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// ITryCatchFinally start string for Fault block
        /// </summary>
        /// <param name="itcfs">ITryCatchFinally</param>
        /// <returns></returns>
        protected override string FaultBlockStart(ITryCatchFinallyStatement itcfs)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("catch");
            sb.AppendLine();
            sb.Append(GetTabString());
            sb.Append("{");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// ITryCatchFinally end string for Fault block
        /// </summary>
        /// <param name="itcfs">ITryCatchFinally</param>
        /// <returns></returns>
        protected override string FaultBlockEnd(ITryCatchFinallyStatement itcfs)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }


        /// <summary>
        /// Catch clause start
        /// </summary>
        /// <param name="icc">catch clause</param>
        /// <returns></returns>
        protected override string CatchClauseStart(ICatchClause icc)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("catch (");
            AppendExpression(sb, icc.Condition);
            sb.Append(")");
            sb.AppendLine();
            sb.Append(GetTabString());
            sb.Append("{");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// Catch clause end
        /// </summary>
        /// <param name="icc">catch clause</param>
        /// <returns></returns>
        protected override string CatchClauseEnd(ICatchClause icc)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// ITryCatchFinally start string for Finally block
        /// </summary>
        /// <param name="itcfs">ITryCatchFinally</param>
        /// <returns></returns>
        protected override string FinallyBlockStart(ITryCatchFinallyStatement itcfs)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("finally");
            sb.AppendLine();
            sb.Append(GetTabString());
            sb.Append("{");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// ITryCatchFinally end string for Finally block
        /// </summary>
        /// <param name="itcfs">ITryCatchFinally</param>
        /// <returns></returns>
        protected override string FinallyBlockEnd(ITryCatchFinallyStatement itcfs)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// UsingStatementStart start string
        /// </summary>
        /// <param name="ius">UsingStatementStart</param>
        /// <returns></returns>
        protected override string UsingStatementStart(IUsingStatement ius)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("using (");
            AppendExpression(sb, ius.Expression);
            sb.Append(")");
            sb.AppendLine();
            sb.Append(GetTabString());
            sb.Append("{");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// UsingStatementStart end string
        /// </summary>
        /// <param name="ius">UsingStatementStart</param>
        /// <returns></returns>
        protected override string UsingStatementEnd(IUsingStatement ius)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// IWhileStatement start string
        /// </summary>
        /// <param name="iws">IWhileStatement</param>
        /// <returns></returns>
        protected override string WhileStatementStart(IWhileStatement iws)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            if (iws is IFusedBlockStatement)
                sb.Append("fused (");
            else
                sb.Append("while (");
            AppendExpression(sb, iws.Condition);
            sb.Append(")");
            sb.AppendLine();
            sb.Append(GetTabString());
            sb.Append("{");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IWhileStatement end string
        /// </summary>
        /// <param name="iws">IWhileStatement</param>
        /// <returns></returns>
        protected override string WhileStatementEnd(IWhileStatement iws)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// IMethodDeclaration start string
        /// </summary>
        /// <param name="imd">IMethodDeclaration</param>
        /// <returns></returns>
        protected override string MethodDeclarationStart(IMethodDeclaration imd)
        {
            StringBuilder sb = new StringBuilder();

            // Documentation
            AppendDocumentation(sb, imd.Documentation);

            // Custom attributes
            foreach (ICustomAttribute attr in imd.Attributes)
            {
                AppendAttribute(sb, attr);
                sb.AppendLine();
            }

            sb.Append(GetTabString());

            // Public or private
            switch (imd.Visibility)
            {
                case MethodVisibility.Private:
                    sb.Append("private ");
                    break;
                case MethodVisibility.Public:
                    sb.Append("public ");
                    break;
                case MethodVisibility.Assembly:
                    sb.Append("internal ");
                    break;
                default:
                    throw new NotImplementedException("unhandled visibility " + imd.Visibility);
            }
            if (imd.Abstract)
                sb.Append("abstract ");
            if (imd.Virtual)
            {
                if (imd.NewSlot)
                    sb.Append("virtual ");
                else
                    sb.Append("override ");
            }
            if (imd.Static)
                sb.Append("static ");

            if (imd is IConstructorDeclaration) sb.Append(((ITypeDeclaration) imd.DeclaringType).Name);
            else
            {
                // The return type
                AppendType(sb, imd.ReturnType.Type);
                sb.Append(" ");

                // The method name
                string methodName = ValidIdentifier(imd.Name);
                sb.Append(methodName);
            }

            // The generic argument types
            //if (Builder.IsMethodInstRef(imd))
            AppendGenericArguments(sb, imd.GenericArguments);
            sb.Append("(");

            // Append the parameters
            if (imd.Parameters != null)
            {
                AppendParameterDeclarationCollection(sb, imd.Parameters);
            }
            sb.Append(")");
            sb.AppendLine();
            sb.Append(GetTabString());
            sb.Append("{");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IMethodDeclaration end string
        /// </summary>
        /// <param name="imd">IMethodDeclaration</param>
        /// <returns></returns>
        protected override string MethodDeclarationEnd(IMethodDeclaration imd)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// IFieldDeclaration start string
        /// </summary>
        /// <param name="ifd">IFieldDeclaration</param>
        /// <returns></returns>
        protected override string FieldDeclarationStart(IFieldDeclaration ifd)
        {
            string fieldName = ValidIdentifier(ifd.Name);

            StringBuilder sb = new StringBuilder();

            // Documentation
            if (ifd.Documentation != null)
            {
                sb.Append(GetTabString());
                sb.AppendLine("/// <summary>" + ifd.Documentation + "</summary>");
            }

            // Custom attributes
            foreach (ICustomAttribute attr in ifd.Attributes)
            {
                AppendAttribute(sb, attr);
                sb.AppendLine();
            }

            sb.Append(GetTabString());

            // Public or private
            switch (ifd.Visibility)
            {
                case FieldVisibility.Private:
                    sb.Append("private ");
                    break;
                default:
                    sb.Append("public ");
                    break;
            }

            // Static
            if (ifd.Static)
            {
                sb.Append("static ");
            }

            // Type and name of the field
            AppendType(sb, ifd.FieldType);
            sb.Append(" ");
            sb.Append(fieldName);

            // Initializer
            if (ifd.Initializer != null)
            {
                sb.Append(" = ");
                AppendExpression(sb, ifd.Initializer);
            }
            sb.Append(";");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IFieldDeclaration end string
        /// </summary>
        /// <param name="ifd">IFieldDeclaration</param>
        /// <returns></returns>
        protected override string FieldDeclarationEnd(IFieldDeclaration ifd)
        {
            return "";
        }

        /// <summary>
        /// IPropertyDeclarationCollection start string
        /// </summary>
        /// <param name="ipdc">IPropertyDeclarationCollection</param>
        /// <returns></returns>
        protected override string PropertyDeclarationCollectionStart(List<IPropertyDeclaration> ipdc)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("#region Properties");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IPropertyDeclarationCollection end string
        /// </summary>
        /// <param name="ipdc">IPropertyDeclarationCollection</param>
        /// <returns></returns>
        protected override string PropertyDeclarationCollectionEnd(List<IPropertyDeclaration> ipdc)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("#endregion");
            sbEnd.AppendLine();
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// IPropertyDeclaration start string
        /// </summary>
        /// <param name="ipd">IPropertyDeclaration</param>
        /// <returns></returns>
        protected override string PropertyDeclarationStart(IPropertyDeclaration ipd)
        {
            string propertyName = ValidIdentifier(ipd.Name);

            StringBuilder sb = new StringBuilder();

            // Documentation
            if (ipd.Documentation != null)
            {
                sb.Append(GetTabString());
                sb.AppendLine("/// <summary>" + ipd.Documentation + "</summary>");
            }

            // Custom attributes
            foreach (ICustomAttribute attr in ipd.Attributes)
            {
                AppendAttribute(sb, attr);
                sb.AppendLine();
            }

            sb.Append(GetTabString());
            sb.Append("public ");
            AppendType(sb, ipd.PropertyType);
            sb.Append(" ");
            sb.Append(propertyName);
            sb.AppendLine();
            sb.Append(GetTabString());
            sb.Append("{");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IPropertyDeclaration end string
        /// </summary>
        /// <param name="ipd">IPropertyDeclaration</param>
        /// <returns></returns>
        protected override string PropertyDeclarationEnd(IPropertyDeclaration ipd)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("}");
            sbEnd.AppendLine();
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// IEventDeclaration start string
        /// </summary>
        /// <param name="ied">IEventDeclaration</param>
        /// <returns></returns>
        protected override string EventDeclarationStart(IEventDeclaration ied)
        {
            string eventName = ValidIdentifier(ied.Name);

            StringBuilder sb = new StringBuilder();

            // Documentation
            if (ied.Documentation != null)
            {
                sb.Append(GetTabString());
                sb.AppendLine("/// <summary>" + ied.Documentation + "</summary>");
            }

            // Custom attributes
            foreach (ICustomAttribute attr in ied.Attributes)
            {
                AppendAttribute(sb, attr);
                sb.AppendLine();
            }

            sb.Append(GetTabString());
            sb.Append("public ");
            sb.Append("event ");
            AppendTypeReference(sb, ied.EventType);
            sb.Append(" ");
            sb.Append(eventName);
            sb.Append(";");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IEventDeclaration end string
        /// </summary>
        /// <param name="ied">IEventDeclaration</param>
        /// <returns></returns>
        protected override string EventDeclarationEnd(IEventDeclaration ied)
        {
            return "";
        }

        /// <summary>
        /// ITypeDeclarationCollection start string
        /// </summary>
        /// <param name="intdc">ITypeDeclarationCollection</param>
        /// <returns></returns>
        protected override string TypeDeclarationCollectionStart(List<ITypeDeclaration> intdc)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("#region Nested types");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// ITypeDeclarationCollection end string
        /// </summary>
        /// <param name="intdc">ITypeDeclarationCollection</param>
        /// <returns></returns>
        protected override string TypeDeclarationCollectionEnd(List<ITypeDeclaration> intdc)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("#endregion");
            sbEnd.AppendLine();
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// IFieldDeclarationCollection start string
        /// </summary>
        /// <param name="ifdc">IFieldDeclarationCollection</param>
        /// <returns></returns>
        protected override string FieldDeclarationCollectionStart(List<IFieldDeclaration> ifdc)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("#region Fields");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IFieldDeclarationCollection end string
        /// </summary>
        /// <param name="ifdc">IFieldDeclarationCollection</param>
        /// <returns></returns>
        protected override string FieldDeclarationCollectionEnd(List<IFieldDeclaration> ifdc)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("#endregion");
            sbEnd.AppendLine();
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// IMethodDeclarationCollection start string
        /// </summary>
        /// <param name="imdc">IMethodDeclarationCollection</param>
        /// <returns></returns>
        protected override string MethodDeclarationCollectionStart(List<IMethodDeclaration> imdc)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("#region Methods");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IMethodDeclarationCollection end string
        /// </summary>
        /// <param name="imdc">IMethodDeclarationCollection</param>
        /// <returns></returns>
        protected override string MethodDeclarationCollectionEnd(List<IMethodDeclaration> imdc)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("#endregion");
            sbEnd.AppendLine();
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// IEventDeclarationCollection start string
        /// </summary>
        /// <param name="iedc">IEventDeclarationCollection</param>
        /// <returns></returns>
        protected override string EventDeclarationCollectionStart(List<IEventDeclaration> iedc)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(GetTabString());
            sb.Append("#region Events");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// IEventDeclarationCollection end string
        /// </summary>
        /// <param name="iedc">IEventDeclarationCollection</param>
        /// <returns></returns>
        protected override string EventDeclarationCollectionEnd(List<IEventDeclaration> iedc)
        {
            StringBuilder sbEnd = new StringBuilder();
            sbEnd.Append(GetTabString());
            sbEnd.Append("#endregion");
            sbEnd.AppendLine();
            sbEnd.AppendLine();
            return sbEnd.ToString();
        }

        /// <summary>
        /// True if the given double is negative zero (-0.0).
        /// </summary>
        /// <param name="x">A double.</param>
        /// <returns>True, if the given double is negative zero (-0.0).</returns>
        private bool IsNegativeZero(double x)
        {
            return BitConverter.DoubleToInt64Bits(x) == NegativeZeroBits;
        }
    }
}