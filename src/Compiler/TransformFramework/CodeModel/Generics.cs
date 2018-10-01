// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A static class providing a GetType method to convert generic types to specific types
    /// based on the specified generic argument providers
    /// </summary>
    internal class Generics
    {
        /// <summary>
        /// Use the generic argument providers to specialize a type
        /// </summary>
        /// <param name="type">The type to resolve</param>
        /// <param name="declaringType">The declaring type as a generic argument provider</param>
        /// <param name="method">The method as a generic argument provider</param>
        /// <returns></returns>
        public static IType GetType(IType type, IGenericArgumentProvider declaringType, IGenericArgumentProvider method)
        {
            ITypeReference itr = type as ITypeReference;
            if (itr != null)
            {
                if (itr.GenericType == null)
                    // The parameter is directly a non-generic type reference
                    return itr;
                else
                {
                    // The parameter is a reference to a generic type. Build an array of types of the
                    // generic arguments by recursion
                    IType[] ita = new IType[itr.GenericArguments.Count];
                    for (int i = 0; i < ita.Length; i++)
                    {
                        ita[i] = GetType(itr.GenericArguments[i], declaringType, method);
                    }
                    // Build the specialization (the 'instance') of the generic type
                    ITypeReference itr2 = new XTypeInstanceReference();
                    itr2.GenericType = itr.GenericType;
                    itr2.GenericArguments.AddRange(ita);
                    ((ISettableTypeDeclaration) itr2).Declaration = itr.Resolve();
                    return itr2;
                }
            }

            // Check to see if the parameter is an array
            IArrayType iat = type as IArrayType;
            if (iat != null)
            {
                // The parameter is an array type. Build an array type and recurse to get its element type
                IArrayType iat2 = new XArrayType();
                iat2.ElementType = GetType(iat.ElementType, declaringType, method);
                iat2.DotNetType = iat.DotNetType;
                return iat2;
            }

            // Check to see if the parameter is a pointer
            IPointerType ipt = type as IPointerType;
            if (ipt != null)
            {
                // The parameter is a pointer type. Build a pointer type and recurse to set its element type
                IPointerType ipt2 = new XPointerType();
                ipt2.ElementType = GetType(ipt.ElementType, declaringType, method);
                return ipt2;
            }

            // Check to see if the parameter is a reference
            IReferenceType irt = type as IReferenceType;
            if (irt != null)
            {
                // The parameter is a reference type. Build a reference type and recurse
                // to set its element type
                IReferenceType irt2 = new XReferenceType();
                irt2.ElementType = GetType(irt.ElementType, declaringType, method);
                return irt2;
            }

            // Check to see if the parameter has an optional modifier
            IOptionalModifier iom = type as IOptionalModifier;
            if (iom != null)
            {
                // The parameter has an optional modifier. Build an optional modifier type and recurse
                // to set its element type
                IOptionalModifier iom2 = new XOptionalModifier();
                iom2.Modifier = (ITypeReference) GetType(iom.Modifier, declaringType, method);
                iom2.ElementType = GetType(iom.ElementType, declaringType, method);
                return iom2;
            }

            // Check to see if the parameter has an required modifier
            IRequiredModifier irm = type as IRequiredModifier;
            if (irm != null)
            {
                // The parameter has a required modifier. Build a required modifier type and recurse
                // to set its element type
                IRequiredModifier irm2 = new XRequiredModifier();
                irm2.Modifier = (ITypeReference) GetType(irm.Modifier, declaringType, method);
                irm2.ElementType = GetType(irm.ElementType, declaringType, method);
                return irm2;
            }

            // Deal with generic parameters
            IGenericParameter igp = type as IGenericParameter;
            IMethodReference imr;
            if (igp != null)
            {
                itr = igp.Owner as ITypeReference;
                imr = igp.Owner as IMethodReference;
                ;
                if (itr == null && imr == null)
                    throw new NotSupportedException("A generic parameter must be owned by a method or type");

                if (itr == null)
                {
                    if (method != null)
                        // Get the parameter type from the method instance
                        return method.GenericArguments[igp.Position];
                    else
                        return igp;
                }
                else
                {
                    if (declaringType != null)
                        // Get the parameter type from the declaring type
                        return declaringType.GenericArguments[igp.Position];
                    else
                        return igp;
                }
            }

            // The only thing left is that the parameter is a generic argument
            IGenericArgument iga = type as IGenericArgument;
            if (iga == null)
                throw new NotSupportedException("Unable to get the parameters type");

            IType it = iga.Resolve();
            if (it == null || !(it is IGenericArgument) || (it is IGenericParameter))
            {
                itr = iga.Owner as ITypeReference;
                imr = iga.Owner as IMethodReference;
                if (itr == null && imr == null)
                    throw new NotSupportedException();

                if (itr == null)
                {
                    if (method != null)
                    {
                        IGenericArgument iga2 = new XGenericArgument();
                        iga2.Owner = method;
                        iga2.Position = iga.Position;
                        return iga2;
                    }
                    else
                        return iga;
                }
                else
                {
                    IGenericArgument iga2 = new XGenericArgument();
                    iga2.Owner = declaringType;
                    iga2.Position = iga.Position;
                    return iga2;
                }
            }

            // Recurse
            return GetType(it, declaringType, method);
        }
    }
}