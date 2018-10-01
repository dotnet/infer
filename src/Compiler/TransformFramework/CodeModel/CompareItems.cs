// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A static class with various methods for comparison and equality testing
    /// </summary>
    public static class CompareItems
    {
        /// <summary>
        /// Compare two type references
        /// </summary>
        /// <param name="itra">Type reference a</param>
        /// <param name="itrb">Type reference b</param>
        /// <returns>-1, 0, or 1</returns>
        internal static int CompareTypeReferences(ITypeReference itra, ITypeReference itrb)
        {
            if (itra == null || itrb == null)
            {
                throw new NotSupportedException();
            }
            int ret = String.Compare(itra.Name, itrb.Name, StringComparison.InvariantCulture);
            if (0 == ret)
                ret = String.Compare(itra.Namespace, itrb.Namespace, StringComparison.InvariantCulture);
            if (0 == ret)
                ret = itra.GenericArguments.Count.CompareTo(itrb.GenericArguments.Count);

            // RTODO: Should we do something with owner also?
            return ret;
        }

        /// <summary>
        /// Determine if two method signatures are equal
        /// </summary>
        /// <param name="imsRef">Reference method signature</param>
        /// <param name="imsDecl">Method declaration signature</param>
        /// <returns></returns>
        internal static bool MethodSignaturesAreEqual2(IMethodSignature imsRef, IMethodSignature imsDecl)
        {
            MethodBase refMI = imsRef.MethodInfo;
            MethodBase decMI = imsDecl.MethodInfo;

            if (refMI != null && refMI.Equals(decMI))
                return true;
            else
                return false;
        }

        /// <summary>
        /// Determine if two method signatures are equal
        /// </summary>
        /// <param name="imsRef">Reference method signature</param>
        /// <param name="imsDecl">Method declaration signature</param>
        /// <returns></returns>
        internal static bool MethodSignaturesAreEqual(IMethodSignature imsRef, IMethodSignature imsDecl)
        {
            // Look at the parameters
            IList<IParameterDeclaration> parmsRef = imsRef.Parameters;
            IList<IParameterDeclaration> parmsDecl = imsDecl.Parameters;

            if (parmsRef.Count != parmsDecl.Count)
            {
                return false;
            }
            else
            {
                for (int i = 0; i < parmsDecl.Count; i++)
                {
                    IType pct = parmsRef[i].ParameterType;
                    IType pdt = parmsDecl[i].ParameterType;
                    if (pct == null)
                    {
                        if (pdt != null)
                            return false;
                    }
                    else if (!TypesAreEqual(pct, pdt))
                        return false;
                }
            }

            // Everything checks out - last thing to do is check the return type
            return TypesAreEqual(imsRef.ReturnType.Type, imsDecl.ReturnType.Type);
        }

        /// <summary>
        /// Determine if two method references are equal, ignoring declaring type
        /// </summary>
        /// <param name="imr1">Method reference 1</param>
        /// <param name="imr2">Method reference 2</param>
        /// <returns>true if equal</returns>
        internal static bool MethodReferencesAreEqualInner(IMethodReference imr1, IMethodReference imr2)
        {
            // Nme and generic arguments must be the same...
            if (imr1.Name != imr2.Name ||
                imr1.GenericArguments.Count != imr2.GenericArguments.Count)
                return false;

            // ... and generic method defining this specialization (if it is one)
            IMethodReference genericRef1 = imr1.GenericMethod;
            IMethodReference genericRef2 = imr2.GenericMethod;
            if (genericRef1 == null)
            {
                if (genericRef2 != null)
                    return false;
            }
            else
            {
                if (genericRef2 == null)
                    return false;

                if (!genericRef1.Equals(genericRef2))
                    return false;

                for (int i = 0; i < imr2.GenericArguments.Count; i++)
                {
                    IType tRef1 = imr1.GenericArguments[i];
                    IType tRef2 = imr2.GenericArguments[i];
                    if (tRef1 == null || tRef2 == null || !tRef1.Equals(tRef2))
                        return false;
                }
            }

            // ... and method signatures
            if (!CompareItems.MethodSignaturesAreEqual(imr1, imr2))
                return false;

            return true;
        }

        /// <summary>
        /// Determine if two method references are equal
        /// </summary>
        /// <param name="imr1">First method reference</param>
        /// <param name="imr2">Second method reference</param>
        /// <returns>true if equal</returns>
        internal static bool MethodReferencesAreEqual(IMethodReference imr1, IMethodReference imr2)
        {
            if (imr1 == imr2)
                return true;

            // Method references and declaring types must be the same
            return (CompareItems.MethodReferencesAreEqualInner(imr1, imr2) &&
                    imr1.DeclaringType.Equals(imr2.DeclaringType));
        }

        /// <summary>
        /// Compare two method references
        /// </summary>
        /// <param name="imr1">First method reference</param>
        /// <param name="imr2">Second method reference</param>
        /// <returns>-1, 0 or 1</returns>
        internal static int CompareMethodReferences(IMethodReference imr1, IMethodReference imr2)
        {
            int ret = imr1.DeclaringType.CompareTo(imr2.DeclaringType);
            if (ret != 0)
                return ret;

            if ((ret = String.Compare(imr1.Name, imr2.Name, StringComparison.InvariantCulture)) != 0)
                return ret;

            if ((ret = imr1.Parameters.Count.CompareTo(imr2.Parameters.Count)) != 0)
                return ret;

            for (int i = 0; i < imr1.Parameters.Count; i++)
            {
                if ((ret = imr1.Parameters[i].ParameterType.CompareTo(imr2.Parameters[i].ParameterType)) != 0)
                    return ret;
            }

            if ((ret = imr1.ReturnType.Type.CompareTo(imr2.ReturnType.Type)) != 0)
                return ret;

            if ((ret = imr1.GenericArguments.Count.CompareTo(imr2.GenericArguments.Count)) != 0)
                return ret;

            return 0;
        }

        /// <summary>
        /// Compare two array types
        /// </summary>
        /// <param name="iat1">First array type</param>
        /// <param name="iat2">Second array type</param>
        /// <returns>-1, 0 or 1</returns>
        internal static int CompareArrayTypes(IArrayType iat1, IArrayType iat2)
        {
            if (iat1.Rank > iat2.Rank)
                return 1;
            else if (iat1.Rank < iat2.Rank)
                return -1;
            else
                return iat1.ElementType.CompareTo(iat2.ElementType);
        }

        /// <summary>
        /// Determine if two array types are equal
        /// </summary>
        /// <param name="iat1">Reference method signature</param>
        /// <param name="iat2">Method declaration signature</param>
        /// <returns></returns>
        internal static bool ArrayTypesAreEqual(IArrayType iat1, IArrayType iat2)
        {
            if (iat1.Rank != iat2.Rank || !iat1.ElementType.Equals(iat2.ElementType))
                return false;
            else
                return true;
        }

        /// <summary>
        /// Determine if two ITypes are equal
        /// </summary>
        /// <param name="it1">First IType</param>
        /// <param name="it2">Second IType</param>
        /// <returns></returns>
        internal static bool TypesAreEqual(IType it1, IType it2)
        {
            if (it1 is IArrayType && it2 is IArrayType)
            {
                return ((it1 as IArrayType).Equals(it2 as IArrayType));
            }
            else if (it1 is ITypeReference && it2 is ITypeReference)
            {
                return ((it1 as ITypeReference).Equals(it2 as ITypeReference));
            }
            else if (it1 is IReferenceType && it2 is IReferenceType)
            {
                return ((it1 as IReferenceType).Equals(it2 as IReferenceType));
            }
            else if (it1 is IGenericParameter && it2 is IGenericParameter)
            {
                return ((it1 as IGenericParameter).Equals(it2 as IGenericParameter));
            }
            else if (it1 is IGenericArgument && it2 is IGenericArgument)
            {
                return ((it1 as IGenericArgument).Equals(it2 as IGenericArgument));
            }
            else if (it1 is IPointerType && it2 is IPointerType)
            {
                return ((it1 as IPointerType).Equals(it2 as IPointerType));
            }
            else if (it1 is IRequiredModifier && it2 is IRequiredModifier)
            {
                return ((it1 as IRequiredModifier).Equals(it2 as IRequiredModifier));
            }
            else
                return it1 == it2;
        }

        /// <summary>
        /// Compare two property references
        /// </summary>
        /// <param name="ipr1">First property reference</param>
        /// <param name="ipr2">Second property reference</param>
        /// <returns>-1, 0 or 1</returns>
        internal static int ComparePropertyReferences(IPropertyReference ipr1, IPropertyReference ipr2)
        {
            int ret = ipr1.DeclaringType.CompareTo(ipr2.DeclaringType);
            if (ret != 0)
                return ret;

            if ((ret = String.Compare(ipr1.Name, ipr2.Name, StringComparison.InvariantCulture)) != 0)
                return ret;

            if ((ret = ipr1.PropertyType.CompareTo(ipr2.PropertyType)) != 0)
                return ret;

            if ((ret = ipr1.Parameters.Count.CompareTo(ipr2.Parameters.Count)) != 0)
                return ret;

            for (int i = 0; i < ipr1.Parameters.Count; i++)
            {
                if ((ret = ipr1.Parameters[i].ParameterType.CompareTo(ipr1.Parameters[i].ParameterType)) != 0)
                    return ret;
            }

            return 0;
        }

        /// <summary>
        /// Determine if two property references are equal
        /// </summary>
        /// <param name="ipr1">Property reference 1</param>
        /// <param name="ipr2">Property reference 2</param>
        /// <returns>true if equal</returns>
        internal static bool PropertyReferencesAreEqual(IPropertyReference ipr1, IPropertyReference ipr2)
        {
            // Names must be the same
            if ((!ipr1.Name.Equals(ipr2.Name)) ||
                (!ipr1.PropertyType.Equals(ipr2.PropertyType)) ||
                (!ipr1.DeclaringType.Equals(ipr2.DeclaringType)) ||
                ipr1.Parameters.Count != ipr2.Parameters.Count)
                return false;

            for (int i = 0; i < ipr1.Parameters.Count; i++)
            {
                if (!ipr1.Parameters[i].Equals(ipr2.Parameters[i]))
                    return false;
            }

            return true;
        }
    }
}