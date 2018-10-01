// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;

namespace Microsoft.ML.Probabilistic.Factors.Attributes
{
    /// <summary>
    /// When applied to an assembly, indicates that the assembly should be searched for message functions.
    /// </summary>
    public class HasMessageFunctionsAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a class, indicates that the class provides message functions for a given factor.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = true)]
    public class FactorMethodAttribute : Attribute
    {
        /// <summary>
        /// Type which contains the factor definition
        /// </summary>
        public Type Type;

        /// <summary>
        /// Method name for the factor
        /// </summary>
        public string MethodName;

        /// <summary>
        /// Parameter types for the factor
        /// </summary>
        public Type[] ParameterTypes;

        /// <summary>
        /// New names for factor parameters overriding the default choice of parameter names
        /// - these are used to name message passing methods
        /// </summary>
        public string[] NewParameterNames;

        /// <summary>
        /// True if this class should override other classes for the same factor.
        /// </summary>
        public bool Default;

        /// <summary>
        /// Creates a new FactorMethod attribute
        /// </summary>
        /// <param name="type">Type which contains the factor definition</param>
        /// <param name="methodName">Method name for the factor</param>
        /// <param name="parameterTypes">Parameter types for the factor</param>
        public FactorMethodAttribute(Type type, string methodName, params Type[] parameterTypes)
        {
            this.Type = type;
            this.MethodName = methodName;
            this.ParameterTypes = parameterTypes;
        }

        /// <summary>
        /// Creates a new FactorMethod attribute
        /// </summary>
        /// <param name="type">Type which contains the factor definition</param>
        /// <param name="methodName">Method name for the factor</param>
        /// <param name="parameterTypes">Parameter types for the factor</param>
        /// <param name="newParameterNames">New names for factor parameters overriding the default choice of
        /// parameter names - these are used to name message passing methods</param>
        public FactorMethodAttribute(string[] newParameterNames, Type type, string methodName, params Type[] parameterTypes)
            : this(type, methodName, parameterTypes)
        {
            this.NewParameterNames = newParameterNames;
        }
    }

    /// <summary>
    /// Marks a factor as hidden
    /// </summary>
    public class HiddenAttribute : Attribute
    {
        /// <summary>
        /// Returns true if the given method has a 'Hidden' attribute
        /// </summary>
        /// <param name="mi">The method info</param>
        /// <returns></returns>
        public static bool IsDefined(MethodInfo mi)
        {
            return Attribute.IsDefined(mi, typeof (HiddenAttribute));
        }

        /// <summary>
        /// Returns true if the given type has a 'Hidden' attribute
        /// </summary>
        /// <param name="ty">The type</param>
        /// <returns></returns>
        public static bool IsDefined(Type ty)
        {
            return Attribute.IsDefined(ty, typeof (HiddenAttribute));
        }
    }

    /// <summary>
    /// Marks a factor as having derivative wrt all inputs equal to 1 always (only applies to deterministic factors)
    /// </summary>
    public class HasUnitDerivative : Attribute
    {
    }

    /// <summary>
    /// When applied to a message operator class, indicates that the message operators may use the named parameters as storage for holding algorithm state.
    /// </summary>
    public class BuffersAttribute : Attribute
    {
        /// <summary>
        /// Names of buffers that may be used as method parameters.
        /// </summary>
        public string[] BufferNames;

        /// <summary>
        /// Creates a new Buffers attribute
        /// </summary>
        /// <param name="names"></param>
        public BuffersAttribute(params string[] names)
        {
            this.BufferNames = names;
        }
    }

    /// <summary>
    /// When applied to a method, overrides the default choice of parameter names.
    /// </summary>
    /// <remarks>
    /// The first parameter is the result.  Thus the method <c>int f(int x)</c> has two parameters, the
    /// result and x.
    /// </remarks>
    public class ParameterNamesAttribute : Attribute
    {
        /// <summary>
        /// The new parameter names
        /// </summary>
        public string[] Names;

        /// <summary>
        /// Creates a new ParameterNames attribute
        /// </summary>
        /// <param name="names"></param>
        public ParameterNamesAttribute(params string[] names)
        {
            Names = names;
        }
    }

    /// <summary>
    /// When applied to a message function parameter, indicates that the function depends on all items
    /// in the message collection except the resultIndex.  The default is all items.
    /// </summary>
    /// <remarks>
    /// This attribute cannot be combined with MatchingIndexAttribute, since they would cancel.
    /// </remarks>
    public class AllExceptIndexAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a message function parameter, indicates that the function depends on the one
    /// item in the message collection at resultIndex.  The default is all items.
    /// </summary>
    public class MatchingIndexAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a message function parameter, indicates that the argument should be indexed by resultIndex.
    /// </summary>
    public class IndexedAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a message function parameter, indicates that the argument's value is cancelled by another argument, thus there is no real dependency.
    /// </summary>
    /// <remarks>
    /// Implies NoInit.
    /// </remarks>
    public class CancelsAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a message function parameter, causes the InducedTarget to depend on the parameter
    /// </summary>
    public class InducedSourceAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a message function parameter, causes the parameter to depend on the InducedSource
    /// </summary>
    public class InducedTargetAttribute : Attribute
    {
    }

    /// <summary>
    /// The distribution must be proper.
    /// </summary>
    /// <remarks>
    /// Applies to message function parameters.  Indicates that the parameter should be a proper distribution,
    /// or else the behavior is undefined.
    /// </remarks>
    public class ProperAttribute : Attribute
    {
    }

    /// <summary>
    /// The result is always uniform.
    /// </summary>
    /// <remarks>
    /// Applies to message functions.  This annotation is optional and allows the inference
    /// engine to skip unnecessary function calls, i.e. ones which would produce uniform results.
    /// </remarks>
    public class SkipAttribute : Attribute
    {
    }

    /// <summary>
    /// The result is uniform (or an exception would be thrown) whenever this parameter is uniform, hence the function call can be skipped.
    /// </summary>
    /// <remarks><para>
    /// Applies to message function parameters.  This annotation is optional and allows the inference
    /// engine to skip unnecessary function calls, i.e. ones which would produce a uniform result or throw
    /// an exception.  For LogAverageFactor and AverageLogFactor, SkipIfUniform means the result would be 0.
    /// </para><para>
    /// When applied to an array parameter, this attribute means the result is uniform (or an exception would be thrown)
    /// whenever all dependent elements in the array are uniform.
    /// For example:
    /// <list type="bullet">
    /// <item><term><c>f([AllExceptIndex,SkipIfUniform] Message[] array, int resultIndex)</c></term>
    /// <description><c>f</c> depends on all elements other than resultIndex, and can be skipped if all of these are uniform.
    /// <c>f</c> does not depend on <c>array[resultIndex]</c> and the uniformity of this element is ignored.
    /// </description>
    /// </item>
    /// </list>
    /// </para></remarks>
    public class SkipIfUniformAttribute : Attribute
    {
    }

    /// <summary>
    /// The result is uniform (or an exception would be thrown) if all elements are uniform, hence the function call can be skipped.
    /// </summary>
    /// <remarks><para>
    /// Applies to message functions and message function array parameters.
    /// This annotation is optional and allows the inference
    /// engine to skip unnecessary function calls, i.e. ones which would produce a uniform result or throw
    /// an exception.
    /// When applied to a message function, it means that the result is uniform (or an exception would be thrown) whenever all parameters to the function are uniform.
    /// When applied to an array parameter, it means that the result is uniform (or an exception would be thrown) whenever all dependent elements in the array are uniform.
    /// Array elements that the function does not depend on are ignored.
    /// </para><para>
    /// It only makes sense to apply this attribute to a method when the method parameters have no SkipIfUniform attributes set.
    /// Otherwise SkipIfAllUniform is automatically implied by SkipIfUniform for the parameter.
    /// </para></remarks>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Parameter, AllowMultiple = true)]
    public class SkipIfAllUniformAttribute : Attribute
    {
        /// <summary>
        /// List of parameter names for which to check for uniformity
        /// </summary>
        public string[] ParameterNames;

        /// <summary>
        /// Creates a new SkipIfAllUniform attribute
        /// </summary>
        public SkipIfAllUniformAttribute()
        {
        }

        /// <summary>
        /// Creates a new SkipIfAllUniform attribute applied to the specified parameter names
        /// </summary>
        public SkipIfAllUniformAttribute(params string[] parameterNames)
        {
            this.ParameterNames = parameterNames;
        }
    }

    /// <summary>
    /// The result is uniform (or an exception would be thrown) if any element is uniform, hence the function call can be skipped.
    /// </summary>
    /// <remarks><para>
    /// Applies to message function array parameters.
    /// This annotation is optional and allows the inference
    /// engine to skip unnecessary function calls, i.e. ones which would produce a uniform result or throw
    /// an exception.
    /// When applied to an array parameter, it means that the result is uniform (or an exception would be thrown) whenever any dependent element in the array is uniform.
    /// Array elements that the function does not depend on are ignored.
    /// When applied to a parameter that is not an array, it has the same meaning as SkipIfUniform.
    /// </para><para>
    /// </para></remarks>
    public class SkipIfAnyUniformAttribute : Attribute
    {
    }

    /// <summary>
    /// The result is uniform (or an exception would be thrown) if any element except
    /// at the result index is uniform, hence the function call can be skipped.
    /// </summary>
    /// <remarks>
    /// Examples:
    /// <list type="bullet">
    /// <item><term><c>f([SkipIfAnyExceptIndexIsUniform] Message[] array, int resultIndex)</c></term>
    /// <description>Since no dependency attribute was given, the default is that <c>f</c> depends on all elements of the array.
    /// Since <c>SkipIfAnyExceptIndexIsUniform</c> was given, <c>f</c> returns uniform (or throws an exception) if any elements except the one at resultIndex is uniform.
    /// Thus <c>array[resultIndex]</c> is a dependency but its uniformity is ignored.</description>
    /// </item>
    /// <item><term><c>f([AllExceptIndex,SkipIfAnyExceptIndexIsUniform] Message[] array, int resultIndex)</c></term>
    /// <description>In this case, <c>f</c> depends on all elements other than resultIndex, and returns uniform (or throws an exception) if any of them are uniform.
    /// Thus it is equivalent to <c>[AllExceptIndex,SkipIfUniform]</c></description>
    /// </item>
    /// <item><term><c>f([MatchingIndex,SkipIfAnyExceptIndexIsUniform] Message[] array, int resultIndex)</c></term>
    /// <description>Here <c>SkipIfAnyExceptIndexIsUniform</c> is ignored since <c>f</c> only depends on <c>array[resultIndex]</c>.
    /// </description>
    /// </item>
    /// </list>
    /// This attribute can be stacked with the other SkipIfUniform attributes, to build up a set of skip cases.
    /// </remarks>
    public class SkipIfAnyExceptIndexIsUniformAttribute : Attribute
    {
    }

    /// <summary>
    /// The result is uniform (or an exception would be thrown) if all elements except
    /// at the result index are uniform, hence the function call can be skipped.
    /// </summary>
    public class SkipIfAllExceptIndexAreUniformAttribute : Attribute
    {
    }

    /// <summary>
    /// The result is uniform (or an exception would be thrown) if the element at
    /// the result index is uniform, hence the function call can be skipped.
    /// </summary>
    public class SkipIfMatchingIndexIsUniformAttribute : Attribute
    {
    }

    /// <summary>
    /// When attached to a method parameter, indicates that the dependency should be ignored by the FactorManager.  Only a declaration dependency will be retained.
    /// </summary>
    public class IgnoreDependencyAttribute : Attribute
    {
    }

    /// <summary>
    /// When attached to a method parameter, indicates that the dependency and the declaration dependency should be ignored by the FactorManager.
    /// </summary>
    public class IgnoreDeclarationAttribute : Attribute
    {
    }

    /// <summary>
    /// When attached to a factor parameter, indicates that the parameter
    /// is constant - i.e. cannot be changed by observation or by inference
    /// </summary>
    [AttributeUsage(AttributeTargets.Parameter)]
    public class ConstantAttribute : Attribute
    {
        /// <summary>
        /// Returns true if this parameter has a Constant attribute
        /// </summary>
        /// <param name="pi"></param>
        /// <returns></returns>
        public static bool IsDefined(ParameterInfo pi)
        {
            return Attribute.IsDefined(pi, typeof (ConstantAttribute));
        }
    }

    /// <summary>
    /// When applied to a method argument, indicates that argument is required to have
    /// been set before calling the method
    /// </summary>
    public class RequiredArgumentAttribute : Attribute
    {
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "RequiredArgument";
        }
    }

    /// <summary>
    /// When applied to a method argument, indicates that the argument will be returned unmodified
    /// as the result of the method.  This automatically implies SkipIfUniform and Trigger.
    /// </summary>
    /// <remarks>
    /// This attribute allows significant optimisations to be undertaken in the compiler.
    /// </remarks>
    [AttributeUsage(AttributeTargets.Parameter)]
    public class IsReturnedAttribute : Attribute
    {
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "IsReturned";
        }
    }

    /// <summary>
    /// When applied to a method argument for methods that return lists, indicates that all elements
    /// of the returned list will be identical and equal to that argument.  This automatically implies SkipIfUniform.
    /// </summary>
    /// <remarks>
    /// This attribute allows significant optimisations to be undertaken in the compiler.
    /// </remarks>
    [AttributeUsage(AttributeTargets.Parameter)]
    public class IsReturnedInEveryElementAttribute : Attribute
    {
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "IsReturnedInEveryElement";
        }
    }

    /// <summary>
    /// When applied to a method argument, indicates that the method's result is invalidated when
    /// a dependent item in that argument changes.
    /// </summary>
    /// <remarks>
    /// The formal definition of trigger is:
    /// If C uses A, and A is triggered by B, then when B changes, A must be updated before C.
    /// </remarks>
    public class TriggerAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a method argument, indicates that the method's result is invalidated when
    /// the array element at resultIndex changes.
    /// </summary>
    public class MatchingIndexTriggerAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a method, indicates that no default triggers should be assigned to its parameters.
    /// The method will explicitly mark its triggers using TriggerAttribute.
    /// A general rule is that a VMP operator that depends on the opposite message must not trigger on that message (i.e. non-conjugate operators).
    /// </summary>
    public class NoTriggersAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a method argument, indicates that the argument must be up-to-date
    /// before invoking the method.
    /// When applied to a method, indicates that the method must be recomputed whenever any of its arguments change.
    /// </summary>
    public class FreshAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a message function, indicates that the function returns the product of its arguments.
    /// </summary>
    public class MultiplyAllAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a method argument, indicates that the argument does not benefit from initialization (by default, initialization is assumed to be beneficial).
    /// This attribute is incompatible with SkipIfUniform or Required.
    /// </summary>
    public class NoInitAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a method argument, indicates that the argument should not participate in backward sequential loops.
    /// </summary>
    public class DiodeAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a factor, indicates that the factor returns a composite array.
    /// </summary>
    public class ReturnsCompositeArrayAttribute : Attribute
    {
    }

    /// <summary>
    /// When applied to a method, indicates that the method will always throw a NotSupportedException.
    /// </summary>
    public class NotSupportedAttribute : Attribute
    {
        /// <summary>
        /// Message for the exception that will be thrown by the method
        /// </summary>
        public string Message;

        /// <summary>
        /// Creates a NotSupported attribute with the given exception message
        /// </summary>
        /// <param name="message"></param>
        public NotSupportedAttribute(string message)
        {
            this.Message = message;
        }
    }

    /// <summary>
    /// Quality bands for Infer.NET components - distributions, operators, factors
    /// </summary>
    public enum QualityBand
    {
        /// <summary>
        /// Unknown. Components which are not marked with a quality band. 
        /// </summary>
        Unknown = 0,

        /// <summary>
        /// Experimental components are work in progress and represent early stage development
        /// of new components. They are typically usable in just a few common scenarios, and
        /// will not have undergone rigorous testing. They are primarily intended for use by
        /// researchers and provide an opportunity for preliminary feedback. It is possible that
        /// a given experimental component may not be included in a future release. These may
        /// be undocumented or only sparsely documented.
        /// </summary>
        Experimental = 1,

        /// <summary>
        /// Preview components are intended to meet most basic usage scenarios. While in
        /// the Preview Quality Band, these components may have a moderate number of breaking
        /// API or behavior changes in response to user feedback and as we learn more
        /// about how they will be used. Users are likely to encounter bugs and
        /// functionality issues for less common scenarios. These will have some documentation
        /// which may be minimal.
        /// </summary>
        Preview = 2, // Must have at least one test
        /// <summary>
        /// Stable components are suitable for a wide range of usage scenarios and will
        /// have incorporated substantial design and functionality feedback. They may continue
        /// evolving via limited bug fixes, fine-tuning, and support for additional scenarios.
        /// Stable components may have a small number of breaking API or behaviour changes
        /// when feedback demands it. These components will have reasonable documentation
        /// which may include examples.
        /// </summary>
        Stable = 3, // Must have a variety of tests, and have been used in > 1 project
        /// <summary>
        /// Mature components are ready for full release, meeting the highest levels of
        /// quality and stability. Future releases of mature components will maintain a high
        /// quality bar with no breaking changes other than in exceptional circumstances.
        /// Users should be confident using mature components, knowing that when they
        /// upgrade from one version of Infer.NET to a newer version it will be a quick and
        /// easy process. These components will have detailed documentation and at least one
        /// example of usage.
        /// </summary>
        Mature = 4
    }

    /// <summary>
    /// Attribute used to label Infer.NET components. They may be attached to algorithm classes,
    /// distribution classes, and operator classes. They may also be attached to any methods on
    /// these classes
    /// </summary>
    /// <remarks>Methods inherit the quality of their class. Classes are, by default, experimental.
    /// <para>
    /// The compiler will convert static quality bands to a quality band for each generated statement
    /// of code. Depending on the error level and warning levels set in the compiler, Errors and/or
    /// warnings will be issued. In addition, a general warning/error may be issued for the algorithm.
    /// </para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.All, AllowMultiple = false)]
    public class Quality : Attribute
    {
        /// <summary>
        /// Quality band
        /// </summary>
        public QualityBand Band;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="band"></param>
        public Quality(QualityBand band)
        {
            Band = band;
        }

        /// <summary>
        /// Gets the quality band associated with a type. If this is an array type
        /// then the quality of the element is returned. If the type has generic parameters
        /// then the quality of the generic type is returned
        /// </summary>
        /// <param name="t">Type</param>
        public static QualityBand GetQualityBand(Type t)
        {
            if (t.IsArray)
                return GetQualityBand(t.GetElementType());

            if (t.IsGenericType)
                t = t.GetGenericTypeDefinition();

            QualityBand qb = QualityBand.Unknown;
            Quality[] qatts = (Quality[]) t.GetCustomAttributes(typeof (Quality), true);
            if (qatts != null && qatts.Length > 0)
                qb = qatts[0].Band;
            return qb;
        }

        /// <summary>
        /// Gets the quality band associated with a member of a class
        /// </summary>
        /// <param name="mi">Member info</param>
        public static QualityBand GetQualityBand(MemberInfo mi)
        {
            if (mi == null)
                return QualityBand.Unknown;
            QualityBand ownerQB = GetQualityBand(mi.ReflectedType);
            Quality[] qatts = (Quality[]) mi.GetCustomAttributes(typeof (Quality), true);
            QualityBand memberQB = QualityBand.Unknown;
            if (qatts != null && qatts.Length > 0)
                memberQB = qatts[0].Band;

            if (memberQB == QualityBand.Unknown)
                return ownerQB;
            else
                return memberQB;
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "Quality Band: " + Band.ToString();
        }
    }
}
