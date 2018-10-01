---
layout: default
---
[Factors and Constraints](Factors and Constraints.md)

## String and character factors

This page lists the built-in methods and operators for creating random variables of type `string` and `char`. For both static methods and operators, you can often pass in random variables as arguments e.g. `Variable<string>` instead of `string`. For compactness, this is not shown in the syntax below.

These methods provide a convenient short alternative to using `Variable<T>.Factor` and passing in the factor method, as described [on this page](Applying functions and operators to variables.md).

**All factors on this page are experimental, and subject to change.**

#### Distribution Factors

A distribution factor creates a random variable from a parameterised distribution.

| **Distribution** | **Syntax** | **Description** |
|-------------------------------------------------|
| _String Distribution_ | `Variable.StringUniform()` | Creates a uniformly distributed **string** random variable. The distribution of the variable is improper. |
| _String Distribution_ | `Variable.StringLower()` `Variable.StringLower(int minLength, int maxLength)` `Variable.StringUpper()` `Variable.StringUpper(int minLength, int maxLength)` `Variable.StringLetters()` `Variable.StringLetters(int minLength, int maxLength)` `Variable.StringDigits()` `Variable.StringDigits(int minLength, int maxLength)` `Variable.StringLettersOrDigits()` `Variable.StringLettersOrDigits(int minLength, int maxLength)` `Variable.StringWhitespace()` `Variable.StringWhitespace(int minLength, int maxLength)` `Variable.StringWord()` `Variable.StringWord(int minLength, int maxLength)` `Variable.StringNonWord()` `Variable.StringNonWord(int minLength, int maxLength)` | Creates a **string** random variable uniformly distributed over all either non-empty strings or strings with length in given bounds, containing lowercase (uppercase, letters, digits, letters and digits, word, non-word) characters only. If the upper length bound is not specified, the distribution of the variable is improper. |
| _String Distribution_ | `Variable.StringCapitalized()` `Variable.StringCapitalized(int minLength, int maxLength)` | Creates a **string** random variable uniformly distributed over all strings starting from an uppercase letter, followed by one or more lowercase letters. If the upper length bound is not specified, the distribution of the variable is improper. |
| _String Distribution_ | `Variable.StringOfLength(int length)` `Variable.StringOfLength(int length, DiscreteChar allowedCharacters)` | Creates a **string** random variable uniformly distributed over all strings of given length. String characters are restricted to be non zero probability characters under a given character distribution, if it is provided. |
| _String Distribution_ | `Variable.String(int minLength, int maxLength)` `Variable.String(int minLength, int maxLength, DiscreteChar allowedCharacters)` | Creates a **string** random variable uniformly distributed over all strings with length in given bounds. String characters are restricted to be non zero probability characters under a given character distribution, if it is provided. If the upper length bound is not specified, the distribution of the variable is improper. |
| _Discrete Char_ | `Variable.Char(Vector probs)` | Creates a **char** random variable from a supplied vector of character probabilities. The probabilities much add up to 1 and be provided for every possible character value, from **char.MinValue** to **char.MaxValue**. |
| _Discrete Char_ | `Variable.CharUniform()` `Variable.CharLower()` `Variable.CharUpper()` `Variable.CharLetter()` `Variable.CharDigit()` `Variable.CharLetterOrDigit()` `Variable.CharWord()` `Variable.CharNonWord()` | Creates a **char** random variable having a uniform distribution over all possible (lowercase, uppercase, letters, digits, letters and digits, word, non-word) characters. |

#### String Operations

| **Operation** | **Syntax** | **Description** |
|----------------------------------------------|
| `Concat(string, string)` `Concat(string, char)` `Concat(char, string)` | a + b | Create a **string** random variable equal to the concatenation of its arguments. |
| _Substring_ | `Variable.Substring(string str, int start, int length)` | Create a **string** random variable array by extracting a substring of a given string. |
| _StringFormat_ | `Variable.StringFormat(string format, string[] args)` | Create a **string** random variable by replacing argument placeholders like {0}, {1} etc. in the provided format string with argument values, similar to what .NET **string.Format** does. For more information, see [this page](StringFormat operation.md). |
