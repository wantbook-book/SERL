# Changelog

## [0.7.0]
### Added
- Added CITATION.cff file with library metadata
- Ensure And don't throw an error on comparisson

### Changed
- Simplified joining of multiple boxed expressions
- Improved assignment resoluion
- Improved handling of And instances in `sympy_compare_relational`
- Qol changes (formatting, etc)

## [0.6.2]
### Changed
- Made parsing timeout to be global not per extraction to avoid too long extraction time
- Changed timeout exception to be a custom exception to prevent default exception handler catching
- Fixed bug with 0,xxx not being parsed as float

## [0.6.0]
- Added support for setting numeric precision for numeric evaluation
- Fixed bug with expression with =
- Deprecated `equations` parameter in `NormalizationConfig`, as it is now handled by the parser
- Fixed processing of Assignment relations
- Bumped latex2sympy2_extended to 1.0.9

## [0.5.4]
- Added logging to grader and parser
- Fixed bug with imports of antlr4 runtime

## [0.5.3]
- Added support for multiple antlr4 runtimes:
    - `antlr4-python3-runtime==4.13.2`
    - `antlr4-python3-runtime==4.11.0`
    - `antlr4-python3-runtime==4.9.3`

## [0.5.3-pre]
- Improve process of running `evaluate_model.py`:
  - Update the README command and installation instructions.
  - Update `inference` dependencies to "lighteval" instead of "lighteval[accelerate]" to reflect the fact that `accelerate` is now a main rather than optional dependency of `lighteval`.
  - Fix import path passed to `lighteval/tasks/registry.py`.
  - Provide `--override-bs` as a CLI parameter as an alternative to automatic batch size selection, which does not work well on all hardware.
  - Use available accelerators.

## [0.5.2]
- Bump latex2sympy2_extended to 1.0.6, which fixes a bug with boxed normalization
- Allow more separators in latex expressions

## [0.5.1]
- Remove empty excepts, so that KeyboardInterrupt is not caught

## [0.5.0]

### Changed
- Replaced `FiniteSet` from `sympy` with `FiniteSet` from `latex2sympy2_extended.sets` in `src/math_verify/grader.py` and `src/math_verify/parser.py`.
- Modified `sympy_deep_compare_set_and_tuple` and `sympy_compare_sets` functions to use `SympyFiniteSet` for better compatibility with `latex2sympy2_extended`.
- Updated `is_assignment_relation` to use `is_expr_of_only_symbols` instead of `is_assignment_symbol`.
- Improved sorting logic in `sympy_deep_compare_set_and_tuple` to handle `TimeoutError`.

### Added
- New test cases in `tests/test_numina_cases.py` for enhanced expression comparison, including complex expressions and boxed expressions.

### Fixed
- Fixed issues with expression comparison logic, ensuring more accurate results when comparing sets and tuples. 

## [0.4.2]
- Bump latex2sympy2_extended to 1.0.2

## [0.4.1]

### Added
- Fix bug with \boxed expressions. Boxed expression have no ending stopped so they will extract until }. This is fine as normalization will then extract the content. Issue was that if we have multiple of them, it would take all of them no matter whether they are connected or not. So "\boxed{1} ahhh no it's \boxed{2}" would be parsed as "\boxed{1,2}. This is now fixed.

## [0.4.0]

### Added
- Support for multiple expressions joined by "and"/"or" in latex parsing
- Support for comparing expressions with different variable names in non-strict mode
- Support for comparing E (euler's number) with symbol 'e'
- Support for comparing concatenated symbols (e.g., 'abc' vs 'a*b*c')
- Support for comparing relations with sets (e.g., '1 < x < 2' equals '(1,2)')
- Support for comparing tuples with sets
- Support for unwrapping function calls to their arguments
- Added new test files:
  - `test_numina_cases.py` for specific test cases
  - `test_open_thoughts.py` for additional test scenarios
  - `test_strict.py` for testing strict vs non-strict comparison modes

### Changed
- Improved latex parsing to handle multiple expressions
- Enhanced set comparison logic to handle more edge cases
- Renamed `sympy_deep_compare_finite_set` to `sympy_deep_compare_set_and_tuple`
- Updated `verify` function to support strict/non-strict comparison modes
- Modified timeout handling in parsing functions
- Improved documentation and type hints

### Fixed
- Fixed handling of percentage notation
- Fixed comparison of intervals with finite sets
- Fixed handling of boxed expressions with multiple values
- Fixed handling of text in latex expressions

### Removed
- Removed redundant `sympy_compare_set_interval` function
- Removed unnecessary string comparison in some cases 
