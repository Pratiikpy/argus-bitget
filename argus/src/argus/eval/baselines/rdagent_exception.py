# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/microsoft/RD-Agent
# Path:    rdagent/core/exception.py (whole file, 82 lines)
# Commit:  32b3d395e73d9db5eee3fe9063d69aec0fdc83bd (2026-09-04)
# Licence: MIT (Copyright (c) Microsoft Corporation) — full text below.
#
# `CodeFormatError`/`CustomRuntimeError`/`NoOutputError` (all real, un-shimmed classes here) are
# the three exceptions the vendored `FactorFBWorkspace.execute()` (in `rdagent_factor.py`) can
# raise when `raise_exception=True`. Self-contained — no imports of its own.
#
# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ============================== VENDORED FROM HERE ==============================
class WorkflowError(Exception):
    """
    Exception indicating an error that the current loop cannot handle, preventing further progress.
    """


class FormatError(WorkflowError):
    """
    After multiple attempts, we are unable to obtain the answer in the correct format to proceed.
    """


class CodeBlockParseError(FormatError):
    """Raised when code block extraction fails after all strategies."""

    def __init__(self, message: str, content: str, language: str) -> None:
        self.message = message
        self.content = content
        self.language = language
        super().__init__(message)


class CoderError(WorkflowError):
    """
    Exceptions raised when Implementing and running code.
    - start: FactorTask => FactorGenerator
    - end: Get dataframe after execution

    The more detailed evaluation in dataframe values are managed by the evaluator.
    """

    # NOTE: it corresponds to the error of **component**
    caused_by_timeout: bool = False  # whether the error is caused by timeout


class CodeFormatError(CoderError):
    """
    The generated code is not found due format error.
    """


class CustomRuntimeError(CoderError):
    """
    The generated code fail to execute the script.
    """


class NoOutputError(CoderError):
    """
    The code fail to generate output file.
    """


class RunnerError(Exception):
    """
    Exceptions raised when running the code output.
    """

    # NOTE: it corresponds to the error of whole **project**


FactorEmptyError = CoderError  # Exceptions raised when no factor is generated correctly

ModelEmptyError = CoderError  # Exceptions raised when no model is generated correctly


class KaggleError(Exception):
    """
    Exceptions raised when calling Kaggle API
    """


class PolicyError(Exception):
    """
    Exceptions raised due to content management policy
    """


class EvaluatorDidNotTerminateError(RuntimeError):
    """
    Evaluator generator did not terminate with a final Feedback.
    """
