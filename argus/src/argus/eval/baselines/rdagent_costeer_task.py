# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/microsoft/RD-Agent
# Path:    rdagent/components/coder/CoSTEER/task.py (whole file, 9 lines)
# Commit:  32b3d395e73d9db5eee3fe9063d69aec0fdc83bd (2026-09-04)
# Licence: MIT (Copyright (c) Microsoft Corporation) — full text below.
#
# `FactorTask` (vendored in `rdagent_factor.py`) subclasses this real, tiny base class. No
# behaviour of its own beyond `base_code` bookkeeping this comparison never touches — included
# because vendoring means the real class, not a duck-typed stand-in.
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
from rdagent.core.experiment import Task


class CoSTEERTask(Task):
    def __init__(self, base_code: str = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: we may upgrade the base_code into a workspace-like thing to know previous.
        # NOTE: (xiao) think we don't need the base_code anymore. The information should be retrieved from the workspace.
        self.base_code = base_code
