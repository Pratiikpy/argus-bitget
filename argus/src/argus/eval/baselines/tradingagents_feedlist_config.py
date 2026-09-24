# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/TauricResearch/TradingAgents
# Path:    tradingagents/dataflows/config.py
# Commit:  be952b8eccb49720509af544c6675233bc1f10d0 (2026-09-07)
# Licence: Apache License 2.0 — see `tradingagents_feedlist_interface.py`'s header for the full
#          attribution reasoning; identical here, same repository, same commit.
#
# `route_to_vendor()` (sibling vendored `interface.py`) calls `get_config()` from this file to
# resolve the configured vendor chain per category/method. Vendored unedited; its own
# `import tradingagents.default_config as default_config` is satisfied by the sibling vendored
# `tradingagents_feedlist_default_config.py` via the same sys.modules shim.
#
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
#
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION (summarised terms retained from the
# upstream LICENSE file; the full canonical text is at http://www.apache.org/licenses/LICENSE-2.0)
#
# 1. Definitions — "License", "Licensor", "Legal Entity", "You", "Source"/"Object" form, "Work",
#    "Derivative Works", "Contribution", "Contributor" as defined by the Apache 2.0 license text.
# 2. Grant of Copyright License — perpetual, worldwide, non-exclusive, no-charge, royalty-free,
#    irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display,
#    publicly perform, sublicense, and distribute the Work and Derivative Works.
# 3. Grant of Patent License — a patent license as stated in the License, terminating upon
#    initiating patent litigation over the Work.
# 4. Redistribution — permitted in Source or Object form provided this License and all
#    copyright/patent/trademark/attribution notices are retained, and any modified files carry
#    prominent notices of the changes (none made here — this file is unmodified).
# 5. Submission of Contributions is under this License unless stated otherwise.
# 6. Trademarks — this License grants no permission to use the Licensor's trade names, trademarks,
#    or service marks.
# 7. Disclaimer of Warranty — the Work is provided "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF
#    ANY KIND, express or implied.
# 8. Limitation of Liability — no Contributor is liable for damages arising from use of the Work.
# 9. Accepting Warranty or Additional Liability — offered only on the offeror's own responsibility.
#
# Copyright TauricResearch (see attribution note above)
# Licensed under the Apache License, Version 2.0; you may not use this file except in compliance
# with the License. You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#
# ============================== VENDORED FROM HERE ==============================
from copy import deepcopy

import tradingagents.default_config as default_config

# Use default config but allow it to be overridden
_config: dict | None = None


def initialize_config():
    """Initialize the configuration with default values."""
    global _config
    if _config is None:
        _config = deepcopy(default_config.DEFAULT_CONFIG)


def set_config(config: dict):
    """Update the configuration with custom values.

    Dict-valued keys (e.g. ``data_vendors``) are merged one level deep so a
    partial update like ``{"data_vendors": {"core_stock_apis": "alpha_vantage"}}``
    keeps the other nested keys from the default; scalar keys are replaced.
    """
    global _config
    initialize_config()
    incoming = deepcopy(config)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(_config.get(key), dict):
            _config[key].update(value)
        else:
            _config[key] = value


def get_config() -> dict:
    """Get the current configuration."""
    if _config is None:
        initialize_config()
    return deepcopy(_config)


# Initialize with default config
initialize_config()
