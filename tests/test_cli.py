"""CLI regression tests.

Guards the documented ``--state k=v k=v k=v`` syntax (one flag, many pairs)
used by the CI smoke step and the demo script, plus the legacy repeated-flag
form. See ci.yml step "Smoke-test the lead-triage workflow".
"""

import unittest

from agentlab.cli import main

LEAD_TRIAGE = "examples/lead_triage.py"


class CliStateParsingTests(unittest.TestCase):
    def test_multiple_pairs_after_single_flag(self):
        """The documented + CI form: --state k1=v1 k2=v2 ..."""
        rc = main(["workflow", LEAD_TRIAGE, "--state",
                   "role_type=ai-agent-engineer", "remote=true", "visa=true"])
        self.assertEqual(rc, 0)

    def test_single_pair(self):
        rc = main(["workflow", LEAD_TRIAGE, "--state", "role_type=x"])
        self.assertEqual(rc, 0)

    def test_repeated_flags_still_accepted(self):
        rc = main(["workflow", LEAD_TRIAGE,
                   "--state", "role_type=x", "--state", "remote=true"])
        self.assertEqual(rc, 0)

    def test_malformed_pair_exits_with_usage_error(self):
        with self.assertRaises(SystemExit):
            main(["workflow", LEAD_TRIAGE, "--state", "noequals"])


if __name__ == "__main__":
    unittest.main()