import unittest

import pr_qwen38_bot as bot


class ConcurrencyAxesTests(unittest.TestCase):
    def test_concurrency_is_scored_and_c1_is_only_a_floor(self):
        # Issue #1080: the unsloth checkpoint's packed step takes FP8 and Q4_K paths no other
        # scored dimension enters, so c2..c32 must be able to earn a tier.
        for c in (2, 4, 8, 16, 32):
            self.assertIn(f"cb-decode@c{c}", bot.SCORING_DIMS)
        # c=1 is measured but never scored: it stops a PR buying scaling by slowing one stream.
        self.assertIn(1, bot.CB_CONCS)
        self.assertNotIn("cb-decode@c1", bot.SCORING_DIMS)
        self.assertIn("prefill@16k", bot.SCORING_DIMS)
        self.assertNotIn("decode@128", bot.SCORING_DIMS)

    def test_schema_is_bumped_so_old_verdicts_re_evaluate(self):
        self.assertNotEqual(bot.EVAL_SCHEMA_VERSION, "v1-nvfp4-decode128")
        self.assertIn(bot.EVAL_SCHEMA_VERSION, bot.MARKER_RE.pattern.replace("\\", ""))

    def test_remote_script_measures_the_ladder_like_the_dspark_bot(self):
        script = bot._remote_script("main", role="main")
        self.assertIn("for CC in 1 2 4 8 16 32; do", script)
        # Same shape and env as pr_dspark_bot.py's ModelOpt rows, so the checkpoints compare.
        self.assertIn('qwen3_gguf_cb_bench "$MODEL_DIR" "$CC" 256 256 512', script)
        for env in ("SPARKINFER_QWEN38_PREFILL_NVFP4=1", "SPARKINFER_QWEN38_DECODE_NVFP4=1",
                    "SPARKINFER_KV_INT8=1"):
            self.assertIn(env, script)
        self.assertIn("qwen3_gguf_cb_bench -j", script)
        for kind in ("AGG", "ITL", "ERR"):
            self.assertIn(f"RESULT_CB${{CC}}_{kind}", script)
        # A harness that measures nothing is infra, never a regression to zero.
        self.assertIn("concurrent decode produced no positive metric", script)

    def test_remote_script_takes_the_harness_from_main(self):
        script = bot._remote_script("pull/1/head", role="pr")
        self.assertIn("git checkout -q origin/main -- runtime/examples/qwen3_gguf_bench.cpp", script)
        self.assertIn("runtime/examples/qwen3_gguf_cb_bench.cpp", script)
        self.assertIn("HARNESS_PINNED", script)
        for path in ("runtime/examples/qwen3_gguf_cb_bench.cpp", "eval/", "bench/scripts/"):
            self.assertIn(path, bot.HARNESS_PATHS)

    def test_parity_gate_is_not_run(self):
        # main fails prefill_parity_check on this checkpoint; an absolute gate main fails would
        # REJECT and auto-close every PR.
        self.assertNotIn('prefill_parity_check.py "', bot._remote_script("main", role="main"))

    def test_ladder_results_parse(self):
        out = bot._parse_remote("RESULT_CB2_AGG 155.9\nRESULT_CB2_ITL 12.59\n"
                                "RESULT_CB32_AGG 246.3\nRESULT_CB32_ERR 2\n")
        self.assertEqual(out["cb2_agg"], 155.9)
        self.assertEqual(out["cb2_itl"], 12.59)
        self.assertEqual(out["cb32_agg"], 246.3)
        self.assertEqual(out["cb32_err"], 2.0)

    def test_a_retryable_ladder_failure_is_retried_and_named(self):
        err = "RETRYABLE_INFRA_FAILURE concurrent-decode harness exited nonzero at c=32\n"
        self.assertTrue(bot._looks_like_hard_kill("", err))
        self.assertIn("c=32", bot._crash_reason("", err))


    def _cb_complete(self, c, tok, err):
        # Run the ladder's own bash check, extracted from the rendered script.
        import re, subprocess
        script = bot._remote_script("main", role="main")
        fn = re.search(r"^cb_complete\(\) \{\n.*?^\}\n", script, re.S | re.M).group(0)
        return subprocess.run(["bash", "-c", fn + f"cb_complete {c} {tok} {err}"]).returncode == 0

    def test_complete_runs_are_accepted(self):
        # Runs measured on main 507017b whose requests all finished or failed outright.
        self.assertTrue(self._cb_complete(1, 264, 0))
        self.assertTrue(self._cb_complete(16, 4104, 0))
        self.assertTrue(self._cb_complete(32, 7936, 2))   # one short request + the long one failed
        self.assertTrue(self._cb_complete(32, 7688, 2))   # two short requests failed
        self.assertTrue(self._cb_complete(32, 8200, 0))   # a PR that fixes the out-of-memory

    def test_a_run_whose_requests_stopped_part_way_is_rejected(self):
        # The c32 run that read 293.0 tok/s instead of ~249: 4,493 tokens with only 2 errors.
        self.assertFalse(self._cb_complete(32, 4493, 2))
        self.assertFalse(self._cb_complete(8, 2000, 0))

    def test_each_width_is_the_median_of_complete_runs(self):
        script = bot._remote_script("main", role="main")
        # c32 on identical code differs by up to 2.7% run to run -- outside the -2% reject band --
        # so one run per width could REJECT and auto-close an unchanged PR.
        self.assertEqual(bot.CB_REPS, 3)
        self.assertIn('while [ "$CB_VALID" -lt 3 ]; do', script)
        self.assertIn("statistics.median", script)
        # A partial run is re-run, never scored; too many of them fail the round as infra.
        self.assertIn("CB_PARTIAL c=$CC", script)
        self.assertIn('if [ "$ATTEMPT" -gt 5 ]; then', script)
        for kind in ("RUNS", "TOK"):
            self.assertIn(f"RESULT_CB${{CC}}_{kind}", script)

    def test_the_median_step_runs(self):
        import re, subprocess
        script = bot._remote_script("main", role="main")
        cmd = re.search(r'CB_AGG=\$\((python3 -c "import statistics.*?") \$CB_AGGS\)', script).group(1)
        out = subprocess.run(["bash", "-c", cmd + " 246.3 293.0 248.6"], capture_output=True, text=True)
        self.assertEqual(float(out.stdout), 248.6)

    def test_runs_parse(self):
        out = bot._parse_remote("RESULT_CB32_RUNS 246.3 250.2 247.9\nRESULT_CB32_AGG 247.9\n")
        self.assertEqual(out["cb32_runs"], [246.3, 250.2, 247.9])
        self.assertEqual(out["cb32_agg"], 247.9)

if __name__ == "__main__":
    unittest.main()
