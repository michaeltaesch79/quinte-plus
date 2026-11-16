"""
Reference candidates for benchmark.py: the frozen v2 model of the pre-registered hold-out (REPORT.md §10),
before the trainer-opinion split and the outsider correction (§12). Used to check that the benchmark
reproduces the report: the "frozen v2" rows must match §10, and production minus frozen v2 must match §12.

    <venv>/python research/benchmark.py --with research/ref_v2.py --tag ref_v2
"""
import benchmark as bm
import evaluate as ev

NO_ODDS, WITH_ODDS = "frozen v2, no odds", "frozen v2, with odds"


def frozen(with_odds):
    def predict(races, feats, months):
        rs, f2 = ev.frozen_v2([dict(r) for r in races], list(feats))  # copies: the shared races keep their columns
        cols = f2 if with_odds else [c for c in f2 if c != "f_mkt_logp"]
        return ev.walk_forward(rs, f2, {"cols": cols, "l2": 10.0 if with_odds else 20.0, "discount": True}, months)
    return predict


MODELS = [{"name": NO_ODDS, "versus": bm.NO_ODDS, "predict": frozen(False)},
          {"name": WITH_ODDS, "versus": bm.WITH_ODDS, "predict": frozen(True)}]
STRATEGIES = [{"name": "top 5 " + NO_ODDS, "versus": "top 5 " + bm.NO_ODDS, "pick": bm.top5(NO_ODDS)},
              {"name": "top 5 " + WITH_ODDS, "versus": "top 5 " + bm.WITH_ODDS, "pick": bm.top5(WITH_ODDS)},
              {"name": "bon rapport (frozen v2)", "pick": bm.bon_rapport(NO_ODDS)},
              {"name": "expected value > 2 € (frozen v2)", "pick": bm.expected_value(WITH_ODDS)}]
