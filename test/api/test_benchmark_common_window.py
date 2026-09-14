import os
import sys
import unittest


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.benchmark import (
    _common_decode_window,
    _per_request_common_rate,
)


def _request(request_id, first_token_time, token_times, end_time, start_time=0.0):
    """One request in the shape _run_batch records."""
    return {
        "request_id": request_id,
        "start_time": start_time,
        "first_token_time": first_token_time,
        "end_time": end_time,
        "output_tokens": len(token_times),
        "token_ids": list(range(len(token_times))),
        "token_times": list(token_times),
    }


class CommonDecodeWindowTest(unittest.TestCase):
    def test_window_starts_at_last_ttft_not_first(self):
        # #0 prefills until t=10 then decodes to t=19. #1 prefills until t=30
        # then decodes to t=39. Only #1 is decoding after t=30.
        #
        # Starting the window at the first TTFT (t=10) spans 29 s and credits
        # 18 decode tokens, most of them emitted while #1 was still prefilling.
        # The common window is [30, 39]: 9 s and exactly the 9 tokens #1
        # emitted after its own prefill.
        requests = [
            _request(0, 10.0, [10.0 + i for i in range(10)], 19.0),
            _request(1, 30.0, [30.0 + i for i in range(10)], 39.0),
        ]

        start, span, tokens = _common_decode_window(requests, batch_end=39.0)

        self.assertEqual(start, 30.0)
        self.assertAlmostEqual(span, 9.0)
        self.assertEqual(tokens, 9)
        self.assertAlmostEqual(tokens / span, 1.0)

    def test_first_token_is_prefill_and_never_counts_as_decode(self):
        # A request that emitted only its first token did no decode work.
        requests = [_request(0, 5.0, [5.0], 5.0)]

        start, span, tokens = _common_decode_window(requests, batch_end=5.0)

        self.assertEqual(start, 5.0)
        self.assertEqual(span, 0.0)
        self.assertEqual(tokens, 0)

    def test_no_tokens_yields_no_window(self):
        requests = [_request(0, None, [], None)]

        start, span, tokens = _common_decode_window(requests, batch_end=0.0)

        self.assertIsNone(start)
        self.assertEqual(span, 0.0)
        self.assertEqual(tokens, 0)

    def test_single_request_window_matches_its_own_decode(self):
        # At C=1 the common window collapses to the request's own decode span,
        # so the corrected metric must not move the single-request number.
        times = [2.0 + 0.5 * i for i in range(11)]  # 11 tokens, 10 intervals
        requests = [_request(0, 2.0, times, 7.0)]

        start, span, tokens = _common_decode_window(requests, batch_end=7.0)

        self.assertEqual(start, 2.0)
        self.assertAlmostEqual(span, 5.0)
        self.assertEqual(tokens, 10)
        self.assertAlmostEqual(tokens / span, 10 / 5.0)

    def test_peer_prefill_is_not_charged_to_decode_at_80k_scale(self):
        # Reconstructed C=2 / 80K shape: two 256-token requests whose prefills
        # serialize, both finishing together. The upstream metric starts at the
        # first TTFT and reports roughly 11 tok/s; the common window reports
        # the rate while both are actually decoding.
        def timeline(first, step, count):
            return [first + step * i for i in range(count)]

        r0 = _request(0, 41.52, timeline(41.52, 0.1817, 256), 87.85)
        r1 = _request(1, 83.14, timeline(83.14, 0.01843, 256), 87.85)
        requests = [r0, r1]

        start, span, tokens = _common_decode_window(requests, batch_end=87.85)

        self.assertAlmostEqual(start, 83.14)
        self.assertAlmostEqual(span, 4.71, places=2)
        self.assertEqual(tokens, 281)
        self.assertGreater(tokens / span, 50.0)

        # And in the same window each request's own rate is measurable, which
        # is what distinguishes "bandwidth sharing" from "one request starved".
        self.assertGreater(_per_request_common_rate(r1, start), 50.0)

    def test_per_request_rate_is_zero_without_a_window(self):
        item = _request(0, 5.0, [5.0, 6.0], 6.0)
        self.assertEqual(_per_request_common_rate(item, None), 0.0)


if __name__ == "__main__":
    unittest.main()
