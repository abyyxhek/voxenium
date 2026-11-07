# Pocket TTS Benchmark

- Text: `Hello, this is a benchmark of Pocket TTS streaming latency and throughput for real-time voice applications.`
- Voice: `alba`
- Platform: `win32` (Windows, local run)

| Model | Language | Quant | Load (s) | RSS after load (MB) | Peak RSS (MB) | TTFT (ms) | Audio (s) | Wall (s) | RTF (audio/wall) | Chunks | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| english_fp32 | english | fp32 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | error: The paging file is too small for this operation to complete. (os error 1455) |
| english_int8 | english | int8 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | error: Windows paging-file/memory pressure during isolated run (os error 1455 / host OOM condition observed) |
| english_18m_fp32 | english_18m_2026-10 | fp32 | 0.76 | n/a | n/a | 95.6 | 5.68 | 1.24 | 4.59 | 34 | ok |
| english_7m_fp32 | english_7m_2026-10 | fp32 | 11.29 | n/a | n/a | 64.3 | 5.84 | 1.11 | 5.27 | 30 | ok |

Notes: RTF > 1 means faster than real-time. RSS metrics were unavailable in these isolated runs (`psutil` not installed / fallback not captured). English FP32/int8 runs failed in isolated subprocesses on this host with Windows paging-file/memory pressure (os error 1455); distilled configs completed.
