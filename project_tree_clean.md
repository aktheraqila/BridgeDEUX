BridgeDEUX/
├── android
├── benchmarks
│   ├── __init__.py
│   ├── checkpoint_manager.py
│   ├── evaluate_speech_translation.py
│   ├── exceptions.py
│   ├── run_asr_benchmark - Copy.py
│   ├── run_asr_benchmark.py
│   └── run_speech_translation_benchmark.py
├── bridge
│   ├── __init__.py
│   ├── audio.py
│   ├── config.py
│   ├── logger.py
│   └── metadata.py
├── datasets
│   ├── builders
│   │   ├── benchmark_subset.py
│   │   └── covost_cache_builder.py
│   ├── cache
│   │   ├── covost
│   │   │   └── de_en
│   │   ├── marianmt
│   │   │   └── marianmt_results_20260712_084824_736816.jsonl.bak
│   │   ├── marianmt_results_20260712_080020_248595.jsonl.bak
│   │   └── marianmt_results_20260712_080551_438641.jsonl.bak
│   ├── downloaders
│   │   ├── __init__.py
│   │   ├── exceptions.py
│   │   ├── hf_downloader.py
│   │   ├── manifest.py
│   │   └── verifier.py
│   ├── logs
│   │   └── bridgedeux.log
│   ├── providers
│   │   ├── base_provider.py
│   │   ├── covost_provider.py
│   │   └── sample.py
│   ├── raw
│   │   ├── Choice1_covost_v2.de_en.tsv
│   │   │   └── covost_v2.de_en.tsv
│   │   ├── covost_v2.en_de.tsv
│   │   │   └── covost_v2.en_de.tsv
│   │   ├── de_en
│   │   │   └── test
│   │   ├── Choice1_covost_v2.de_en.tsv.tar.gz
│   │   └── covost_v2.en_de.tsv.tar.gz
│   ├── reports
│   ├── validators
│   └── __init__.py
├── evaluation
│   ├── asr
│   ├── __init__.py
│   ├── evaluator.py
│   ├── exceptions.py
│   ├── metrics.py
│   ├── report_generator.py
│   └── result.py
├── evaluation_reports
│   └── evaluation_summary.md
├── experiments
│   ├── results
│   ├── 00_test_framework.py
│   ├── 01_inspect_covost.py
│   ├── 01_test_metadata.py
│   ├── 02_test_hf_covost.py
│   ├── 03_extract_covost_text.py
│   ├── 04_diagnostic_parquet.py
│   ├── 05_test_hf_download.py
│   ├── __init__.py
│   ├── analyze_mt_baseline.py
│   ├── analyze_whisper_baseline.py
│   ├── benchmark_marian.py
│   ├── build_full_benchmark.py
│   ├── check_generation_config.py
│   ├── check_subset.py
│   ├── evaluate_models.py
│   ├── missing_dataset.py
│   ├── run_benchmark.py
│   ├── smoke_test_comet.py
│   ├── test_audio_decode.py
│   ├── test_benchmark_subset.py
│   ├── test_build_subset.py
│   ├── test_m2m100.py
│   ├── test_marian.py
│   ├── test_provider.py
│   ├── test_translator_api.py
│   ├── test_vosk_inference.py
│   ├── test_whisper_inference.py
│   ├── test_wrapper.py
│   ├── validate_onnx_export.py
│   └── verify_marian_onnx.py
├── models
│   ├── asr
│   │   ├── __init__.py
│   │   ├── base_asr.py
│   │   ├── result.py
│   │   ├── vosk.py
│   │   └── whisper_cpp.py
│   ├── onnx
│   │   ├── opus_mt_de_en
│   │   │   ├── source.spm
│   │   │   └── target.spm
│   │   └── opus_mt_de_en_opt_extended
│   │       ├── source.spm
│   │       └── target.spm
│   ├── translators
│   │   ├── __init__.py
│   │   ├── base_translator.py
│   │   ├── exceptions.py
│   │   ├── m2m100.py
│   │   ├── marian.py
│   │   ├── marian_onnx.py
│   │   └── result.py
│   ├── vosk
│   │   └── vosk-model-small-de-0.15
│   │       ├── am
│   │       │   └── final.mdl
│   │       ├── conf
│   │       │   ├── mfcc.conf
│   │       │   └── model.conf
│   │       ├── graph
│   │       │   ├── phones
│   │       │   │   └── word_boundary.int
│   │       │   ├── disambig_tid.int
│   │       │   ├── Gr.fst
│   │       │   └── HCLr.fst
│   │       ├── ivector
│   │       │   ├── final.dubm
│   │       │   ├── final.ie
│   │       │   ├── final.mat
│   │       │   ├── global_cmvn.stats
│   │       │   ├── online_cmvn.conf
│   │       │   └── splice.conf
│   │       ├── COPYING
│   │       └── README
│   ├── whisper
│   └── marian_onnx.py
├── results
│   ├── analysis_graphs
│   │   ├── mt_latency_comparison.png
│   │   ├── mt_token_speed_comparison.png
│   │   ├── whisper_latency_distribution.png
│   │   └── whisper_wer_distribution.png
│   ├── cascaded_whisper.cpp (base)_marianmt_test
│   │   ├── cascaded_whisper.cpp (base)_marianmt_test_results_20260717_192147_096604.jsonl.bak
│   │   ├── cascaded_whisper.cpp (base)_marianmt_test_results_20260717_192908_458576.jsonl.bak
│   │   └── cascaded_whisper.cpp (base)_marianmt_test_results_20260717_193658_017558.jsonl.bak
│   ├── m2m100_covost2_de_en_test
│   │   └── m2m100_covost2_de_en_test_results_20260713_151113_049214.jsonl.bak
│   ├── marianmt-onnx_benchmark_subset_100
│   │   └── marianmt-onnx_benchmark_subset_100_results_20260724_192524_752659.jsonl.bak
│   ├── marianmt-onnx_opus_mt_de_en_benchmark_subset_100
│   │   └── marianmt-onnx_opus_mt_de_en_benchmark_subset_100_results_20260725_221453_200173.jsonl.bak
│   ├── marianmt-onnx_opus_mt_de_en_opt_extended_benchmark_subset_100
│   │   └── marianmt-onnx_opus_mt_de_en_opt_extended_benchmark_subset_100_results_20260725_221700_561714.jsonl.bak
│   ├── marianmt_benchmark_subset_100
│   │   └── marianmt_benchmark_subset_100_results_20260724_173854_024368.jsonl.bak
│   ├── marianmt_covost2_de_en_test
│   │   ├── marianmt_covost2_de_en_test_results_20260712_173309_727441.jsonl.bak
│   │   ├── marianmt_covost2_de_en_test_results_20260712_200852_148534.jsonl.bak
│   │   └── marianmt_covost2_de_en_test_results_20260712_204815_654890.jsonl.bak
│   ├── vosk_test
│   │   └── vosk_test_results_20260716_134712_609335.jsonl.bak
│   └── whisper.cpp (base)_test
│       ├── whisper.cpp (base)_test_results_20260714_215322_040673.jsonl.bak
│       └── whisper.cpp (base)_test_results_20260716_203250_747588.jsonl.bak
├── scheduler
├── scripts
│   ├── __init__.py
│   ├── analyze_onnx.py
│   ├── compare_onnx_reports.py
│   ├── inspect_bridge.py
│   ├── optimize_onnx.py
│   └── sanity_check.py
├── temp
├── utils
│   ├── __init__.py
│   └── experiment_logger.py
├── .gitignore
├── 01_validate_mt_base.py
├── 02_validate_m2m100.py
├── 03_validate_nllb.py
├── check_audio.py
├── check_schema.py
├── extract_samples.py
├── LICENSE
├── project_tree_clean.md
└── tree.py
