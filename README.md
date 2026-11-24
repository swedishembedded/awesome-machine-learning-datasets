<!--
SPDX-License-Identifier: CC0-1.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

Swedish Embedded AB implements solutions for building licensed, deduplicated and
evaluation-isolated training-data pipelines for its clients. If your team needs
expertise in machine learning data engineering then you can procure our services
by sending an email to info@swedishembedded.com.
-->

# Awesome Machine Learning Datasets

> A curated, role-oriented catalog of open datasets for training **language models** and **voice/speech models** from scratch: pretraining corpora, supervised demonstrations, preference judgments, verifiable tasks, executable environments and evaluation benchmarks.

Last reviewed: November 2026.

**Organization:** datasets are grouped first by **model type** (large language models, video, audio and speech, images) and then by **training stage in chronological order**: pretraining, mid-training, post-training, and evaluation. The language-model section additionally covers continual learning.

A competitive model needs more than one big download. Each stage of development uses a different data type and training objective, so this list is organized by **what each dataset enables**, not by popularity. Evaluation sets are listed separately from training sets on purpose.

## Contents

- [How to use this list](#how-to-use-this-list)
- [Large Language Models](#large-language-models)
  - [Lifecycle overview](#lifecycle-overview)
  - [Pretraining](#pretraining)
    - [General-purpose corpora](#general-purpose-corpora)
    - [Code corpora](#code-corpora)
    - [Multilingual corpora](#multilingual-corpora)
  - [Mid-training](#mid-training)
    - [Mathematics and science corpora](#mathematics-and-science-corpora)
    - [Agentic mid-training](#agentic-mid-training)
    - [Humor and comedy corpora](#humor-and-comedy-corpora)
  - [Post-training](#post-training)
    - [Supervised fine-tuning](#supervised-fine-tuning)
    - [Function calling and tool use](#function-calling-and-tool-use)
    - [Preference and reward learning](#preference-and-reward-learning)
    - [Verifiable reinforcement learning](#verifiable-reinforcement-learning)
    - [Agentic training](#agentic-training)
    - [Safety and defensive alignment](#safety-and-defensive-alignment)
  - [Continual learning and self-improvement](#continual-learning-and-self-improvement)
  - [Evaluation](#evaluation)
- [Video](#video)
  - [Pretraining](#pretraining-1)
  - [Mid-training](#mid-training-1)
    - [Action recognition and temporal understanding](#action-recognition-and-temporal-understanding)
    - [Segmentation, tracking, depth, flow and 3D](#segmentation-tracking-depth-flow-and-3d)
  - [Post-training](#post-training-1)
    - [Video instruction and understanding](#video-instruction-and-understanding)
    - [Video editing and human-centric generation](#video-editing-and-human-centric-generation)
    - [World models and action-conditioned video](#world-models-and-action-conditioned-video)
    - [Video preference and reward data](#video-preference-and-reward-data)
  - [Evaluation](#evaluation-1)
- [Audio and Speech](#audio-and-speech)
  - [Pretraining](#pretraining-2)
    - [Largest speech corpora](#largest-speech-corpora)
    - [Audio-only speech pretraining](#audio-only-speech-pretraining)
  - [Mid-training](#mid-training-2)
    - [Automatic speech recognition](#automatic-speech-recognition)
    - [Text-to-speech and voice generation](#text-to-speech-and-voice-generation)
    - [Swedish and Nordic speech](#swedish-and-nordic-speech)
    - [Speaker identity, verification and diarization](#speaker-identity-verification-and-diarization)
    - [Speech translation and speech-to-speech](#speech-translation-and-speech-to-speech)
    - [Noise, reverberation, enhancement and separation](#noise-reverberation-enhancement-and-separation)
  - [Post-training](#post-training-2)
    - [Expressive speech, emotion and vocalizations](#expressive-speech-emotion-and-vocalizations)
    - [Conversational speech and full-duplex](#conversational-speech-and-full-duplex)
    - [Speech-language understanding](#speech-language-understanding)
    - [Singing and specialized vocal production](#singing-and-specialized-vocal-production)
    - [Stand-up comedy, timing and laughter](#stand-up-comedy-timing-and-laughter)
  - [Evaluation](#evaluation-2)
  - [Training strategy and shortlist](#training-strategy-and-shortlist)
- [Images](#images)
  - [Pretraining](#pretraining-3)
  - [Mid-training](#mid-training-3)
    - [Detection, segmentation and grounding](#detection-segmentation-and-grounding)
    - [OCR, documents, charts and diagrams](#ocr-documents-charts-and-diagrams)
  - [Post-training](#post-training-3)
    - [Vision-language instruction mixtures](#vision-language-instruction-mixtures)
    - [Image editing and controllable generation](#image-editing-and-controllable-generation)
    - [Image preference and reward data](#image-preference-and-reward-data)
  - [Evaluation](#evaluation-3)
- [Practice](#practice)
  - [Recommended starter set for language models](#recommended-starter-set-for-language-models)
  - [Dataset registry](#dataset-registry)
  - [Licensing and source overlap](#licensing-and-source-overlap)
  - [Dataset intake checklist](#dataset-intake-checklist)
- [Related Lists](#related-lists)
- [Contributing](#contributing)
- [License](#license)

## How to use this list

**Legend**

| Mark | Meaning |
| ---- | ------- |
| 🔒 | Gated, contract-based or approval-required access |
| ⚠️ | Restrictive terms (non-commercial, research-only, no-derivatives, third-party model-output terms) or unverified entry; read the license before training |
| 🧪 | Primarily an evaluation resource; keep it out of training data |
| 🔗 | Metadata, labels or instructions only; the underlying audio or documents are fetched and licensed separately |

**Rules of thumb**

- **Do not add scale figures together.** Many corpora reuse or relabel the same upstream sources (FineWeb, DCLM, Zyda and Dolma share web sources; Granary, MOSEL, YODAS and Emilia-YODAS share recordings). Deduplicate at document or recording level before mixing.
- **A repository license is not a content license.** `apache-2.0` metadata does not license the underlying web pages, code or recordings. Teacher-model outputs add their own terms.
- **Keep evaluation isolated.** Check every training mixture for overlap with the benchmarks you report on.
- **Size** columns give the most useful unit available (tokens, documents, examples, trajectories, tasks, hours of audio, images or clips). Figures are approximate (`~`), taken from dataset cards or papers at the review date, and shown as `—` where no reliable figure was found.
- Sizes are approximate and taken from dataset cards or papers at the review date. Entries marked ⚠️ *unverified* could not be confirmed against a canonical publisher; see [Contributing](#contributing).

## Large Language Models

Stages follow the order in which data is used: pretraining, mid-training, post-training (SFT, preference and reward learning, verifiable RL, agentic training, safety), continual learning, then evaluation.

### Lifecycle overview

| # | Stage | Primary data | Typical objective |
| - | ----- | ------------ | ----------------- |
| 1 | Pretraining | Web, books, science, code, multilingual, math text | Next-token prediction |
| 2 | Mid-training (continued pretraining) | Domain, long-context, newer or specialized text | Causal LM, with replay |
| 3 | Post-training: supervised fine-tuning | Instruction, chat, reasoning, coding, tool use | SFT, distillation |
| 4 | Post-training: preference and reward learning | Human or AI judgments, rankings, process labels | DPO, KTO, reward modeling |
| 5 | Post-training: verifiable RL | Math, code execution, constraints, environment rewards | GRPO, PPO, RLVR |
| 6 | Post-training: agentic training | Tool trajectories, repositories, terminals, browsers | Trajectory SFT, agent RL |
| 7 | Continual learning | Explore, generate, verify, learn, evaluate, retain | Online RL, self-training, replay |

An independent **evaluation and regression track** runs across every stage.

| Stage | Representative datasets |
| ----- | ----------------------- |
| Pretraining | FineWeb-Edu, DCLM, Dolma, The Stack |
| Mid-training | FineMath, peS2o, domain corpora |
| SFT | Tülu 3, UltraChat, OpenHermes |
| Reasoning SFT | OpenMathInstruct, NuminaMath |
| Preference training | UltraFeedback, HelpSteer, HH-RLHF |
| Reward learning | HelpSteer2, PRM800K, RewardBench |
| RLVR | DeepScaleR, Tülu 3 RLVR, APPS |
| Agentic SFT/RL | SWE-Gym, AgentGym, SWE-Smith |
| Continual learning | TiC-LM, TRACE, custom task streams |
| Safety | WildGuardMix, BeaverTails, XSTest |
| Evaluation | MMLU-Pro, LiveCodeBench, SWE-bench |

**Tülu 3 is a useful reference pipeline**: AI2 released the SFT mixture, preference data, RLVR tasks and training recipes, so every stage can be studied as one coherent system. For autonomous improvement, **SWE-Gym** and **SWE-Smith** matter because they provide executable tasks with measurable outcomes, which static SFT corpora cannot.

### Pretraining

Corpora used to teach language, world knowledge, programming syntax and mathematical notation through causal language modeling.

#### General-purpose corpora

| Dataset | Size | Content | Purpose |
| ---- | ---- | ---- | ---- |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ~1.3T tokens | Educationally filtered web text (core release) | High-quality English foundation pretraining |
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | ~15T tokens | Large-scale cleaned Common Crawl | Broad web-text coverage |
| [DCLM Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | ~4T tokens | Classifier-filtered web | Reproducible large-scale pretraining; data-filtering studies |
| [Dolma](https://huggingface.co/datasets/allenai/dolma) | ~3T tokens | Web, books, papers, code, encyclopedic text | Diverse foundation corpus |
| [Dolma 3 Pool](https://huggingface.co/datasets/allenai/dolma3_pool) | — | Component datasets for newer OLMo training | Controlled data-mixture experiments |
| [Dolma 3 Mix](https://huggingface.co/datasets/allenai/dolma3_mix-6T) | ~6T tokens | Prepared multi-domain mixture | Larger-scale foundation training |
| [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2) | ~5T tokens | Cross-deduplicated mixture of major open corpora | Pretraining with reduced cross-corpus duplication |
| [RedPajama V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) | ~30T tokens (100B+ docs) | Common Crawl with quality signals | Custom quality-filtering experiments |
| [Common Pile](https://huggingface.co/collections/common-pile) | ~8 TB text | Public-domain and openly licensed corpus family | License-conscious pretraining |
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | ~25B tokens (30M files) | Synthetic educational text | Supplementing natural web text |
| [SmolLM Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | — | Curated multi-source corpus | Efficient small-model pretraining |
| [Dolma 3](https://huggingface.co/datasets/allenai/dolma3) | — | Large mixed pretraining corpus | Strong general-purpose corpus |
| [Nemotron-CC v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) ⚠️ | — | Curated and rephrased web text | Large-scale candidate; review terms |
| [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) | 627B tokens | Deduplicated mixed text | Smaller-scale pretraining |
| [The Pile](https://huggingface.co/datasets/EleutherAI/pile) ⚠️ | ~825 GiB (~300B tokens) | Research, code, books, web and discussions | Historical baseline; rights review |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | — | Encyclopedic knowledge | General knowledge and retrieval |
| [Common Pile (dataset)](https://huggingface.co/datasets/EleutherAI/common-pile) | ~8 TB text | Public-domain and openly licensed material | Rights-conscious training |

> **Do not blindly concatenate these.** Without document-level and near-duplicate detection, apparent diversity becomes repeated training examples.

#### Code corpora

| Dataset | Size | Content | Purpose |
| ---- | ---- | ---- | ---- |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | ~3B files, 600+ languages | Source-code repositories (Software Heritage IDs; content must be fetched separately 🔗) | Programming-language pretraining |
| [The Stack v2 Dedup](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) | — | Deduplicated Stack v2 derivative | Reducing repository-level duplication |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) | ~250B tokens | Source code, notebooks, commits, issues | Code-specialized foundation models |
| [The Stack](https://huggingface.co/datasets/bigcode/the-stack) | ~6 TB, 358 languages | Earlier code corpus | Historical comparison, reproducibility |
| [Nemotron Pretraining Code v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1) | — | NVIDIA code pretraining corpus | Advanced coding foundation training |
| [Nemotron CC Code v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1) | — | Code-oriented web corpus | Supplementary programming knowledge |
| [GitHub Code](https://huggingface.co/datasets/codeparrot/github-code) | 115M files (~1 TB) | Source code across programming languages | Code pretraining |
| [CodeParrot Clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) | ~50 GB | Python code | Python code pretraining |
| [CommitPack](https://huggingface.co/datasets/bigcode/commitpack) | ~4 TB, 350 languages | Repository commit history and modifications | Change-aware code pretraining |
| [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) | ~702K examples | Commit-based instruction-tuning material | Commit instruction tuning |
| [Stack v3 Train](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train) ⚠️ | — | More recent large-scale code corpus; audit first | Code pretraining |
| [Nemotron Pretraining Code v3](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v3) | — | NVIDIA code pretraining corpus | Code foundation training |
| [KStack](https://huggingface.co/datasets/JetBrains/KStack) | — | Curated Kotlin code from JetBrains | Code pretraining |
| [CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net) | ~2M function-doc pairs | Code paired with natural-language documentation | Code-to-text association |
| [CodeXGLUE Completion](https://huggingface.co/datasets/google/code_x_glue_cc_code_completion_token) | — | Token and line completion tasks | Completion (supplemental) |
| [CodeXGLUE Code-to-Text](https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text) | — | Code summarization data | Summarization (supplemental) |

Preserve file extensions, language labels, repository provenance and file boundaries; they are needed later for completion, repository understanding and task generation.

#### Multilingual corpora

| Dataset | Size | Coverage | Purpose |
| ---- | ---- | ---- | ---- |
| [FineWeb 2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | ~3T words, 1,000+ languages | 1,000+ languages and varieties | Broad multilingual training |
| [FineTranslations](https://huggingface.co/datasets/HuggingFaceFW/finetranslations) | — | Translated web documents | Cross-language transfer |
| [FineTranslations-Edu](https://huggingface.co/datasets/HuggingFaceFW/finetranslations-edu) | — | Educational multilingual translations | Multilingual instruction readiness |
| [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) | ~251B tokens | Indian languages | Regional specialization |
| [mC4 (multilingual C4)](https://huggingface.co/datasets/allenai/c4) | C4 en ~365M docs; mC4 ~6.3T tokens | Multilingual Common Crawl | Multilingual baselines |

For Swedish and other Nordic languages FineWeb 2 is a sound base, but supplement it with separately licensed native-language material instead of relying on translated English.

### Mid-training

Mid-training continues next-token prediction on higher-quality, specialized, long-context or newer data after the main run, normally mixed with replay of the original distribution.

| Objective | Dataset families | Signal |
| --------- | ---------------- | ------ |
| Knowledge specialization | peS2o, legal corpora, medical literature | Next-token prediction |
| Programming specialization | The Stack v2, Nemotron Code | Next-token prediction |
| Mathematical specialization | FineMath, OpenWebMath | Next-token prediction |
| Multilingual specialization | FineWeb 2, FineTranslations | Next-token prediction |
| Temporal knowledge updates | Time-indexed news/web corpora, [TiC-LM](https://aclanthology.org/2025.acl-long.1551/) | Next-token prediction |
| Long-context adaptation | Long documents, concatenated multi-document sequences | Long-sequence LM |
| Retention | Replay subsets of the original mixture | Mixed original and new objectives |

Keep a **replay set** drawn from the original distribution; training only on a narrow domain can improve it while degrading general ability. Record provenance, publication timestamps and licenses at this stage.

#### Mathematics and science corpora

| Dataset | Size | Content | Purpose |
| ---- | ---- | ---- | ---- |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | ~54B tokens | Filtered mathematical web documents | Math-heavy pretraining |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | ~14.7B tokens | Mathematical web text | Notation and knowledge |
| [Proof-Pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) | ~55B tokens | Scientific and mathematical text mixtures | Technical foundation training |
| [peS2o](https://huggingface.co/datasets/allenai/peS2o) | ~40B tokens | Academic publications | Scientific domain adaptation |
| [AutoMathText](https://huggingface.co/datasets/math-ai/AutoMathText) | ~200 GB | Mathematics-focused corpus | Mathematical pretraining |
| [Nemotron CC Math v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | — | Mathematics-focused web data | Technical-domain enrichment |
| [Nemotron Math Proofs v1](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1) | — | Proofs and solutions | Proof-oriented training; inspect sample format first |
| [Proof-Pile](https://huggingface.co/datasets/hoskinson-center/proof-pile) | ~13 GB | Mathematics and proofs | Mathematical pretraining |
| [Stack Exchange](https://huggingface.co/datasets/stackexchange/stackexchange) | — | Technical questions and solutions | Technical Q&A knowledge |

Mathematical *knowledge* pretraining is not mathematical *reasoning* SFT: papers teach notation and concepts, problem-solution sets teach how to construct answers.

These corpora can also be mixed into main pretraining; they are listed here because they are most often used for domain-focused continued training.

#### Agentic mid-training

Large, mixed-quality interaction corpora used to learn the action/observation distribution before quality-focused SFT. Many attempts are imperfect, so these are not success-only imitation data. Several overlap with each other and with the trajectory sets in [Agentic training](#agentic-training); deduplicate at the task and trajectory level before mixing.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [SWE-ZERO 12M](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories) | 12.3M trajectories (~111B tokens) | Mixed-quality coding-agent trajectories | Large-scale agent mid-training |
| [AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove) | ~1.7M traces | Mixed agent traces (coding, terminal, math, computer use) | Diverse agent mid-training / SFT |
| [Nemotron Terminal Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) | 100Ks of trajectories | Hundreds of thousands of terminal trajectories | Shell and coding agent behavior |
| [Mini Coder Trajectories 400k](https://huggingface.co/datasets/ricdomolm/mini-coder-trajs-400k) | ~400K trajectories | Code-agent trajectories | Code-agent mid-training |

#### Humor and comedy corpora

Raw joke text for humor language modeling. The Reddit-derived and short-joke sets overlap heavily; deduplicate before combining or the model will memorize common jokes. Noisy and not independently quality-verified.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [One Million Reddit Jokes](https://huggingface.co/datasets/SocialGrep/one-million-reddit-jokes) ⚠️ | ~1M posts | Reddit joke posts | Broad humor continued pretraining |
| [rJokes](https://aclanthology.org/2020.lrec-1.753/) ⚠️ | 550K+ | Reddit jokes with community feedback | Joke generation, weak preference signals |
| [Short Jokes Dataset](https://huggingface.co/datasets/Maximofn/short-jokes-dataset) | ~232K | Short jokes | One-liners, setup/punchline SFT |
| [Short Jokes](https://huggingface.co/datasets/ysharma/short_jokes) | 200K+ | Short jokes | Joke language modeling |
| [10K Reddit Jokes](https://huggingface.co/datasets/Yashodhar29/10k-jokes-dataset) ⚠️ | 10K | Reddit jokes | Small curated experiments |

### Post-training

Post-training teaches the pretrained model to behave: supervised demonstrations first, then preference and reward learning, then verifiable and agentic reinforcement learning, then safety alignment.

#### Supervised fine-tuning

##### General instruction following and chat

| Dataset | Size | Character | Use |
| ---- | ---- | ---- | ---- |
| [Tülu 3 SFT Mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) ⚠️ | ~939K examples | Some outputs from third-party models with extra terms | Reproducible general post-training baseline |
| [UltraChat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | ~200K dialogues | Curated multi-turn dialogue | Dialogue, instruction following |
| [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | Synthetic and mixed-source examples | ~1M examples | Diverse synthetic assistant responses |
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | ~161K messages | Human-written conversation trees | Dialogue, comparisons |
| [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | ~10K | Human-written | High-quality instruction following |
| [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) | ~4.2M | FLAN-derived prompts with synthetic explanations | Explanatory responses |
| [Orca AgentInstruct](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1) | ~1M | Synthetic instructions | Task diversity |
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | ~1M | Mixed post-training corpus | Small-model instruction tuning |
| [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) ⚠️ | ~52K | Synthetic instructions | Reference SFT baseline |
| [Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | ~15K | Human-authored | Small supervised baseline |
| [LIMA](https://huggingface.co/datasets/GAIR/lima) | ~1K | Curated | Data-efficiency experiments |
| [Tülu 3 Personas](https://huggingface.co/datasets/allenai/tulu-3-personas-instruction-following) | — | Synthetic, diverse-persona instruction following | Instruction following |
| [Nemotron Instruction Following Chat v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) | — | Constraint adherence and chat (synthetic) | Constraint-following SFT |
| [OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) | — | Multilingual conversations with ratings and preferences | Dialogue, preference signals |
| [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) ⚠️ | ~1M conversations | Real user prompts and conversations; sensitive content and usage terms | Realistic prompt distributions |
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) ⚠️ | ~1M conversations | Real-world conversations; privacy and terms review | Realistic prompt distributions |

Tülu 3 is the preferred reference for a reproducible SFT pipeline; No Robots and LIMA are useful for separating the effects of sample quality from sample count. Public availability is not unrestricted commercial permission.

##### Reasoning and mathematics

| Dataset | Size | Capability | Use |
| ---- | ---- | ---- | ---- |
| [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) | ~1.8M | Worked mathematics solutions | Math SFT |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | ~14M | Large synthetic math reasoning | Scaling math SFT |
| [NuminaMath CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | ~860K | Competition math reasoning | Worked-solution SFT; RLVR with extra checkers |
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | ~395K | Mathematical instruction variants | Generalization |
| [Orca Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | ~200K | Math word problems | Instruction tuning |
| [LIMO](https://huggingface.co/datasets/GAIR/LIMO) | 817 | Small curated reasoning set | Data-efficiency studies |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | ~8.5K (1,319 test) | Grade-school math with solutions | SFT, simple RLVR baseline, 🧪 evaluation |
| [MATH](https://huggingface.co/datasets/hendrycks/competition_math) | ~12.5K (5,000 test) | Competition math | SFT, RLVR, 🧪 evaluation |
| [OpenThoughts 3](https://huggingface.co/datasets/OpenThoughts/OpenThoughts3-1.2M) | ~1.2M | Teacher-generated reasoning demonstrations | Reasoning SFT |
| [DeepSeek-R1 distillation resources](https://huggingface.co/datasets/deepseek-ai/DeepSeek-R1-Distill) ⚠️ | — | Reasoning supervision; confirm dataset availability separately from models | Reasoning SFT |

Distinguish **final-answer correctness** from **reasoning-trace quality**. A right answer can come from invalid steps, which matters for rejection-sampling fine-tuning and STaR.

##### Coding and software engineering

| Dataset | Size | Focus | Use |
| ---- | ---- | ---- | ---- |
| [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | ~5M | Coding instructions | Broad coding SFT |
| [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) | ~66K | Coding with feedback and corrections | Coding, debugging |
| [Magicoder OSS-Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | ~75K | Instruction synthesis from open-source code | Code SFT |
| [WizardLM Evol-Instruct V2](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) | ~196K | General-purpose instruction-evolution set (not code-only) | Instruction-complexity experiments |
| [Nemotron SFT OpenCode v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-OpenCode-v1) | — | Code-centered instruction data | Coding post-training |
| [Nemotron SFT SWE v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3) | — | Software engineering trajectories | Repository-level agent SFT |
| [Open SWE Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | 200K+ trajectories | Software engineering agent traces | Tool-trajectory imitation |
| [Nemotron Competitive Programming v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2) | — | Solutions, code completion and critique | Competitive-programming SFT, critique supervision |
| [CodeFeedback Filtered Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | ~157K | Filtered coding instructions | Coding SFT, quality filtering |
| [Magicoder Evol-Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K) | ~110K | Evolved coding instructions | Code SFT |
| [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | ~735K | Coding problem solving with reasoning traces | Reasoning demonstrations |
| [OpenCodeReasoning II](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-II) | — | Subsequent code-reasoning dataset | Reasoning demonstrations |
| [Nemotron SWE v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2) | — | Multi-candidate software engineering solutions | SWE SFT |
| [Nemotron SWE v3.5](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3.5) | — | Newer SWE instruction and agent data | SWE SFT |
| [Nemotron Cascade 2 SFT](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) | — | Large blended SFT corpus including code and agents | Supplemental SFT |
| [CodeXGLUE Refinement](https://huggingface.co/datasets/google/code_x_glue_cc_code_refinement) | — | Correcting broken code | Bug fixing |
| [CodeXGLUE Defect Detection](https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection) | — | Defect identification | Defect detection |
| [CodeXGLUE Translation](https://huggingface.co/datasets/google/code_x_glue_cc_code_to_code_trans) | — | Translating implementations between languages | Code translation |

Treat code data at three levels: single-function generation from a spec; file-level edits that preserve surrounding code; and repository-level tasks that inspect a project, use tools, change several files and validate behavior. The third is closest to what a software-engineering agent needs.

##### Multilingual, scientific and domain-specific

| Dataset | Size | Domain | Use |
| ---- | ---- | ---- | ---- |
| [Aya Dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) | ~204K | Multilingual instructions | Cross-lingual alignment |
| [SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF) | ~137K | Scientific information extraction and reasoning | Scientific assistant SFT |
| [SciInstruct](https://huggingface.co/datasets/TIGER-Lab/SciInstruct) ⚠️ | ~254K | Scientific problem solving | Scientific reasoning |
| [LegalBench](https://huggingface.co/datasets/nguha/legalbench) 🧪 | 162 tasks | Legal reasoning tasks (related to, not, LawInstruct) | Legal evaluation, task taxonomy |
| [Medical Meadow flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) | ~34K | Medical flashcard Q&A (related to, not, MedInstruct-52K) | Medical specialization |
| [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | ~27K | Customer-service conversations | Support agents |

LawInstruct and MedInstruct-52K are not listed because no canonical release was verified. Pair any domain SFT with independent evaluation; fluent domain vocabulary is not evidence of accurate conclusions.

##### Humor, sarcasm and conversational comedy

Teaches humor on demand (not in every response) and recognition of sarcasm and context. Recognizing sarcasm is a separate objective from using it appropriately.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [Humor ShareGPT 5K](https://huggingface.co/datasets/briancconnelly/humor-sharegpt-5k) | 5,234 conversations | Joke conversations | Instruction SFT for humor |
| [Jokes: Setup and Punchline](https://huggingface.co/datasets/hltoh/jokes) | — | Paired short jokes | Completion and punchline generation |
| [Anekdots Dialogs](https://huggingface.co/datasets/igorktech/anekdots_dialogs) | — | Conversational jokes | Dialogue-format humor |
| [Comedy Style Instruct](https://huggingface.co/datasets/2stacks/comedy-style-instruct) ⚠️ | — | Small style-conditioned corpus; non-commercial, includes copyrighted stand-up excerpts | Comedy style controls |
| [FunnyBench](https://huggingface.co/datasets/fchaubard/funny_bench) ⚠️ | 24,438 transcripts | Stand-up transcripts with audience/laughter tags plus 12,218 engagement-based preference pairs; Apache-2.0 card but TikTok-derived performances need rights review | Stand-up SFT and weak preferences |
| [FunnyData](https://huggingface.co/datasets/SajilAwale/FunnyData) | — | Reddit jokes with humor, offense and sentiment annotations | Multitask humor selection |
| [Humor Generation Experiments](https://huggingface.co/datasets/halaction/humor-generation) | — | Generated candidates and evaluations | Generative humor methodology |
| [LLM Humor Research Outputs](https://huggingface.co/datasets/fchaubard/llm_humor_research_outputs) | 14 methods × held-out prompts | Outputs of humor generation methods | Comparing humor decoders and reward models |
| [SARC](https://huggingface.co/datasets/CreativeLang/SARC_Sarcasm) | ~1.3M comments | Reddit sarcastic comments with nonsarcastic comparisons (varies by release) | Sarcasm detection, contextual irony |
| [Contextual Humor Dataset](https://github.com/SimonMarinov/ContextualHumorDataset) | — | Joke spans with surrounding context (laughter-aligned) | Humor recognition in longer conversations |
| [HaHackathon](https://aclanthology.org/2021.semeval-1.9/) | — | Text with humor and offensiveness scores from human raters | Humor with social-awareness labeling |
| [Harm or Humor](https://github.com/insait-institute/harm-or-humor) | — | Text, image, video; multilingual | Humor vs harmful-content distinctions |
| [Conversational Sarcasm Benchmark](https://huggingface.co/datasets/darksyntax0/conversational-sarcasm-benchmark) 🧪 | — | Conversation/speech sarcasm | Best treated as evaluation, not comedy SFT |

##### Humor reasoning, puns and wordplay

Teaches mechanisms (phonetic similarity, double meanings, incongruity, reinterpretation) rather than only labels. Reserve held-out sets such as Phunny for evaluation.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [ExPUNations](https://aclanthology.org/2022.emnlp-main.304/) | — | Puns with keywords, explanations and funniness ratings | Generate and explain wordplay; pun-quality reward |
| [Phunny](https://huggingface.co/datasets/disi-unibo-nlp/Phunny) 🧪 | 350 | Novel structured puns, contamination-checked | Held-out pun generation/reasoning evaluation |
| [Puns 205](https://davidbau.github.io/puns/DATASET.html) | — | Pun vs literal cloze completions | Contrastive wordplay modeling |
| [Humicroedit](https://cs.rochester.edu/u/nhossain/humicroedit.html) ⚠️ | — | Minimal humorous headline edits with ratings and pairwise comparisons ([HF mirror](https://huggingface.co/datasets/tasksource/humicroedit)) | Misdirection from one-word changes; ranking |
| [FunLines](https://cs.rochester.edu/u/nhossain/funlines.html) ⚠️ | — | Crowd-written funny headline edits with funniness ratings | Constrained joke rewriting, human-guided optimization |
| [HumorGen](https://github.com/edwardajayi/HumorGen) | — | Constrained generation experiments; includes Humor Transfer Bench (400 prompts, eight styles) | Humor-generation curricula and benchmarks |
| [Incongruity–Resolution Supervision](https://cyberiada.github.io/NYCC-Thinking/) | — | Cartoon scene interpretation with incongruity and resolution traces | Structured humor reasoning |
| [D-Humor (text labels)](https://huggingface.co/datasets/UVSKKR/D-Humor) ⚠️ | 4,379 memes | Dark-humor intensity labels | Style/intensity prediction |

#### Function calling and tool use

Teaches schema interpretation, argument generation, call/no-call decisions, parallel calls and use of tool results. Executed and independently checked calls are stronger evidence than correct-looking synthetic ones; keep schema correctness, execution correctness and task success distinct.

| Dataset | Size | Capability |
| ---- | ---- | ---- |
| [xLAM Function Calling 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60k | Structured calls, multiple functions, argument correctness |
| [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 113k | Single/multi-turn calls and function responses |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | ~11k | High-quality synthetic API calling |
| [Hermes Function Calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | ~11.6k | Hermes-formatted tool use |
| [Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | ~1.5M | Diverse synthetic MCP/tool trajectories |
| [Nemotron Agentic v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | — | Conversational and multi-step agent traces |
| [Nemotron SFT Agentic v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) | — | Synthetic single/multi-turn tool workflows |
| [Nemotron SFT Agentic v4](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v4) | — | Multi-turn tool selection and orchestration |
| [Nemotron RL Conversational Tool Use](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1) | — | Tool-use reward training |
| [When2Call](https://huggingface.co/datasets/zanand/When2Call) | — | When to call, answer, clarify or abstain |
| [xLAM Irrelevance](https://huggingface.co/datasets/Salesforce/xlam-irrelevance-7.5k) | 7.5k | Avoiding irrelevant tool use |
| [UltraData SFT Agent 2609](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609) | ~240k rows | Agent-focused instruction and reasoning |
| [APIGen / xLAM](https://github.com/salesforce/xLAM) | — | Verified function-call data |
| [APIGen-MT](https://apigen-mt.github.io/) | — | Multi-turn tool use |
| [APIBench](https://huggingface.co/datasets/gorilla-llm/APIBench) | 16,450 instruction–API pairs | API documentation understanding |
| [TaskBench](https://github.com/microsoft/TaskBench) | — | Planner training |
| [APIGen-MT-5K](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) | ~5K | Multi-turn API interaction — Multi-turn tool use |
| [Agent-FLAN](https://huggingface.co/datasets/internlm/Agent-FLAN) | — | Agent trajectories and capability-focused instruction data — Agent SFT |
| [Nemotron RL Function Calling](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1) | — | Verifiable function-calling training tasks — Function-calling RL |
| [CodeAct / CodeActInstruct](https://huggingface.co/datasets/xingyaoww/code-act) | — | Code execution as an agent action — Executable-action agents |
| [τ2-bench trajectories](https://huggingface.co/datasets/AgentSuite/tau2-bench-trajectories) | — | Stateful multi-turn tool workflows — Tool-use SFT |
| [Nemotron Terminal RL](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1) | — | Terminal-oriented agent tasks — Terminal RL |
| [Terminal-Bench trajectories](https://huggingface.co/datasets/yoonholee/terminalbench-trajectories) ⚠️ | — | Terminal tasks with recorded action sequences — Terminal agent SFT; audit evaluation overlap |
| [Terminal-Lego 15K](https://huggingface.co/datasets/PrimeIntellect/Terminal-Lego-15k) | ~15K trajectories | Terminal agent trajectories — Terminal agent SFT |
| [Agent-SFT](https://huggingface.co/datasets/voidful/agent-sft) | — | Aggregated and normalized agent data — Agent SFT |

#### Preference and reward learning

Preference data answers: given two or more responses, which should the model prefer?

##### Preference datasets

| Dataset | Size | Annotation | Algorithms |
| ---- | ---- | ---- | ---- |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | ~64K prompts, ~256K responses | Multiple responses with model-generated ratings | Reward modeling, DPO |
| [UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | ~61K pairs | Chosen/rejected pairs (loses rating detail) | DPO, ORPO |
| [Tülu 3 datasets](https://huggingface.co/collections/allenai/tulu-3-datasets) | — | On- and off-policy preference mixtures | DPO, reward modeling |
| [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) | ~37K responses | Human-rated response attributes | Multi-attribute reward models |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | ~21K responses (~10K prompts) | Human ratings and preferences | Reward modeling, DPO |
| [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | — | Later NVIDIA preference generation | Preference and reward learning |
| [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | ~170K comparisons | Helpful/harmless chosen-rejected dialogues | DPO, reward modeling |
| [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar) | ~183K prompts | Multiple ranked responses per prompt | Ranking-based rewards |
| [Stanford SHP](https://huggingface.co/datasets/stanfordnlp/SHP) | ~385K | Community-derived comparisons | Preference optimization |
| [OpenAI Summarize from Feedback](https://huggingface.co/datasets/openai/summarize_from_feedback) | — | Human comparisons of summaries | Reward models |
| [Orca DPO Pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) | ~12.9K | Synthetic pairs | DPO experiments |
| [Capybara Preferences](https://huggingface.co/datasets/argilla/Capybara-Preferences) | ~15K | Multi-turn preferences | Conversational DPO |
| [CodeUltraFeedback](https://huggingface.co/datasets/RLHFlow/CodeUltraFeedback-standard) | — | Coding preferences | Code DPO |
| [UltraInteract Pairs](https://huggingface.co/datasets/openbmb/UltraInteract_pair) | — | Reasoning and interaction preferences | Reasoning preference tuning |
| [Math-Step-DPO-10K](https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K) | ~10K | Mathematical step preferences | Step-oriented DPO |
| [UltraFeedback Binarized (cleaned)](https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned) | — | Cleaned chosen/rejected pairs | DPO, reward modeling |
| [Argilla UltraFeedback](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | ~61K pairs | Cleaned preference pairs | DPO |
| [TRL UltraFeedback](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) | — | Common DPO training representation | DPO |
| [Curriculum DPO Preferences](https://huggingface.co/datasets/ServiceNow-AI/Curriculum_DPO_preferences) | — | Preference pairs with difficulty progression | Curriculum DPO |

##### Reward models and process supervision

| Resource | Size | Signal | Application |
| ---- | ---- | ---- | ---- |
| [PRM800K](https://github.com/openai/prm800k) | ~800K step labels | Human step-level correctness | Process reward modeling |
| [Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd) | — | Step-level mathematical supervision | Process reward modeling |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | ~21K responses (~10K prompts) | Multi-dimensional human ratings | Scalar or attribute rewards |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | ~64K prompts, ~256K responses | AI-generated quality scores | Outcome reward models |
| [RewardBench](https://huggingface.co/datasets/allenai/reward-bench) 🧪 | ~3K prompts | Held-out preference comparisons | Reward-model evaluation |
| [Agent Reward Bench](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench) 🧪 | ~1.3K trajectories | Agent outcome assessments | Agent reward evaluation |
| [AgentJudgeBench](https://huggingface.co/datasets/ServiceNow-AI/AgentJudgeBench) 🧪 | — | Agent trajectory assessment | Judge evaluation |

Keep reward types distinct and never collapse them into one unqualified score:

- **Outcome reward**: did the completed solution succeed?
- **Process reward**: were the intermediate steps valid?
- **Preference reward**: which candidate is preferred?
- **Verifiable reward**: does the answer satisfy an independently checkable condition?

A verifier-backed reward avoids treating another model's subjective preference as ground truth.

##### Humor and funniness preferences

Candidates written for the same setup, context or image give the cleanest comparisons. Engagement-based pairs across different clips are a weaker signal. Evaluate by language and audience, since averaged scores hide disagreement.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [New Yorker Caption Ranking](https://huggingface.co/datasets/yguooo/newyorker_caption_ranking) ⚠️ | 250M+ ratings, 2.2M captions | Human ratings of cartoon captions; research-only, not for direct commercial product training | Reward modeling, ranking, research DPO |
| [Rapidata Multilingual Joke Preferences](https://huggingface.co/datasets/Rapidata/multilingual-llm-jokes-4o-claude-gemini) | 1M+ votes on ~10K jokes | Human funniness votes in English, Arabic, Japanese, Vietnamese, Portuguese; check license and generated-output rights | Audience-conditioned funniness reward |
| [Humor Evals](https://github.com/kristopolous/humor-evals) ⚠️ | 195K+ jokes | Reddit jokes with scores | Weak ranking and comparison |

#### Verifiable reinforcement learning

RLVR needs a task plus an independent way to decide whether an attempt succeeded; it does not need a reference trajectory. That is the key difference from offline SFT.

##### Mathematics and symbolic verification

| Dataset | Size | Verification | Use |
| ---- | ---- | ---- | ---- |
| [Tülu 3 RLVR (GSM/MATH/IF mixed)](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints) | ~30K prompts | Answer checking plus instruction constraints | Reproducible RLVR; recipe in [Open Instruct](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md) |
| [DeepScaleR Preview](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | ~40K | Math answer verification | GRPO/RLVR experiments |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | ~8.5K (1,319 test) | Numerical answer extraction | Simple RLVR baseline |
| [MATH](https://huggingface.co/datasets/hendrycks/competition_math) | ~12.5K (5,000 test) | Mathematical answer checking | Reasoning RL |
| [NuminaMath CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | ~860K | Reference solutions | Verifiable reasoning with extra checkers |
| [Nemotron Math v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) | — | Math problems and reasoning data | SFT or RL depending on subset |

Exact string match is rarely a sufficient math verifier: handle numeric normalization, symbolic equivalence, multiple valid forms and answer-extraction errors. Recommended first implementation: Tülu 3 RLVR.

##### Coding with executable rewards

| Dataset / environment | Size | Verification | Role |
| ---- | ---- | ---- | ---- |
| [APPS](https://huggingface.co/datasets/codeparrot/apps) | 10K problems | Execution against test cases | Coding RL |
| [CodeContests](https://huggingface.co/datasets/deepmind/code_contests) | ~13K problems | Competitive-programming tests | Harder coding RL |
| [TACO](https://huggingface.co/datasets/BAAI/TACO) | ~26K problems | Executable checks | Coding RL |
| [KodCode](https://huggingface.co/datasets?search=KodCode) | ~447K | Problems with test-based verification | Programmatic RL rewards (search link; confirm canonical repo) ⚠️ |
| [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) 🧪 | 1,000+ (rolling) | Time-separated problems with tests | Held-out evaluation |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) | 2,438 instances | Repository tests, task outcomes | Agentic coding RL |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | ~50K task instances | Generated repository tasks and environments | Scaling task generation |
| [Terminal-Bench](https://github.com/laude-institute/terminal-bench) | ~100 tasks | Containerized terminal tasks | Terminal-agent RL and evaluation |
| [DeepCoder Preview](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) | ~24K problems | Verifiable competitive-programming problems | Code RL |
| [SWE-bench Multilingual](https://github.com/SWE-bench/SWE-bench) 🧪 | 300 tasks | Multilingual software repair tasks | Cross-language generalization |
| [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) | ~8K tasks | Executable repo-level environments | Coding RL and SFT |
| [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) | 21K+ tasks | Reconstructed software engineering tasks | Training/validation with exclusions |
| [Longtail-SWE](https://huggingface.co/datasets/nebius/Longtail-SWE) | — | Difficult software tasks | Hard-task generation and evaluation |
| [Harbor](https://github.com/laude-institute/harbor) | — | Environment execution and grading infrastructure | Generates trajectories; not itself a dataset |
| [CodeBenchGen](https://github.com/THUDM/CodeBenchGen) | — | Generated executable coding tasks | Scalable program verification |
| [Verifiable Coding Problems](https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems) | — | RL-oriented verifiable coding prompts | Code RL |
| [Nemotron Competitive Coding RL](https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding) | — | Coding RL task distribution | Competitive-coding RL |
| [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2) | ~32K tasks, 20 languages | Expanded multilingual repository tasks | Broader SWE RL coverage |
| [Open-Instruct SWE-smith](https://huggingface.co/datasets/allenai/open-instruct-swe-smith) | — | SWE-smith task and grading integrations | SWE RL |
| [Nemotron Agentic SWE RL](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1) | — | SWE-oriented RL training data | SWE RL |
| [Nemotron Cascade RL SWE](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-RL-SWE) | — | Software engineering RL task data | SWE RL |
| [Nemotron Cascade 2 RL](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data) | — | Broader RL mixture containing SWE tasks | Mixed RL |
| [Terminal-Bench 2.0](https://huggingface.co/datasets/harborframework/terminal-bench-2.0) 🧪 | — | Complex terminal tasks; keep protected for evaluation | Terminal evaluation |
| [Verified Terminal-Bench 2](https://huggingface.co/datasets/zai-org/terminal-bench-2-verified) 🧪 | — | Audited benchmark task collection | Terminal evaluation |

Track the strength of each test suite; passing visible tests is not the same as a correct solution.

| Verification level | Confidence |
| ------------------ | ---------- |
| Formatting or string match | Low |
| Compilation or syntax checks | Low to moderate |
| Public unit tests | Moderate |
| Hidden independent tests | Higher |
| Property-based and metamorphic tests | Higher |
| Formal specification or proof checking | Highest, for the verified properties |

##### Constraint following

| Resource | Size | Task | Reward |
| ---- | ---- | ---- | ---- |
| [IFEval](https://huggingface.co/datasets/google/IFEval) 🧪 | ~541 prompts | Instruction constraints | Programmatic compliance |
| [Tülu RLVR IFEval](https://huggingface.co/datasets/allenai/RLVR-IFeval) | ~15K | Instruction-following training tasks | Verifier-based |
| [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | — | Function-call correctness | API call validation |

These give fast, unambiguous feedback and are excellent for checking that an RL implementation works. They are not evidence of general agentic improvement.

#### Agentic training

The core artifact is a trajectory: `τ = (s0, a0, o1, a1, o2, …, aT, r)` with initial state `s0`, actions `a`, environment observations `o` and a task outcome or reward `r`. Trajectories support imitation learning, outcome filtering, process reward learning and on-policy RL. Preserve native tool arguments, results, timestamps, errors and environment references; flattening to plain text loses what reproduction and RL need.

##### Software engineering agents

| Resource | Size | Provides | Use |
| ---- | ---- | ---- | ---- |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) | 2,438 instances | Repository tasks, sampled trajectories, verifier data | SFT and RL |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | ~50K task instances | Scalable repository-level task generation | Expanding executable environments |
| [SWE-bench](https://github.com/SWE-bench/SWE-bench) | 2,294 instances | Real issue-resolution tasks | Held-out evaluation, selected training splits |
| [OpenHands SFT Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) | 491 trajectories | Agent-tool trajectories | Imitation learning |
| [OpenHands Sampled Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories) | ~6K attempts | Multiple sampled attempts | Outcome filtering, offline learning |
| [Open SWE Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | 200K+ trajectories | Agent interaction traces | Trajectory SFT |
| [Nemotron SFT SWE v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3) | — | Software engineering agent data | Long-horizon SFT |
| [daVinci-Agency](https://huggingface.co/datasets/GAIR/daVinci-Agency) | — | Small curated long-horizon examples | Long-horizon behavior study (value is structure, not scale) |
| [NVIDIA SWE-Zero OpenHands](https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories) | 318K trajectories | OpenHands trajectories | SWE agent SFT |
| [SWE-Hero OpenHands](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories) | 34K trajectories | OpenHands trajectories | Higher-quality supervised agent learning |
| [SWE-smith Trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | 5,017 trajectories | Original training-set trajectories | Repository issue resolution |
| [Nebius SWE-agent Trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | ~80K trajectories | Tens of thousands | SWE-agent execution and problem solving |
| [SWE-rebench OpenHands](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | ~67K trajectories | Repository repair trajectories | Repository repair, test-aware actions |
| [R2E-Gym SFT](https://huggingface.co/datasets/R2E-Gym/R2EGym-SFT-Trajectories) | ~3K | Executed repository-level repairs | Executed repository-level repairs |
| [Klear SWE-smith 66k](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k) | ~66K | Small agent scaffold trajectories | SWE SFT |
| [Terminal-Bench 2 traces](https://huggingface.co/datasets/AlienKevin/terminal-bench-2-sft-traces) ⚠️ | 100s of traces | Hundreds of traces | Terminal demonstrations; audit evaluation overlap |
| [Longtail-SWE Trajectories](https://huggingface.co/datasets/nebius/Longtail-SWE-trajectories) | — | Hard software tasks | Failure analysis, challenging trajectories |
| [Thoughtworks Agentic Coding Trajectories](https://huggingface.co/datasets/thoughtworks/agentic-coding-trajectories) | — | Multi-framework trace aggregation | Comparative agent behavior |
| [SFT Benchmark Trajectories](https://huggingface.co/datasets/Heisenger/sft-benchmark-trajectories) | — | Multiple published trajectory sources | Comparison and normalization |
| [Microsoft Orchard](https://huggingface.co/datasets/microsoft/Orchard) | ~107K trajectories (SWE) | Partial-progress trajectory segments from incomplete runs | Credit assignment, recovery learning |
| [SI2CA Training Trajectories](https://huggingface.co/datasets/Self-Improving-Coding-Agents/SI2CA-Training-Trajectories) | 32,340 trajectories | Alternative curation strategies over the same executable tasks | Trajectory-selection research |

##### General tool use and multi-domain agents

| Resource | Size | Environment / data | Use |
| ---- | ---- | ---- | ---- |
| [AgentGym](https://github.com/woooodyy/AgentGym) | — | Multiple interactive agent environments | General agent RL; best treated as an environment-interface reference |
| [AgentGym RL Data (ID)](https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID) | ~186K instances | RL instances | Multi-domain agent training |
| [ALFWorld](https://github.com/alfworld/alfworld) | 3,553 train tasks | Text-based household tasks | Planning, sequential action |
| [WebArena](https://github.com/web-arena-x/webarena) | 812 tasks | Web application environments | Browser agents |
| [VisualWebArena](https://github.com/web-arena-x/visualwebarena) | 910 tasks | Visual web tasks | Multimodal browser agents |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym) | — | Unified browser environments | Browser-agent RL and benchmarking |
| [ToolBench](https://github.com/OpenBMB/ToolBench) | ~126K instances, 16K+ APIs | Tool-use instructions and API interactions | Tool-call SFT |
| [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) | 314 dialogues | API invocation conversations | Tool planning and calling |
| [τ²-Bench](https://github.com/sierra-research/tau2-bench) | — | Stateful customer-service environments | Multi-turn tool agents |
| [BFCL](https://github.com/ShishirPatil/gorilla) | — | Function-calling evaluation | Tool-call correctness |

##### Deep research and long-horizon execution

| Resource | Size | Capability | Role |
| ---- | ---- | ---- | ---- |
| [OpenResearcher Dataset](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset) | ~97K trajectories | Research trajectories | Search and evidence-gathering SFT |
| [Long-Horizon-Execution](https://huggingface.co/datasets/arvindh75/Long-Horizon-Execution) | — | Synthetic multi-step execution | Long-horizon diagnostics |
| [KLong](https://huggingface.co/papers/2602.17547) ⚠️ | — | Long-horizon reasoning research | Methodology reference; dataset access unverified |
| [Research-Factory](https://huggingface.co/papers/2603.20278) ⚠️ | — | Long-horizon research generation | Methodology reference; confirm released artifacts |

##### Static trajectories versus live environments

| Static trajectory dataset | Executable environment |
| ------------------------- | ---------------------- |
| Recorded actions and observations | Generates new observations |
| Supports teacher imitation | Supports policy exploration |
| Teaches recovery patterns seen in demonstrations | Can discover novel recovery strategies |
| Outcomes are recorded | Outcomes can be re-verified |
| Risks imitating the teacher's policy | Trains against the learner's own behavior |
| Cannot establish on-policy improvement alone | Enables on-policy RL |

100,000 successful trajectories are not necessarily worth more than an environment that can generate unlimited novel, independently verifiable tasks.

##### Web browsing and research agents

Search, navigation, extraction and multi-step web workflows. Static retrieval QA is not equivalent to live browsing, where pages go missing and tool calls change the environment.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [AgentTrek](https://huggingface.co/datasets/xlangai/AgentTrek) | ~52K trajectories | Synthetic trajectories from web tutorials | Web navigation |
| [WebLINX](https://huggingface.co/datasets/McGill-NLP/WebLINX) | 2.3K demonstrations (~100K turns) | Multi-turn demonstrations with browser actions | Web interaction |
| [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | 2,350 tasks, 137 sites | Human demonstrations across websites; [project](https://github.com/OSU-NLP-Group/Mind2Web) | Website navigation |
| [WorkArena](https://github.com/ServiceNow/WorkArena) | 33 task types (~19.9K instances) | Enterprise browser workflows with evaluators | Stateful web tasks |
| [MiniWoB++](https://github.com/Farama-Foundation/miniwob-plusplus) | ~100 environments | Simple browser interaction | UI-control RL |
| [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) | 1,384 queries | Long-horizon, reasoning-intensive retrieval | Retrieval supervision and evaluation |

##### Desktop, GUI and mobile agents

Screenshots, accessibility trees, pointer and key actions. Distinguish grounding samples (screenshot → target), action demonstrations, full trajectories and executable tasks with state-based verifiers.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [AgentNet](https://huggingface.co/datasets/xlangai/AgentNet) | 22.6K tasks | Human-annotated desktop tasks (Windows, macOS, Ubuntu) | Multimodal agent SFT |
| [Ubuntu OSWorld Trajectories](https://huggingface.co/datasets/xlangai/ubuntu_osworld2_trajs) | — | Executed desktop task traces | Supervised agent learning |
| [Verified OSWorld Trajectories](https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs) | — | Verified computer-use executions | Supervised agent learning |
| [OSWorld 2.0 Trajectories](https://huggingface.co/datasets/xlangai/osworld2.0-trajectory) | — | Newer computer-control tasks | Supervised agent learning |
| [CUA-Gym](https://huggingface.co/datasets/xlangai/CUA-Gym) | — | Computer-use agent training | Computer-use RL/SFT |
| [Aguvis Stage 1](https://huggingface.co/datasets/xlangai/aguvis-stage1) | — | Visual grounding and computer use | GUI grounding |
| [Aguvis Stage 2](https://huggingface.co/datasets/xlangai/aguvis-stage2) | — | Multi-step GUI behavior | GUI planning and action |
| [AndroidControl](https://huggingface.co/datasets/google-research-datasets/AndroidControl) | ~15K tasks | Android UI commands and screenshots | Mobile agents |
| [Android in the Wild](https://github.com/google-research/google-research/tree/master/android_in_the_wild) | 715K episodes | Large-scale Android action traces | Mobile agents |
| [GUI-Odyssey](https://github.com/OpenGVLab/GUI-Odyssey) | 7,735 episodes | Cross-app mobile trajectories | Mobile agents |
| [Computer Agent Arena](https://huggingface.co/datasets/xlangai/computer-agent-arena) | — | Computer-use task data | Computer-use agents |
| [SeeClick](https://github.com/njucckevin/SeeClick) | — | Visual GUI grounding | GUI grounding |
| [OS-Atlas](https://github.com/OS-Copilot/OS-Atlas) | ~13M elements, 2.3M screenshots | Screenshot/action grounding | GUI grounding |
| [OSWorld](https://github.com/xlang-ai/OSWorld) 🧪 | 369 tasks | Executable desktop tasks (including OSWorld 2) | Mainly evaluation |
| [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) 🧪 | 1,581 samples | Challenging element grounding | Reserve for evaluation |

##### Stateful applications, databases and enterprise agents

Success depends on the final environment state, not a plausible answer.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [AppWorld](https://github.com/StonyBrookNLP/appworld) | 750 tasks, 457 APIs | Multiple apps, 457 APIs, stateful tasks | Training environment and RL |
| [τ-bench](https://github.com/sierra-research/tau-bench) | 165 tasks | Customer-support policies and tools | Stateful tool-use |
| [ToolSandbox](https://github.com/apple/ToolSandbox) | 1,032 scenarios | Stateful tool calling with hidden dependencies | Tool reasoning |
| [EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym) | — | Enterprise operations tasks | Agent RL and evaluation |
| [GitLab Tasks](https://huggingface.co/datasets/ServiceNow-AI/gitlab-tasks) | — | GitLab operations | Development workflow agents |
| [ERPNext Tasks](https://huggingface.co/datasets/ServiceNow-AI/erpnext-tasks) | — | Business operations | Enterprise agent actions |
| [ServiceNow Tasks](https://huggingface.co/datasets/ServiceNow-AI/servicenow-tasks) | — | Ticketing and service management | Stateful enterprise tasks |
| [Spider 2.0 / Spider2-V](https://github.com/xlang-ai/Spider2) | 632 problems (Spider2-V: 494) | Enterprise-scale SQL and data workflows; visual data interaction | Data-engineering agents |
| [Spider2-Lite](https://huggingface.co/datasets/xlangai/spider2-lite) | 547 tasks | Data-engineering problems | Evaluation and task generation |
| [Spider2 LocalDB](https://huggingface.co/datasets/xlangai/spider2-localdb) | — | Local database tasks | Executable data workflows |
| [BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness) | — | Code/test tasks | Build additional verified environments |

##### Planning, interactive reasoning and simulated worlds

Environments whose solutions are computed or checked by code let a policy act and receive feedback instead of replaying teacher demonstrations.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [ScienceWorld](https://github.com/allenai/ScienceWorld) | 30 task types | Scientific experimentation and procedural reasoning | Planning RL |
| [WebShop](https://github.com/princeton-nlp/WebShop) | ~12K instructions, 1.18M products | Shopping and task planning under constraints | Online RL |
| [TextWorld](https://github.com/microsoft/TextWorld) | — | Text-based interactive world modeling | Interactive RL |
| [AgentBench](https://github.com/THUDM/AgentBench) 🧪 | 8 environments | Interactive environments across several domains | Evaluation |
| [MINT-Bench](https://github.com/OSU-NLP-Group/MINT) 🧪 | — | Multi-turn interaction and feedback | Evaluation |
| [AgentInstruct](https://huggingface.co/datasets/zai-org/AgentInstruct) | 1,866 trajectories | Agent trajectories across virtual environments | Agent SFT |
| [TaskTrove](https://huggingface.co/datasets/open-thoughts/TaskTrove) | ~1.9M | Agentic task examples | Generation and practice seeds |

##### Long-context, memory and retrieval

Weaker ecosystem than coding agents; persistent memory typically needs generated longitudinal tasks.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [Agent Data Collection](https://huggingface.co/datasets/neulab/agent-data-collection) | — | Web, knowledge-base, software and household agent traces | Cross-domain agent training |
| [ServiceNow documentation](https://huggingface.co/datasets/ServiceNow-AI/servicenow-docs) | — | Grounded enterprise question answering | Retrieval and tool-use task generation |
| [GitLab documentation](https://huggingface.co/datasets/ServiceNow-AI/gitlab-docs) | — | Tool/API documentation | Technical agent learning |
| [LongBench](https://huggingface.co/datasets/THUDM/LongBench) 🧪 | 4,750 instances | Long-context understanding | Primarily evaluation |
| [LongBench v2](https://huggingface.co/datasets/THUDM/LongBench-v2) 🧪 | 503 questions | Long-context reasoning | Reserve for evaluation |

##### Embodied and robotics agents

Synchronized vision, proprioception, actions and outcomes, normally for a separate vision-language-action model or action head. Reuse depends on each dataset's permissions and the target robot's action space.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [Open X-Embodiment](https://robotics-transformer-x.github.io/) | 1M+ trajectories, 22 embodiments | Large cross-robot manipulation corpus | Manipulation pretraining |
| [DROID](https://droid-dataset.github.io/) | ~76K trajectories | Real-world manipulation demonstrations | Manipulation imitation |
| [LeRobot datasets](https://huggingface.co/lerobot) | — | Standardized robot demonstration collections | Imitation learning |
| [AgiBot World](https://agibot-world.com/) ⚠️ | 1M+ trajectories | Large-scale humanoid and dual-arm demonstrations | Humanoid manipulation |
| [robomimic](https://github.com/ARISE-Initiative/robomimic) | — | Imitation-learning demonstrations | Imitation learning |
| [LIBERO](https://libero-project.github.io/) | 130 tasks | Long-horizon manipulation tasks | Lifelong robot learning |
| [ManiSkill](https://github.com/haosulab/ManiSkill) | — | Simulation-based robot skill learning | Simulated RL |
| [RoboCasa](https://github.com/robocasa/robocasa) | 100 tasks | Household robot simulation | Simulated RL |
| [BEHAVIOR-1K](https://behavior.stanford.edu/) | 1,000 activities | Long-horizon household activities | Embodied planning |
| [MimicGen](https://github.com/ARISE-Initiative/MimicGen) | 50K+ demos | Synthetic robot demonstrations | Data generation |

#### Safety and defensive alignment

Pair refusal examples with harmless requests that should be answered normally, or refusal training turns into over-refusal.

| Dataset | Size | Purpose | Role |
| ---- | ---- | ---- | ---- |
| [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ~83K (v1) | Helpfulness and harmlessness preferences | Safety preference training |
| [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | ~330K QA pairs | Harmful-request/response classifications | Safety model training |
| [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) | ~92K | Prompt harmfulness, response safety, refusal detection | Safety classifier training |
| [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) | ~10K | Toxicity in user interactions | Input classification |
| [XSTest](https://huggingface.co/datasets/allenai/xstest-response) 🧪 | 450 prompts | Safe requests that may trigger refusals | Over-refusal evaluation |
| [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) 🧪 | ~80K prompts | Over-refusal behavior | Safety evaluation |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) 🧪 | 400 behaviors | Robustness to adversarial requests | Defensive red-team evaluation |
| [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) 🧪 | 100 behaviors | Standardized jailbreak robustness | Defensive evaluation |
| [AgentDojo](https://github.com/ethz-spylab/agentdojo) 🧪 | 97 tasks, 629 security cases | Prompt-injection attacks against tool-using agents | Agent robustness evaluation |
| [Nemotron Agentic Safety](https://huggingface.co/datasets/nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0) | — | Agentic safety judgments and examples | Agent safety post-training |
| [PyRIT](https://github.com/microsoft/pyrit) | — | Red-team scenario generation and safety testing | Defensive evaluation tooling |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) | — | Security and agent evaluation infrastructure | Evaluation tooling |

Label semantics differ (preferences vs. moderation labels); do not mix these sets without an explicit label-conversion policy.

### Continual learning and self-improvement

Most open datasets are static. Continual learning needs time-ordered tasks, freshly generated examples, feedback and explicit retention tests.

| Resource | Size | Measures | Use |
| ---- | ---- | ---- | ---- |
| [TiC-LM](https://aclanthology.org/2025.acl-long.1551/) 🧪 | — | Temporal updates vs. retention in continued pretraining | Time-aware pretraining |
| [TRACE](https://github.com/BeyonderXX/TRACE) 🧪 | 8 tasks | Sequential adaptation across diverse tasks | Forgetting and transfer |
| [Long-Horizon-Execution](https://huggingface.co/datasets/arvindh75/Long-Horizon-Execution) | — | Degradation over extended executions | Agent endurance, state tracking |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | ~50K task instances | Task generation across repositories | Continually refreshed practice |
| [AgentGym](https://github.com/woooodyy/AgentGym) | — | Repeated interactive task solving | Online policy improvement |
| [Terminal-Bench](https://github.com/laude-institute/terminal-bench) | ~100 tasks | Executable terminal tasks | Continual task evaluation |

STaR, ReST-EM, expert iteration, GRPO and self-play are **methods, not datasets**; the same raw data serves several of them.

| Method | Input | Generated artifact | Main risk |
| ------ | ----- | ------------------ | --------- |
| Rejection-sampling SFT | Prompts + verifier | Successful completions | Selection bias, limited novelty |
| STaR | Problems + correct answers | Successful or rationalized solutions | Wrong reasoning, right answer |
| ReST-EM | Prompts + reward function | Repeatedly sampled, filtered completions | Distribution narrowing |
| Expert iteration | Tasks + search + evaluator | Improved trajectories from search | Search gains may not transfer |
| GRPO / RLVR | Problems + verifiable rewards | On-policy samples and advantages | Reward overfitting |
| Self-play | Task generator + solver + evaluator | New problems and verified solutions | Collusion, degenerate curriculum |
| Curriculum learning | Tasks with difficulty scores | Ordered task progression | Over-specialization |
| Continual replay | Old and new task distributions | Mixed rehearsal | Poor replay selection, retention loss |

**Requirements for a credible continuously generated task**

- Reproducible definition, environment version and seed.
- Independent success verification.
- Difficulty and capability labels.
- Provenance of the task and any generated solution, including which model produced each attempt.
- Isolation from held-out evaluation tasks.
- A way to replay or reconstruct the original environment.

Measure three outcomes separately: **plasticity** (learning new tasks), **retention** (keeping old abilities) and **generalization** (improving on tasks never trained on). Rising training-task scores with flat held-out transfer and falling retention is a common failure that more SFT examples will not fix.

### Evaluation

Keep these out of training. Use three pools: a fixed **regression suite**, an untouched **generalization suite**, and a rolling **forward-time suite** drawn from genuinely new tasks.

| Capability | Resource | Size | Metric |
| ---- | ---- | ---- | ---- |
| General knowledge | [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | ~12K questions | Accuracy |
| General reasoning | [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) | 448 questions | Accuracy |
| Mathematical reasoning | [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH](https://huggingface.co/datasets/hendrycks/competition_math) | ~8.5K (1,319 test) / ~12.5K (5,000 test) | Answer correctness |
| Live coding | [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | 1,000+ (rolling) | Execution pass rate |
| Function calling | [BFCL](https://github.com/ShishirPatil/gorilla) | — | Call correctness |
| Repository engineering | [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | 500 tasks | Resolved tasks |
| Terminal use | [Terminal-Bench](https://github.com/laude-institute/terminal-bench) | ~100 tasks | Task success |
| Browser interaction | [WebArena](https://github.com/web-arena-x/webarena) | 812 tasks | Goal completion |
| Long-context retrieval | [RULER](https://github.com/NVIDIA/RULER) | — | Length-dependent accuracy |
| Multilingual reasoning | [MMMLU](https://huggingface.co/datasets/openai/MMMLU) | 14 languages | Per-language accuracy |
| Reward models | [RewardBench](https://huggingface.co/datasets/allenai/reward-bench) | ~3K prompts | Ranking accuracy |
| Safety classification | [WildGuardTest](https://huggingface.co/datasets/allenai/wildguardmix) | 1,725 | Classification quality |
| Refusal calibration | [XSTest](https://huggingface.co/datasets/allenai/xstest-response) | 450 prompts | False refusal rate |
| Continual pretraining | [TiC-LM](https://aclanthology.org/2025.acl-long.1551/) | — | Temporal adaptation and retention |
| Desktop computer use | [OSWorld](https://github.com/xlang-ai/OSWorld) | 369 tasks | Task completion |
| Desktop computer use | [AgentNetBench](https://huggingface.co/datasets/xlangai/AgentNetBench) | — | Desktop agent evaluation |
| GUI grounding | [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) | 1,581 samples | Grounding accuracy |
| Enterprise browser workflows | [WorkArena](https://github.com/ServiceNow/WorkArena) | 33 task types (~19.9K instances) | Task success |
| Stateful customer/tool interaction | [τ²-bench](https://github.com/sierra-research/tau2-bench) | — | Task success |
| Cross-domain interaction | [AgentBench](https://github.com/THUDM/AgentBench) | 8 environments | Task success |
| Long-context understanding | [LongBench](https://huggingface.co/datasets/THUDM/LongBench) | 4,750 instances | Accuracy |
| Long-context reasoning | [LongBench v2](https://huggingface.co/datasets/THUDM/LongBench-v2) | 503 questions | Accuracy |
| Agent-trajectory judges | [AgentJudgeBench](https://huggingface.co/datasets/ServiceNow-AI/AgentJudgeBench) | — | Judge accuracy |
| Prompt-injection resistance | [AgentDojo](https://github.com/ethz-spylab/agentdojo) | 97 tasks, 629 security cases | Attack success and utility |
| Humor generation | [SemEval 2026 MWAHAHA](https://pln-fing-udelar.github.io/semeval-2026-humor-gen/) | — | Constrained jokes in English, Spanish, Chinese and humorous GIF captions; human pairwise preferences |
| Pun generation | [Phunny](https://huggingface.co/datasets/disi-unibo-nlp/Phunny) | 350 | Novel pun generation and resolution |
| Practical function-level coding | [BigCodeBench](https://github.com/bigcode-project/bigcodebench) 🧪 | — | Execution pass rate |
| Stronger HumanEval/MBPP tests | [EvalPlus](https://github.com/evalplus/evalplus) 🧪 | — | Execution pass rate |
| Multilingual function generation | [HumanEval-X](https://github.com/THUDM/CodeGeeX) 🧪 | — | Execution pass rate |
| Harder SWE evaluation | [SWE-bench Pro](https://www.swebench.com/) 🧪 | — | Resolved tasks |

Some sets appear in both training and evaluation literature. Training on MATH train problems is compatible with testing on held-out problems; training on your final test set invalidates the test.

## Video

> **Seed section.** Unlike the language-model and audio sections, these entries have not been through a license and provenance review. Several web-scraped video sets have been restricted or taken down by their publishers; confirm current availability and terms before use.

| Stage | Data type | Typical objective |
| ----- | --------- | ----------------- |
| Pretraining | Large video-text corpora, action and instructional video | Video representation learning, text-to-video diffusion or autoregressive pretraining |
| Mid-training | Filtered, high-quality, high-resolution clips with dense captions | Quality and motion refinement |
| Post-training | Video instruction data, human or model quality judgments | Video-language SFT, preference tuning |
| Evaluation | Held-out benchmarks | Generation quality, video understanding |

### Pretraining

| Dataset | Size | Content | Notes |
| ---- | ---- | ---- | ---- |
| [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/) | ~136M clips (1.2M videos) | Narrated instructional clips | ASR-derived captions are noisy; source-video rights apply ⚠️ |
| [InternVid](https://huggingface.co/datasets/OpenGVLab/InternVid) | 7M videos, 234M clips | Large web video-text corpus with generated captions | Web-sourced; review terms ⚠️ |
| [Panda-70M](https://snap-research.github.io/Panda-70M/) | ~70M clips | Captioned clips | Research use; source-video rights apply ⚠️ |
| [HD-VILA-100M](https://github.com/microsoft/XPretrain) | ~100M clips | High-resolution clip-text pairs | Research use ⚠️ |
| [WebVid-10M](https://github.com/m-bain/webvid) | ~10M | Stock-footage video-caption pairs | Distribution has been restricted; verify availability ⚠️ |
| [Kinetics-700](https://github.com/cvdfoundation/kinetics-dataset) | ~650K clips | Action-labeled clips, 700 classes | Action recognition pretraining |
| [Something-Something v2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) | ~220K clips | Object-interaction clips | Temporal reasoning; license terms apply ⚠️ |
| [Ego4D](https://ego4d-data.org/) 🔒 | 3,670 h | Egocentric video with audio | Gated access |
| [LAION-BVD](https://laion.ai/blog/bvd/) 🔗⚠️ | 1.3B URLs; ~13.5M h collected | Video URL discovery corpus (~80M source videos, 55M processed clips); [paper](https://arxiv.org/abs/2608.24845) unverified | Video foundation pretraining; media access is restricted and complex |
| [VideoUFO](https://huggingface.co/datasets/VideoUFO/VideoUFO) | 1.09M clips | Short and detailed captions across 1,291 user-oriented topics ([paper](https://arxiv.org/abs/2503.01739)) | Text-to-video |
| [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) | 43,751 videos (3,425 h) | Scene descriptions, speech transcripts, scene divisions and questions | Video-text, video QA |
| [ViMix-14M](https://arxiv.org/abs/2511.18382) ⚠️ | ~13.7M pairs (~22.8K h) | Long-form captions across mixed sources | Video-text training |
| [Vript](https://github.com/Vchitect/Vript) | 12K videos, 420K+ clips | Dense narrative video captions | Video captioning |
| [SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID) ⚠️ | 2.7M clips (7,089 h) | Real video with estimated depth, camera pose and motion annotations; non-commercial conditions on key components | Controllable and geometry-aware video |

### Mid-training

| Dataset | Size | Content | Notes |
| ---- | ---- | ---- | ---- |
| [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) | ~1M | High-quality text-video pairs | Quality-filtered for text-to-video |
| [Koala-36M](https://koala36m.github.io/) | ~36M clips | Clips with fine-grained captions and quality scoring | Curation reference ⚠️ |
| [MiraData](https://github.com/mira-space/MiraData) | 788K clips | Long clips with structured captions | Motion strength and long-take focus ⚠️ |

#### Action recognition and temporal understanding

Backbone data for video-language and temporal reasoning. Keep protected test splits of benchmarks out of training.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) | 13,320 clips | Action classification | Baseline |
| [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) | 6,766 clips | Human motion recognition | Baseline |
| [Ego-Exo4D](https://ego-exo4d-data.org/) 🔒 | — | Synchronized first- and third-person actions | Cross-view correspondence |
| [EPIC-KITCHENS](https://epic-kitchens.github.io/) | ~100 h (EPIC-KITCHENS-100) | Object manipulation and actions | Egocentric action understanding |

#### Segmentation, tracking, depth, flow and 3D

Geometry supervision for generation that should understand space rather than copy appearance. Combine exact synthetic ground truth (Kubric), measured RGB-D and large estimated annotations.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [SA-V](https://ai.meta.com/datasets/segment-anything-video/) | 51K videos, 643K masklets | Video segmentation with persistent object masks | Segmentation and tracking |
| [TAP-Vid](https://github.com/google-research/tapnet) | — | Point tracking through video | Tracking |
| [Kubric](https://github.com/google-research/kubric) | Generated | Rendered ground-truth flow, depth and segmentation | Synthetic geometry |
| [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) | 1,449 labeled RGB-D | Indoor RGB-D | Depth |
| [KITTI](https://www.cvlibs.net/datasets/kitti/) | — | Driving geometry and motion | Depth, flow, driving |
| [ScanNet](https://www.scan-net.org/) | 1,513 scans | Indoor 3D scenes, RGB-D, camera paths | 3D reconstruction |
| [DL3DV-10K](https://dl3dv-10k.github.io/) | 10,510 scenes | Multiview 3D visual geometry | 3D and novel views |
| [Google Scanned Objects](https://google.github.io/scanned-objects/) | — | Object geometry and multiview images | Object geometry |
| [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) | 22,872 pairs | Synthetic optical flow | Flow |
| [MPI Sintel](https://lmb.informatik.uni-freiburg.de/resources/datasets/Sintel.en.html) | — | Synthetic video optical flow | Flow |
| [Human3.6M](http://vision.imar.ro/human3.6m/) ⚠️ | 3.6M poses | 3D human pose; site availability unverified | 3D pose |
| [AMASS](https://amass.is.tue.mpg.de/) | 40+ h | Human motion sequences | Human motion |
| [SMPL](https://smpl.is.tue.mpg.de/) | — | Parametric human body model | Body geometry |

### Post-training

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) ⚠️ | 178K videos, 1.3M samples | Video instruction-following data | Video-language SFT; includes model-generated annotations |
| [VideoFeedback](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback) | ~37.6K videos | Human ratings of generated videos | Video quality reward models |
| [SMILE](https://github.com/kaist-ami/SMILE-Dataset) ⚠️ | 887 clips | Sitcom and TED clips | Understanding laughter in video |
| [FunnyNet-W](https://link.springer.com/article/10.1007/s11263-024-02000-2) ⚠️ | — | Sitcom video, audio and subtitles | Funny-moment localization |
| [MHD](https://delta-lab-iitk.github.io/Multimodal-Humor-Dataset/) ⚠️ | — | Sitcom laughter tracks | Laughter prediction |
| [UR-FUNNY](https://github.com/declare-lab/ur-funny) ⚠️ | — | TED-talk segments with multimodal context | Humor from speech and delivery |
| [MUStARD](https://lit.eecs.umich.edu/downloads.html) ⚠️ | — | Sitcom video, audio, text with context ([resources](https://github.com/soujanyaporia/MUStARD)) | Audiovisual sarcasm |
| [MultiHuSE](https://arxiv.org/abs/2609.11322) ⚠️ | — | Multimodal humor styles and emotions; unverified paper link | Acting, delivery and affect |

#### Video instruction and understanding

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [VideoChat2 training data](https://github.com/ShowLab/VideoChat2) | — | Video instruction tuning data | Video-language SFT |
| [Video-ChatGPT instructions](https://github.com/Video-ChatGPT/Video-ChatGPT) | — | Video QA and descriptions | Video-language SFT |
| [ShareGPT4Video](https://github.com/ShareGPT4Omni/ShareGPT4Video) | — | Video captions and instruction generation | Video captioning SFT |
| [Cambrian-S-3M](https://github.com/cambrian-mllm/cambrian-s) | 3M | Video instruction data | Video-language SFT |

#### Video editing and human-centric generation

Keep temporal correspondence (original and edited video, masks, tracked IDs, camera trajectory, timestamps, preservation constraints); framewise image edits lose identity, background continuity and motion.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [OpenVE-3M](https://huggingface.co/datasets/Lewandofski/OpenVE-3M) | 3M | Instruction-guided video editing across eight categories (style, background, local, add/remove, subtitles, multishot camera, creative) | Video editing |
| [OpenHumanVid](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_OpenHumanVid_A_Large-Scale_High-Quality_Dataset_for_Enhancing_Human-Centric_Video_Generation_CVPR_2025_paper.pdf) ⚠️ | 52.3M samples | Human-centric video with pose and speech annotations; check source availability and redistribution | Pose/speech-conditioned human video |

#### World models and action-conditioned video

Action-conditioned models need states, actions and transitions, not only captions. Embodiments, coordinate frames and action definitions differ, so record the action-space semantics. See also the embodied datasets under [Agentic training](#agentic-training).

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) | ~60K trajectories | Manipulation states and actions | Imitation, world models |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | Synthetic | Reproducible robot simulation | Control and manipulation |
| [CALVIN](https://github.com/mees/calvin) | — | Language-conditioned manipulation | Long-horizon robotic behavior |
| [EgoDex](https://github.com/apple/ml-egodex) | ~829 h | Hand motion and dexterity | Dexterous manipulation priors |
| [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset) ⚠️ | — | Autonomous-driving multi-camera sequences | Temporal perception |
| [nuScenes](https://www.nuscenes.org/) ⚠️ | — | Driving sequences with RGB and LiDAR | Scene geometry and motion |

#### Video preference and reward data

Separate motion plausibility from frame quality; individually attractive frames can still show impossible trajectories or identity changes.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [VideoScore](https://github.com/TIGER-AI-Lab/VideoScore) | — | Video quality reward modeling (with VideoFeedback) | Reward models |
| [MJ-VIDEO](https://github.com/aiming-lab/MJ-Video) | — | Fine-grained video preferences | Preference tuning |

### Evaluation

| Resource | Size | Measures |
| ---- | ---- | ---- |
| [VBench](https://github.com/Vchitect/VBench) 🧪 | 16 dimensions | Video generation quality across dimensions |
| [Video-MME](https://video-mme.github.io/) 🧪 | 900 videos, 2,700 QA | Video understanding by multimodal models |
| [VideoPhy](https://github.com/ShowLab/VideoPhy) 🧪 | — | Physical commonsense in video generation |
| [Video-Bench](https://github.com/Video-Bench/Video-Bench) 🧪 | — | Video QA and quality comparison |
| [VideoGen-Eval](https://github.com/VideoGen-Eval/VideoGen-Eval) 🧪 | — | Generation quality and adherence |
| [LongVideoBench](https://github.com/longvideobench/LongVideoBench) 🧪 | 6,678 questions | Long-form video grounding |
| [Video-MMMU](https://github.com/JUNJIE99/Video-MMMU) 🧪 | 300 videos, 900 questions | Expert video reasoning |
| [EgoSchema](https://github.com/egoschema/EgoSchema) 🧪 | 5K+ questions | Egocentric temporal reasoning |
| [TVQA](https://github.com/jayleicn/TVQA) 🧪 | ~152K QA | Video QA with subtitles |
| [MSR-VTT](https://github.com/MILVLG/msrvtt) 🧪 | 10K clips | Caption retrieval and video QA |

## Audio and Speech

A transcription dataset teaches recognition, not natural generation. A clean TTS corpus will not teach interruption handling, overlap or noisy-microphone robustness. The practical approach is **one unified speech corpus with several training views**: audio-only pretraining, audio-to-text, text-to-audio, speaker conditioning, speech-to-speech, conversation modeling and task supervision.

| Capability | Question it answers |
| ---------- | ------------------- |
| ASR | Audio → transcript |
| TTS | Text → generated speech |
| Speaker identity | Who is speaking? |
| Expressive speech | How is it being said? |
| Conversation | Turn-taking and interruptions |
| Translation | Speech across languages |
| Acoustic robustness | Noise, echo, reverberation |
| Speech reasoning | Audio → meaning and action |

**Stage mapping for audio**

| Stage | Data used | Objective |
| ----- | --------- | --------- |
| Pretraining | Very large raw or weakly labeled multilingual speech; codec and tokenizer audio | Acoustic and linguistic representations, audio codec |
| Mid-training | Clean transcribed, speaker-labeled and translated corpora; language-specific and robustness data | Supervised ASR, TTS, speaker conditioning, translation |
| Post-training | Expressive, conversational and instruction data | Style control, full-duplex interaction, spoken-request following |
| Evaluation | Held-out benchmarks | ASR, TTS, understanding and dialogue quality |

### Pretraining

Very large corpora for representation learning and weakly supervised multilingual recognition. Quality filtering, source-overlap checks and rights review are mandatory preprocessing here.

#### Largest speech corpora

| Dataset | Size | Best use | Access / licensing |
| ---- | ---- | ---- | ---- |
| [YODAS v3](https://huggingface.co/datasets/espnet/yodas3) | 1.1M h, 147 languages | Multilingual pretraining, ASR, generation | CC BY 3.0 release; underlying web-source rights need review ⚠️ |
| [NVIDIA Granary v2](https://huggingface.co/datasets/nvidia/Granary-v2) 🔗 | 463k h, 48 languages | Multilingual ASR and speech translation | Labels under OpenMDW 1.1; original audio fetched and licensed separately |
| [MOSEL](https://huggingface.co/datasets/FBK-MT/mosel) 🔗 | 444k h of pseudo-labels | Multilingual ASR | Overlaps source corpora; audio rights apply |
| [RealSpeech Multilingual 350k](https://huggingface.co/datasets/TTS-AGI/tts-realspeech-multilingual-350k-moss-v2) 🔒 | 350k h | Codec-based multilingual TTS | Gated; mixed per-source rights |
| [YODAS v1](https://huggingface.co/datasets/espnet/yodas) | ~370k h | Weakly labeled multilingual ASR | Largely superseded by and overlapping with newer YODAS |
| [Emilia2](https://arxiv.org/abs/2611.13614) ⚠️ | 229k unique h, 41 languages | Long-form TTS, dialogue, expressive voice | Recent research release; unverified, confirm distribution and license |
| [Emilia-Large](https://huggingface.co/datasets/amphion/Emilia-Dataset) ⚠️ | 215.6k h | Multilingual speech generation | Original Emilia subset is CC BY-NC 4.0 |
| [WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech) | 65k+ h, 127 language-region variants | Human-transcribed multilingual ASR | Mixed origin; review provenance |
| [Libri-Light](https://github.com/facebookresearch/libri-light) | 60k+ h | Self-supervised pretraining | Confirm audio rights separately from repo license |
| [Multilingual LibriSpeech](https://www.openslr.org/94/) | 50k+ h, 8 languages | Multilingual ASR, TTS adaptation | CC BY 4.0 |
| [The People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech) | 30k+ h | English ASR | License differs across source subsets |
| [WenetSpeech](https://wenet-e2e.github.io/WenetSpeech/) ⚠️ | 22.4k+ h | Mandarin ASR | Additional non-commercial terms |
| [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) ⚠️ | ~10k transcribed h | English ASR | Research and source-use restrictions |

**Never sum these figures.** Granary, MOSEL, YODAS and Emilia-YODAS reuse or relabel recordings that appear in other releases. The biggest dataset is not the most useful: weak captions, noisy segmentation, artifacts and licensing limits can make hundreds of thousands of hours worth less for TTS than a smaller curated collection.

#### Audio-only speech pretraining

| Dataset | Size | Use |
| ---- | ---- | ---- |
| [YODAS v3](https://huggingface.co/datasets/espnet/yodas3) | 1.1M h | Large-scale multilingual representation learning |
| [YODAS2](https://huggingface.co/datasets/espnet/yodas2) | — | Longer-context audio with timestamps |
| [VoxPopuli unlabeled](https://github.com/facebookresearch/voxpopuli) | ~400k h | Multilingual self-supervised pretraining |
| [Libri-Light](https://github.com/facebookresearch/libri-light) | 60k+ h | English representations, speech-tokenizer training |
| [LibriHeavy](https://github.com/k2-fsa/libriheavy) | ~50k h | Long-form LibriVox-derived ASR segments |
| [Granary source collection](https://huggingface.co/datasets/nvidia/Granary-v2) 🔗 | — | Pretraining after source-audio acquisition |

### Mid-training

Supervised corpora with trustworthy labels, grouped by task, followed by language-specific and robustness data.

#### Automatic speech recognition

##### General English and multilingual

| Dataset | Size | Characteristics |
| ---- | ---- | ---- |
| [LibriSpeech](https://www.openslr.org/12/) | 960 h, English | Standard clean/other audiobook ASR |
| [LibriSpeech-PC](https://www.openslr.org/145/) | ~960 h | Punctuated, capitalized labels |
| [Multilingual LibriSpeech (MLS)](https://www.openslr.org/94/) | 50k+ h, 8 languages | Read speech |
| [The People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech) | 30k+ h | Diverse public sources |
| [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) ⚠️ | 10k transcribed h | Podcasts, audiobooks, online video |
| [WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech) | 65k+ h | Human transcripts with quality scores |
| [Granary v2](https://huggingface.co/datasets/nvidia/Granary-v2) 🔗 | 463k h | Synthetic transcription and translation labels |
| [MOSEL](https://huggingface.co/datasets/FBK-MT/mosel) 🔗 | 444k h | Multilingual pseudo-labels |
| [YODAS v3](https://huggingface.co/datasets/espnet/yodas3) | 1.1M h | Weakly labeled, 48 kHz multichannel |
| [OWSM v4 cleaned YODAS](https://huggingface.co/datasets/espnet/yodas_owsmv4) | — | Cleaner YODAS training data |
| [Mozilla Common Voice](https://commonvoice.mozilla.org/) | Multi-language, version-dependent | Crowdsourced read speech, CC0 releases |
| [VoxPopuli](https://github.com/facebookresearch/voxpopuli) | ~1,800 labeled h; ~400k unlabeled h | European parliamentary speech |
| [SPGISpeech](https://huggingface.co/datasets/kensho/spgispeech) ⚠️ | 5,000 h, English | Earnings calls; research-use agreement |
| [TED-LIUM 3](https://www.openslr.org/51/) | ~450 h, English | Presentations and lectures |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) 🧪 | ~1,200 h, 102 languages | Multilingual evaluation and adaptation |
| [ESB (Earnings-21/22 and others)](https://huggingface.co/datasets/esb/datasets) 🧪 | — | Primarily evaluation |

LibriSpeech and MLS are comparatively straightforward audiobook corpora. GigaSpeech's access terms are stricter than its repository metadata suggests. Granary v2 is mostly labels and instructions that point to independently distributed audio.

##### Chinese, Arabic, Indic and other

| Dataset | Size | Notes |
| ---- | ---- | ---- |
| [THCHS-30](https://www.openslr.org/18/) | 30 h, Mandarin | Small clean baseline |
| [AISHELL-1](https://www.openslr.org/33/) | 178 h, Mandarin | High-quality read speech |
| [AISHELL-2](https://www.aishelltech.com/aishell_2) ⚠️ | ~1,000 h, Mandarin | Academic access restrictions |
| [ST-CMDS](https://www.openslr.org/38/) ⚠️ | ~110 h, Mandarin | Non-commercial, no-derivatives |
| [Primewords](https://www.openslr.org/47/) ⚠️ | ~99 h, Mandarin | Restricted derivative/commercial rights |
| [MAGICDATA Mandarin](https://www.openslr.org/68/) | ~755 h | Read speech |
| [MAGICDATA-RAMC](https://www.openslr.org/123/) | ~180 h | Conversational Mandarin |
| [WenetSpeech](https://wenet-e2e.github.io/WenetSpeech/) ⚠️ | 22k+ h | Multiple domains; subset labels vary |
| [GigaSpeech 3](https://huggingface.co/datasets/speechcolab/gigaspeech3) ⚠️ | 39,028 h, five Arabic dialects | Weak labels; research and evaluation only |
| [FalAR](https://huggingface.co/datasets/inesc-id/FalAR) | ~4,700 h, Arabic | Also indexed by Granary v2 |
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/indicvoices) | Indian languages | Speech diversity; [project page](https://github.com/AI4Bharat/IndicVoices) |

#### Text-to-speech and voice generation

TTS audio must preserve speaker identity, pronunciation, rate, prosody, pauses, microphone conditions and emotional style.

| Dataset | Size | Advantage |
| ---- | ---- | ---- |
| [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | 24 h | Classic clean single-speaker English |
| [LibriTTS](https://www.openslr.org/60/) | 585 h | Multi-speaker English with transcripts |
| [LibriTTS-R](https://www.openslr.org/141/) | ~585 h | Restored audio quality |
| [Hi-Fi TTS](https://www.openslr.org/109/) | ~292 h | High-quality multi-speaker audiobooks |
| [HiFiTTS-2](https://huggingface.co/datasets/nvidia/hifitts-2) | ~36.7k h | Higher-bandwidth LibriVox-derived speech |
| [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) | ~44 h | 109 English speakers, accent diversity |
| [AISHELL-3](https://www.openslr.org/93/) | 85 h | Mandarin multi-speaker |
| [M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) | ~1,000 h, 9 languages | Audiobook speakers in several languages |
| [CSS10](https://github.com/Kyubyong/css10) | 10 languages | Single-speaker multilingual baseline |
| [DataBaker](https://www.data-baker.com/data/index/source) | — | Phonetically balanced speech |
| [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) ⚠️ | 101.7k h | In-the-wild TTS; original hours non-commercial |
| [Emilia-YODAS](https://huggingface.co/datasets/amphion/Emilia-Dataset) | 113.9k h | Six-language TTS; CC BY 4.0, subject to underlying rights |
| [Emilia2](https://arxiv.org/abs/2611.13614) ⚠️ | 229k unique h | Long monologues and multi-speaker dialogue with timestamps, 4.77M style instructions, nonverbal labels |
| [RealSpeech 350k](https://huggingface.co/datasets/TTS-AGI/tts-realspeech-multilingual-350k-moss-v2) 🔒 | 350k h | Speech, speaker features, codec tokens |
| [Parler-TTS Data-Speech collection](https://huggingface.co/collections/parler-tts/open-source-speech-datasets-annotated-using-data-speech) | — | Natural-language voice/style descriptions |
| [Nord-Parl-TTS](https://huggingface.co/datasets/Aalto-Speech-Synthesis/Nord-Parl-TTS) 🔒⚠️ | ~5,990 h | Finnish and Swedish parliament speech |

LJSpeech, LibriTTS(-R), VCTK and HiFiTTS are good first TTS corpora. Emilia's views (short utterances, monologues, dialogues) overlap and are not independent recordings; do not treat Emilia2 as commercially cleared before inspecting its final license and provenance.

#### Swedish and Nordic speech

| Dataset | Size | Use | Licensing |
| ---- | ---- | ---- | ---- |
| [RixVox](https://huggingface.co/datasets/KBLab/rixvox) | 5,500 h parliamentary speech, 2003–2023 | ASR, speaker modeling | CC BY 4.0. Transcripts are protocol-aligned, not always verbatim; verify before TTS use |
| [Nord-Parl-TTS](https://huggingface.co/datasets/Aalto-Speech-Synthesis/Nord-Parl-TTS) | 5,090 h Swedish + 900 h Finnish | Nordic TTS | 🔒⚠️ CC BY-NC-ND 4.0, gated |
| [KTH/NST Swedish ASR](https://huggingface.co/datasets/KTH/nst) | — | ASR | Check original language-bank terms |
| [Waxholm](https://huggingface.co/datasets/KTH/waxholm) | — | Spoken dialogue, ASR research | Verify source terms |
| [SpeechDat Swedish](https://huggingface.co/datasets/KTH/speechdat) | — | Telephony ASR | Verify access and rights |
| [Common Voice Swedish](https://commonvoice.mozilla.org/) | Version-dependent | Read speech | CC0 |
| [Swedish Common Voice processed](https://huggingface.co/datasets/LokaalHub/sv-SE-asr-cv) | Derived from CV22 | ASR | CC0 source; check processing |
| [TTS-Swedish](https://huggingface.co/datasets/datadriven-company/TTS-Swedish) ⚠️ | — | Speech generation | Verify card license |
| [YodaLingua Swedish](https://huggingface.co/datasets/Thomcles/YodaLingua-Swedish) | Tens of thousands of samples | ASR/TTS | Verify provenance and license |
| [Swedish Conversational Speech](https://huggingface.co/datasets/SpeechDataAI/swedish-conversational-speech) 🔒 | Advertised 500 h | Natural conversation | Paid license; audio not on HF |
| [FLEURS Swedish](https://huggingface.co/datasets/google/fleurs) 🧪 | ~12 h | Evaluation | CC BY 4.0 |
| [VoxPopuli Swedish](https://huggingface.co/datasets/facebook/voxpopuli) | Small labeled, larger unlabeled | Parliamentary ASR, pretraining | Review source terms |

#### Speaker identity, verification and diarization

| Dataset | Size | Purpose | Notes |
| ---- | ---- | ---- | ---- |
| [VoxCeleb1 and VoxCeleb2](https://mm.kaist.ac.kr/datasets/voxceleb/) ⚠️ | 7,363 speakers, 1M+ utterances | Speaker recognition and representation training | Research release with attribution; not blanket voice-cloning rights. [Original resources](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) |
| [VoxConverse](https://mm.kaist.ac.kr/datasets/voxconverse/) | ~64 h | Diarization | Multispeaker natural recordings |
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) | ~100 h | Meeting diarization | Speaker turns and overlap |
| [AliMeeting](https://www.openslr.org/119/) | ~119 h | Mandarin meeting diarization | Multiple microphones |
| [DIHARD III](https://dihardchallenge.github.io/dihard3/) 🔒 | — | Diarization in difficult conditions | LDC-linked resources |
| [CHiME-6](https://chimechallenge.github.io/chime6/) | ~50 h | Distant-microphone diarization | Real domestic conditions |
| [Ego4D](https://ego4d-data.org/) 🔒 | 3,670 h | Egocentric speaking and listening | Gated audiovisual source |

Also useful for speaker work: [RixVox](https://huggingface.co/datasets/KBLab/rixvox) (Swedish) and [Basis Conversations 1500](https://withbasis.co/open/conversations-1500) (multi-track).

#### Speech translation and speech-to-speech

| Dataset | Size | Data type | Coverage |
| ---- | ---- | ---- | ---- |
| [CoVoST 2](https://ai.meta.com/tools/covost/) | 2,880 h | Speech → translated text | Many language pairs |
| [MuST-C](https://ict.fbk.eu/must-c/) | ~400+ h per language | Speech → translated text | TED speech with translations (v1 and v2) |
| [Europarl-ST](https://www.mllp.upv.es/europarl-st/) | — | Multilingual speech translation | European parliament |
| [CVSS](https://github.com/google-research-datasets/cvss) | — | Speech → translated speech | Common Voice-derived targets |
| [SeamlessAlign](https://github.com/facebookresearch/seamless_communication) | ~470k h | Mined speech/text alignments | Very large multilingual |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) 🧪 | ~1,200 h | Parallel spoken content | 102-language benchmarking |
| [VoxPopuli interpretations](https://huggingface.co/datasets/facebook/voxpopuli) | — | Simultaneous interpretation | Spoken language pairs |
| [Granary v2 AST](https://huggingface.co/datasets/nvidia/Granary-v2) 🔗 | — | Audio → translation instructions | Multilingual synthetic labels |

Translations and synthesized target audio may carry different rights from the source recordings.

#### Noise, reverberation, enhancement and separation

A model trained only on clean utterances fails on microphone noise, reverberation, overlapping speakers and background sound.

| Dataset | Size | Capability |
| ---- | ---- | ---- |
| [Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge) | — | Noise suppression; mixing infrastructure and license info |
| [MUSAN](https://www.openslr.org/17/) | ~109 h | Music, speech, noise augmentation |
| [WHAM! / WHAMR!](https://wham.whisper.ai/) | — | Noisy and reverberant speech separation |
| [LibriMix](https://github.com/JorisCos/LibriMix) | — | Synthetic overlapping speech mixtures |
| [VoiceBank + DEMAND](https://datashare.ed.ac.uk/handle/10283/1942) | ~11.6K train utterances | Paired noisy/clean speech |
| [DEMAND](https://zenodo.org/records/1227121) | 18 environments | Real environmental noise |
| [AudioSet](https://research.google.com/audioset/) | ~2M clips (~5.8k h) | Sound-event recognition |
| [FSD50K](https://zenodo.org/records/4060432) | 51,197 clips (~108 h) | Environmental audio classification (mixed CC terms; filter per file) |
| [Room impulse responses](https://www.openslr.org/28/) | — | Reverberation augmentation |
| [URGENT Challenge data](https://urgent-challenge.github.io/urgent2025/data/) | — | Universal speech enhancement |

### Post-training

Style, interaction and instruction data that sit on top of a model that can already recognize and produce speech.

#### Expressive speech, emotion and vocalizations

ASR corpora give little supervision for delivery style, laughter, breathing, hesitations or pacing. Pipelines often strip these as noise, yet for natural generation they are the target.

| Dataset | Size | Capability | Access / caution |
| ---- | ---- | ---- | ---- |
| [Basis Global Emotions](https://huggingface.co/datasets/basis-ai/basis-global-emotions) | 525 h | Multilingual emotion and delivery, 108 locales | Review dataset agreement |
| [Expresso](https://huggingface.co/collections/parler-tts/parler-tts-expresso) | ~47 h | Expressive speech styles | Research corpus; inspect rights |
| [Parler-TTS MLS annotations](https://huggingface.co/datasets/parler-tts/mls_eng_10k) | ~10k h | Voice descriptions, acoustic tags | MLS-derived; check version |
| [Emilia-NV / NVSpeech](https://huggingface.co/datasets/amphion/Emilia-NV) | — | Laughter, breathing, interjections | Mandarin nonverbal annotations |
| [NonverbalTTS](https://huggingface.co/datasets/deepvk/NonverbalTTS) | ~17 h | Breathing, sighing, coughing, laughing | ~17 h, English |
| [Vocal Bursts](https://huggingface.co/datasets/TTS-AGI/vocal-bursts) | — | Non-speech human vocal sounds | Verify per-source licenses |
| [ExpressiveSpeech](https://huggingface.co/datasets/FreedomIntelligence/ExpressiveSpeech) | — | Emotional conversational speech | Mixed-source lineage |
| [CREMA-D](https://huggingface.co/datasets/CREMA-D/CREMA-D) ⚠️ | 7,442 clips | Emotion recognition/conditioning | Check original terms |
| [RAVDESS](https://zenodo.org/records/1188976) | 7,356 files | Acted emotional speech and song | Commercial license is separate |
| [IEMOCAP](https://sail.usc.edu/iemocap/) 🔒 | ~12 h | Emotion, conversational prosody | Academic/request access |
| [MSP-Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) 🔒 | — | Natural speech emotions | Access approval |
| [MSP-IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-IMPROV.html) 🔒 | ~9.8 h | Improvised expressive speech | Restricted |
| [EmoV-DB](https://github.com/numediart/EmoV-DB) | — | Emotional voice synthesis | Check recording rights |
| [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) | ~29 h | English and Mandarin emotional TTS | Verify original terms |
| [ESC-50](https://huggingface.co/datasets/ashraq/esc50) | 2,000 clips | Environmental sound recognition | Auxiliary audio understanding |

#### Conversational speech and full-duplex

Real-time voice agents need continuous streams, timestamps, multiple speakers, overlap, interruptions, backchannels and natural pauses. Turn-based ASR/TTS pairs cannot teach these.

| Dataset | Size | Value |
| ---- | ---- | ---- |
| [Basis Conversations 1500](https://withbasis.co/open/conversations-1500) | 1,502 h, 22 languages | Up to four simultaneous speakers, channel-separated 48 kHz, ~100 h of detailed annotation; publisher states commercial training is permitted under its release terms |
| [Open Yap 1K](https://huggingface.co/datasets/TheAgenticDataCompany/open-yap-1k) 🔒 | 1,000 h | Dual-channel full-duplex; public sample is 8.9 h, full corpus needs a data-use agreement; commercial training allowed, cloning dataset speakers prohibited |
| [SmoothConv](https://github.com/qualialabsAI/SmoothConv-DuplexConv) | 100 h | Human-annotated conversational dynamics |
| [DuplexConv](https://github.com/qualialabsAI/SmoothConv-DuplexConv) | 2,000 h | Annotated Chinese conversation pretraining |
| [Fisher English Part 1](https://catalog.ldc.upenn.edu/LDC2004S13) 🔒 | 984 h | Telephone conversation (LDC) |
| [Fisher English Part 2](https://catalog.ldc.upenn.edu/LDC2005S13) 🔒 | 975 h | Telephone conversation (LDC) |
| [Switchboard-1](https://catalog.ldc.upenn.edu/LDC97S62) 🔒 | ~260 h | Turns and disfluency (LDC) |
| [CALLHOME corpora](https://catalog.ldc.upenn.edu/search) 🔒 | — | Real telephone conversations (LDC) |
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | ~100 h | Meetings, overlapping speech |
| [AliMeeting](https://www.openslr.org/119/) | ~119 h | Mandarin meetings, diarization |
| [CHiME-6](https://chimechallenge.github.io/chime6/) | ~50 h | Multichannel dinner-party conversations |
| [VoxConverse](https://www.mm.kaist.ac.kr/datasets/voxconverse/) | ~50–64 h | In-the-wild speaker turns |
| [otoSpeech full-duplex](https://www.oto.earth/) 🔒 | — | Channel-separated conversation; access varies |
| [Basis consented conversation collections](https://withbasis.co/) 🔒 | — | Conversation, speaker and preference annotations |

Modern multichannel recordings keep far more acoustic information than Fisher/Switchboard for realistic synthesis and interaction modeling. For broader discovery see the [full-duplex speech dataset index](https://www.fullduplex.ai/datasets) (159 datasets at review time).

#### Speech-language understanding

| Dataset | Size | Capability |
| ---- | ---- | ---- |
| [MINDS-14](https://huggingface.co/datasets/PolyAI/minds14) | — | Spoken intent classification |
| [Speech Commands](https://huggingface.co/datasets/google/speech_commands) | 105,829 utterances | Keyword recognition |
| [SUPERB](https://huggingface.co/datasets/superb) 🧪 | — | Speech representation evaluation |
| [AudioBench](https://huggingface.co/datasets/AudioLLMs/AudioBench) 🧪⚠️ | 26 datasets | Audio understanding benchmark suite |
| [Granary v2 instruction targets](https://huggingface.co/datasets/nvidia/Granary-v2) 🔗 | — | Contextual ASR, captioning, QA, translation |

Grounded instruction data can be derived from audio with trusted transcripts (transcribe, summarize, extract action items, find a stated deadline, answer questions). Keep derived records linked to the original audio and transcript and carry the source license. Generated QA is not human-verified labeling; use the latter for evaluation.

#### Singing and specialized vocal production

| Dataset | Size | Purpose |
| ---- | ---- | ---- |
| [GTSinger](https://huggingface.co/datasets/GTSinger/GTSinger) | ~80 h | Multilingual singing, technique, scores; separates pitch, phonemes, technique and speaker |
| [OpenCPOP](https://wenet-e2e.github.io/opencpop/) | ~5.2 h | Mandarin singing with note/phoneme alignment |
| [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger) | — | Singing voice synthesis resources |
| [VocalSet](https://zenodo.org/records/1193957) | ~10.1 h | Sung vocal techniques and phonation |
| [CrawlSinger-OS](https://huggingface.co/datasets/pymaster/CrawlSinger-OS) | — | Processed singing/audio-score material |

#### Stand-up comedy, timing and laughter

Timing, pauses, prosody and audience response, which large TTS corpora lack. These projects mostly release annotations and metadata, not reusable audio or video; the underlying performances carry their own rights.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [StandUp4AI](https://github.com/Standup4AI/dataset) 🔗 | 334 h, 3,617 performances, 130K+ laughter labels | Multilingual stand-up with laughter timestamps and performance features; media not redistributed | Jokes, audience response, timing |
| [TIC-TALK](https://huggingface.co/datasets/ENC-PSL/TIC-TALK) | 5,416 aligned records | Verbal material aligned with audio, laughter and body movement (2026 research release) | Timing, laughter and kinesics |
| [Indonesian Stand-up Dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC12538016/) | — | Stand-up transcripts with laughter annotations | Non-English comedy timing |

### Evaluation

Keep these out of training data.

| Resource | Size | Measures |
| ---- | ---- | ---- |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | ~1,200 h | Multilingual ASR and speech translation; includes Swedish |
| [ESB benchmark](https://huggingface.co/datasets/esb/datasets) | — | English ASR across domains (earnings calls and others) |
| [SUPERB](https://huggingface.co/datasets/superb) | — | Speech representation quality |
| [AudioBench](https://huggingface.co/datasets/AudioLLMs/AudioBench) ⚠️ | 26 datasets | Audio understanding and speech question answering |
| [VoxConverse](https://mm.kaist.ac.kr/datasets/voxconverse/) | ~64 h | Diarization |
| [CHiME-6](https://chimechallenge.github.io/chime6/) | ~50 h | Distant-microphone conversational recognition |

### Training strategy and shortlist

| Training stage | Data | Objective |
| -------------- | ---- | --------- |
| Audio tokenizer / codec | High-fidelity audio, many speakers and conditions | Reconstruct audio with minimal perceptual loss |
| Representation pretraining | Large multilingual raw speech | Acoustic and linguistic representations |
| ASR | Human transcripts, then quality-controlled pseudo-labels | Audio → text |
| TTS | Clean transcript/audio/speaker pairs | Text → audio |
| Speaker-conditioned synthesis | Multi-speaker recordings with reference clips | Voice conditioning |
| Expressive synthesis | Style-labeled, emotional, nonverbal speech | Prosody and style control |
| Speech-language instruction tuning | Audio plus grounded task-response pairs | Understand spoken requests |
| Speech-to-speech interaction | Multichannel dialogue with turn annotations | Listening, speaking, interruption handling |
| Specialized adaptation | Swedish, engineering vocabulary, noisy environments | Domain and language accuracy |

Quality thresholds differ by task. A noisy recording can serve ASR if the transcript is accurate but be unusable for TTS (noise, clipping, music, overlap). For dialogue, overlap and interruptions are signal and must be preserved. Build several **views** over the same recordings instead of one universal "clean audio" filter.

**Suggested shortlist** (intake candidates, not an assertion that all are commercially licensed):

- *ASR foundation*: LibriSpeech, Multilingual LibriSpeech, WorldSpeech, RixVox, Common Voice; YODAS v3 and Granary v2 after rights review.
- *TTS and generation*: LibriTTS-R, HiFiTTS-2, VCTK, LJSpeech; Emilia-YODAS, Emilia2 and RealSpeech 350k after provenance/license review.
- *Natural conversation*: Basis Conversations 1500, Open Yap 1K, AMI, SmoothConv/DuplexConv.
- *Expressiveness and robustness*: Basis Global Emotions, NonverbalTTS, Emilia-NV, MUSAN, Microsoft DNS Challenge.

Discovery beyond this list: [Audio/Speech Datasets catalog](https://github.com/weimeng23/audio-speech-datasets) for smaller regional and specialized corpora.

## Images

> **Seed section.** Like the video section, these entries have not been through a full license and provenance review. Large web-scraped image-text sets contain content with unresolved copyright and, historically, content that publishers have had to remove; review current versions and terms before use.

| Stage | Data type | Typical objective |
| ----- | --------- | ----------------- |
| Pretraining | Billions of web image-text pairs, labeled image collections | Contrastive, captioning or diffusion pretraining |
| Mid-training | Aesthetic, high-resolution and re-captioned subsets, dense annotation | Quality, resolution and caption-fidelity refinement |
| Post-training | Visual instructions, editing pairs, human preferences | Vision-language SFT, instruction editing, preference tuning |
| Evaluation | Held-out benchmarks | Generation and understanding quality |

### Pretraining

| Dataset | Size | Content | Notes |
| ---- | ---- | ---- | ---- |
| [LAION-5B](https://laion.ai/blog/laion-5b/) | ~5.85B pairs | CLIP-filtered image-text pairs (URLs and metadata) | 🔗⚠️ Metadata only; use the cleaned Re-LAION release where available |
| [DataComp](https://www.datacomp.ai/) | up to 12.8B pairs | Benchmark and pools of image-text pairs | 🔗 Designed for studying data filtering |
| [COYO-700M](https://github.com/kakaobrain/coyo-dataset) | ~747M pairs | Image-text pairs | 🔗⚠️ URLs and metadata |
| [Conceptual 12M](https://github.com/google-research-datasets/conceptual-12m) | ~12M pairs | Image-text pairs | 🔗 URLs and captions |
| [ImageNet](https://www.image-net.org/) | 1.28M train images (1K classes) | 1K-class training images; larger 21K release | ⚠️ Research and non-commercial terms |
| [Open Images](https://storage.googleapis.com/openimages/web/index.html) | ~9M images | Images with labels, boxes, relations | Annotations CC BY; images listed as CC BY |
| [Re-LAION-5B](https://laion.ai/blog/relaion-5b/) 🔗 | 5.5B links | Cleaned image-text links (metadata and URLs) | Web-scale image-language pretraining |
| [DataComp-1B](https://huggingface.co/datasets/mlfoundations/datacomp_1b) 🔗 | ~1.38B candidates | Image-text candidate pool | Filtering research |
| [DataComp-HQ](https://huggingface.co/datasets/laion/datacomp-hq) 🔗 | — | Curated image-text metadata | Quality-aware corpus selection |
| [Re-LAION Caption 19M](https://huggingface.co/datasets/supermodelresearch/Re-LAION-Caption19M) 🔗 | ~19M | Recaptioned image metadata | Image-generation data |
| [Conceptual Captions 3M](https://github.com/google-research-datasets/conceptual-captions) 🔗 | ~3.3M pairs | Image-caption pairs (URLs and captions) | Caption alignment |
| [WIT (Wikipedia-based Image Text)](https://github.com/google-research-datasets/wit) 🔗 | 37M+ examples | Multilingual images with contextual text | Multilingual image-text |
| [RedCaps](https://github.com/redcaps-dataset/redcaps) ⚠️ | ~12M pairs | Reddit image-caption pairs | Image-caption distribution |
| [JourneyDB](https://huggingface.co/datasets/JourneyDB/JourneyDB) ⚠️ | 4.43M images | Generated images with prompts, captions and VQA | Prompt following, stylized imagery |
| [DiffusionDB](https://huggingface.co/datasets/jobs-git/diffusiondb) | 14M images | Generated images with prompts | Prompt-image supervision |
| [FFHQ](https://github.com/NVlabs/ffhq-dataset) ⚠️ | 70K images | High-quality faces | Face reconstruction (VAE/tokenizer) |
| [CelebA / CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) ⚠️ | 200K+ photos; 30K HQ | Face photos | Portrait reconstruction |
| [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | 1K images | High-resolution images | Fine detail, super-resolution |
| [Flickr30K](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) ⚠️ | ~31K images | Image-caption pairs | Small caption training/validation |

### Mid-training

| Dataset | Size | Content | Notes |
| ---- | ---- | ---- | ---- |
| [Recap-DataComp-1B](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B) | ~1B pairs | DataComp-1B images with model-generated recaptions | Caption-quality improvement |
| [LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/) | — | Aesthetic-score-filtered LAION subsets | 🔗⚠️ Inherits LAION caveats |
| [COCO](https://cocodataset.org/) | ~330K images | Images with captions, boxes, segmentation | Annotations CC BY 4.0; images carry Flickr terms |
| [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/) | ~108K images | Dense region descriptions and relationships | Grounded, dense annotation |
| [SA-1B](https://ai.meta.com/datasets/segment-anything/) | ~11M images, 1.1B masks | Segmentation masks on images | Segmentation foundation models |
| [HumorDB](https://github.com/kreimanlab/HumorDB) ⚠️ | 3,542 images | Controlled funny vs minimally different non-funny image pairs with ratings; verify source-image permissions | Visual humor, contrastive funniness |
| [MemeLens](https://huggingface.co/datasets/QCRI/MemeLens) ⚠️ | 46 tasks | Multilingual meme sets | Meme explanation and contextual understanding |
| [AHA-MEMES](https://huggingface.co/datasets/QCRI/AHA-MEMES) ⚠️ | 5,000 memes | Arabic memes | Humor, cultural references, interpretation |
| [Meme Text Corpus](https://huggingface.co/datasets/Alogotron/meme-text-corpus) ⚠️ | — | Aggregated meme OCR and labels | Meme text pretraining |
| [HUMOR-Datasets](https://huggingface.co/datasets/AnonySubmission/HUMOR-Datasets) ⚠️ | — | Memes and less-humorous rewrites | Contrastive humor learning |

#### Detection, segmentation and grounding

Structured spatial supervision.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [Objects365](https://www.objects365.org/) | 365 categories | Large-vocabulary object detection | Detection |
| [LVIS](https://www.lvisdataset.org/) | 1,203 categories | Long-tail instance segmentation | Segmentation |
| [RefCOCO family](https://refcoco.github.io/) | — | Language-conditioned object grounding | Referring expressions |
| [Localized Narratives](https://github.com/google-research-datasets/LocalizedNarratives) | ~849K narratives | Grounded natural-language descriptions | Grounded captioning |

#### OCR, documents, charts and diagrams

General natural-image data does not supply dense text supervision; these matter for PDFs, screenshots, technical drawings and documentation.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [TextOCR](https://rrc.cvc.uab.es/?ch=13) | ~28K images | Text detection and recognition | OCR |
| [DocVQA](https://www.docvqa.org/) | 12K images, 50K questions | Question answering about documents | Document VQA |
| [Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix) | ~2.5M | Generated document questions | Document instruction tuning |
| [DoclingMatix](https://huggingface.co/datasets/HuggingFaceM4/DoclingMatix) | ~1.27M | Document understanding | Document instruction tuning |
| [AI2D](https://github.com/allenai/ai2d) | ~5K diagrams | Diagram interpretation | Diagram QA |
| [ChartQA](https://huggingface.co/datasets/ChartQA/ChartQA) | ~32.7K questions | Chart understanding | Chart QA |
| [InfographicVQA](https://github.com/google-research-datasets/infographic-vqa) | 5.5K images, 30K questions | Infographics and complex layouts | Infographic QA |
| [CORD](https://github.com/clovaai/cord) | 1,000 receipts | Receipt and document parsing | Document parsing |
| [PixMo-Docs](https://huggingface.co/datasets/allenai/pixmo-docs) | — | Document-heavy visual understanding | Document VQA |
| [DataComp-VLM](https://arxiv.org/abs/2606.28551) ⚠️ | 160 datasets, ~6T tokens | Image-caption, interleaved, text-only and instruction mixtures with a controlled 8B-model benchmark; unverified paper link | Foundation mixture design |

### Post-training

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) ⚠️ | ~150K | Visual instruction-following (model-generated) | Vision-language SFT |
| [InstructPix2Pix dataset](https://github.com/timothybrooks/instruct-pix2pix) | ~450K triplets | Instruction-based image-edit triplets | Instruction editing |
| [Pick-a-Pic v2](https://huggingface.co/datasets/yuvalkirstain/pickapic_v2) ⚠️ | ~959K pairs | Human preferences over generated images | Preference tuning, reward models |
| [ImageReward](https://github.com/THUDM/ImageReward) | ~137K pairs | Human rankings for text-to-image output | Reward modeling |
| [New Yorker Caption Contest Benchmarks](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest) ⚠️ | — | Cartoon-caption challenge tasks | Matching, explanation, ranking |
| [Cartoon Caption Generation](https://github.com/yguooo/cartoon-caption-generation) | — | Training and evaluation methodology; builds on [New Yorker Caption Ranking](https://huggingface.co/datasets/yguooo/newyorker_caption_ranking) | Preference-based cartoon caption models |

#### Vision-language instruction mixtures

Alignment comes from captions; precise questions, OCR, localization and spatial reasoning need structured instruction data.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) | 24M+ examples | Large curated multimodal instruction mixture | Vision-language instruction tuning |
| [LLaVA-OneVision-1.5](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data) | 22M examples | Large instruction corpus | Large VLM post-training |
| [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) | ~1.9M | Multimodal instruction datasets | Multimodal instructions |
| [PixMo-Cap](https://huggingface.co/datasets/allenai/pixmo-cap) | — | Human image descriptions | Detailed visual captioning |
| [LLaVA-Instruct-665K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-665K) ⚠️ | ~665K | General VQA and instruction data | VLM instruction tuning |
| [PerceptionLM Data](https://ai.meta.com/datasets/plm-data/) | Multiple datasets | Detailed image and video comprehension | VLM instruction tuning |

#### Image editing and controllable generation

Keep separate projections for instruction edits, inpainting, control signals (depth, pose, edges), multi-reference composition and preservation-aware editing; do not collapse them into caption data. Generated edit targets and derivatives often carry upstream non-commercial terms.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [Pico-Banana-400K](https://github.com/apple/pico-banana-400k) ⚠️ | 400K edits | Open Images-derived edit pairs with quality scoring; non-commercial terms | Instruction-guided editing |
| [GPT-Image-Edit-1.5M](https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M) ⚠️ | ~1.5M edits | Instruction-edit supervision; upstream components include CC BY-NC and BY-NC-SA | Broad editing |
| [HQ-Edit](https://huggingface.co/datasets/UCSC-VLAA/HQ-Edit) | 197,350 edits | High-quality local and global edits | Instruction editing |
| [UltraEdit](https://huggingface.co/datasets/BleachNick/UltraEdit) | Million-scale | Editing triplets | Instruction editing |
| [RCEdit-500K](https://huggingface.co/datasets/carpedkm/RCEdit-500K) | ~472K rows | Image-conditioned editing | Instruction editing |
| [VIVID-10M](https://huggingface.co/datasets/KlingTeam/VIVID-10M) | ~10M-class | Localized and interactive editing; check modality subsets | Image/video editing |
| [ConceptEdit-12M](https://huggingface.co/datasets/inclusionAI/ConceptEdit-12M) | Million-scale | Concept-based changes | Instruction editing |
| [CrispEdit-2M](https://huggingface.co/datasets/WeiChow/CrispEdit-2M) | ~2M | Instruction edits | Instruction editing |
| [MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush) | ~10K edits | Human-annotated instruction editing | Editing fine-tuning |

#### Image preference and reward data

Human preference teaches what people prefer, not necessarily anatomy, text legibility or geometry; learn separate scores for quality, adherence, preservation, artifacts and safety.

| Dataset | Size | Content | Use |
| ---- | ---- | ---- | ---- |
| [HPS v2](https://github.com/tgxs002/HPSv2) | ~800K choices | Human preference choices | Preference prediction |
| [Text-to-Image Preferences 2M](https://huggingface.co/datasets/datapointai/text-2-image-human-preferences-2m) | ~2M judgments | Pairwise human judgments | Large-scale alignment |
| [Rapidata text-to-image feedback benchmark](https://huggingface.co/datasets/Rapidata/text-to-image-human-feedback-benchmark) | — | Human comparisons | Preference assessment |

### Evaluation

| Resource | Size | Measures |
| ---- | ---- | ---- |
| [GenEval](https://github.com/djghosh13/geneval) 🧪 | 553 prompts | Compositional text-to-image alignment |
| [MMMU](https://mmmu-benchmark.github.io/) 🧪 | 11.5K questions | Multimodal expert-level understanding |
| [Humor-Bench](https://github.com/ReubenNarad/Humor-Bench) 🧪 | — | Expert-annotated cartoon explanations for visual-humor reasoning |
| [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) 🧪 | — | Attribute binding and compositions |
| [ImagenHub](https://github.com/TIGER-AI-Lab/ImagenHub) 🧪 | — | Multi-task image generation and editing evaluation |

## Practice

### Recommended starter set for language models

A starting catalog, not a recommendation to train on all fifteen at once. Begin with the smallest suitable subsets and add data only when evaluation shows a benefit.

| # | Dataset | Size | Role |
| ---- | ---- | ---- | ---- |
| 1 | FineWeb-Edu | ~1.3T tokens | Foundation pretraining baseline |
| 2 | DCLM Baseline | ~4T tokens | Independent data-quality comparison |
| 3 | The Stack v2 | ~3B files, 600+ languages | Code foundation and continued pretraining |
| 4 | FineMath | ~54B tokens | Math-focused continued pretraining |
| 5 | Tülu 3 SFT Mixture | ~939K examples | General SFT and distillation |
| 6 | OpenCodeInstruct | ~5M | Coding instruction tuning |
| 7 | OpenMathInstruct-2 | ~14M | Reasoning SFT |
| 8 | UltraFeedback | ~64K prompts, ~256K responses | Preference learning, reward modeling |
| 9 | HelpSteer2 | ~21K responses (~10K prompts) | Human-annotated reward modeling |
| 10 | Tülu 3 RLVR | ~30K prompts | Verifiable RL baseline |
| 11 | SWE-Gym | 2,438 instances | Executable software engineering training |
| 12 | SWE-smith | ~50K task instances | Continual generation of repository tasks |
| 13 | AgentGym | — | General interactive agent learning |
| 14 | TiC-LM | — | Time-continual learning and retention |
| 15 | RewardBench | ~3K prompts | Held-out reward-model validation |

**Suggested order of work**

- **A. Foundation baselines**: small models on controlled FineWeb-Edu and DCLM subsets, then add code and math; measure perplexity, downstream scores and mixture effects.
- **B. SFT to preference pipeline**: Tülu 3 SFT, UltraFeedback or the Tülu preference mixture, evaluated on RewardBench.
- **C. Verifiable RL**: Tülu 3 RLVR and checkable math tasks; verify reward improvement, KL behavior, reference-policy computation and retention before moving to agents.
- **D. Executable agents**: SWE-Gym, SWE-smith and AgentGym with independent success verification; keep SFT demonstrations and learner-generated attempts distinct.
- **E. Continual improvement**: fresh task generation, replay buffers, retention safeguards and forward generalization tests; train only when a candidate survives evaluation on excluded tasks.

Every stage should yield a model that measurably improves a capability without silently destroying earlier ones.

### Dataset registry

Prefer a versioned, machine-readable registry over scripts that each load datasets differently. Separate three things: the **source** (origin and exact revision), the **transformation** (normalize, filter, deduplicate, split, tokenize) and the **use** (which stages consume the artifact).

```yaml
id: allenai/tulu-3-sft-mixture
kind: dataset
version: 1

source:
  provider: huggingface
  repository: allenai/tulu-3-sft-mixture
  revision: "<pin-commit-sha>"
  license_review: required

classification:
  stages: [sft]
  domains: [general, math, code]
  modality: text
  format: conversational

processing:
  normalize: messages
  preserve_source_metadata: true
  deduplicate: true
  evaluation_contamination_check: true

splits:
  train: train
  validation: generated_holdout

training:
  objective: causal_lm
  assistant_only_loss: true
```

The example is conceptual; the source may not provide every field or split shown.

**Canonical record types for text and agent data**

| Record | Essential fields |
| ------ | ---------------- |
| `Document` | Text, source, timestamp, license, content hash |
| `Conversation` | Ordered messages, roles, attachments |
| `Demonstration` | Prompt, reference response, provenance |
| `Preference` | Prompt, candidates, rankings/ratings, annotator |
| `VerifiableTask` | Prompt, verifier, environment specification |
| `Trajectory` | Ordered actions, observations, state, outcome |
| `EvaluationCase` | Task, expected behavior, scoring method, isolation policy |

Large pretraining corpora should stay in streamable formats with normalized metadata alongside; not everything needs one universal JSON schema.

**Canonical speech manifest**

```json
{
  "schema": "speech.record/v1",
  "id": "sample-000001",
  "source": {
    "dataset": "example-corpus",
    "revision": "pinned-revision",
    "recording_id": "recording-123",
    "license_id": "source-license",
    "consent_scope": "documented",
    "split_group": "speaker-42"
  },
  "audio": {
    "uri": "artifacts/audio/000001.flac",
    "sample_rate": 48000,
    "channels": 1,
    "start_ms": 0,
    "end_ms": 7420
  },
  "text": {
    "transcript": "Hello, how can I help you?",
    "normalized": "hello how can i help you",
    "language": "en",
    "confidence": 0.99
  },
  "speaker": { "id": "speaker-42", "reference": null },
  "annotations": {
    "emotion": "neutral",
    "delivery": "conversational",
    "word_timestamps": [],
    "vocalizations": [],
    "speaker_turns": []
  },
  "quality": { "snr_db": null, "dnsmos": null, "alignment_verified": false },
  "lineage": {
    "source_audio_hash": "sha256:...",
    "transcription_model": null,
    "processing_recipe": "speech-ingest/v1"
  }
}
```

In production, use Parquet metadata plus immutable audio objects, or WebDataset shards for large-scale training; JSONL is for debugging and interchange. Each task selects only the fields it needs:

- **ASR**: audio and text.
- **TTS**: text, audio, speaker, style.
- **Voice cloning**: target audio plus a separate reference segment.
- **Dialogue**: timestamped, channel-preserving multichannel audio with turn structure.
- **Speech understanding**: audio plus a grounded instruction and answer.
- **Audio codec**: audio only.

**Speech data pipeline**

1. **Import and verify**: source hashes, license and consent, duplicates, integrity.
2. **Normalize**: decode, resample, keep the original high-quality master.
3. **Analyze**: voice activity, diarization, language ID, quality scores.
4. **Align**: ASR, forced alignment, text normalization, timestamps.
5. **Enrich**: style, emotion, vocalizations, speaker references.
6. **Curate**: task-specific filters, source-level splits, contamination controls.
7. **Project**: ASR, TTS, codec, speaker, dialogue and instruction datasets.
8. **Train and evaluate**: objective-specific benchmark suites.

Keep original audio immutable and derive task views from it. For voice systems that learn from their own interactions, track permissions for incoming user audio, training consent, retention and speaker identity separately; having a recording does not imply permission to train on it.

### Licensing and source overlap

| Class | Meaning | Policy |
| ----- | ------- | ------ |
| Permissive verified | Source rights and model-training permission checked | Eligible for production training |
| Attribution / share-alike | Commercial use possible with obligations | Eligible after obligations review |
| Mixed source | Different records have different rights | Filter per record or recording |
| Non-commercial | Research use only | Exclude from the commercial pool |
| Gated contract | Needs acceptance or specific permission | Exclude until approved |
| Unclear | No authoritative source rights | Quarantine |
| Evaluation-only | Intended for benchmarking | Keep out of training |

Two mistakes to avoid: assuming a repository license (such as `apache-2.0`) covers underlying recordings or documents, and treating generated transcripts as conferring new rights over the source audio. Examples: Granary v2's labels are OpenMDW 1.1 while recordings keep their distributors' terms; Emilia's original 101k hours are non-commercial; Nord-Parl-TTS is CC BY-NC-ND.

### Dataset intake checklist

- [ ] **Publisher and source**: canonical publisher verified, exact version pinned.
- [ ] **License**: dataset, underlying content and teacher-output restrictions reviewed.
- [ ] **Integrity**: checksums or revision hashes recorded.
- [ ] **Provenance**: source documents, synthetic generators and attribution retained.
- [ ] **Contamination**: overlap with benchmark and evaluation sets checked.
- [ ] **Schema**: roles, turns, labels, rewards and tool calls validated.
- [ ] **Quality**: duplication, malformed records and unreliable annotations measured.
- [ ] **Reproducibility**: preprocessing, split generation and sampling seeds recorded.
- [ ] **Privacy**: personal data and sensitive information risks assessed.
- [ ] **Evaluation isolation**: train, validation, test and private retention sets separated.

## Related Lists

Maintained indexes for finding smaller and newer resources.

- [Awesome Agent Dataset](https://github.com/voidful/awesome-agent-dataset): tool, SWE, browser and trajectory data.
- [Agent RL Environments](https://github.com/V01dMur10c/awesome-agent-rl-environments): executable task suites and verifiable training environments.
- [AwesomeCUA datasets](https://awesomecua.com/categories/dataset/): computer-use and GUI trajectories.
- [NVIDIA agentic collection](https://huggingface.co/collections/nvidia/nemotron-agentic-and-tool-use): agentic SFT and RL releases.
- [OpenThoughts datasets](https://huggingface.co/open-thoughts): AgentTrove, TaskTrove and released agent training subsets.
- [OpenEuroLLM data catalogue](https://github.com/OpenEuroLLM/training-data-catalogue): open foundation-model pretraining sources and metadata.
- [Audio/Speech Datasets catalog](https://github.com/weimeng23/audio-speech-datasets) and [full-duplex speech dataset index](https://www.fullduplex.ai/datasets).

## Contributing

Contributions are welcome. Open a pull request that:

- Adds one dataset per row, in the section that matches what it **enables** (pretraining, SFT, preference, RLVR, agentic, evaluation, speech task, …).
- Links the canonical publisher page, not a mirror, and states scale, content and license or access terms.
- Uses the legend marks (🔒 ⚠️ 🧪 🔗) where they apply.
- Notes overlap with existing entries instead of listing the same upstream data twice.

Entries marked ⚠️ as unverified (for example the Emilia2 paper link, KLong, Research-Factory, and links whose Hugging Face repositories returned an access error at review time) should be confirmed or removed. The Video and Images sections are seed lists awaiting the same review as the rest.

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, the contributors have waived all copyright and related rights to this list. The datasets themselves remain under their own licenses.

---

Swedish Embedded AB implements solutions for building licensed, deduplicated and evaluation-isolated training-data pipelines for its clients. If your team needs expertise in machine learning data engineering, voice AI or language-model post-training, you can procure our services by sending an email to info@swedishembedded.com.

