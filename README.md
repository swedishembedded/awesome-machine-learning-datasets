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
  - [Post-training](#post-training)
    - [Supervised fine-tuning](#supervised-fine-tuning)
    - [Preference and reward learning](#preference-and-reward-learning)
    - [Verifiable reinforcement learning](#verifiable-reinforcement-learning)
    - [Agentic training](#agentic-training)
    - [Safety and defensive alignment](#safety-and-defensive-alignment)
  - [Continual learning and self-improvement](#continual-learning-and-self-improvement)
  - [Evaluation](#evaluation)
- [Video](#video)
  - [Pretraining](#pretraining-1)
  - [Mid-training](#mid-training-1)
  - [Post-training](#post-training-1)
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
  - [Evaluation](#evaluation-2)
  - [Training strategy and shortlist](#training-strategy-and-shortlist)
- [Images](#images)
  - [Pretraining](#pretraining-3)
  - [Mid-training](#mid-training-3)
  - [Post-training](#post-training-3)
  - [Evaluation](#evaluation-3)
- [Practice](#practice)
  - [Recommended starter set for language models](#recommended-starter-set-for-language-models)
  - [Dataset registry](#dataset-registry)
  - [Licensing and source overlap](#licensing-and-source-overlap)
  - [Dataset intake checklist](#dataset-intake-checklist)
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

| Dataset | Content | Purpose |
| ------- | ------- | ------- |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ~1.3T educationally filtered tokens (core release) | High-quality English foundation pretraining |
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | Large-scale cleaned Common Crawl | Broad web-text coverage |
| [DCLM Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | ~4T tokens, classifier-filtered web | Reproducible large-scale pretraining; data-filtering studies |
| [Dolma](https://huggingface.co/datasets/allenai/dolma) | Web, books, papers, code, encyclopedic text | Diverse foundation corpus |
| [Dolma 3 Pool](https://huggingface.co/datasets/allenai/dolma3_pool) | Component datasets for newer OLMo training | Controlled data-mixture experiments |
| [Dolma 3 Mix](https://huggingface.co/datasets/allenai/dolma3_mix-6T) | Prepared multi-domain mixture | Larger-scale foundation training |
| [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2) | Cross-deduplicated mixture of major open corpora | Pretraining with reduced cross-corpus duplication |
| [RedPajama V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) | Common Crawl with quality signals | Custom quality-filtering experiments |
| [Common Pile](https://huggingface.co/collections/common-pile) | Public-domain and openly licensed corpus family | License-conscious pretraining |
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | Synthetic educational text | Supplementing natural web text |
| [SmolLM Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | Curated multi-source corpus | Efficient small-model pretraining |

> **Do not blindly concatenate these.** Without document-level and near-duplicate detection, apparent diversity becomes repeated training examples.

#### Code corpora

| Dataset | Content | Purpose |
| ------- | ------- | ------- |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | Source-code repositories (Software Heritage IDs; content must be fetched separately 🔗) | Programming-language pretraining |
| [The Stack v2 Dedup](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) | Deduplicated Stack v2 derivative | Reducing repository-level duplication |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) | Source code, notebooks, commits, issues | Code-specialized foundation models |
| [The Stack](https://huggingface.co/datasets/bigcode/the-stack) | Earlier code corpus | Historical comparison, reproducibility |
| [Nemotron Pretraining Code v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1) | NVIDIA code pretraining corpus | Advanced coding foundation training |
| [Nemotron CC Code v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1) | Code-oriented web corpus | Supplementary programming knowledge |

Preserve file extensions, language labels, repository provenance and file boundaries; they are needed later for completion, repository understanding and task generation.

#### Multilingual corpora

| Dataset | Coverage | Purpose |
| ------- | -------- | ------- |
| [FineWeb 2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | 1,000+ languages and varieties | Broad multilingual training |
| [FineTranslations](https://huggingface.co/datasets/HuggingFaceFW/finetranslations) | Translated web documents | Cross-language transfer |
| [FineTranslations-Edu](https://huggingface.co/datasets/HuggingFaceFW/finetranslations-edu) | Educational multilingual translations | Multilingual instruction readiness |
| [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) | Indian languages | Regional specialization |
| [mC4 (multilingual C4)](https://huggingface.co/datasets/allenai/c4) | Multilingual Common Crawl | Multilingual baselines |

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

| Dataset | Content | Purpose |
| ------- | ------- | ------- |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | Filtered mathematical web documents | Math-heavy pretraining |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | Mathematical web text | Notation and knowledge |
| [Proof-Pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) | Scientific and mathematical text mixtures | Technical foundation training |
| [peS2o](https://huggingface.co/datasets/allenai/peS2o) | Academic publications | Scientific domain adaptation |
| [AutoMathText](https://huggingface.co/datasets/math-ai/AutoMathText) | Mathematics-focused corpus | Mathematical pretraining |
| [Nemotron CC Math v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | Mathematics-focused web data | Technical-domain enrichment |
| [Nemotron Math Proofs v1](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1) | Proofs and solutions | Proof-oriented training; inspect sample format first |

Mathematical *knowledge* pretraining is not mathematical *reasoning* SFT: papers teach notation and concepts, problem-solution sets teach how to construct answers.

These corpora can also be mixed into main pretraining; they are listed here because they are most often used for domain-focused continued training.

### Post-training

Post-training teaches the pretrained model to behave: supervised demonstrations first, then preference and reward learning, then verifiable and agentic reinforcement learning, then safety alignment.

#### Supervised fine-tuning

##### General instruction following and chat

| Dataset | Scale / character | Use |
| ------- | ----------------- | --- |
| [Tülu 3 SFT Mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) ⚠️ | ~939K examples; some outputs from third-party models with extra terms | Reproducible general post-training baseline |
| [UltraChat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | Curated multi-turn dialogue | Dialogue, instruction following |
| [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | ~1M examples | Diverse synthetic assistant responses |
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Human-written conversation trees | Dialogue, comparisons |
| [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | ~10K human-written | High-quality instruction following |
| [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) | FLAN-derived prompts with synthetic explanations | Explanatory responses |
| [Orca AgentInstruct](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1) | ~1M synthetic instructions | Task diversity |
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | Mixed post-training corpus | Small-model instruction tuning |
| [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) ⚠️ | ~52K synthetic instructions | Reference SFT baseline |
| [Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | ~15K human-authored | Small supervised baseline |
| [LIMA](https://huggingface.co/datasets/GAIR/lima) | ~1K curated | Data-efficiency experiments |

Tülu 3 is the preferred reference for a reproducible SFT pipeline; No Robots and LIMA are useful for separating the effects of sample quality from sample count. Public availability is not unrestricted commercial permission.

##### Reasoning and mathematics

| Dataset | Capability | Use |
| ------- | ---------- | --- |
| [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) | Worked mathematics solutions | Math SFT |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | Large synthetic math reasoning | Scaling math SFT |
| [NuminaMath CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Competition math reasoning | Worked-solution SFT; RLVR with extra checkers |
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | Mathematical instruction variants | Generalization |
| [Orca Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | Math word problems | Instruction tuning |
| [LIMO](https://huggingface.co/datasets/GAIR/LIMO) | Small curated reasoning set | Data-efficiency studies |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Grade-school math with solutions | SFT, simple RLVR baseline, 🧪 evaluation |
| [MATH](https://huggingface.co/datasets/hendrycks/competition_math) | Competition math | SFT, RLVR, 🧪 evaluation |

Distinguish **final-answer correctness** from **reasoning-trace quality**. A right answer can come from invalid steps, which matters for rejection-sampling fine-tuning and STaR.

##### Coding and software engineering

| Dataset | Focus | Use |
| ------- | ----- | --- |
| [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | ~5M coding instructions | Broad coding SFT |
| [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) | Coding with feedback and corrections | Coding, debugging |
| [Magicoder OSS-Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | Instruction synthesis from open-source code | Code SFT |
| [WizardLM Evol-Instruct V2](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) | General-purpose instruction-evolution set (not code-only) | Instruction-complexity experiments |
| [Nemotron SFT OpenCode v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-OpenCode-v1) | Code-centered instruction data | Coding post-training |
| [Nemotron SFT SWE v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3) | Software engineering trajectories | Repository-level agent SFT |
| [Open SWE Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | Software engineering agent traces | Tool-trajectory imitation |

Treat code data at three levels: single-function generation from a spec; file-level edits that preserve surrounding code; and repository-level tasks that inspect a project, use tools, change several files and validate behavior. The third is closest to what a software-engineering agent needs.

##### Multilingual, scientific and domain-specific

| Dataset | Domain | Use |
| ------- | ------ | --- |
| [Aya Dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) | Multilingual instructions | Cross-lingual alignment |
| [SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF) | Scientific information extraction and reasoning | Scientific assistant SFT |
| [SciInstruct](https://huggingface.co/datasets/TIGER-Lab/SciInstruct) ⚠️ | Scientific problem solving | Scientific reasoning |
| [LegalBench](https://huggingface.co/datasets/nguha/legalbench) 🧪 | Legal reasoning tasks (related to, not, LawInstruct) | Legal evaluation, task taxonomy |
| [Medical Meadow flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) | Medical flashcard Q&A (related to, not, MedInstruct-52K) | Medical specialization |
| [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | Customer-service conversations | Support agents |

LawInstruct and MedInstruct-52K are not listed because no canonical release was verified. Pair any domain SFT with independent evaluation; fluent domain vocabulary is not evidence of accurate conclusions.

#### Preference and reward learning

Preference data answers: given two or more responses, which should the model prefer?

##### Preference datasets

| Dataset | Annotation | Algorithms |
| ------- | ---------- | ---------- |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | Multiple responses with model-generated ratings | Reward modeling, DPO |
| [UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | Chosen/rejected pairs (loses rating detail) | DPO, ORPO |
| [Tülu 3 datasets](https://huggingface.co/collections/allenai/tulu-3-datasets) | On- and off-policy preference mixtures | DPO, reward modeling |
| [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) | Human-rated response attributes | Multi-attribute reward models |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | Human ratings and preferences | Reward modeling, DPO |
| [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | Later NVIDIA preference generation | Preference and reward learning |
| [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | Helpful/harmless chosen-rejected dialogues | DPO, reward modeling |
| [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar) | Multiple ranked responses per prompt | Ranking-based rewards |
| [Stanford SHP](https://huggingface.co/datasets/stanfordnlp/SHP) | Community-derived comparisons | Preference optimization |
| [OpenAI Summarize from Feedback](https://huggingface.co/datasets/openai/summarize_from_feedback) | Human comparisons of summaries | Reward models |
| [Orca DPO Pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) | Synthetic pairs | DPO experiments |
| [Capybara Preferences](https://huggingface.co/datasets/argilla/Capybara-Preferences) | Multi-turn preferences | Conversational DPO |
| [CodeUltraFeedback](https://huggingface.co/datasets/RLHFlow/CodeUltraFeedback-standard) | Coding preferences | Code DPO |
| [UltraInteract Pairs](https://huggingface.co/datasets/openbmb/UltraInteract_pair) | Reasoning and interaction preferences | Reasoning preference tuning |
| [Math-Step-DPO-10K](https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K) | Mathematical step preferences | Step-oriented DPO |

##### Reward models and process supervision

| Resource | Signal | Application |
| -------- | ------ | ----------- |
| [PRM800K](https://github.com/openai/prm800k) | Human step-level correctness | Process reward modeling |
| [Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd) | Step-level mathematical supervision | Process reward modeling |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | Multi-dimensional human ratings | Scalar or attribute rewards |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | AI-generated quality scores | Outcome reward models |
| [RewardBench](https://huggingface.co/datasets/allenai/reward-bench) 🧪 | Held-out preference comparisons | Reward-model evaluation |
| [Agent Reward Bench](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench) 🧪 | Agent outcome assessments | Agent reward evaluation |

Keep reward types distinct and never collapse them into one unqualified score:

- **Outcome reward**: did the completed solution succeed?
- **Process reward**: were the intermediate steps valid?
- **Preference reward**: which candidate is preferred?
- **Verifiable reward**: does the answer satisfy an independently checkable condition?

A verifier-backed reward avoids treating another model's subjective preference as ground truth.

#### Verifiable reinforcement learning

RLVR needs a task plus an independent way to decide whether an attempt succeeded; it does not need a reference trajectory. That is the key difference from offline SFT.

##### Mathematics and symbolic verification

| Dataset | Verification | Use |
| ------- | ------------ | --- |
| [Tülu 3 RLVR (GSM/MATH/IF mixed)](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints) | Answer checking plus instruction constraints | Reproducible RLVR; recipe in [Open Instruct](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md) |
| [DeepScaleR Preview](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | Math answer verification | GRPO/RLVR experiments |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Numerical answer extraction | Simple RLVR baseline |
| [MATH](https://huggingface.co/datasets/hendrycks/competition_math) | Mathematical answer checking | Reasoning RL |
| [NuminaMath CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Reference solutions | Verifiable reasoning with extra checkers |
| [Nemotron Math v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) | Math problems and reasoning data | SFT or RL depending on subset |

Exact string match is rarely a sufficient math verifier: handle numeric normalization, symbolic equivalence, multiple valid forms and answer-extraction errors. Recommended first implementation: Tülu 3 RLVR.

##### Coding with executable rewards

| Dataset / environment | Verification | Role |
| --------------------- | ------------ | ---- |
| [APPS](https://huggingface.co/datasets/codeparrot/apps) | Execution against test cases | Coding RL |
| [CodeContests](https://huggingface.co/datasets/deepmind/code_contests) | Competitive-programming tests | Harder coding RL |
| [TACO](https://huggingface.co/datasets/BAAI/TACO) | Executable checks | Coding RL |
| [KodCode](https://huggingface.co/datasets?search=KodCode) | Problems with test-based verification | Programmatic RL rewards (search link; confirm canonical repo) ⚠️ |
| [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) 🧪 | Time-separated problems with tests | Held-out evaluation |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) | Repository tests, task outcomes | Agentic coding RL |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | Generated repository tasks and environments | Scaling task generation |
| [Terminal-Bench](https://github.com/laude-institute/terminal-bench) | Containerized terminal tasks | Terminal-agent RL and evaluation |

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

| Resource | Task | Reward |
| -------- | ---- | ------ |
| [IFEval](https://huggingface.co/datasets/google/IFEval) 🧪 | Instruction constraints | Programmatic compliance |
| [Tülu RLVR IFEval](https://huggingface.co/datasets/allenai/RLVR-IFeval) | Instruction-following training tasks | Verifier-based |
| [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | Function-call correctness | API call validation |

These give fast, unambiguous feedback and are excellent for checking that an RL implementation works. They are not evidence of general agentic improvement.

#### Agentic training

The core artifact is a trajectory: `τ = (s0, a0, o1, a1, o2, …, aT, r)` with initial state `s0`, actions `a`, environment observations `o` and a task outcome or reward `r`. Trajectories support imitation learning, outcome filtering, process reward learning and on-policy RL. Preserve native tool arguments, results, timestamps, errors and environment references; flattening to plain text loses what reproduction and RL need.

##### Software engineering agents

| Resource | Provides | Use |
| -------- | -------- | --- |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) | Repository tasks, sampled trajectories, verifier data | SFT and RL |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | Scalable repository-level task generation | Expanding executable environments |
| [SWE-bench](https://github.com/SWE-bench/SWE-bench) | Real issue-resolution tasks | Held-out evaluation, selected training splits |
| [OpenHands SFT Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) | Agent-tool trajectories | Imitation learning |
| [OpenHands Sampled Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories) | Multiple sampled attempts | Outcome filtering, offline learning |
| [Open SWE Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | Agent interaction traces | Trajectory SFT |
| [Nemotron SFT SWE v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3) | Software engineering agent data | Long-horizon SFT |
| [daVinci-Agency](https://huggingface.co/datasets/GAIR/daVinci-Agency) | Small curated long-horizon examples | Long-horizon behavior study (value is structure, not scale) |

##### General tool use and multi-domain agents

| Resource | Environment / data | Use |
| -------- | ------------------ | --- |
| [AgentGym](https://github.com/woooodyy/AgentGym) | Multiple interactive agent environments | General agent RL; best treated as an environment-interface reference |
| [AgentGym RL Data (ID)](https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID) | ~186K RL instances | Multi-domain agent training |
| [ALFWorld](https://github.com/alfworld/alfworld) | Text-based household tasks | Planning, sequential action |
| [WebArena](https://github.com/web-arena-x/webarena) | Web application environments | Browser agents |
| [VisualWebArena](https://github.com/web-arena-x/visualwebarena) | Visual web tasks | Multimodal browser agents |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym) | Unified browser environments | Browser-agent RL and benchmarking |
| [ToolBench](https://github.com/OpenBMB/ToolBench) | Tool-use instructions and API interactions | Tool-call SFT |
| [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) | API invocation conversations | Tool planning and calling |
| [τ²-Bench](https://github.com/sierra-research/tau2-bench) | Stateful customer-service environments | Multi-turn tool agents |
| [BFCL](https://github.com/ShishirPatil/gorilla) | Function-calling evaluation | Tool-call correctness |

##### Deep research and long-horizon execution

| Resource | Capability | Role |
| -------- | ---------- | ---- |
| [OpenResearcher Dataset](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset) | ~97K research trajectories | Search and evidence-gathering SFT |
| [Long-Horizon-Execution](https://huggingface.co/datasets/arvindh75/Long-Horizon-Execution) | Synthetic multi-step execution | Long-horizon diagnostics |
| [KLong](https://huggingface.co/papers/2602.17547) ⚠️ | Long-horizon reasoning research | Methodology reference; dataset access unverified |
| [Research-Factory](https://huggingface.co/papers/2603.20278) ⚠️ | Long-horizon research generation | Methodology reference; confirm released artifacts |

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

#### Safety and defensive alignment

Pair refusal examples with harmless requests that should be answered normally, or refusal training turns into over-refusal.

| Dataset | Purpose | Role |
| ------- | ------- | ---- |
| [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | Helpfulness and harmlessness preferences | Safety preference training |
| [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Harmful-request/response classifications | Safety model training |
| [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) | Prompt harmfulness, response safety, refusal detection | Safety classifier training |
| [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) | Toxicity in user interactions | Input classification |
| [XSTest](https://huggingface.co/datasets/allenai/xstest-response) 🧪 | Safe requests that may trigger refusals | Over-refusal evaluation |
| [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) 🧪 | Over-refusal behavior | Safety evaluation |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) 🧪 | Robustness to adversarial requests | Defensive red-team evaluation |
| [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) 🧪 | Standardized jailbreak robustness | Defensive evaluation |

Label semantics differ (preferences vs. moderation labels); do not mix these sets without an explicit label-conversion policy.

### Continual learning and self-improvement

Most open datasets are static. Continual learning needs time-ordered tasks, freshly generated examples, feedback and explicit retention tests.

| Resource | Measures | Use |
| -------- | -------- | --- |
| [TiC-LM](https://aclanthology.org/2025.acl-long.1551/) 🧪 | Temporal updates vs. retention in continued pretraining | Time-aware pretraining |
| [TRACE](https://github.com/BeyonderXX/TRACE) 🧪 | Sequential adaptation across diverse tasks | Forgetting and transfer |
| [Long-Horizon-Execution](https://huggingface.co/datasets/arvindh75/Long-Horizon-Execution) | Degradation over extended executions | Agent endurance, state tracking |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | Task generation across repositories | Continually refreshed practice |
| [AgentGym](https://github.com/woooodyy/AgentGym) | Repeated interactive task solving | Online policy improvement |
| [Terminal-Bench](https://github.com/laude-institute/terminal-bench) | Executable terminal tasks | Continual task evaluation |

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

| Capability | Resource | Metric |
| ---------- | -------- | ------ |
| General knowledge | [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | Accuracy |
| General reasoning | [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) | Accuracy |
| Mathematical reasoning | [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH](https://huggingface.co/datasets/hendrycks/competition_math) | Answer correctness |
| Live coding | [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | Execution pass rate |
| Function calling | [BFCL](https://github.com/ShishirPatil/gorilla) | Call correctness |
| Repository engineering | [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | Resolved tasks |
| Terminal use | [Terminal-Bench](https://github.com/laude-institute/terminal-bench) | Task success |
| Browser interaction | [WebArena](https://github.com/web-arena-x/webarena) | Goal completion |
| Long-context retrieval | [RULER](https://github.com/NVIDIA/RULER) | Length-dependent accuracy |
| Multilingual reasoning | [MMMLU](https://huggingface.co/datasets/openai/MMMLU) | Per-language accuracy |
| Reward models | [RewardBench](https://huggingface.co/datasets/allenai/reward-bench) | Ranking accuracy |
| Safety classification | [WildGuardTest](https://huggingface.co/datasets/allenai/wildguardmix) | Classification quality |
| Refusal calibration | [XSTest](https://huggingface.co/datasets/allenai/xstest-response) | False refusal rate |
| Continual pretraining | [TiC-LM](https://aclanthology.org/2025.acl-long.1551/) | Temporal adaptation and retention |

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

| Dataset | Content | Notes |
| ------- | ------- | ----- |
| [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/) | ~136M narrated instructional clips | ASR-derived captions are noisy; source-video rights apply ⚠️ |
| [InternVid](https://huggingface.co/datasets/OpenGVLab/InternVid) | Large web video-text corpus with generated captions | Web-sourced; review terms ⚠️ |
| [Panda-70M](https://snap-research.github.io/Panda-70M/) | ~70M captioned clips | Research use; source-video rights apply ⚠️ |
| [HD-VILA-100M](https://github.com/microsoft/XPretrain) | ~100M high-resolution clip-text pairs | Research use ⚠️ |
| [WebVid-10M](https://github.com/m-bain/webvid) | ~10M stock-footage video-caption pairs | Distribution has been restricted; verify availability ⚠️ |
| [Kinetics-700](https://github.com/cvdfoundation/kinetics-dataset) | ~650K action-labeled clips, 700 classes | Action recognition pretraining |
| [Something-Something v2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) | ~220K object-interaction clips | Temporal reasoning; license terms apply ⚠️ |
| [Ego4D](https://ego4d-data.org/) 🔒 | Egocentric video with audio | Gated access |

### Mid-training

| Dataset | Content | Notes |
| ------- | ------- | ----- |
| [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) | ~1M high-quality text-video pairs | Quality-filtered for text-to-video |
| [Koala-36M](https://koala36m.github.io/) | ~36M clips with fine-grained captions and quality scoring | Curation reference ⚠️ |
| [MiraData](https://github.com/mira-space/MiraData) | Long clips with structured captions | Motion strength and long-take focus ⚠️ |

### Post-training

| Dataset | Content | Use |
| ------- | ------- | --- |
| [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) ⚠️ | Video instruction-following data | Video-language SFT; includes model-generated annotations |
| [VideoFeedback](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback) | Human ratings of generated videos | Video quality reward models |

### Evaluation

| Resource | Measures |
| -------- | -------- |
| [VBench](https://github.com/Vchitect/VBench) 🧪 | Video generation quality across dimensions |
| [Video-MME](https://video-mme.github.io/) 🧪 | Video understanding by multimodal models |

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

| Dataset | Scale | Best use | Access / licensing |
| ------- | ----- | -------- | ------------------ |
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

| Dataset | Use |
| ------- | --- |
| [YODAS v3](https://huggingface.co/datasets/espnet/yodas3) | Large-scale multilingual representation learning |
| [YODAS2](https://huggingface.co/datasets/espnet/yodas2) | Longer-context audio with timestamps |
| [VoxPopuli unlabeled](https://github.com/facebookresearch/voxpopuli) | Multilingual self-supervised pretraining |
| [Libri-Light](https://github.com/facebookresearch/libri-light) | English representations, speech-tokenizer training |
| [LibriHeavy](https://github.com/k2-fsa/libriheavy) | Long-form LibriVox-derived ASR segments |
| [Granary source collection](https://huggingface.co/datasets/nvidia/Granary-v2) 🔗 | Pretraining after source-audio acquisition |

### Mid-training

Supervised corpora with trustworthy labels, grouped by task, followed by language-specific and robustness data.

#### Automatic speech recognition

##### General English and multilingual

| Dataset | Size / coverage | Characteristics |
| ------- | --------------- | --------------- |
| [LibriSpeech](https://www.openslr.org/12/) | 960 h, English | Standard clean/other audiobook ASR |
| [LibriSpeech-PC](https://www.openslr.org/145/) | ~960 h | Punctuated, capitalized labels |
| [Multilingual LibriSpeech (MLS)](https://www.openslr.org/94/) | 50k+ h, 8 languages | Read speech |
| [The People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech) | 30k+ h | Diverse public sources |
| [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) ⚠️ | 10k transcribed h | Podcasts, audiobooks, online video |
| [WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech) | 65k+ h | Human transcripts with quality scores |
| [Granary v2](https://huggingface.co/datasets/nvidia/Granary-v2) 🔗 | 463k h | Synthetic transcription and translation labels |
| [MOSEL](https://huggingface.co/datasets/FBK-MT/mosel) 🔗 | 444k h | Multilingual pseudo-labels |
| [YODAS v3](https://huggingface.co/datasets/espnet/yodas3) | 1.1M h | Weakly labeled, 48 kHz multichannel |
| [OWSM v4 cleaned YODAS](https://huggingface.co/datasets/espnet/yodas_owsmv4) | Large multilingual subset | Cleaner YODAS training data |
| [Mozilla Common Voice](https://commonvoice.mozilla.org/) | Multi-language, version-dependent | Crowdsourced read speech, CC0 releases |
| [VoxPopuli](https://github.com/facebookresearch/voxpopuli) | ~1,800 labeled h; ~400k unlabeled h | European parliamentary speech |
| [SPGISpeech](https://huggingface.co/datasets/kensho/spgispeech) ⚠️ | 5,000 h, English | Earnings calls; research-use agreement |
| [TED-LIUM 3](https://www.openslr.org/51/) | ~450 h, English | Presentations and lectures |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) 🧪 | ~1,200 h, 102 languages | Multilingual evaluation and adaptation |
| [ESB (Earnings-21/22 and others)](https://huggingface.co/datasets/esb/datasets) 🧪 | Smaller sets, accented financial speech | Primarily evaluation |

LibriSpeech and MLS are comparatively straightforward audiobook corpora. GigaSpeech's access terms are stricter than its repository metadata suggests. Granary v2 is mostly labels and instructions that point to independently distributed audio.

##### Chinese, Arabic, Indic and other

| Dataset | Scale / language | Notes |
| ------- | ---------------- | ----- |
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

| Dataset | Scale | Advantage |
| ------- | ----- | --------- |
| [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | 24 h | Classic clean single-speaker English |
| [LibriTTS](https://www.openslr.org/60/) | 585 h | Multi-speaker English with transcripts |
| [LibriTTS-R](https://www.openslr.org/141/) | ~585 h | Restored audio quality |
| [Hi-Fi TTS](https://www.openslr.org/109/) | ~292 h | High-quality multi-speaker audiobooks |
| [HiFiTTS-2](https://huggingface.co/datasets/nvidia/hifitts-2) | Large multi-speaker corpus | Higher-bandwidth LibriVox-derived speech |
| [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) | ~44 h | 109 English speakers, accent diversity |
| [AISHELL-3](https://www.openslr.org/93/) | 85 h | Mandarin multi-speaker |
| [M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) | Multilingual | Audiobook speakers in several languages |
| [CSS10](https://github.com/Kyubyong/css10) | 10 languages | Single-speaker multilingual baseline |
| [DataBaker](https://www.data-baker.com/data/index/source) | Mandarin | Phonetically balanced speech |
| [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) ⚠️ | 101.7k h | In-the-wild TTS; original hours non-commercial |
| [Emilia-YODAS](https://huggingface.co/datasets/amphion/Emilia-Dataset) | 113.9k h | Six-language TTS; CC BY 4.0, subject to underlying rights |
| [Emilia2](https://arxiv.org/abs/2611.13614) ⚠️ | 229k unique h | Long monologues and multi-speaker dialogue with timestamps, 4.77M style instructions, nonverbal labels |
| [RealSpeech 350k](https://huggingface.co/datasets/TTS-AGI/tts-realspeech-multilingual-350k-moss-v2) 🔒 | 350k h | Speech, speaker features, codec tokens |
| [Parler-TTS Data-Speech collection](https://huggingface.co/collections/parler-tts/open-source-speech-datasets-annotated-using-data-speech) | Several datasets | Natural-language voice/style descriptions |
| [Nord-Parl-TTS](https://huggingface.co/datasets/Aalto-Speech-Synthesis/Nord-Parl-TTS) 🔒⚠️ | ~5,990 h | Finnish and Swedish parliament speech |

LJSpeech, LibriTTS(-R), VCTK and HiFiTTS are good first TTS corpora. Emilia's views (short utterances, monologues, dialogues) overlap and are not independent recordings; do not treat Emilia2 as commercially cleared before inspecting its final license and provenance.

#### Swedish and Nordic speech

| Dataset | Scale | Use | Licensing |
| ------- | ----- | --- | --------- |
| [RixVox](https://huggingface.co/datasets/KBLab/rixvox) | 5,500 h parliamentary speech, 2003–2023 | ASR, speaker modeling | CC BY 4.0. Transcripts are protocol-aligned, not always verbatim; verify before TTS use |
| [Nord-Parl-TTS](https://huggingface.co/datasets/Aalto-Speech-Synthesis/Nord-Parl-TTS) | 5,090 h Swedish + 900 h Finnish | Nordic TTS | 🔒⚠️ CC BY-NC-ND 4.0, gated |
| [KTH/NST Swedish ASR](https://huggingface.co/datasets/KTH/nst) | Swedish corpus | ASR | Check original language-bank terms |
| [Waxholm](https://huggingface.co/datasets/KTH/waxholm) | Small dialogue corpus | Spoken dialogue, ASR research | Verify source terms |
| [SpeechDat Swedish](https://huggingface.co/datasets/KTH/speechdat) | Telephone speech | Telephony ASR | Verify access and rights |
| [Common Voice Swedish](https://commonvoice.mozilla.org/) | Version-dependent | Read speech | CC0 |
| [Swedish Common Voice processed](https://huggingface.co/datasets/LokaalHub/sv-SE-asr-cv) | Derived from CV22 | ASR | CC0 source; check processing |
| [TTS-Swedish](https://huggingface.co/datasets/datadriven-company/TTS-Swedish) ⚠️ | Small/medium | Speech generation | Verify card license |
| [YodaLingua Swedish](https://huggingface.co/datasets/Thomcles/YodaLingua-Swedish) | Tens of thousands of samples | ASR/TTS | Verify provenance and license |
| [Swedish Conversational Speech](https://huggingface.co/datasets/SpeechDataAI/swedish-conversational-speech) 🔒 | Advertised 500 h | Natural conversation | Paid license; audio not on HF |
| [FLEURS Swedish](https://huggingface.co/datasets/google/fleurs) 🧪 | ~12 h | Evaluation | CC BY 4.0 |
| [VoxPopuli Swedish](https://huggingface.co/datasets/facebook/voxpopuli) | Small labeled, larger unlabeled | Parliamentary ASR, pretraining | Review source terms |

#### Speaker identity, verification and diarization

| Dataset | Purpose | Notes |
| ------- | ------- | ----- |
| [VoxCeleb1 and VoxCeleb2](https://mm.kaist.ac.kr/datasets/voxceleb/) ⚠️ | Speaker recognition and representation training | Research release with attribution; not blanket voice-cloning rights. [Original resources](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) |
| [VoxConverse](https://mm.kaist.ac.kr/datasets/voxconverse/) | Diarization | Multispeaker natural recordings |
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) | Meeting diarization | Speaker turns and overlap |
| [AliMeeting](https://www.openslr.org/119/) | Mandarin meeting diarization | Multiple microphones |
| [DIHARD III](https://dihardchallenge.github.io/dihard3/) 🔒 | Diarization in difficult conditions | LDC-linked resources |
| [CHiME-6](https://chimechallenge.github.io/chime6/) | Distant-microphone diarization | Real domestic conditions |
| [Ego4D](https://ego4d-data.org/) 🔒 | Egocentric speaking and listening | Gated audiovisual source |

Also useful for speaker work: [RixVox](https://huggingface.co/datasets/KBLab/rixvox) (Swedish) and [Basis Conversations 1500](https://withbasis.co/open/conversations-1500) (multi-track).

#### Speech translation and speech-to-speech

| Dataset | Data type | Coverage |
| ------- | --------- | -------- |
| [CoVoST 2](https://ai.meta.com/tools/covost/) | Speech → translated text | 2,880 h, many pairs |
| [MuST-C](https://ict.fbk.eu/must-c/) | Speech → translated text | TED speech with translations (v1 and v2) |
| [Europarl-ST](https://www.mllp.upv.es/europarl-st/) | Multilingual speech translation | European parliament |
| [CVSS](https://github.com/google-research-datasets/cvss) | Speech → translated speech | Common Voice-derived targets |
| [SeamlessAlign](https://github.com/facebookresearch/seamless_communication) | Mined speech/text alignments | Very large multilingual |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) 🧪 | Parallel spoken content | 102-language benchmarking |
| [VoxPopuli interpretations](https://huggingface.co/datasets/facebook/voxpopuli) | Simultaneous interpretation | Spoken language pairs |
| [Granary v2 AST](https://huggingface.co/datasets/nvidia/Granary-v2) 🔗 | Audio → translation instructions | Multilingual synthetic labels |

Translations and synthesized target audio may carry different rights from the source recordings.

#### Noise, reverberation, enhancement and separation

A model trained only on clean utterances fails on microphone noise, reverberation, overlapping speakers and background sound.

| Dataset | Capability |
| ------- | ---------- |
| [Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge) | Noise suppression; mixing infrastructure and license info |
| [MUSAN](https://www.openslr.org/17/) | Music, speech, noise augmentation |
| [WHAM! / WHAMR!](https://wham.whisper.ai/) | Noisy and reverberant speech separation |
| [LibriMix](https://github.com/JorisCos/LibriMix) | Synthetic overlapping speech mixtures |
| [VoiceBank + DEMAND](https://datashare.ed.ac.uk/handle/10283/1942) | Paired noisy/clean speech |
| [DEMAND](https://zenodo.org/records/1227121) | Real environmental noise |
| [AudioSet](https://research.google.com/audioset/) | Sound-event recognition |
| [FSD50K](https://zenodo.org/records/4060432) | Environmental audio classification (mixed CC terms; filter per file) |
| [Room impulse responses](https://www.openslr.org/28/) | Reverberation augmentation |
| [URGENT Challenge data](https://urgent-challenge.github.io/urgent2025/data/) | Universal speech enhancement |

### Post-training

Style, interaction and instruction data that sit on top of a model that can already recognize and produce speech.

#### Expressive speech, emotion and vocalizations

ASR corpora give little supervision for delivery style, laughter, breathing, hesitations or pacing. Pipelines often strip these as noise, yet for natural generation they are the target.

| Dataset | Capability | Access / caution |
| ------- | ---------- | ---------------- |
| [Basis Global Emotions](https://huggingface.co/datasets/basis-ai/basis-global-emotions) | Multilingual emotion and delivery, 525 h, 108 locales | Review dataset agreement |
| [Expresso](https://huggingface.co/collections/parler-tts/parler-tts-expresso) | Expressive speech styles | Research corpus; inspect rights |
| [Parler-TTS MLS annotations](https://huggingface.co/datasets/parler-tts/mls_eng_10k) | Voice descriptions, acoustic tags | MLS-derived; check version |
| [Emilia-NV / NVSpeech](https://huggingface.co/datasets/amphion/Emilia-NV) | Laughter, breathing, interjections | Mandarin nonverbal annotations |
| [NonverbalTTS](https://huggingface.co/datasets/deepvk/NonverbalTTS) | Breathing, sighing, coughing, laughing | ~17 h, English |
| [Vocal Bursts](https://huggingface.co/datasets/TTS-AGI/vocal-bursts) | Non-speech human vocal sounds | Verify per-source licenses |
| [ExpressiveSpeech](https://huggingface.co/datasets/FreedomIntelligence/ExpressiveSpeech) | Emotional conversational speech | Mixed-source lineage |
| [CREMA-D](https://huggingface.co/datasets/CREMA-D/CREMA-D) ⚠️ | Emotion recognition/conditioning | Check original terms |
| [RAVDESS](https://zenodo.org/records/1188976) | Acted emotional speech and song | Commercial license is separate |
| [IEMOCAP](https://sail.usc.edu/iemocap/) 🔒 | Emotion, conversational prosody | Academic/request access |
| [MSP-Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) 🔒 | Natural speech emotions | Access approval |
| [MSP-IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-IMPROV.html) 🔒 | Improvised expressive speech | Restricted |
| [EmoV-DB](https://github.com/numediart/EmoV-DB) | Emotional voice synthesis | Check recording rights |
| [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) | English and Mandarin emotional TTS | Verify original terms |
| [ESC-50](https://huggingface.co/datasets/ashraq/esc50) | Environmental sound recognition | Auxiliary audio understanding |

#### Conversational speech and full-duplex

Real-time voice agents need continuous streams, timestamps, multiple speakers, overlap, interruptions, backchannels and natural pauses. Turn-based ASR/TTS pairs cannot teach these.

| Dataset | Scale | Value |
| ------- | ----- | ----- |
| [Basis Conversations 1500](https://withbasis.co/open/conversations-1500) | 1,502 h, 22 languages | Up to four simultaneous speakers, channel-separated 48 kHz, ~100 h of detailed annotation; publisher states commercial training is permitted under its release terms |
| [Open Yap 1K](https://huggingface.co/datasets/TheAgenticDataCompany/open-yap-1k) 🔒 | 1,000 h | Dual-channel full-duplex; public sample is 8.9 h, full corpus needs a data-use agreement; commercial training allowed, cloning dataset speakers prohibited |
| [SmoothConv](https://github.com/qualialabsAI/SmoothConv-DuplexConv) | 100 h | Human-annotated conversational dynamics |
| [DuplexConv](https://github.com/qualialabsAI/SmoothConv-DuplexConv) | 2,000 h | Annotated Chinese conversation pretraining |
| [Fisher English Part 1](https://catalog.ldc.upenn.edu/LDC2004S13) 🔒 | 984 h | Telephone conversation (LDC) |
| [Fisher English Part 2](https://catalog.ldc.upenn.edu/LDC2005S13) 🔒 | 975 h | Telephone conversation (LDC) |
| [Switchboard-1](https://catalog.ldc.upenn.edu/LDC97S62) 🔒 | ~260 h | Turns and disfluency (LDC) |
| [CALLHOME corpora](https://catalog.ldc.upenn.edu/search) 🔒 | Multiple languages | Real telephone conversations (LDC) |
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | ~100 h | Meetings, overlapping speech |
| [AliMeeting](https://www.openslr.org/119/) | ~119 h | Mandarin meetings, diarization |
| [CHiME-6](https://chimechallenge.github.io/chime6/) | ~50 h | Multichannel dinner-party conversations |
| [VoxConverse](https://www.mm.kaist.ac.kr/datasets/voxconverse/) | ~50–64 h | In-the-wild speaker turns |
| [otoSpeech full-duplex](https://www.oto.earth/) 🔒 | Several subsets | Channel-separated conversation; access varies |
| [Basis consented conversation collections](https://withbasis.co/) 🔒 | Multiple | Conversation, speaker and preference annotations |

Modern multichannel recordings keep far more acoustic information than Fisher/Switchboard for realistic synthesis and interaction modeling. For broader discovery see the [full-duplex speech dataset index](https://www.fullduplex.ai/datasets) (159 datasets at review time).

#### Speech-language understanding

| Dataset | Capability |
| ------- | ---------- |
| [MINDS-14](https://huggingface.co/datasets/PolyAI/minds14) | Spoken intent classification |
| [Speech Commands](https://huggingface.co/datasets/google/speech_commands) | Keyword recognition |
| [SUPERB](https://huggingface.co/datasets/superb) 🧪 | Speech representation evaluation |
| [AudioBench](https://huggingface.co/datasets/AudioLLMs/AudioBench) 🧪⚠️ | Audio understanding benchmark suite |
| [Granary v2 instruction targets](https://huggingface.co/datasets/nvidia/Granary-v2) 🔗 | Contextual ASR, captioning, QA, translation |

Grounded instruction data can be derived from audio with trusted transcripts (transcribe, summarize, extract action items, find a stated deadline, answer questions). Keep derived records linked to the original audio and transcript and carry the source license. Generated QA is not human-verified labeling; use the latter for evaluation.

#### Singing and specialized vocal production

| Dataset | Purpose |
| ------- | ------- |
| [GTSinger](https://huggingface.co/datasets/GTSinger/GTSinger) | Multilingual singing, technique, scores; separates pitch, phonemes, technique and speaker |
| [OpenCPOP](https://wenet-e2e.github.io/opencpop/) | Mandarin singing with note/phoneme alignment |
| [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger) | Singing voice synthesis resources |
| [VocalSet](https://zenodo.org/records/1193957) | Sung vocal techniques and phonation |
| [CrawlSinger-OS](https://huggingface.co/datasets/pymaster/CrawlSinger-OS) | Processed singing/audio-score material |

### Evaluation

Keep these out of training data.

| Resource | Measures |
| -------- | -------- |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | Multilingual ASR and speech translation; includes Swedish |
| [ESB benchmark](https://huggingface.co/datasets/esb/datasets) | English ASR across domains (earnings calls and others) |
| [SUPERB](https://huggingface.co/datasets/superb) | Speech representation quality |
| [AudioBench](https://huggingface.co/datasets/AudioLLMs/AudioBench) ⚠️ | Audio understanding and speech question answering |
| [VoxConverse](https://mm.kaist.ac.kr/datasets/voxconverse/) | Diarization |
| [CHiME-6](https://chimechallenge.github.io/chime6/) | Distant-microphone conversational recognition |

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

| Dataset | Content | Notes |
| ------- | ------- | ----- |
| [LAION-5B](https://laion.ai/blog/laion-5b/) | ~5.85B CLIP-filtered image-text pairs (URLs and metadata) | 🔗⚠️ Metadata only; use the cleaned Re-LAION release where available |
| [DataComp](https://www.datacomp.ai/) | Benchmark and pools up to 12.8B image-text pairs | 🔗 Designed for studying data filtering |
| [COYO-700M](https://github.com/kakaobrain/coyo-dataset) | ~747M image-text pairs | 🔗⚠️ URLs and metadata |
| [Conceptual 12M](https://github.com/google-research-datasets/conceptual-12m) | ~12M image-text pairs | 🔗 URLs and captions |
| [ImageNet](https://www.image-net.org/) | 1.28M training images (1K classes); larger 21K release | ⚠️ Research and non-commercial terms |
| [Open Images](https://storage.googleapis.com/openimages/web/index.html) | ~9M images with labels, boxes, relations | Annotations CC BY; images listed as CC BY |

### Mid-training

| Dataset | Content | Notes |
| ------- | ------- | ----- |
| [Recap-DataComp-1B](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B) | DataComp-1B images with model-generated recaptions | Caption-quality improvement |
| [LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/) | Aesthetic-score-filtered LAION subsets | 🔗⚠️ Inherits LAION caveats |
| [COCO](https://cocodataset.org/) | ~330K images with captions, boxes, segmentation | Annotations CC BY 4.0; images carry Flickr terms |
| [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/) | Dense region descriptions and relationships | Grounded, dense annotation |
| [SA-1B](https://ai.meta.com/datasets/segment-anything/) | ~1.1B masks on ~11M images | Segmentation foundation models |

### Post-training

| Dataset | Content | Use |
| ------- | ------- | --- |
| [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) ⚠️ | Visual instruction-following (model-generated) | Vision-language SFT |
| [InstructPix2Pix dataset](https://github.com/timothybrooks/instruct-pix2pix) | Instruction-based image-edit triplets | Instruction editing |
| [Pick-a-Pic v2](https://huggingface.co/datasets/yuvalkirstain/pickapic_v2) ⚠️ | Human preferences over generated images | Preference tuning, reward models |
| [ImageReward](https://github.com/THUDM/ImageReward) | Human rankings for text-to-image output | Reward modeling |

### Evaluation

| Resource | Measures |
| -------- | -------- |
| [GenEval](https://github.com/djghosh13/geneval) 🧪 | Compositional text-to-image alignment |
| [MMMU](https://mmmu-benchmark.github.io/) 🧪 | Multimodal expert-level understanding |

## Practice

### Recommended starter set for language models

A starting catalog, not a recommendation to train on all fifteen at once. Begin with the smallest suitable subsets and add data only when evaluation shows a benefit.

| # | Dataset | Role |
| - | ------- | ---- |
| 1 | FineWeb-Edu | Foundation pretraining baseline |
| 2 | DCLM Baseline | Independent data-quality comparison |
| 3 | The Stack v2 | Code foundation and continued pretraining |
| 4 | FineMath | Math-focused continued pretraining |
| 5 | Tülu 3 SFT Mixture | General SFT and distillation |
| 6 | OpenCodeInstruct | Coding instruction tuning |
| 7 | OpenMathInstruct-2 | Reasoning SFT |
| 8 | UltraFeedback | Preference learning, reward modeling |
| 9 | HelpSteer2 | Human-annotated reward modeling |
| 10 | Tülu 3 RLVR | Verifiable RL baseline |
| 11 | SWE-Gym | Executable software engineering training |
| 12 | SWE-smith | Continual generation of repository tasks |
| 13 | AgentGym | General interactive agent learning |
| 14 | TiC-LM | Time-continual learning and retention |
| 15 | RewardBench | Held-out reward-model validation |

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

