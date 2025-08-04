---
title: Reinforcement Learning
date: 2025-07-26T12:04:01-04:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # ai/ml
  - agents
  - AGI
  - generative AI
  - LLMs
series: []
layout: single
toc: true
math: false
plotly: false
draft: true
---

I had intended to start this post by proclaiming _"2026 will be the year of reinforcement learning"_ as 2025 is "the year of agents"... But model and research releases over the past several weeks indicate that it might be "H2 2025" is when reinforcement learning for agentic AI really takes off. Open model releases such as Qwen3 (particularly the recent Qwen3 0725 updates), Kimi K2, GLM-4.5, gold-level performance from OpenAI and Google at the International Math Olympiad, and a rapid increase in reinforcement learning algorithm research and refinement indicate that focus of the industry has shifted from scaling raw data and compute to reinforcement learning (RL).

## What is reinforcement learning?

Reinforcement learning (RL) is a process that allows an algorithm to continue learning _without explicit examples_. Whereas supervised learning tasks (such as classification or prediction) requires sets of input-output pairs that teach the model to generate the output based on the input, RL teaches the model to maximize the reward it receives for actions it takes in response to arbitrary inputs. Thus, RL requires a separate reward function or reward model that can determine the appropriate reward (and reward intensity) for any action the model may take.[^wiki]

## Language Model Training

Language models leverage both supervised and reinforcement learning approaches at different points during their training phase.

### Pre-training

A base model is trained on enormous amounts of text using a self-supervised approach. The process is self-supervised because the text corpus itself provides the training examples and the ground truth answer. During pre-training, the model tries to predict the next ~~word~~ token in a sentence or paragraph ("The cat in \_\_\_", "The cat in the \_\_\_", "The cat in the hat \_\_\_"). It does so by predicting the probability of _all_ tokens, and then receives feedback about the correct answer, allowing it to update its internal weights to make the correct answer more probable.

By the end of pre-training, the model has internalized how language "works". It is capable of continuing an initial "seed" phrase with fluent grammar and logical and syntactic consistency at the sentence and paragraph level. Because the training data encompassed broad, undirected swaths of text-encoded human knowledge, the model is familiar with general facts about the world that are consistent across its training data. Depending on the data mixture used in pre-training, the model may also exhibit some initial ability to follow instructions or continue patterns.

{{% details title="Next-token prediction" %}}

```mermaid
flowchart TD
  %% Styling definitions
  classDef corpus fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
  classDef model fill:#fff3e0,stroke:#ff6f00,stroke-width:2px,color:#000
  classDef loss fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000
  classDef update fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000

  subgraph Corpus["Unsupervised Text Corpus"]
      TextData["Training Examples:
      'The cat sat on the mat'
      'To be or not to be'
      'Machine learning is...'"]:::corpus
  end

  subgraph Pretraining["Pre-training Process"]
      InputTokens["Input Context:
      ['The', 'cat', 'sat', 'on', 'the']"]:::model

      Model["Language Model
      (learns statistical patterns)"]:::model

      Predictions["Next Token Predictions:
      • 'mat': 0.91
      • 'rug': 0.04
      • 'floor': 0.03
      • 'couch': 0.02"]:::model

      GroundTruth["Actual Next Token:
      'mat'"]:::corpus

      Loss["Cross-Entropy Loss:
      -log(0.91) ≈ 0.094"]:::loss

      Update["Parameter Update:
      Increase P('mat' | context)
      Decrease P(other tokens | context)"]:::update

      InputTokens --> Model
      Model --> Predictions
      Predictions --> Loss
      GroundTruth --> Loss
      Loss --> Update
      Update -.->|"Backpropagation"| Model
  end

  Corpus --> InputTokens
```

{{% /details %}}

### Mid-training

Mid-training may involve _context extension_, _language extension_, and/or _domain extension_, enabling the pre-trained model to be able to comprehend and work with longer contexts, to understand and reply in different languages, and/or to integrate _knowledge_ from high-quality datasets rather than simply produce coherent output, respectively. Mid-training is also likely to include instruction-following training. The idea is to refine the model's predictions by showing it many examples of how we want it to respond. Depending on the subtype of training during the mid-training phase, the model may get feedback after each token is predicted (i.e., the same next-token prediction task used in pre-training), or it may get feedback only after it generates a complete response (which is more typical of supervised fine-tuning).

{{% details title="Instruction tuning" closed="false" %}}

```mermaid
flowchart TD
    %% Styling definitions
    classDef dataset fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef model fill:#fff3e0,stroke:#ff6f00,stroke-width:2px,color:#000
    classDef loss fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000
    classDef update fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000

    subgraph LabeledData["Supervised Instruction Dataset"]
        Pair["Example:
        Input: 'How do I boil an egg?'
        Target: 'Place the egg in boiling water for 7–9 minutes.'"]:::dataset
    end

    subgraph SFT["Supervised Fine-Tuning Process"]
        Prompt["Input Prompt:
        'How do I boil an egg?'"]:::model

        PretrainedModel["Pre-trained Model
        (already knows language patterns)"]:::model

        Generated["Generated Response:
        'Place the egg in boiling water for 7–9 minutes.'"]:::model

        Target["Target Response:
        'Place the egg in boiling water for 7–9 minutes.'"]:::dataset

        TokenComparison["Token-by-Token Comparison:
        Generated: ['Place', 'the', 'egg', 'in', 'boiling', 'water', ...]
        Target:    ['Place', 'the', 'egg', 'in', 'boiling', 'water', ...]"]:::loss

        CrossEntropy["Cross-Entropy Loss:
        Σ -log P(target_token | context)
        averaged over sequence length"]:::loss

        Backprop["Gradient Update:
        Adjust model weights to increase
        probability of target tokens"]:::update

        Prompt --> PretrainedModel
        PretrainedModel --> Generated
        Generated --> TokenComparison
        Target --> TokenComparison
        TokenComparison --> CrossEntropy
        CrossEntropy --> Backprop
        Backprop -.->|"Parameter updates"| PretrainedModel
    end

    Pair --> Prompt
    Pair --> Target
```

{{% /details %}}

The distinction between pre-training, mid-training, and post-training is fuzzy; I listened to a podcast where the definition of mid-training was "not pre-training and not post-training" (_A/N: sorry, I can't find the reference_). I think of mid-training as "adding utility" but not the final polish; capabilities introduced in mid-training, such as instruction-following and long-context understanding, are often required in post-training.

### Post-training

Post-training focuses the model on being able to complete tasks in alignment with human expectations. Various reinforcement learning (RL) techniques are used (RLHF - RL with Human Feedback, RLVR - RL with Verifiable Rewards) that assess the full model response rather than simply determining whether the next-token-prediction is correct. _Post-training focuses on assessing the full model response in its entirety; pre- and mid-training focus on next-token accuracy._

#### RLHF

> We first collect a dataset of human-written demonstrations on prompts submitted to our API, and use this to train our supervised learning baselines. Next, we collect a dataset of human-labeled comparisons between two model outputs on a larger set of API prompts. We then train a reward model (RM) on this dataset to predict which output our labelers would prefer. Finally, we use this RM as a reward function and fine-tune our GPT‑3 policy to maximize this reward using the PPO algorithm⁠.
> [Aligning language models to follow instructions | OpenAI](https://openai.com/index/instruction-following/) 27 Jan 2022

```mermaid
flowchart TD
    %% Styling definitions
    classDef dataset fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef model fill:#fff3e0,stroke:#ff6f00,stroke-width:2px,color:#000
    classDef loss fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000
    classDef update fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef ai fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef human fill:#fff8e1,stroke:#f57f17,stroke-width:2px,color:#000

    subgraph DataCollection["Phase 1: Preference Data Collection"]
        direction TB

        HumanPrompts["Base Prompts:<br/>• Instruction tasks<br/>• Conversation scenarios"]:::dataset
        ModelCandidates["Model Responses:<br/>• Multiple quality levels<br/>• Diverse approaches"]:::dataset
        HumanAnnotators["Human Annotators:<br/>• Rate responses<br/>• Rank preferences"]:::human
        AIJudge["AI Judge (Optional):<br/>• Scale human feedback<br/>• Cover edge cases"]:::ai

        HumanPreferences["Human Preferences:<br/>• Quality rankings<br/>• Safety assessments"]:::human
        AIPreferences["AI Preferences:<br/>• Broader coverage<br/>• Consistent scoring"]:::ai
    end

    subgraph RewardTraining["Phase 2: Reward Model Training"]
        direction TB

        HumanPairs["Human Pairwise Data:<br/>• ~10K examples<br/>• High quality"]:::dataset
        AIPairs["AI Pairwise Data:<br/>• ~100K examples<br/>• Broad coverage"]:::dataset
        CombinedData["Combined Training Set:<br/>• Human (high weight)<br/>• AI (lower weight)"]:::dataset

        RewardModel["Reward Model Training:<br/>• Preference learning<br/>• f(prompt, response) → score"]:::model
        TrainedRM["Trained Reward Model:<br/>• Calibrated on human prefs<br/>• Validated performance"]:::model

        HumanPairs --> CombinedData
        AIPairs --> CombinedData
        CombinedData --> RewardModel
        RewardModel --> TrainedRM
    end

    subgraph PolicyOptimization["Phase 3: Policy Optimization"]
        direction TB

        PolicyModel["Policy Model"]:::model
        ResponseGen["Response Generation:<br/>• Multiple candidates<br/>• Diverse sampling"]:::model
        RewardEval["Reward Evaluation:<br/>• Score with reward model<br/>• Batch processing"]:::update
        PolicyUpdate["Policy Update:<br/>• PPO/DPO optimization<br/>• Maximize rewards"]:::update
        ConvergenceCheck{"Converged?"}:::loss
        OptimizedModel["Optimized Model"]:::update

        PolicyModel --> ResponseGen
        ResponseGen --> RewardEval
        RewardEval --> PolicyUpdate
        PolicyUpdate --> ConvergenceCheck
        ConvergenceCheck -->|No| PolicyModel
        ConvergenceCheck -->|Yes| OptimizedModel
    end

    %% Cross-phase connections
    HumanPrompts --> ModelCandidates
    ModelCandidates --> HumanAnnotators
    ModelCandidates --> AIJudge
    HumanAnnotators --> HumanPreferences
    AIJudge --> AIPreferences
    HumanPreferences --> HumanPairs
    AIPreferences --> AIPairs
    TrainedRM --> RewardEval

    %% Phase flow indicators
    DataCollection -.->|"Preference data"| RewardTraining
    RewardTraining -.->|"Reward function"| PolicyOptimization
```

#### RL (RLVR)

```mermaid
flowchart TD
    %% Styling definitions
    classDef dataset fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef model fill:#fff3e0,stroke:#ff6f00,stroke-width:2px,color:#000
    classDef loss fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000
    classDef update fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef verify fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px,color:#000
    classDef execute fill:#f1f8e9,stroke:#33691e,stroke-width:2px,color:#000

    subgraph TaskCollection["Phase 1: Verifiable Task Collection"]
        direction TB

        CodingProblems["Coding Problems:<br/>• Algorithm challenges<br/>• Code completion"]:::dataset
        MathProblems["Math Problems:<br/>• Competition problems<br/>• Proof verification"]:::dataset
        ToolProblems["Tool Use Problems:<br/>• API interactions<br/>• Web navigation"]:::dataset

        UnitTests["Unit Tests:<br/>• Test suites<br/>• Expected outputs"]:::verify
        MathVerifiers["Math Verifiers:<br/>• Solution checkers<br/>• Proof validators"]:::verify
        EnvironmentTests["Environment Tests:<br/>• State verification<br/>• Response validation"]:::verify
    end

    subgraph RewardGeneration["Phase 2: Automated Reward Generation"]
        direction TB

        subgraph ExecutionEngine["Execution & Verification Engine"]
            direction LR
            CodeExecutor["Code Execution:<br/>Sandboxed environment"]:::execute
            MathExecutor["Math Execution:<br/>Solution verification"]:::execute
            WebExecutor["Web/Tool Execution:<br/>Browser automation"]:::execute
        end

        subgraph RewardComputation["Reward Computation"]
            direction TB
            PassFailReward["Pass/Fail Rewards:<br/>• Binary scores<br/>• Partial credit"]:::verify
            QualityMetrics["Quality Metrics:<br/>• Code style<br/>• Solution elegance"]:::verify
            FinalReward["Final Reward Signal:<br/>Weighted combination"]:::update

            PassFailReward --> FinalReward
            QualityMetrics --> FinalReward
        end

        CodeExecutor --> PassFailReward
        MathExecutor --> PassFailReward
        WebExecutor --> PassFailReward
    end

    subgraph PolicyOptimization["Phase 3: Policy Optimization"]
        direction TB

        subgraph Generation["Response Generation"]
            PolicyModel["Policy Model"]:::model
            TaskSampling["Task Sampling:<br/>Multi-modal selection"]:::model
            ResponseGen["Response Generation:<br/>Chain-of-thought"]:::model

            PolicyModel --> TaskSampling
            TaskSampling --> ResponseGen
        end

        subgraph Evaluation["Automated Evaluation"]
            ExecutionEval["Execution Evaluation:<br/>Run code/actions"]:::verify
            PerformanceEval["Performance Analysis:<br/>Success rate scoring"]:::verify

            ResponseGen --> ExecutionEval
            ExecutionEval --> PerformanceEval
        end

        subgraph Update["Policy Update"]
            PPOUpdate["PPO Update:<br/>Policy optimization"]:::update
            ConvergenceCheck{"Converged?"}:::loss
            OptimizedModel["Optimized Model"]:::update

            PerformanceEval --> PPOUpdate
            PPOUpdate --> ConvergenceCheck
            ConvergenceCheck -->|No| PolicyModel
            ConvergenceCheck -->|Yes| OptimizedModel
        end
    end

    %% Task-to-verifier connections
    CodingProblems --> UnitTests
    MathProblems --> MathVerifiers
    ToolProblems --> EnvironmentTests

    %% Verifier-to-executor connections
    UnitTests --> CodeExecutor
    MathVerifiers --> MathExecutor
    EnvironmentTests --> WebExecutor

    %% Cross-phase connections
    FinalReward --> PerformanceEval

    %% Phase flow indicators
    TaskCollection -.->|"Verifiable tasks"| RewardGeneration
    RewardGeneration -.->|"Automated rewards"| PolicyOptimization

    %% Multi-modal task sampling
    CodingProblems --> TaskSampling
    MathProblems --> TaskSampling
    ToolProblems --> TaskSampling
```

[^wiki]: [Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
