# Fugu / Conductor / Trinity — Análise do vídeo + Arquitetura

> Investigação do vídeo **"FUGU Proves Intelligence Isn't What You Think"**
> (DEEPTECH AI LABS, 5:48) cruzada com as **fontes primárias**. Mapeia o que o
> vídeo afirma → o que é verdade → como se liga à **Amalia** (a nossa reimpl).
> Data: 24 jun 2026.

---

## 0. TL;DR

- O vídeo é **substancialmente correto** mas é hype de canal pequeno (4.3K views,
  motion-graphics) com **2 erros factuais** e mistura de números *paper* vs *produto*.
- **Fugu existe mesmo**: produto comercial da Sakana AI, lançado 22 jun 2026.
  É um **orquestrador 7B** que encaminha tarefas por um pool de modelos frontier.
- Sustentado por **dois papers ICLR 2026 reais**: **Trinity** (coordenador
  evolutivo) e **Conductor** (orquestrador treinado por RL/GRPO).
- **A Amalia é a nossa reimplementação local do ramo Conductor.** Este vídeo é
  literalmente sobre o paper que o nosso projeto replica.

---

## 1. O que o vídeo afirma (transcript) vs realidade

| # | Afirmação do vídeo | Veredicto | Fonte primária |
|---|---|---|---|
| 1 | "7B model beat GPT-5.5 e Opus 4.8 ao mesmo tempo" | ✅ no sentido **orquestrador**, não modelo solo | sakana.ai/fugu-release |
| 2 | "Nunca treinou num dataset frontier" | ✅ o orquestrador treina em **outcomes**, não em dados destilados dos workers | Conductor §3 |
| 3 | "Traffic cop / scoring head, **zero geração** na decisão" | ⚠️ **mistura Trinity com Conductor**. O *Trinity* usa selection-head sobre hidden-state (sem geração). O *Conductor* **gera** o workflow em texto (com geração) | Trinity abstract vs Conductor §3.1 |
| 4 | "SWE-Bench Pro 73.7 vs 69.2; LCB 93.2; GPQA-D 95.5; HLE 50.0" | ✅ números reais — mas do **Technical Report do produto** (Fugu Ultra), não dos papers | arXiv 2606.21228 |
| 5 | "**Fable 5, OpenAI's best**" | ❌ **ERRADO** — Fable 5 é da **Anthropic** | CNBC 12/06/2026 + sakana.ai |
| 6 | "Dois papers ICLR 2026: Trinity + Conductor" | ✅ ambos reais, ambos aceites | OpenReview 5HaRjXai12 |
| 7 | "GRPO, treina em outcomes, sem human feedback" | ✅ exato | Conductor §2-3 |
| 8 | "Comunicação emergente agent-to-agent que ninguém programou" | ✅ o paper mostra emergência de planner/verifier/debate | Conductor §3.1, Fig.3 |
| 9 | "SVD fine-tuned selection head, low-rank" | ⚠️ é do **Trinity/Fugu-base** (selection-head), não do Conductor | Trinity |
| 10 | "Pool swappable: GPT-5.5, Opus 4.8, Gemini 3.1" | ✅ é o argumento central ("adaptive worker selection") | Conductor §3.2 |
| 11 | "Fugu reescreve à volta de export controls; não disponível na UE/EEA" | ✅ **real e oportuno** — EUA forçaram Anthropic a desligar Fable/Mythos a 12/06/2026 | CNBC |
| 12 | "Fugu chama-se a si mesmo recursivamente (emergente)" | ✅ **recursão é real**, mas é uma **extensão deliberada** (finetune com 1 chamada recursiva em metade do batch), não puramente "emergente" | Conductor §3.2 |

**Conclusão**: o esqueleto técnico do vídeo está certo. Os erros são (a) atribuir
Fable 5 à OpenAI, (b) fundir os mecanismos de Trinity (selection-head, SVD, "sem
geração") com os do Conductor (geração de workflow em texto), e (c) vender a
recursão e a topologia como 100% "emergentes" quando são extensões desenhadas que
*depois* produzem comportamento emergente.

---

## 2. Os dois papers (fontes primárias)

### 2.1 Trinity — *An Evolved LLM Coordinator* (ICLR 2026, Poster; ratings 6/4/6/6)
- Autores: Xu, Sun, Schwendeman, Nielsen, Cetin, Tang (Sakana AI + U.Michigan + IST).
- Coordenador = **LM compacto (~0.6B) + head (~10K params)**.
- Otimizado por **estratégia evolutiva (CMA-ES separável)** — não RL.
- A cada turno atribui **um de 3 papéis** (Thinker / Worker / Verifier) a um LLM selecionado.
- Encaminha via **hidden-state → selection head** (logits-to-agent), **sem gerar texto** na decisão.
- SOTA: **86.2% LiveCodeBench**.
- Argumento teórico: CMA-ES > RL sob alta dimensionalidade + budget apertado (block-ε-separabilidade).

### 2.2 Conductor — *Learning to Orchestrate Agents in Natural Language* (ICLR 2026)
- Mesmo grupo. arXiv:2512.04388.
- Coordenador = **7B (de checkpoint Qwen2.5)**, treinado com **GRPO** (RL).
- Em vez de selection-head, **gera um workflow em linguagem natural**:
  - cada *step* = `(subtask string, worker_id int, access_list[int])`
  - parse de 3 listas Python do output após chain-of-thought.
- **Reward progressivo (2 condições)**:
  1. **Format**: `r=0` se as 3 listas não fazem parse.
  2. **Correctness**: `r=1` se o output final do workflow == solução; `r=0.5` caso contrário.
- **Sem KL regularization**, batch 256, **só 200 iterações GRPO** até convergir.
- Pool de treino: Gemini-2.5-Pro, Claude-Sonnet-4, GPT-5 + DeepSeek-R1-Distill-Qwen-32B, Gemma3-27B, Qwen3-32B.
- Dataset: 960 problemas (MATH, MMLU, RLPR, LiveCodeBench V1).
- **Extensões**:
  - **Adaptive worker selection** — finetune com subconjunto-k aleatório do pool → generaliza a qualquer pool.
  - **Recursão** — Conductor escolhe-se a si próprio como worker → test-time scaling por nº de chamadas recursivas.
- **Resultados (Tabela 1)**: Conductor 7B **Avg 77.27** vs GPT-5 74.78, Gemini-2.5-Pro 70.97. SOTA in- e out-of-domain.

### 2.3 Por que escolhemos o ramo Conductor (decisão de design da Amalia)
- Trinity/Fugu-base precisam de **acesso aos pesos** do orquestrador (selection-head sobre hidden-state) → não embrulha workers só-API.
- O **Conductor encaminha por prompting** (input → texto → parse) → funciona sobre **qualquer endpoint OpenAI-compatible**. É o que o torna plug-and-play.
- Expressa todo o espaço de coordenação (chain / best-of-N / tree) + recursão → não perdemos nada.

---

## 3. O contexto geopolítico (por que "soberania" não é só marketing)

- **12 jun 2026**: o governo dos EUA emitiu uma export-control directive obrigando a
  **Anthropic a desligar Fable 5 e Mythos 5 globalmente** para qualquer estrangeiro
  (CNBC, TechTimes, etc.).
- Resultado: quem dependia de um único vendor frontier ficou subitamente sem acesso.
- O pitch do Fugu — **pool swappable que reescreve à volta de cortes de acesso** — é
  uma resposta direta a este evento. Por isso o Fugu **não está disponível na UE/EEA**
  (carve-out de compliance), e os baselines Fable 5 / Mythos **não estão no pool**
  (não são publicamente acessíveis; só usados como vara de medição).

---

## 4. Arquitetura — Fugu/Conductor ↔ Amalia

### 4.1 O padrão (igual nos dois)

```
                 ┌──────────────────────────────────────────────┐
   client ─────► │  ENDPOINT ÚNICO (OpenAI-compatible)           │
   (IDE/agent)   │  Fugu: api.sakana.ai  │  Amalia: :8900        │
                 └───────────────┬──────────────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  ORQUESTRADOR 7B         │   ← a "policy"
                    │  (Qwen2.5-7B)            │
                    │  lê prompt → emite       │
                    │  WORKFLOW em texto:      │
                    │   subtasks[]             │
                    │   worker_ids[]           │
                    │   access_list[]          │
                    └───────────┬─────────────┘
                                ▼  parse + validate (= format reward)
                    ┌─────────────────────────┐
                    │  WORKFLOW ENGINE         │
                    │  DAG por ondas topológicas│
                    │  chain │ best-of-N │ tree │
                    │  + recursão (≤N rondas)  │
                    └───────────┬─────────────┘
                                ▼  por step: subtask + contexto do access_list
                    ┌─────────────────────────┐
                    │  WORKER POOL (swappable) │
                    │  Model 0,1,2… = qualquer │
                    │  endpoint OpenAI/Responses│
                    └─────────────────────────┘
```

### 4.2 Mapa de equivalências

| Conceito do paper | Fugu (produto) | Amalia (nós) | Ficheiro |
|---|---|---|---|
| Orquestrador 7B | proprietário | **Qwen2.5-7B** em llama.cpp Vulkan | `conductor.py` |
| Workflow (subtask/id/access) | interno | parser de brackets balanceados + validação | `parser.py` |
| Execução por topologia | interno | **DAG executor com ondas** (tree paralelo / best-of-N) | `engine.py` |
| Recursão / test-time scaling | "self-calling" | rondas de recursão ≤N (refine/verify) | `conductor.py` |
| Pool swappable | GPT-5.5/Opus/Gemini | **ordinal-only** (Model 0,1,2…) + hint | `workers.py` |
| Reward = format + correctness | GRPO interno | **exec_reward** (executa workflow nos workers reais) | `training/grpo_real.py` |
| Treino GRPO 200 iter | interno | GRPO real em ROCm, validado | `training/grpo_real.py` |
| Endpoint OpenAI-compatible | api.sakana.ai | `amalia-v1` @ :8900 | `server.py` |

### 4.3 O que já temos a funcionar (Phase 1 ✅ + Phase 2 em curso)
- Topologias **emergentes do `access_list`** (não hard-coded): chain, best-of-N, tree.
- Pool **swappable** por config; orquestrador só vê ordinais → troca de provider sem retreinar.
- Recursão E2E verificada (ex.: `is_balanced` → código executável correto).
- **GRPO real validado** no hardware (Strix Halo, 96 GB VRAM iGPU): ~63 s/step com
  exec-reward, format_reward=1.0, exec_reward 0.22–1.0, gradientes a fluir com contraste.
- O **exec_reward** é o nosso diferencial fiel ao paper: o orquestrador é premiado por
  **orquestrar workflows que produzem a resposta CERTA** (executados nos workers reais
  via bridge :4141), não por sintaxe válida.

### 4.4 Diferenças deliberadas Amalia vs Fugu
- **Local-first**: orquestrador corre na nossa máquina; workers podem ser locais ou API.
- **Training-free por defeito**: prompt + few-shot já induz orquestração (o paper confirma);
  GRPO é o passo opcional de afiação (Phase 2).
- **Sem selection-head/SVD**: ficámos no ramo Conductor (puro prompting) de propósito —
  é o que permite embrulhar workers só-API.

---

## 5. Próximos passos candidatos (a decidir com o João)

1. **Fechar Phase 2** — treino GRPO "validação séria" (200 steps, ~3.5 h) + A/B honesto
   do orquestrador treinado vs baseline. Pre-flight já está 100% verde.
2. **Pool heterogéneo real** — em vez de 3× Qwen, usar capacidades genuinamente
   distintas via bridge :4141 (Claude/Gemini/GPT) para dar à orquestração algo real
   para explorar (a causa-raiz do delta nulo na primeira corrida).
3. **Adaptive worker selection** (extensão do paper) — treinar com subconjunto-k
   aleatório → robustez a pool variável, exatamente o pitch de "soberania" do Fugu.
4. **Eval em benchmark público** — replicar uma fatia do LiveCodeBench/GPQA-D para ter
   um número comparável (honesto) com a Tabela 1 do paper.

---

## Fontes

- Sakana, *Fugu: One Model to Command Them All* — https://sakana.ai/fugu-release/
- Nielsen, Cetin, Schwendeman, Sun, Xu, Tang, *Learning to Orchestrate Agents in Natural Language with the Conductor* — arXiv:2512.04388 (ICLR 2026)
- Xu, Sun, Schwendeman, Nielsen, Cetin, Tang, *Trinity: An Evolved LLM Coordinator* — OpenReview 5HaRjXai12 (ICLR 2026, Poster)
- *Sakana Fugu Technical Report* — arXiv:2606.21228
- CNBC, *Anthropic disables access to Fable 5 and Mythos 5…* — 12 jun 2026
- Vídeo analisado: "FUGU Proves Intelligence Isn't What You Think", DEEPTECH AI LABS (youtu.be/D8P_lcyYrsw)
