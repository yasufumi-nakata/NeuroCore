# Deployment

NeuroCore の実運用は core package、decoder、permissioned executor を分けて配置します。

## Local Research Setup

1. EEG acquisition process が raw samples を取得します。
2. Device adapter が timestamp、sampling rate、channels、unit を含む `NeuroFrame` を作ります。
3. NeuroCore pipeline が validation と deterministic features を出します。
4. 外部 decoder が intent と confidence を返します。
5. NeuroCore `ControlRouter` が action envelope を作ります。
6. 別プロセスの executor が human-arm、OS permission、agent policy、audit log を確認して実行します。

## Recommended Runtime Policy

- `SafetyPolicy.require_human_arm` を default で有効にし、trusted state の `human_armed` は operator UI や permissioned executor から渡します。
- low confidence、rate limit、emergency stop、unknown binding は blocked action として記録します。
- agent payload は `trust: untrusted_decoded_intent` のまま渡します。
- decoded text をそのまま shell、browser、AI agent prompt に入れません。
- 研究ログは被験者 ID と raw EEG を分離し、public issue には redacted fixture だけを載せます。

## Service Split

| Layer | Responsibility | NeuroCore scope |
| --- | --- | --- |
| Device | EEG stream acquisition | Adapter only |
| Runtime | Frame validation, kernels, routing, audit envelope | Yes |
| Decoder | Intent weights and model inference | No |
| Executor | Mouse, keyboard, agent side effects | No |
| Operator UI | Settings, self-test, route simulation | Local console |

## Preflight

```bash
neurocore self-test --json
neurocore stream-demo --json
neurocore simulate-intents samples/intent_commands.json --human-armed --json
```

preflight が落ちる場合、EEG device を接続する前に settings、signal plan、safety policy、binding を直してください。
