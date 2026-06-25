# Changelog

## 0.2.0

- Enforced the human-arm requirement in `ControlRouter` for Issue #26: added trusted `SafetyPolicy.human_armed` state with a default of false, made `route()` block decoded intents with reason `human_arm_required` until armed before confidence, rate-limit, binding, and agent-payload checks, added the `--human-armed` CLI flag for `route-intent` and `simulate-intents`, and added the `human_arm_required_blocks_until_armed` self-test.
- Updated settings, API, frontend safety panel, docs, and README for consistency while leaving existing safety boundaries unchanged.
- Updated frontend dependencies: `react` and `react-dom` to 19.2.7, `vite` to 8.1.0, `@vitejs/plugin-react` to 6.0.3, and `lucide-react` to 1.21.0.

## 0.1.0

- Added the initial EEG control runtime package.
- Added `NeuroFrame`, deterministic CPU kernels, pipeline reports, settings, stream windows, signal quality reports, and dry-run action audit logs.
- Added CLI, FastAPI API, React settings UI, CI, and self-tests.
- Added tutorial, public Pages generation, repository health checks, and OSS operations files.
