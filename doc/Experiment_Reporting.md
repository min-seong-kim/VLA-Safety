# 실험 단계 완료 자동 보고

2026-11-12 사용자 요청: 질문을 기다리지 않고 모델·실험의 주요 단계가 완료되면
한국어로 보고한다. 반복적인 step 진행률 보고는 생략한다.

## 보고 시점

- 학습 시연 준비 및 전체 검증 완료.
- 모델별 학습과 freeze 검증 완료.
- 모델별 체크포인트 검증 완료 및 최종 체크포인트 선택 완료.
- 모델별 최종 평가 완료, 전체 실험 및 보고서 완료.
- 오류 발생, 자동 복구 시도, 복구 중단, 사용자 요청에 따른 중지.

## 실행 방식

`scripts/semantic_safety/experiment_notifications.py`가 완료 증거 JSON만 확인한다.
평상시에는 LLM을 호출하지 않으며, 새 이벤트가 있을 때만
`codex queue --thread <현재 대화 ID> --message <보고 요청> --resume`를 호출한다.
원래 대화의 모델을 사용하고 별도의 학습·평가 실행 권한을 추가하지 않는다.
v5 supervisor에는 이 호출이 연결되어 있다. 이후 다른 형식의 실험은 완료 파일
형식에 맞게 연결한 뒤 실행한다.

설정은 `runs/experiment_reporting.json`에 있다. 실행별 `reporting/state.json`으로
중복을 방지하고, `reporting/events/`에 근거, `reporting/dispatch/`에 큐 등록 결과를
보존한다. 실제 보고 수신 확인은 `reporting/reported/`에 별도로 남긴다.
전송 결과가 불명확하면 중복 전송하지 않고 기록을 확인한다.

이미 보고한 과거 실행에 처음 연결할 때만 다음 명령으로 기존 이벤트를 등록한다.

```bash
openpi/.venv/bin/python scripts/semantic_safety/experiment_notifications.py \
  --run-dir runs/semantic_expansion_v5_lora_v1 --initialize
```

감시만 별도 실행할 때는 동일 명령에서 `--initialize` 대신 `--watch`를 사용한다.
완료·중지·확인 필요 상태에서는 감시가 종료된다.

## 동작 조건

서버와 Codex daemon이 실행 중이어야 같은 대화를 다시 실행할 수 있다.
서버가 꺼져 있으면 즉시 보고하지 못하며, 새 서버에서 감시를 다시 시작해야 한다.
큐 등록 성공과 실제 대화 수신은 구분한다. `--resume` 실패는 종료 코드가 0이어도
경고로 남을 수 있다. [OpenAI Docs의 queue 명령 설명](https://learn.chatgpt.com/docs/developer-commands#codex-queue).

현재 추론 서버의 대기 상태는 실험 진행을 뜻하지 않는다. 자동 보고를 위해
GPU 추론 서버를 계속 실행할 필요는 없다.
