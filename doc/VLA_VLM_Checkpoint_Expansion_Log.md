# VLA 체크포인트 추가 평가 기록

작성일: 2026-11-05 · 결과 디렉터리: `runs/backbone_eval_v2_checkpoints`

## 목적과 범위

사용자 요청에 따라 VLA마다 공개 체크포인트를 하나씩 추가 평가한다. 사용자는 VLA를 직접 fine-tuning하는 대신 **공개 체크포인트 추가 평가**를 선택했다. 모델 자체는 고정하고, 기존과 동일하게 첫 답변 토큰 likelihood와 마지막 표현의 선형 probe를 측정한다.

| 기존 조건 | 추가 조건 | 비교의 의미 |
|---|---|---|
| `pi0`: pi0_base | `pi0_libero`: gs://openpi-assets/checkpoints/pi0_libero | Base와 LIBERO 체크포인트 비교 |
| `pi05`: pi05_base | `pi05_libero`: gs://openpi-assets/checkpoints/pi05_libero | Base와 LIBERO 체크포인트 비교 |
| `openvla`: openvla-7b | `openvla_libero_spatial`: openvla/openvla-7b-finetuned-libero-spatial | Base와 LIBERO-Spatial fine-tuned 모델 비교 |
| `openvla_oft`: OFT LIBERO-Spatial | `openvla_oft_goal`: moojink/openvla-7b-oft-finetuned-libero-goal | OFT 학습 suite 차이 비교 |

OFT-Goal은 OFT-Spatial에서 이어서 학습한 모델이라고 가정하지 않는다. 독립 suite 체크포인트를 비교한다. Pi LIBERO와 OpenVLA의 개별 suite는 학습 범위가 같지 않으므로 완전히 통제된 계열 간 비교도 아니다. PaliGemma와 Prismatic은 기준 VLM으로 기존 결과를 유지한다.

공식 근거: [openpi LIBERO 예제](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md), [openpi 학습 설정](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py), [OpenVLA-Spatial 모델 카드](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial), [OFT-Goal 모델 카드](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-goal).

## 기존 실험과 맞춘 조건

- `data_eval/`의 원본 이미지·질문·선택지·label·분할·seed는 유지한다.
- POPE, MME, SQA-IMG, VQAv2-MC, COCO-CaptionChoice, SIUO-MCQA, VLGuard-request의 7개 데이터셋을 평가한다.
- 기존과 같은 전처리, prompt, 후보 토큰, 마지막 위치, FP32 출력 점수 계산, probe 설정을 사용한다.
- OFT는 single-image VLM-only 경로와 native 양방향 attention을 유지한다. Action head/proprioception은 사용하지 않는다.
- `pi05_libero`는 upstream 설정의 `action_horizon=10`, `discrete_state_input=False`를 사용한다. 평가에는 action/state 입력을 넣지 않는다.
- 새 smoke와 전체 결과는 모두 v2 디렉터리에 저장한다. 기존 `backbone_eval_v1`은 덮어쓰지 않는다.

## 추가한 검증

POPE의 중복 source ID는 메모리에서 원본 manifest 행 번호를 붙여 구분한다. 문항·정답·행 순서·split은 바꾸지 않는다. 새 셔플과 범주별 집계는 고유 ID를 사용하며, 기존 POPE 보조 집계를 전후 비교에 사용하지 않는다.

기존 추출기가 일부 검사 실패나 split 추출 예외 뒤에도 정상 종료할 수 있었으므로, 이번 실행에는 `--strict-validation`을 적용한다. 후보 토큰, padding, attention, label 독립성, parameter fingerprint, 필수 split 추출을 확인하고 실패 시 해당 lane을 중단한다. Smoke는 16문항, 셔플은 8문항으로 별도 출력하며 통과 후 전체 평가한다.

완료 후에는 28개 모델·데이터 조합의 probe 수렴, class 누락, 예측과 metric 일치, 기준/추가 모델의 test population 일치를 검사한다. SIUO는 각 계열의 기준 모델과 동일한 test ID인지 확인한다. 모델 계열 사이의 26/32문항 차이는 유지되므로 계열 간 순위는 직접 비교하지 않는다.

## 준비 기록

- 기존 `pi0_libero`, `pi05_libero` 파일이 프로젝트의 `checkpoints/`에 있음을 확인했다. 실제 로딩은 Slurm smoke에서 확인한다.
- OpenVLA-Spatial revision `962318ce...`, OFT-Goal revision `c2d0f9fb...`를 고정한 파일 목록을 `checkpoints/hf/*.filelist.tsv`에 저장했다.
- 공개 merged model 가중치와 tokenizer/config/code만 dm01에서 다운로드한다. 별도 LoRA adapter와 action/proprio head는 이번 VLM-only 평가에 불필요하므로 받지 않는다.
- 호스트 아키텍처는 aarch64다. 기존 openpi 환경 및 사이트 PyTorch aarch64 컨테이너를 사용하며 새 바이너리를 설치하지 않는다.
- 모델 원본 파일 크기와 작은 metadata/code hash를 `checkpoint_sources.json`에 기록한다. 대용량 가중치 전체 hash 재계산은 로그인 노드에서 수행하지 않는다.
- 작은 문법 검사 및 POPE ID 보존·실패 gate 회귀 검사를 통과했다.

## 실행과 산출물

Slurm 진입점은 [checkpoint_eval.sbatch](../jobs/checkpoint_eval.sbatch)다. natStr의 한 노드, GPU 4개를 요청하고 각 GPU에서 추가 모델 하나를 평가한다. 제출 ID는 [submitted-jobs.tsv](../runs/submitted-jobs.tsv)에 기록한다.

최종 확인 파일:

- `runs/backbone_eval_v2_checkpoints/summary.csv`, `summary.md`
- `validation_report.md`, `diagnostics.md`, `models_manifest.json`
- `checkpoint_comparison.csv`, `checkpoint_comparison.md`
- `completion.json`: 모든 요구 검사를 통과한 경우에만 생성

상태와 결과 해석은 실제 작업 종료 후 이 문서에 덧붙인다. 제출 또는 다운로드 완료를 평가 완료로 간주하지 않는다.

## 다운로드 완료와 작업 제출

- OFT-Goal의 두 번째 가중치 shard 전송이 HTTP/2 연결 오류로 중단됐다. 부분 파일을 보존하고 HTTP/1.1 이어받기로 완료했다.
- Pi0-LIBERO 19개 파일(12.01 GB), Pi0.5-LIBERO 16개(12.44 GB), OpenVLA-Spatial 15개(15.09 GB), OFT-Goal 19개(15.09 GB)의 크기를 검증했다.
- OFT-Goal의 configuration/modeling/processing Python 코드와 tokenizer·전처리 파일이 기존 OFT-Spatial의 대응 파일과 동일함을 확인한 뒤 실행 준비를 마쳤다.
- 실행 코드의 작은 사본은 `runs/backbone_eval_v2_checkpoints/code_snapshot/`, 설정과 해시는 `run_config.json`에 보관했다.
- **2026-11-05 18:23 KST: 작업 1221183 제출.** `natStr`, 1노드/GPU 4개, 최대 45분. 최초 상태 `PENDING (Priority)`.
- 작업의 `WorkDir`과 `Command`가 이 프로젝트 및 `jobs/checkpoint_eval.sbatch`인 것을 확인했다. 제출 기록은 `runs/submitted-jobs.tsv`에 추가했다.

## 첫 실행과 BF16 수치 진단

작업 1221183은 18:35 KST에 시작해 4분 36초 후 실패 상태로 종료했다. π0-LIBERO와 π0.5-LIBERO는 smoke 및 전체 7개 데이터셋 추출을 완료했다. 실패 원인은 추가 OpenVLA 계열의 batch 검증이었다. 데이터나 두 Pi 모델의 추출 결과가 사라진 것은 아니다.

- OpenVLA-Spatial의 SQA 검증: singleton/padded-batch hidden 상대차 2.30%, 후보 log-probability 최대 차이 0.680. 네 검증 문항 중 후보 argmax 불일치가 있었다.
- OFT-Goal의 SIUO 검증: hidden 상대차 2.09%, 후보 log-probability 최대 차이 0.263. 후보 argmax 불일치가 있었다.
- 두 모델의 가중치 로딩·언어 head·attention 방향 검사는 통과했다. 낮은 정확도를 이유로 실패 처리한 것이 아니다.

작업 **1221331**에서 추가 모델 두 개와 기존 OpenVLA/OFT 기준 모델을 각각 GPU에 올려 native SDPA, masked padding 추가, eager attention+masked padding을 비교했다. 추가 모델의 argmax 민감도는 세 경로 모두에서 남았다. 기존 두 기준 모델은 검사한 문항에서 세 경로 모두 argmax가 같았다. 수치는 `runs/backbone_eval_v2_checkpoints/numerics/`에 저장했다. 이 작업에서는 이미 추출한 두 Pi 모델의 probe 학습도 완료했다.

배치 검증 기준을 낮추는 대신 **OpenVLA 계열의 primary 실행을 물리적 batch size 1로 고정**했다. 여러 문항을 adapter에 한 번에 넘기는 경우에도 각각 단독 forward한 뒤 합친다. 따라서 이 실행의 합성 호출 일치 검증은 원래 padded batch의 안정성을 증명하는 검사가 아니다. 원래 batch 경로의 실패와 진단을 그대로 보존한다.

같은 조건의 비교를 위해 기존 `openvla`, `openvla_oft` 기준 모델도 batch size 1로 재평가한다. 이 두 기준은 새 체크포인트가 아니라 수치 조건을 맞추기 위한 재실행이다. Pi 기준은 기존 v1 결과를 사용한다. 모델 가중치 dtype은 BF16, head/log-softmax는 FP32로 유지한다.

작업 **1221385**를 `checkpoint_eval_hf_single.sbatch`로 제출했다. 진단/Pi probe 작업 1221331 성공 후 실행되며, GPU 4개에서 기존 HF 기준 2개와 추가 HF 모델 2개를 평가한다. 최대 시간은 1시간이다. 수정된 실행 코드는 `code_snapshot_hf_singleton/`에 별도로 보관해 첫 실행의 snapshot을 보존했다.

## 전체 추출 완료와 최종 수렴 점검

작업 1221385는 18:59:49 KST에 node1301에서 시작해 **30분, COMPLETED(0:0)**로 종료했다. 네 HF 모델 모두 batch size 1 smoke 및 전체 추출을 통과했다. 추출 전후 parameter fingerprint가 같았고, 기록된 검증 실패는 없었다. 이후 HF probe 28개와 결과 집계를 생성했다.

부모 프로세스의 warning 포착만으로 수렴 여부를 판단하던 코드에서, 병렬 worker의 `ConvergenceWarning`을 놓치는 경우를 발견했다. 실제 `n_iter`를 확인하니 **OFT 기준 SQA probe**와 **원 v1 π0 기준 SQA probe**의 선택된 C가 각각 2,000회 상한에 도달해 있었다. 다른 이번 비교의 선택된 probe들은 상한 미만이었다.

`train_linear_probe.py`에 `--max_iter`와 `--feature_run_dir`를 추가하고, 단일 최적화 작업에서 warning 및 실제 반복 횟수를 함께 검사하도록 수정했다. 두 기준 SQA probe만 같은 특징·분할·C grid로 최대 10,000회 재최적화했다. 원 v1 파일은 읽기만 하고 새 probe와 예측은 v2 아래에 저장했다.

| 작업 | 수행 내용 | 결과 |
|---|---|---|
| 1221605 | OFT 기준 SQA probe 재최적화, 집계 | COMPLETED, 1분 23초 |
| 1221624 | π0 기준 SQA probe를 v1 캐시로 재최적화, 최종 집계 | COMPLETED, 52초 |

OFT SQA probe 정확도는 이번 단독 실행의 36.3%에서 36.2%로, π0 기준 SQA probe는 원 v1의 37.2%에서 37.7%로 바뀌었다. 최종 비교에서는 보완한 값을 사용한다. 수렴 여부는 기록된 boolean뿐 아니라 선택된 C의 `n_iter < max_iter`도 확인했다.

## 최종 상태: 완료

- 추가 체크포인트 4개 × 데이터셋 7개의 **28개 비교**를 완료했다.
- v2에는 새 모델 4개와 HF 기준 2개, 총 **42개 조합의 특징 추출**이 있다. π0 기준 SQA는 기존 특징을 재사용한 probe 보완이므로 추가 추출로 세지 않는다.
- 예측 파일에서 다시 계산한 정확도, 기준/추가 모델의 test population, 실제 최적화 수렴, 모델 고정, 원본 manifest/v1 summary 해시가 최종 검증을 통과했다.
- 같은 평가 조합이라도 baseline source는 다를 수 있다. Pi는 주로 v1을 사용하되 π0 SQA probe는 v1 특징으로 보완한 v2 결과를, HF는 모두 v2 단독 실행 기준을 사용한다. `checkpoint_comparison.csv`의 `baseline_run`이 해당 경로를 표시한다.
- SIUO는 공통 26문항 재집계를 `siuo_common26.csv`에 추가했다.
- 읽기 쉬운 최종 설명은 [추가 체크포인트 평가 결과](VLA_VLM_Checkpoint_Expansion_Results.md)에 정리했다.
