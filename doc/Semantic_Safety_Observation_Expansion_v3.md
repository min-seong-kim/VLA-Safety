# LIBERO Semantic Safety 확장 장면 v3

검토 후보 378쌍, 실제 MuJoCo 렌더링 756조건, RGB 2268장.

`preview.html`에서 양쪽 agentview·wrist·근접 검토 시점을 볼 수 있다. 원본 PNG는 이미지 클릭으로 연다. 검토 상태와 메모는 브라우저에 저장되며 JSON으로 내보낼 수 있다.

## 구현 범위

canonical LIBERO의 tabletop·MountedPanda·카메라를 사용했다. 모든 물체는 초기 구도 검토를 위해 고정 fixture로 배치했다. 지시는 원 카탈로그를 유지하되 도구가 이미 손에 들려 있다는 상태는 집기 전 테이블 배치로 정규화했다. 이 버전은 물체를 실제로 조작하는 완성된 task 환경이 아니다. 동작 학습·관절·접촉 경로·성공 채점은 후속 단계다.

기존 mesh와 프로젝트에서 만든 단순화된 rigid geometry를 혼용한다. 사람·동물·식물은 정적인 시각 대체물이다. 실제 생체·유체·열·전기·오염·변형을 시뮬레이션하지 않는다. 재질·물질·온도·통전·알레르기·소유/사용 맥락은 텍스트 조건 또는 추가 장면 구현이 필요하며, RGB가 해당 사실을 증명한다고 볼 수 없다. 각 카드의 구현 자산과 검토 사항을 보고 물체 식별성 및 라벨 타당성을 확인한다.

양쪽 이미지가 동일한 항목은 필터로 따로 볼 수 있다. 의도한 동일 장면의 지시 대조 외에도 비시각적 상태, 서로 다른 물체에 같은 단순화 형상을 사용한 경우가 포함된다. 이런 항목은 관측만으로 label 차이가 성립한다고 가정하지 말고, 필요한 텍스트 조건 또는 추가 자산 구현을 검토한다.

## 파일

- `observations/<id>/agentview.png`, `wrist.png`: 512×512 실제 렌더. 안전/위험 글자나 검토용 표시는 없다.
- `overview.png`: 같은 시뮬레이터 상태의 768×768 근접 검토 카메라.
- `initial_observation.npz`: raw RGB 2개, 8차원 robot state, qpos/qvel 및 fixture transform.
- `model.xml`, `bddl/<id>.bddl`, `assets/`: 장면 재구성 자료. XML의 기존 mesh 의존성은 현재 저장소 경로를 사용한다.
- `scenes.jsonl`, `asset_bindings.jsonl`, `assets/manifest.json`: 배치·자산 출처·단순화 여부.
- `validation.json`: 수치 상태, 객체 가시성, 카메라, 로봇 접촉 검사. 의미적 검수 완료를 뜻하지 않는다.

## 재생성

```bash
envs/libero-render/bin/python scripts/semantic_safety/build_expansion_v3_observations.py prepare
VLA_EGL=/NHNHOME/26msit001_B/BASE/edge_ai_lab/minseong/.persist/nvidia-egl
__EGL_VENDOR_LIBRARY_FILENAMES="$VLA_EGL/10_nvidia.json" LD_LIBRARY_PATH="$VLA_EGL" envs/libero-render/bin/python scripts/semantic_safety/build_expansion_v3_observations.py render --workers 4
envs/libero-render/bin/python scripts/semantic_safety/build_expansion_v3_observations.py package
```

기존 v6 데이터와 모델, 원 카탈로그 및 검토 기록은 변경하지 않는다. 검토용 ZIP은 웹 화면과 이미지를 위한 묶음이며 전체 LIBERO 실행 환경을 포함하지 않는다.
