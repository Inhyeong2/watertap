# RO 제약조건 설정 내역 및 근거 (DPR_flowsheet_v2_spiral.py)

`DPR_flowsheet_v2_spiral.py`는 `DPR_flowsheet_v2.py`(flat-sheet)를 복사해 RO를
**spiral-wound + 현실적 운전 제약**으로 수정한 버전이다. 이 문서는 무엇을 바꿨고,
왜 그렇게 설정했는지, 그 근거와 검증 결과를 정리한다.

작성 맥락: DPR RO(역삼투) 단(stage)이 단일패스(single-stage)인데, 원래 flat 모델은
회수율(recovery)이 ~79%까지 비현실적으로 높게 나왔다(실무 단단 한계 ~40~67%).
원인은 모델이 **농축수(출구)측 수리 제약을 걸지 않아서**였고, 이를 제조사 설계
가이드라인에 맞춰 보정했다.

---

## 0. 모델 개요 (변경 안 함, 참고용)

- RO 유닛은 **1개(single-stage / single-pass)**. `RO_post`(2단)는 주석 처리됨.
- WaterTAP `ReverseOsmosis0D`, 옵션: `pressure_change_type=calculated`,
  `mass_transfer_coefficient=calculated`, `concentration_polarization_type=calculated`.
- 플랜트 전체 막을 **하나의 등가 평판/권취 채널**(length × width)로 lumping한다.
  - `length` = 흐름 방향 유로 길이, `width` = (모든 병렬 leaf 폭의 합) 등가 폭.
  - `area` = 막 표면적(투과 면), `feed_side.area` = 흐름 단면적(= channel_height × width × porosity, 유속 결정).

---

## 1. module_type = spiral_wound

**변경:** `build_RO`의 `ReverseOsmosis0D(...)`에 `module_type=ModuleType.spiral_wound` 추가
(디폴트는 `flat_sheet`였음). import: `from watertap.core.membrane_channel_base import ModuleType`.

**flat ↔ spiral 차이는 딱 두 가지** (나머지 물리는 동일):

| | flat_sheet | spiral_wound |
|---|---|---|
| 막 면적식 (`eq_area`) | `area = L × W` | `area = L × 2 × W` |
| Darcy 마찰계수 | `f = 0.42 + 189.3/Re` | `f = 6.23·Re^-0.3` (Schock & Miquel, 1987) |

**근거/이유:**
- 실제 시스템이 spiral-wound이므로 막 면적·압력강하 상관식을 spiral로 맞춤.
- spiral은 막 봉투(leaf)가 **반으로 접혀 감기**므로, 폭 W 하나당 막 면이 **양쪽 2장** → 면적 ×2.
- 마찰계수는 spiral 스페이서 채널 실험 상관식(Schock & Miquel) 사용.

**검증:** 해(解)에서 `area / (L × W) = 2.00`로 spiral 면적식이 정확히 적용됨을 확인.
초기값·스케일링·최종해 세 단계 모두 일관. (`area`는 하드코딩이 아니라 `eq_area`
제약식에서 역산되므로 ×2가 자동 반영됨.)

---

## 2. 초기 펌프 압력 40 bar → 20 bar (`set_operating_conditions`)

**변경:** `RO_main.pump.outlet.pressure[0].fix(40*101325)` → `fix(20*101325)`

**근거/이유:**
- spiral은 같은 W·L로 막 면적이 2배 → 같은 회수율에 **약 절반 압력**이면 충분.
- flat용 초기값 40 bar를 그대로 쓰면 2배 면적 + 40 bar에서 회수율이 0.9를 넘어
  **RO.initialize()가 비feasible로 실패**함.
- 초기 압력별 pre-optimize 회수율(저TDS 0.5 kg/m³ 기준) 탐색 결과:

  | 초기 압력 | pre-opt recovery |
  |---|---|
  | 15 bar | 0.40 |
  | **20 bar** | **0.56** |
  | 25 bar | 0.71 |
  | 30 bar | 0.85 |
  | 40 bar | init 실패 |

  → 20 bar가 flat(40 bar에서 ~0.60)과 유사한 시작점이면서 안정적으로 초기화됨.
- 이건 **초기값(initial guess)**일 뿐, 최적화에서 압력은 1~83 bar로 풀린다.

---

## 3. scale_system 마찰계수 식을 spiral로 변경

**변경:** `scale_system` 내 하드코딩된 마찰계수
`f_in = 0.42 + 189.3/re_in` (및 `f_out`) → `f = 6.23·re^-0.3`

**근거/이유:**
- 원본 `scale_system`은 **flat-sheet 마찰식**을 하드코딩해 `friction_factor_darcy`,
  `dP_dx`, `deltaP`의 **스케일링 인자**를 계산한다.
- module_type을 spiral로 바꾸면 실제 제약식은 `6.23·Re^-0.3`을 쓰므로, 스케일링도
  같은 식으로 맞춰 솔버 컨디셔닝의 일관성을 유지. (스케일링은 해의 정확도가 아니라
  수렴성에 영향; 두 식 값은 Re~300에서 비슷한 크기라 영향은 작지만 일관성을 위해 변경.)

---

## 4. RO length 상한 10 → 8 m (`optimize_operation`, RBAT)

**변경:** `RO_main.RO.length.setub(10)` → `setub(8)` (하한 6 m 유지)

**근거/이유:**
- spiral element 1개 ≈ 1.016 m(8040, 40인치). 표준 압력용기(pressure vessel) 1개에
  **element 6~8개를 직렬**로 끼움(8개가 대형 플랜트 표준).
- 따라서 `length` 6~8 m = **직렬 element 6~8개 = 표준 용기 1개**에 대응.
- 기존 상한 10 m(= 10 elements)는 표준 용기(최대 8개)를 초과 → 비현실적이라 8 m로 하향.

---

## 5. ⭐ RO 출구(농축수) cross-flow velocity 하한 추가 (핵심)

**추가:** `RO_main.RO.feed_side.velocity[0, 1].setlb(0.1)`  (단위 m/s)
- `velocity[0, 0]` = 입구 유속, `velocity[0, 1]` = **출구(농축수) 유속**.
- 기존엔 입구 유속만 0.1~0.3으로 제약했고, 출구는 무제약이었음.

**근거/이유 (실무 규정):**
- 단일패스 RO 회수율은 제조사 가이드라인상 **element당 최대 회수율**과
  **베셀당 최소 농축수 유량(minimum concentrate flow)**으로 제한된다.
  - FilmTec/DuPont: element당 최대 회수율 — 지표수(SDI<5) **13~15%**, 지하수 17~19%
    → 용기(6~8 elements 직렬) 누적 시 단단 현실 회수율 **~45~67%**.
  - 8" element 최소 농축수 유량 ≈ **2.7~3.6 m³/h**, element당 최대 dP ≈ **1.0 bar**.
- 선속도(velocity)로는: 농축수 선속도 보통 **0.05~0.15 m/s**, **권장 최소 ≈ 0.1 m/s**.
  이보다 낮으면 막 표면 농축계수가 **100을 초과** → 투과 플럭스 급감, 농도분극·스케일 심화.
- lumped 0D 모델에선 "베셀당 최소 농축수 유량"을 **등가 채널의 출구 선속도 하한**으로
  거는 것이 가장 자연스러운 번역이다.

**효과 (정량):** 질량수지상 `v_exit = v_in × (1 − recovery)`이므로
```
recovery_max = 1 − v_exit_min / v_in_max
```
- 현재 설정(`v_exit ≥ 0.10`, `v_in ≤ 0.30`) → **recovery ≤ 0.667**
- 더 보수적으로 가려면: `v_exit ≥ 0.12` → recovery ≤ 0.60, `v_exit ≥ 0.15` → recovery ≤ 0.50

**왜 이 방식이 좋은가:** "recovery 상한을 그냥 0.6으로 박는" 것보다, *왜* 그 값인지를
물리(최소 농축수 유속)로 설명하므로 방어력이 높다. 또 추후 다단(concentrate staging)으로
확장하면 **같은 제약을 단별로 복제**하면 된다(새 물리 아님).

### 5.1 per-element 회수율 제약을 따로 두지 않는 이유

실무 가이드라인엔 "element당 최대 회수율(지표수 13~15%)"도 있지만, 본 모델엔 이를 따로
걸지 않는다. 괜찮은 이유:

1. **0D lumped 모델은 element를 분해하지 않는다.** 전체 막을 하나의 등가 채널(입·출구
   2점)로 보므로 element 단위 회수율 변수 자체가 없어 제약을 걸 대상이 없다.
2. **per-element 회수율 한계 ≈ 최소 농축수 유량(= 출구 유속 하한).** 둘 다 "국부적으로
   물을 너무 빨리 빼서 농축수 유속이 떨어지는 것"을 막는 같은 물리다. 실제 베셀에서
   마지막 element의 농축수 유량이 가장 낮아 binding이 되는데, 본 모델은 그 지점(시스템
   출구)을 §5에서 직접 제약하므로 같은 물리를 시스템 레벨에서 더 직접 잡는다.
3. **수치적으로도 동일한 캡을 준다.** 시스템 회수율 = `1 − (1−r)^n` (r=element당 회수율,
   n=직렬 element 수):

   | per-element r | 8개 직렬 → 시스템 recovery |
   |---|---|
   | 13% | 1 − 0.87⁸ = 0.672 |
   | 15% | 1 − 0.85⁸ = 0.728 |
   | **12.8%** | **0.667** |

   §5의 출구유속 제약이 준 캡 0.667은 정확히 "element당 12.8% × 8단"과 같다(지표수
   권장 13~15% 범위 내). 즉 두 제약이 사실상 같은 결과 → per-element를 따로 안 걸어도 됨.

**단, 출구유속 제약이 못 잡는 것:** 유로 방향 **플럭스 분포**(선두 element 과플럭스 →
파울링)는 못 잡는다. 이건 per-element 회수율이 아니라 **최대 플럭스 제약**(§9)이 맞는 도구다.

---

## 6. 그대로 유지한 기존 제약 (참고)

`optimize_operation`(RBAT)에서 기존대로 유지:
- 입구 유속 `velocity[0,0]`: 0.10 ~ 0.30 m/s
- 운전 압력(pump out): 1 ~ 83 bar (8.3e6 Pa; 압력용기 burst 한계)
- recovery: 0.30 ~ 0.90 (실질적으로는 §9 플럭스 제약이 가장 빡빡해 ~0.45에서 binding;
  플럭스 제약이 없으면 §5 출구유속 제약이 0.667에서 binding)
- 투과수 TDS: ≤ 0.5 kg/m³ (= 500 mg/L)
- rejection(TDS): ≥ 0.99
- 막 상수(input, 고정): A_comp = 4.2e-12 m/s/Pa, B_comp = 3.5e-8 m/s (기수막 값)

---

## 7. 검증 결과

### 7.1 회수율 캡 (여러 조건) — ※ 아래는 §9 플럭스 제약 적용 *전*(velocity-only) 결과
(플럭스 제약 적용 후의 현실적 회수율 ~0.45 / LCOW는 §9.1~9.2 참조)
CA/CO × flow {10, 49, 99 MGD} × TDS {0.5, 1.0 kg/m³}에서:
- **recovery = 0.667 (모든 조건 동일)**, `v_in = 0.300`(상한), `v_exit = 0.100`(하한) 둘 다 binding.
- 0.667 = 1 − 0.1/0.3 (예측과 정확히 일치).
- 모든 조건에서 동일한 이유: 저염이라 삼투압이 작아 회수율을 막는 건 osmotic이 아니라
  **수리 제약(농축수 유속)**. 저염 DPR에선 타당한 거동.
- LCOW는 규모의 경제 뚜렷 (CA 10→99 MGD: 2.25 → 1.62 $/m³), TDS 영향은 미미.

### 7.2 고TDS 한계 (CA, 10 MGD, feed TDS 스캔)
- **TDS 25,000 mg/L까지 수렴** (recovery 0.667 유지, 압력 28 → 73.6 bar 상승,
  투과수 TDS 최대 202.6 mg/L로 500 한참 아래).
- **30,000 mg/L 이상 초기화 실패** — 원인:
  1. 초기 압력 20 bar가 고TDS 삼투압(피드 ~24 bar, 농축수 ~72 bar)보다 낮아 init 비feasible.
  2. recovery 0.667 기준 농축수 삼투압이 83 bar 상한에 근접(25 g/L에서 이미 73.6 bar).
- DPR feed는 500~1,000 mg/L이므로 25 g/L는 설계값의 25~50배 → **관련 범위 완전 커버**.
- 더 높은 TDS가 필요하면: (a) 초기 펌프압을 TDS 기반으로 상향, (b) 막 상수 A/B를 해수막 값으로 변경,
  (c) recovery를 압력-제한으로 하향 운전.

### 7.3 기타
- per-point IPOPT knife-edge(특정 유량에서 단발 solver error)는 flat 모델과 동일한 특성.
  이웃 점은 매끄럽게 수렴하며, sweep의 warm-start + rescue(perturbation 재시도)가 처리한다.
- CBAT(비RO)는 RO 변경에 영향 없음 (검증 완료).

---

## 8. 회수율 타깃을 바꾸려면

아래는 **플럭스 제약(§9)이 binding이 아닐 때** velocity만으로 결정되는 캡이다
(v_in 상한 0.3 기준, `recovery_max = 1 − v_exit/v_in`):

| v_exit 하한 | recovery 캡 (velocity-only) |
|---|---|
| 0.10 m/s (현재) | 0.667 |
| 0.12 m/s | 0.60 |
| 0.15 m/s | 0.50 |

**단, 현재는 §9의 플럭스 제약(≤20 LMH)이 더 빡빡해 실제 binding 회수율은 ~0.45**이다
(§9.1 참조). 즉 velocity-only 캡(0.667)은 상한일 뿐이고, flux 제약이 단단 recovery를
~0.45~0.5로 더 낮춘다. 회수율을 더 올리려면 다단(concentrate staging)이 필요하다.

---

## 9. 최대 설계 플럭스 제약 (적용됨)

**적용:** `optimize_operation`(RBAT)에 평균 볼류메트릭 플럭스 상한 추가
```python
max_design_flux = convert(20 L/m²/hr → m/s)   # ≈ 5.556e-6 m/s
RO.eq_max_design_flux:  mixed_permeate.flow_vol_phase["Liq"] <= max_design_flux * RO.area
```

**값의 근거 (20 LMH):** DPR/tertiary 폐수 RO 설계 플럭스.
- **GWRS(OCWD, 대표 DPR 플랜트): 20.4 LMH (12 gfd)** — 직접 벤치마크
- 도시폐수 RO 평균 17~20 LMH, 표준 설계 ~18~19 LMH (11 gfd)
- FilmTec filtered tertiary effluent 19~22 LMH

**왜 cross-velocity로 커버 안 되나:** 플럭스 `Jw = A·(ΔP−Δπ)`는 **압력**이, 유속은
**회수율·CP·압력강하**가 지배 → 독립. 제약 전에는 옵티마이저가 면적을 과소설계해
평균 38 LMH(선두 40~47)로 운전 → 권장의 ~2배(파울링·낙관적). 별도 제약이 필요.

### 9.1 ⭐ 핵심 결과: 진짜 최적 recovery ≈ 0.45 (flux + rejection + length가 캡)

플럭스 제약을 켜면 **recovery가 0.667 → 0.452로 내려간다** (CA 10 MGD, TDS 0.5).
이는 artifact가 아니라 **물리적으로 올바른 단단 한계**이며, 진짜 최적이다.

**진짜 최적임을 확정한 근거:**
1. 자유최적화가 recovery 0.452에서 **ok=True**로 수렴.
2. recovery 0.50에서 출발해 풀어주면 옵티마이저가 **다시 0.452로 내려가 수렴** — 최소화
   솔버가 더 비싼 쪽으로 가는 유일한 이유는 싼 쪽(>0.452)이 **infeasible**이기 때문.
3. recovery 0.46~0.51 고정 시 전부 **ok=False(IPOPT: locally infeasible)**. 그때 보이는
   낮은 LCOW(3.2~3.5)는 비수렴/infeasible 점의 무의미한 값이다.

**최적에서 동시에 binding하는 세 제약** (recovery 0.452 기준):
- **flux = 20 LMH** (≤20)
- **rejection = 0.990** (≥0.99) — recovery↑ → 브라인 농축 → 염 투과↑ → rejection↓
- **length = 8 m** (≤8) — 면적을 더 못 키움

→ 이 셋이 동시에 빡빡해지는 모서리가 recovery 0.452. 그 위로는 세 제약이 충돌(infeasible).
**참고: velocity는 binding이 아니다** (v_exit 0.127 > 0.1, v_in 0.231 < 0.30). 즉 flux 제약을
넣은 뒤로는 recovery를 캡하는 것이 **velocity가 아니라 flux + rejection + length**다.
(§5 velocity 제약은 flux가 없을 때만 0.667에서 binding했다 — §8 참조.)

결과적으로 **현실적 플럭스(20 LMH) + 99% 염제거 + 단일 베셀 길이(≤8 elements)**가 함께
단단 recovery를 **~45%**로 제한하며, 이는 교과서적 단단 RO 한계(40~50%)와 일치한다.
75~85%를 원하면 다단(concentrate staging)이 필요하다(GWRS가 3단으로 85% 하는 이유).

### 9.2 현실적 LCOW (flux≤20 + 모든 제약, 수렴 OK)

| state | 10 MGD | 50 MGD | 100 MGD |
|---|---|---|---|
| CA | 3.58 | 2.88 | 2.66 |
| CO | 3.28 | 2.80 | 2.60 |
| FL | 3.27 | 2.79 | 2.59 |

(recovery ≈ 0.45, flux 20.0 LMH 전 조건 binding; feed TDS 0.5↔1.0 차이 미미)
플럭스 제약 전(recovery 0.79, flux 38) 대비 LCOW 상승은 현실적: 면적 2배(막비↑) +
recovery 하락(생산수↓)이 함께 작용.

---

## 10. Optimized vs Conservative 비교 (LCOW vs system capacity)

비용 최적화의 "효용"을 정량화하기 위한 비교. 같은 모델(spiral, §1~9 제약 모두 적용)에서
운전조건을 자유 최적화한 경우와 보수적으로 고정한 경우의 LCOW를 system capacity(1~100 MGD)에
대해 비교한다. 스크립트: `dpr_spiral_analysis.py` (rbat/cbat/envelopes 명령). 출력: `output/optimal_vs_conserv/
optimal_vs_conserv_<STATE>.{csv,png}` (STATE = CA / CO / FL).

### 10.1 공통 입력 조건 + 케이스 정의

**모든 케이스 공통 입력 (sweep 고정):**
| 항목 | 값 |
|---|---|
| feed flow (sweep 축) | 1 ~ 100 MGD |
| feed TOC | 0.01 kg/m³ = **10 mg/L** (기본값) |
| feed TDS | 0.5 kg/m³ = **500 mg/L** (고정) |
| feed nitrate/nitrite | 기본값 (DPR_initial_setting) |
| TOC 유출 목표 `toc_eff` | **CA 0.5, CO/FL 2 mg/L** |
| 막 상수 (RBAT) | A=4.2e-12, B=3.5e-8, channel_height 1mm, spacer_porosity 0.85 |
| Cl contact_time | 30 min |
| 물리·수질 제약 (RBAT, §1~9) | flux≤20 LMH, rejection(TDS)≥0.99, v_in 0.1~0.3, v_exit≥0.1, length 6~8, pressure 1~83 bar |

**RBAT 케이스 (`dpr_spiral_analysis.py rbat`, STATE=CA/CO/FL):**
| 케이스 | 상류(Ozone/BAF/UV) | RO recovery |
|---|---|---|
| **Optimized** | 자유 최적화 | 자유 → ~0.45 (flux+rejection 캡) |
| **Conserv rec=0.45** | 보수 강제/고정 | 0.45 (≈최적, 상류효과 isolation용) |
| **Conserv rec=0.40** | 보수 강제/고정 | 0.40 |
| **Conserv rec=0.35** | 보수 강제/고정 | 0.35 |
| **Conserv rec=0.30** | 보수 강제/고정 | 0.30 |

**CBAT 케이스 (`dpr_spiral_analysis.py cbat`, STATE=CO/FL — CA는 규정상 불가):**
| 케이스 | 상류(Ozone/BAF/UV) | GAC |
|---|---|---|
| **Optimized** | 자유 최적화 | 자유 (removal은 total_toc_removal로 목표 맞춤, EBCT 자유→10) |
| **Conservative** | 보수 강제/고정 | removal 0.80 고정 + EBCT 20 고정 + required_BV 물리계산-고정 |

### 10.2 각 운전조건/변수를 무엇으로 고정했나

**보수(Conservative) 케이스 — 고정값:**
| 유닛 | 변수 | 보수 고정값 | (최적화가 고른 값) |
|---|---|---|---|
| Ozone | `contact_time` | 10 min (강제) | CA: LRV고정 6.39; CO/FL: 5~7 |
| Ozone | `O3toTOC` | 1.0 (강제) | CA: 1(제약); CO/FL: ~0.79 |
| BAF | `EBCT` | 30 min | CA **20**(하한), CO/FL **30**(상한) |
| UV-AOP | `hydrogen_peroxide_dose` | 10 mg/L | **2**(RBAT 하한) |
| RO | `recovery_vol_phase` | 0.45/0.40/0.35/0.30 (고정) | ~0.452 (자유) |

> **Ozone 강제 방식(중요).** ozone 모델에는 `O3toTOC_ratio_constraint`가 있어, 규정 LRV를
> 맞추도록 **O3toTOC를 contact_time의 함수로 결정**한다(예: CO에서 contact 7 → O3toTOC 0.79;
> CA는 항상 O3toTOC==1). 따라서 O3toTOC는 자유 설계변수가 아니다. 보수 케이스는 "규정-튜닝
> 도징을 무시하고 최대로 과투입"하려는 것이므로, **이 제약을 deactivate한 뒤 contact_time=10,
> O3toTOC=1로 강제**한다(둘 다 최댓값 → LRV는 초과 충족). 제약을 끄지 않고 O3toTOC=1로
> 고정하면 CO/FL에서 제약(≈0.94 요구)과 충돌해 `locally infeasible`이 된다.
>
> conservative 재solve는 optimized 해를 warm-start로 사용한다.

**CBAT의 GAC 보수화 (RBAT엔 GAC 없음):** CBAT 보수 케이스는 GAC를 카본·자본 양쪽으로 보수화:
- `total_toc_removal` **제거**(BAF/GAC 분배 최적화 끔) + `GAC.removal_frac[toc] = 0.80` 고정
  (카본비 보수; required_BV↓ → 카본↑). → 유출 TOC가 출력이 되며 ~1.37 mg/L로 목표(2) 만족.
- `GAC.EBCT = 20` 고정(자본비 보수; capital ∝ EBCT).
- `required_BV`는 `required_BV_constraint`(cubic)에서 한 번 계산(`calculate_variable_from_constraint`)
  해 **고정**하고 그 제약을 **deactivate**. 이유: CBAT는 RO slack이 없어 전부 고정 시 0-DOF인데,
  cubic required_BV 제약이 살아있으면 동시 솔브가 발산한다. required_BV를 "고정 숫자"로 만들면
  (pre-optimize square 모델과 동일 구조) 수렴한다. 값은 removal 0.80의 물리값이라 일관성 유지.
- **GAC 비용 구조 근거**: EBCT → 자본비(`0.0043·flow·EBCT`), removal → required_BV → 카본 OPEX
  (`∝flow/required_BV`). 둘은 독립이라 양쪽 다 고정해야 진짜 보수적.

- 종속변수(고정 안 함, 제약으로 결정): BAF `removal_frac[toc]`(link 제약), UV `uv_dose`
  (required_UV_dose 제약), RO `pressure·area·width·velocity`(transport+flux 제약),
  RBAT는 GAC 없음 / CBAT의 GAC `removal`·`required_BV`(total_toc_removal·required_BV 제약).
- `cl_contact_time`은 두 케이스 모두 30 min 고정(원래부터 input).

**Optimized 케이스 — 자유(unfixed)인 것:** BAF EBCT, UV H2O2, RO recovery·pressure·length·
velocity(+area·width), Ozone O3toTOC. **고정인 것:** Ozone contact_time(CA, LRV), 막상수 A/B,
channel_height, spacer_porosity, cl_contact_time.

> 주의(CA): `_setup_ozone_optimization`이 CA에선 ozone contact_time을 LRV 규정값으로 **고정**
> 하므로, CA 최적화에서도 ozone contact_time은 자유가 아니다. CO/FL은 ozone도 자유 최적화됨.

### 10.3 RBAT 결과 (최종, 3개 state) 및 페널티 분해

LCOW($/m³), 괄호는 Optimized 대비 보수 페널티(%). 전부 수렴(CO/FL은 각 1점 knife-edge NaN).

**CA** (toc_eff 0.5):
| 용량 | Optimized | 보수 0.45 | 보수 0.40 | 보수 0.35 | 보수 0.30 |
|---|---|---|---|---|---|
| 10 MGD | 3.58 | 3.93 (+10%) | 4.50 (+26%) | 5.24 (+46%) | 6.22 (+74%) |
| 50 MGD | 2.88 | 3.07 (+6%) | 3.54 (+23%) | 4.14 (+44%) | 4.94 (+71%) |
| 100 MGD | 2.66 | 2.83 (+6%) | 3.26 (+22%) | 3.83 (+44%) | 4.58 (+72%) |

**CO / FL** (toc_eff 2; 두 state 거의 동일):
| 용량 | Optimized | 보수 0.45 | 보수 0.40 | 보수 0.35 | 보수 0.30 |
|---|---|---|---|---|---|
| 10 MGD | 3.28 | 3.71 (+13%) | 4.25 (+30%) | 4.95 (+51%) | 5.87 (+79%) |
| 50 MGD | 2.80 | 3.04 (+9%) | 3.50 (+25%) | 4.09 (+46%) | 4.89 (+75%) |
| 100 MGD | 2.60 | 2.81 (+8%) | 3.24 (+25%) | 3.81 (+47%) | 4.56 (+75%) |

**페널티 분해 (10 MGD): 전체 +26%(CA 보수 0.40) = 상류 ~+10% × recovery 0.45→0.40 ~+15%.**
- **상류(BAF EBCT→30, UV H2O2→10, ozone 강제)만의 페널티 ≈ +10~13%** (= "보수 0.45" 곡선과
  Optimized 곡선의 간격).
- **recovery가 지배적 레버**: 0.45→0.30으로 낮추면 +72~79%. 저염 RO에선 회수율이 LCOW를
  가장 크게 좌우(생산수·브라인·전처리 단위비 모두 영향).
- 모든 곡선은 규모의 경제(소용량 급등, ~20 MGD 이후 완만).

### 10.4 실무 RO 운전 기준 메모 (왜 recovery를 레버로?)

실무 RO는 recovery(설계 마스터 변수)·design flux·crossflow·feed pressure(운영 조절)·
product 전도도(rejection)·Δp(telescoping)·CIP 주기로 운영. 설계 비교(LCOW vs 용량)에선
recovery가 1차 설계 레버이자 가장 지배적이라 conservative 레버로 적합.
- **design flux를 낮추는 보수화는 이 막에선 불가** (flux<20 → rejection<0.99, §9 참조).
- 더 높은 recovery(75~85%)는 **다단(concentrate staging)** 필요(미구현, 향후 확장).

---

## 11. CBAT 비교 (CO/FL — CA는 규정상 불가)

CBAT는 RO가 없어 recovery/flux/rejection 레버가 없다. 보수화는 ozone 강제(§10.2) + BAF EBCT 30
+ UV H2O2 10 + **GAC removal 0.80·EBCT 20·required_BV 물리계산-고정**(§10.2 GAC 단락).
유출 TOC는 출력값이며 보수 케이스에서 **~1.37 mg/L** (목표 2 만족, 여유 OK).

**규정 주의:** CBAT는 CA의 virus 20 LRV/4공정을 못 맞춘다(RO 없이 최대 18 LRV/3공정) → CA는 RBAT 필수.

### 11.1 결과 (최종, 전부 수렴)

LCOW($/m³), 괄호는 보수 페널티(%):
| 용량 | CO Opt | CO 보수 | FL Opt | FL 보수 |
|---|---|---|---|---|
| 10 MGD | 1.47 | 1.74 (+18%) | 1.42 | 1.74 (+22%) |
| 50 MGD | 1.07 | 1.27 (+18%) | 1.06 | 1.30 (+22%) |
| 100 MGD | 0.96 | 1.15 (+20%) | 0.95 | 1.18 (+24%) |

### 11.2 RBAT vs CBAT 보수 페널티 비교 (핵심 시사점)

- **RBAT 보수 페널티(최대 +72~79%) ≫ CBAT(+18~24%).**
- 이유: RBAT는 **RO recovery**라는 강력한 단일 레버가 있어 최적화 여지가 큼. CBAT는 recovery 레버가
  없고, 옵티마이저가 이미 BAF EBCT를 최대로 쓰므로(BAF가 TOC 제거에 GAC보다 쌈) 최적화 여지가 작음.
- 즉 **운전 최적화의 효용은 RBAT(특히 RO recovery)에서 훨씬 크다.**

---

## 12. Monte Carlo (spiral) + worst/best envelope

§1~9의 spiral·제약 RO를 반영한 Monte Carlo. 스크립트: `DPR_sweep_v2_spiral.py`
(= `DPR_sweep_v2.py`를 spiral 플로우시트 import로 바꾼 복사본; 원본 불변). 출력은 TAG
`ROspiral_add_const`로 구분해 flat 결과를 덮어쓰지 않음.

### 12.1 설정 및 flat 대비
- RBAT, state별 CA/CO/FL, 각 **1000 샘플**(seed=0 → 세 state 동일 입력).
- MC config(예전 RBAT와 동일): feed_flow 1~100 MGD, feed_toc 7~15 mg/L, feed_tds 0.5~1.0,
  brine_disposal_cost 0.05~0.66. (CBAT는 RO가 없어 불변 → 재실행 안 함; 기존 CBAT MC 사용.)
- 전부 1000/1000 수렴. recovery는 0.45로 캡(spiral 제약).
- **flat 대비 LCOW 중앙값 +62%** (CA 1.33→2.16, CO 1.27→2.06, FL 1.27→2.06). 즉 이전 flat MC는
  비현실적 RO(recovery 0.76, flux 38)로 LCOW를 ~38% 과소평가했음.

### 12.2 worst/best optimal envelope 커브
각 산점도 위에, MC 입력범위의 **LCOW 최대(worst)·최소(best) 코너**에서의 **최적 운전 LCOW**를
system capacity의 함수로 올림 (스크립트 `dpr_spiral_analysis.py envelopes`). 즉 클라우드의 상·하단 envelope.

| envelope | RBAT 입력 | CBAT 입력 |
|---|---|---|
| **worst (상단, 실선)** | TOC 15 mg/L, TDS 1000 mg/L, brine 0.66 $/m³ | TOC 15 mg/L |
| **best (하단, 점선)** | TOC 7 mg/L, TDS 500 mg/L, brine 0.05 $/m³ | TOC 7 mg/L |

worst/best optimal LCOW ($/m³):
| 케이스 | 1 MGD | 10 MGD | 50 MGD | 100 MGD |
|---|---|---|---|---|
| RBAT CA | 9.60 / 8.81 | 3.21 / 2.41 | 2.51 / 1.71 | 2.29 / 1.49 |
| RBAT CO | 6.37 / 5.60 | 2.89 / 2.12 | 2.41 / 1.64 | 2.21 / 1.44 |
| RBAT FL | 6.36 / 5.60 | 2.89 / 2.12 | 2.40 / 1.63 | 2.20 / 1.43 |
| CBAT CO | 5.76 / 4.41 | 2.28 / 1.18 | 1.86 / 0.81 | 1.75 / 0.70 |
| CBAT FL | 5.53 / 4.38 | 2.30 / 1.15 | 1.94 / 0.79 | 1.83 / 0.68 |

해석: 두 envelope 사이 폭 = 수질(TOC/TDS)·brine 비용 불확실성이 LCOW에 주는 범위.
**CBAT는 best가 매우 낮아(0.68~0.70) 폭이 넓고**(브라인·TDS가 없어 TOC만으로도 변동 큼),
RBAT는 worst/best 간격이 상대적으로 좁다.

### 12.3 산출물 (output/monte_carlo/)
- 산점도 PNG 5개: `mc_v2_{CA,CO,FL}_RBAT_ROspiral_add_const_n1000_flow_vs_LCOW.png`,
  `mc_v2_{CO,FL}_CBAT_n1000_flow_vs_LCOW.png` — **통일 스타일**: 공통 x·y축, 단색
  (RBAT 연두 yellowgreen / CBAT 연보라 mediumpurple), worst(실선)·best(점선) envelope 오버레이.
- envelope 데이터 CSV: 케이스별 `..._envelope.csv` + 통합 `mc_envelopes_all.csv`
  (train, state, flow_MGD, worst_LCOW, best_LCOW).
- 1000-샘플 MC 원자료 CSV: `mc_v2_<...>_n1000.csv` (각 행=샘플 1개, 입력 4 + 출력 8).
- 플롯 함수 `plot_monte_carlo.plot_flow_vs_lcow`: `color_by=None`(단색), `point_color`,
  `xlim`/`ylim`(공통축), `curves`(다중 오버레이 커브) 인자 지원.

---

## 13. CAPEX ratio (optimal vs conservative)

LCOW 외에 자본/운영비 구성비를 보는 지표. opt-vs-conserv 비교(§10/§11)에 추가됨.

### 13.1 정의
```
capex_ratio = annualized_capex / (annualized_capex + opex)
annualized_capex = total_capital_cost × CRF
opex            = total_operating_cost  ( + brine_disposal_cost,  RBAT만 )
```
- **CRF (Capital Recovery Factor) = 0.10000** (WACC 9.31%, plant lifetime 30년).
  `CRF = i(1+i)^n / ((1+i)^n − 1)`. zo·ro costing의 WACC·수명이 동일하게 설정돼 **zo CRF = ro CRF**
  → `total_capital_cost × zo CRF`로 연환산(LCOW 분자의 capex 항과 동일 방식).
- **RBAT는 opex에 `brine_disposal_cost`를 명시적으로 더한다.** 이 모델에서 brine disposal은
  `total_operating_cost`가 아니라 `total_externalities`에 들어있기 때문(§코드 line 1554/1593).
  CBAT는 brine 없음 → opex = total_operating_cost.
- 구현: `dpr_spiral_analysis._capex_ratio(m, train)`. CSV에 `capex_ratio` 열, 그래프
  `optimal_vs_conserv_<...>_capexratio.png`.

### 13.2 결과 (대표값)
RBAT (CA; CO/FL는 RBAT에선 거의 동일):
| 용량 | Opt | 0.45 | 0.40 | 0.35 | 0.30 |
|---|---|---|---|---|---|
| 10 MGD | 0.423 | 0.449 | 0.436 | 0.424 | 0.412 |
| 50 MGD | 0.319 | 0.337 | 0.323 | 0.311 | 0.299 |
| 100 MGD | 0.275 | 0.292 | 0.279 | 0.266 | 0.255 |

CBAT (50 MGD): CO Opt 0.334 / Cons 0.303 ; FL Opt 0.304 / Cons 0.297.

### 13.3 관찰
- **규모↑ → capex_ratio↓**: 소규모 ~0.7(CAPEX-heavy) → 대규모 ~0.25(OPEX-heavy). 규모의 경제로
  CAPEX가 분산되어 OPEX 비중이 커짐.
- **recovery↓ → capex_ratio↓** (RBAT): 낮은 recovery는 brine disposal·feed 비례 OPEX를 키움.
- **CBAT: CO는 opt↔cons capex_ratio gap이 크고(~0.03), FL은 거의 0.** 원인은 **규제 LRV 차이**
  (FL: ozone crypto LRV 2 / Cl virus 2 > CO: 1 / 1). FL은 LRV가 높아 **최적화해도 ozone·Cl 소독을
  많이** 써야 함 → optimized가 이미 OPEX-heavy → conservative(소독 최대 강제)와 비용구조가 비슷
  → gap 작음. CO는 LRV가 낮아 optimized가 소독을 아껴 CAPEX-heavy → conservative와 gap 큼.
  (RBAT은 RO recovery·brine이 capex_ratio를 지배해 LRV 영향이 묻혀 CO=FL로 동일.)

---

## 참고 문헌 / 출처

- FILMTEC™ Membrane System Design Guidelines for Commercial Elements
  (Dow/DuPont, Form No. 609-02054) — element당 최대 회수율, 최소 농축수 유량,
  element당 최대 압력강하, 최대 feed 압력.
  https://www.lenntech.com/Data-sheets/Filmtec-Membrane-Design-Guidlines-L.pdf
- DuPont FilmTec 8-inch Membrane System Design Guidelines (Form 45-D01695).
  https://www.dupont.com/content/dam/water/amer/us/en/water/public/documents/en/RO-NF-FilmTec-Membrane-Sys-Design-Guidelines-8inch-Manual-Exc-45-D01695-en.pdf
- 농축수 선속도 0.05~0.15 m/s, 권장 최소 0.1 m/s (US Patent 10,252,219 — RO 막장치 운전법).
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10252219
- Schock & Miquel (1987), "Mass transfer and pressure loss in spiral wound modules" —
  spiral-wound 마찰계수 상관식.
- Kurita America, RO Membrane System Design.
  https://www.kuritaamerica.com/the-splash/membrane-system-design-reverse-osmosis
