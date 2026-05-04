# Nucleo-F334R8 — Conveyor + 28BYJ-48 + ULN2003 + TCRT5000

**Production code files:** use `stm32_conveyor/conveyor_control.h`, `stm32_conveyor/conveyor_control.c`, and follow `stm32_conveyor/INTEGRATION.txt` in STM32CubeIDE.

Below is an older single-file paste reference (same logic).

---

## 1. Wiring (câblage)

### Power
| Circuit | Connection |
|--------|------------|
| ULN2003 module `+` and `-` | External **5 V** supply (not Nucleo USB if motor draws high current). ≥ 1 A recommended. |
| Motor supply **GND** | **Nucleo GND** + **IR module GND** (common ground required). |

### ULN2003 ↔ Nucleo F334R8
| Driver pin | MCU pin |
|------------|---------|
| IN1 | **PA8** |
| IN2 | **PA9** |
| IN3 | **PA10** |
| IN4 | **PB10** |

### 28BYJ-48
- White connector → ULN2003 board socket.

### TCRT5000 (`G`, `V+`, `S`)
| Module | Connection |
|--------|------------|
| G | GND |
| V+ | **3.3 V** Nucleo (or **5 V** if module allows) |
| S | **PA4** |

Adjust the blue potentiometer until filter present/absent gives a clear logic change.

**Logic (modular `stm32_conveyor/conveyor_control.*`):** the belt runs when the **run latch** is set and the sensor does not see a filter; **PA4** stops the stepper and clears the latch. **PC13** (USER) can re-arm after a stop; a **UART** host (e.g. Raspberry Pi) can send `RUN` / `STOP` lines — see `stm32_conveyor/INTEGRATION.txt`.

---

## 2. STM32CubeMX configuration

1. **New Project** → Board **NUCLEO-F334R8** (or MCU **STM32F334R8Tx**).
2. **Project Manager** → Toolchain **STM32CubeIDE** → Generate.

### RCC / Clock
- Use **HSI** + PLL to target **SYSCLK = 72 MHz** if CubeMX allows for this part.
- **SYS → Debug**: **Serial Wire**.

### GPIO outputs (stepper)
| Pin | Mode | User label |
|-----|------|------------|
| PA8 | GPIO_Output | STEP_IN1 |
| PA9 | GPIO_Output | STEP_IN2 |
| PA10 | GPIO_Output | STEP_IN3 |
| PB10 | GPIO_Output | STEP_IN4 |

Initial output **Low**. Speed Medium or Low.

### Sensor + EXTI
- **PA4**: **GPIO_EXTI4**
- Trigger: Rising/Falling as needed after tests (start with **Falling** or **Rising**).
- Pull: **Pull-up** or **Pull-down** per module behavior.

**NVIC**: enable **EXTI line4 interrupt**.

### Timer (step cadence)
- **TIM6**: enable, enable global interrupt.
- Example start values if TIM6 clock ≈ 72 MHz: **PSC = 71999**, **ARR = 9** (~100 Hz tick → tune for belt speed).

### Generate code
**Project → Generate Code**, then merge the code below into `main.c`.

---

## 3. Where to paste in `main.c`

1. **`USER CODE BEGIN PV`** — variables / macros from the block.
2. **`USER CODE BEGIN 0`** — static helper functions.
3. **`USER CODE BEGIN 2`** — after `MX_TIM6_Init();`: start timer interrupt + motor init line.
4. **`USER CODE BEGIN WHILE`** — optional user-button restart loop (or replace with your logic).
5. **`USER CODE BEGIN 4`** — `HAL_TIM_PeriodElapsedCallback` and `HAL_GPIO_EXTI_Callback` (add if missing).

---

## 4. Complete firmware snippet (single block)

Legacy **all-in-`main.c`** example below (uses `g_run_motor`). Prefer **`stm32_conveyor/conveyor_control.h`** / **`.c`** for current behaviour (no switch, sensor-only stop).

```c
/* ========== USER CODE BEGIN PV — paste variables here ========== */
static volatile uint8_t g_run_motor = 1;

static const uint8_t HALF_SEQ[8][4] = {
  {1,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,1,0},
  {0,0,1,0}, {0,0,1,1}, {0,0,0,1}, {1,0,0,1}
};
static uint8_t g_step_idx = 0;

/* Set to GPIO_PIN_SET if filter detected = HIGH on PA4, else GPIO_PIN_RESET */
#define FILTER_DETECT_LEVEL   GPIO_PIN_SET
/* ========== USER CODE END PV ========== */

/* ========== USER CODE BEGIN 0 — paste helpers here ========== */
static void StepOutputs(uint8_t a, uint8_t b, uint8_t c, uint8_t d)
{
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8,  a ? GPIO_PIN_SET : GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_9,  b ? GPIO_PIN_SET : GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_10, c ? GPIO_PIN_SET : GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, d ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

static void MotorAllLow(void)
{
  StepOutputs(0, 0, 0, 0);
}

static uint8_t FilterPresent(void)
{
  GPIO_PinState s = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_4);
  return (s == FILTER_DETECT_LEVEL) ? 1u : 0u;
}

static void MotorStepForward(void)
{
  const uint8_t *w = HALF_SEQ[g_step_idx];
  StepOutputs(w[0], w[1], w[2], w[3]);
  g_step_idx = (uint8_t)((g_step_idx + 1u) % 8u);
}
/* ========== USER CODE END 0 ========== */

/* ========== USER CODE BEGIN 2 — after MX_TIM6_Init(); ========== */
MotorAllLow();
HAL_TIM_Base_Start_IT(&htim6);
/* ========== USER CODE END 2 ========== */

/* ========== USER CODE BEGIN WHILE — inside while(1) or replace loop body ========== */
while (1)
{
  /* USER BUTTON PC13 often active LOW when pressed — verify Nucleo UM */
  if (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_RESET)
  {
    HAL_Delay(50);
    while (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_RESET) {
      HAL_Delay(10);
    }
    if (!FilterPresent())
      g_run_motor = 1;
  }
  HAL_Delay(10);
}
/* ========== USER CODE END WHILE ========== */

/* ========== USER CODE BEGIN 4 — add these callbacks if not generated ========== */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  if (htim->Instance == TIM6)
  {
    if (!g_run_motor)
      return;

    if (FilterPresent())
    {
      g_run_motor = 0;
      MotorAllLow();
      return;
    }

    MotorStepForward();
  }
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
  if (GPIO_Pin == GPIO_PIN_4)
  {
    if (FilterPresent())
    {
      g_run_motor = 0;
      MotorAllLow();
    }
  }
}
/* ========== USER CODE END 4 ========== */
```

---

## 5. Notes

- Wrong belt direction: use `(g_step_idx + 7) % 8` instead of `+1` in `MotorStepForward`.
- Tune **TIM6 PSC/ARR** for smooth speed without stalling.
- If EXTI fires noisy: rely mainly on timer + `FilterPresent()` or add software debounce.
