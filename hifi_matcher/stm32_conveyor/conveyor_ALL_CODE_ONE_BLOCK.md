# STM32 conveyor — one copy block + separate `.h` / `.c`

You already have **full individual files** in this folder:

- `conveyor_control.h`
- `conveyor_control.c`

The fenced block below is **everything at once** (same content): split into `.h` / `.c` using the `SAVE AS` comments at the top of each section. The **`main.c` glue** at the bottom — paste into CubeMX **USER CODE** regions. Enable **USART2** (or your VCP UART) in CubeMX for Pi `RUN` / `STOP` lines; see `INTEGRATION.txt`.

```c
/* ======================================================================
 * SAVE AS: conveyor_control.h
 * ======================================================================
 */

/**

 * Conveyor + 28BYJ-48 (ULN2003) + TCRT5000 — Nucleo-F334R8

 *

 * Behaviour:

 *   - Stepper advances on each TIM tick while the sensor does not report a filter.

 *   - Sensor sees filter → immediate stop (coils released); clear belt / adjust sensor before motion again.

 *

 * Pins (CubeMX):

 *   STEP_IN1..3 -> PA8, PA9, PA10

 *   STEP_IN4    -> PB10

 *   SENSOR (TCRT5000 S) -> PA4 (+ EXTI line 4 recommended)

 *

 *   RUN / re-arm switch (optional): default Nucleo USER button PC13 (momentary,

 *   active low). After a filter stop, press and release to allow motion again

 *   if PA4 is clear. Override with CONVEYOR_RUN_SWITCH_* macros; set

 *   CONVEYOR_USE_RUN_SWITCH to 0 to disable polling (call Conveyor_RequestRun()).

 *

 * Override before #include "conveyor_control.h" if needed:

 *   CONVEYOR_FILTER_DETECT_LEVEL

 *   CONVEYOR_USE_RUN_SWITCH, CONVEYOR_RUN_SWITCH_GPIO_PORT / _GPIO_PIN

 *   CONVEYOR_RUN_SWITCH_PRESSED_LEVEL

 *

 *   Pi / PC UART commands (optional): connect a USART to the host (e.g. USB VCP).

 *   Text lines ending in CR or LF (case-insensitive):

 *     RUN  -> Conveyor_RequestRun() (belt may move again if PA4 clear)

 *     STOP -> Conveyor_Stop() (coils off, latch cleared)

 *   From HAL_UART_RxCpltCallback (or similar), call Conveyor_OnUartRxByte() or

 *   Conveyor_OnUartRxBytes() for each received byte/block.

 */



#ifndef CONVEYOR_CONTROL_H

#define CONVEYOR_CONTROL_H



#include "stm32f3xx_hal.h"



#ifndef CONVEYOR_FILTER_DETECT_LEVEL

#define CONVEYOR_FILTER_DETECT_LEVEL GPIO_PIN_SET

#endif



#ifndef CONVEYOR_USE_RUN_SWITCH

#define CONVEYOR_USE_RUN_SWITCH 1

#endif



#ifndef CONVEYOR_RUN_SWITCH_GPIO_PORT

#define CONVEYOR_RUN_SWITCH_GPIO_PORT GPIOC

#endif



#ifndef CONVEYOR_RUN_SWITCH_GPIO_PIN

#define CONVEYOR_RUN_SWITCH_GPIO_PIN GPIO_PIN_13

#endif



#ifndef CONVEYOR_RUN_SWITCH_PRESSED_LEVEL

#define CONVEYOR_RUN_SWITCH_PRESSED_LEVEL GPIO_PIN_RESET

#endif



void Conveyor_Init(TIM_HandleTypeDef *htim_step);

void Conveyor_Start(void);



/** Optional idle loop — debounced RUN switch poll + delay (sensor in timer ISR). */

void Conveyor_MainLoopPoll(void);



/** Re-arm belt motion after a stop (ignored while sensor still sees a filter). */

void Conveyor_RequestRun(void);



void Conveyor_OnTimerPeriod(TIM_HandleTypeDef *htim);

void Conveyor_OnGpioExti(uint16_t GPIO_Pin);



/** Forces coils off immediately (same effect as sensor trip while running). */

void Conveyor_Stop(void);



/** Feed one UART byte from host (Pi); lines: RUN, STOP (see file header). */

void Conveyor_OnUartRxByte(uint8_t b);



void Conveyor_OnUartRxBytes(const uint8_t *data, uint16_t len);



#endif /* CONVEYOR_CONTROL_H */

/* ======================================================================
 * SAVE AS: conveyor_control.c
 * ======================================================================
 */

/**

 * Conveyor control — run when latched + no filter; PA4 filter clears latch.

 * RUN switch (default PC13) re-arms in MainLoopPoll, or use Conveyor_RequestRun().

 * Host UART lines RUN/STOP (see Conveyor_OnUartRxByte) map to RequestRun / Stop.

 */



#include "conveyor_control.h"

#include <string.h>



static TIM_HandleTypeDef *s_htim = NULL;



static const uint8_t k_half_seq[8][4] = {

    {1U, 0U, 0U, 0U}, {1U, 1U, 0U, 0U}, {0U, 1U, 0U, 0U}, {0U, 1U, 1U, 0U},

    {0U, 0U, 1U, 0U}, {0U, 0U, 1U, 1U}, {0U, 0U, 0U, 1U}, {1U, 0U, 0U, 1U},

};

static uint8_t s_step_idx = 0U;



static volatile uint8_t s_run_latched = 1U;



#define UART_CMD_LINE_MAX 24U



static char s_uart_line[UART_CMD_LINE_MAX];

static uint8_t s_uart_line_len = 0U;



static void uart_line_to_lower(char *dst, const char *src, uint16_t max_out)

{

    uint16_t i = 0U;

    while (i + 1U < max_out && src[i] != '\0') {

        char c = src[i];

        if (c >= 'A' && c <= 'Z') {

            c = (char)(c + 32);

        }

        dst[i] = c;

        i++;

    }

    dst[i] = '\0';

}



static void uart_process_complete_line(void)

{

    char low[UART_CMD_LINE_MAX];

    if (s_uart_line_len == 0U) {

        return;

    }

    s_uart_line[s_uart_line_len] = '\0';

    uart_line_to_lower(low, s_uart_line, UART_CMD_LINE_MAX);

    {

        const char *p = low;

        while (*p == ' ' || *p == '\t') {

            p++;

        }

        if (*p == '\0') {

            return;

        }

        if (!strcmp(p, "run")) {

            Conveyor_RequestRun();

            return;

        }

        if (!strcmp(p, "stop") || !strcmp(p, "halt")) {

            Conveyor_Stop();

            return;

        }

        if (p[0] == 'r' && p[1] == '\0') {

            Conveyor_RequestRun();

            return;

        }

        if (p[0] == 's' && p[1] == '\0') {

            Conveyor_Stop();

            return;

        }

    }

}



static void step_outputs(uint8_t a, uint8_t b, uint8_t c, uint8_t d)

{

    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, a ? GPIO_PIN_SET : GPIO_PIN_RESET);

    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_9, b ? GPIO_PIN_SET : GPIO_PIN_RESET);

    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_10, c ? GPIO_PIN_SET : GPIO_PIN_RESET);

    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, d ? GPIO_PIN_SET : GPIO_PIN_RESET);

}



static void motor_all_low(void)

{

    step_outputs(0U, 0U, 0U, 0U);

}



static uint8_t filter_present(void)

{

    GPIO_PinState s = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_4);

    return (s == CONVEYOR_FILTER_DETECT_LEVEL) ? 1U : 0U;

}



#if CONVEYOR_USE_RUN_SWITCH

static uint8_t run_switch_pressed(void)

{

    return (HAL_GPIO_ReadPin(CONVEYOR_RUN_SWITCH_GPIO_PORT, CONVEYOR_RUN_SWITCH_GPIO_PIN)

            == CONVEYOR_RUN_SWITCH_PRESSED_LEVEL)

               ? 1U

               : 0U;

}

#endif



static void motor_step_forward(void)

{

    const uint8_t *w = k_half_seq[s_step_idx];

    step_outputs(w[0], w[1], w[2], w[3]);

    s_step_idx = (uint8_t)((s_step_idx + 1U) % 8U);

}



void Conveyor_Init(TIM_HandleTypeDef *htim_step)

{

    s_htim = htim_step;

    s_step_idx = 0U;

    motor_all_low();

}



void Conveyor_Start(void)

{

    motor_all_low();

    if (s_htim != NULL) {

        (void)HAL_TIM_Base_Start_IT(s_htim);

    }

}



void Conveyor_Stop(void)

{

    s_run_latched = 0U;

    motor_all_low();

}



void Conveyor_RequestRun(void)

{

    if (!filter_present()) {

        s_run_latched = 1U;

    }

}



void Conveyor_MainLoopPoll(void)

{

#if CONVEYOR_USE_RUN_SWITCH

    if (run_switch_pressed()) {

        HAL_Delay(50);

        while (run_switch_pressed()) {

            HAL_Delay(10);

        }

        if (!filter_present()) {

            s_run_latched = 1U;

        }

    }

#endif

    HAL_Delay(10);

}



void Conveyor_OnTimerPeriod(TIM_HandleTypeDef *htim)

{

    if (s_htim == NULL || htim != s_htim) {

        return;

    }



    if (!s_run_latched) {

        motor_all_low();

        return;

    }



    if (filter_present()) {

        s_run_latched = 0U;

        motor_all_low();

        return;

    }



    motor_step_forward();

}



void Conveyor_OnGpioExti(uint16_t GPIO_Pin)

{

    if (GPIO_Pin != GPIO_PIN_4) {

        return;

    }

    if (filter_present()) {

        s_run_latched = 0U;

        motor_all_low();

    }

}



void Conveyor_OnUartRxByte(uint8_t b)

{

    if (b == (uint8_t)'\r' || b == (uint8_t)'\n') {

        uart_process_complete_line();

        s_uart_line_len = 0U;

        return;

    }

    if (s_uart_line_len < (UART_CMD_LINE_MAX - 1U)) {

        s_uart_line[s_uart_line_len] = (char)b;

        s_uart_line_len++;

    }

}



void Conveyor_OnUartRxBytes(const uint8_t *data, uint16_t len)

{

    uint16_t i;

    if (data == NULL) {

        return;

    }

    for (i = 0U; i < len; i++) {

        Conveyor_OnUartRxByte(data[i]);

    }

}
/* ======================================================================
 * PASTE INTO main.c — split where CubeMX USER CODE tags go.
 * Replace htim6 if your step timer handle name differs.
 * For Pi UART: enable USART in CubeMX; start first IT receive after UART Init.
 * ======================================================================
 */

/* USER CODE BEGIN Includes */
#include "conveyor_control.h"
/* USER CODE END Includes */

/* USER CODE BEGIN PV */
static uint8_t uart_rx_byte = 0;
/* USER CODE END PV */

/* USER CODE BEGIN 2 */
Conveyor_Init(&htim6);
Conveyor_Start();
/* After MX_USART2_UART_Init(): HAL_UART_Receive_IT(&huart2, &uart_rx_byte, 1); */
/* USER CODE END 2 */

/* Inside main(): USER CODE BEGIN 3 */
Conveyor_MainLoopPoll();
/* USER CODE END 3 */

/* USER CODE BEGIN 4 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    Conveyor_OnTimerPeriod(htim);
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    Conveyor_OnGpioExti(GPIO_Pin);
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2) {
        Conveyor_OnUartRxByte(uart_rx_byte);
        (void)HAL_UART_Receive_IT(&huart2, &uart_rx_byte, 1);
    }
}
/* USER CODE END 4 */

```
