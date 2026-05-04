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

