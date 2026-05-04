# HIFI Matcher — Raspberry Pi configuration

This guide is the **step-by-step Raspberry Pi setup** for `rayen mansali bellehy emchy.py` (desktop GUI, camera, OCR) and optional **UART** control of an STM32 Nucleo conveyor (`stm32_conveyor/`).

---

## Before you start (what you need)

| Item | Notes |
|------|--------|
| **Raspberry Pi 5** (or 4) | 8 GB RAM recommended for Pi 5 |
| **microSD** | 32 GB or larger |
| **Monitor + keyboard/mouse**, or **VNC** with a full desktop | The app uses **Tkinter**; a desktop session is required |
| **Camera** | Pi Camera (CSI) and/or **USB webcam** |
| **Good power supply** | Official PSU recommended; **fan** helps under OCR load |
| **Optional:** Nucleo F334 + 3 wires | UART to Pi for belt **RUN** / **STOP** (see Step 9) |

---

## Raspberry Pi — step by step

Follow these steps **in order**. After each step that says **reboot**, log back in and continue.

### Step 1 — Flash the operating system

1. On a PC, install **Raspberry Pi Imager**.
2. Choose **Raspberry Pi OS (64-bit)** — use the image that includes **Desktop** (not Lite only).
3. Write the image to the microSD card, eject it, put it in the Pi, power on.

---

### Step 2 — First boot

1. Complete the first-boot wizard: country, user, password, Wi‑Fi (if needed).
2. Wait until you reach the **desktop**.

---

### Step 3 — Update the system

Open **Terminal** on the Pi and run:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

After reboot, open Terminal again.

---

### Step 4 — Enable the camera

1. Run:

   ```bash
   sudo raspi-config
   ```

2. Go to **Interface Options** → **Camera** (wording may vary slightly by image) → **Enable**.
3. Go back to the main menu. **Do not reboot yet** if you still need Step 5 on the same run (you can do both then reboot once).

---

### Step 5 — Enable serial UART (for wired Nucleo on GPIO pins)

Still in **`sudo raspi-config`**:

1. Go to **Interface Options** → **Serial Port**.
2. Set **Login shell over serial** → **No** (the matcher must own the port, not a login shell).
3. Set **Serial port hardware** → **Yes**.
4. Choose **Finish**. When asked, **Yes** to **reboot**.

If you skipped Step 4, enable the camera in the same `raspi-config` session before finishing, then reboot once.

---

### Step 6 — (Optional) Enable SSH

Only if you want remote shell access (the GUI still needs a desktop or VNC):

```bash
sudo raspi-config
```

**Interface Options** → **SSH** → **Enable** → **Finish**.

---

### Step 7 — Check that the camera works

In Terminal:

```bash
rpicam-hello -t 3000
```

If that command is missing, try:

```bash
libcamera-hello -t 3000
```

You should see a **preview window** for a few seconds. If yes, the camera stack is OK.

---

### Step 8 — Check serial and permissions (for Nucleo over GPIO UART)

1. List serial devices:

   ```bash
   ls -l /dev/serial0 /dev/ttyAMA0 2>/dev/null || true
   ```

   Often **`/dev/serial0`** exists after Step 5.

2. Allow your user to open the port without `sudo`:

   ```bash
   sudo usermod -aG dialout "$USER"
   ```

3. **Log out and log in again** (or reboot) so the `dialout` group applies.

---

### Step 9 — (Optional) Wire the Raspberry Pi to the STM32 Nucleo

Only if you control the belt over **GPIO UART** (not required if you use USB to the Nucleo only).

**Use 3.3 V logic only.** Do not connect 5 V TTL USB–UART adapters to the Pi’s GPIO UART.

| Pi 40-pin | Connect to Nucleo (USART2 example) |
|-----------|--------------------------------------|
| **Pin 6** (GND) | **GND** |
| **Pin 8** (GPIO14 = **UART TX**) | **PA3** (MCU **RX**) |
| **Pin 10** (GPIO15 = **UART RX**) | **PA2** (MCU **TX**) |

Rule: Pi **TX** → MCU **RX**; Pi **RX** ← MCU **TX**. Always connect **GND** together.

Flash the STM32 with firmware that decodes **`RUN`** / **`STOP`** lines (see **`stm32_conveyor/INTEGRATION.txt`**). USART baud in CubeMX should be **115200**.

**Note:** Nucleo **PA2/PA3** often share the ST-Link USB serial. Do not have two devices **transmitting** on the same MCU RX at the same time.

If you **do not** use GPIO wires and only use **USB** from the Pi to the Nucleo, skip this step and in Step 12 use:

`HIFI_CONVEYOR_SERIAL=/dev/ttyACM0 bash run_matcher.sh`  
(adjust device name if needed: `ls /dev/serial/by-id/`)

---

### Step 10 — Put the project folder on the Pi

Pick **one** method:

**A) Git clone (only if the project is on GitHub / GitLab / etc.)**

1. Open your project page in the browser (example: `https://github.com/yourname/hifi_matcher`).
2. Click the green **Code** button.
3. Copy the URL shown — either:
   - **HTTPS:** `https://github.com/yourname/hifi_matcher.git`
   - **SSH:** `git@github.com:yourname/hifi_matcher.git` (needs SSH keys set up on the Pi)

4. On the Pi:

   ```bash
   cd ~
   git clone https://github.com/yourname/hifi_matcher.git
   cd hifi_matcher
   ```

Replace the URL with **your** real link. If you never uploaded the project to the internet, use method **B** instead.

**B) Copy the folder (no Git URL needed)**

Copy the whole **`hifi_matcher`** folder to the Pi with a USB stick, **WinSCP** / **FileZilla** (SFTP), **Samba** share, or `scp` from your PC, then:

```bash
cd ~/hifi_matcher
```

(Use your real folder path if it is not `~/hifi_matcher`.)

---

### Step 11 — Install all software (one command)

From **inside** the project folder:

```bash
cd ~/hifi_matcher
bash setup_pi.sh
```

Wait until it finishes without errors. It installs system packages (Python, Tk, Tesseract, OpenCV, …), creates **`.venv`**, and runs **`pip install -r requirements.txt`**.

---

### Step 12 — Run the HIFI Matcher

From the project folder:

```bash
cd ~/hifi_matcher
bash run_matcher.sh
```

**`run_matcher.sh`** activates the virtual environment and, by default, sets:

- `HIFI_CONVEYOR_SERIAL=/dev/serial0`
- `HIFI_CONVEYOR_BAUD=115200`

so a **wired** Nucleo on GPIO UART works without extra exports.

**Overrides:**

```bash
HIFI_CONVEYOR_SERIAL=/dev/ttyACM0 bash run_matcher.sh   # Nucleo on USB only
HIFI_CONVEYOR_SERIAL= bash run_matcher.sh               # do not send RUN/STOP
```

**Manual run** (without the script):

```bash
cd ~/hifi_matcher
source .venv/bin/activate
export HIFI_CONVEYOR_SERIAL=/dev/serial0
export HIFI_CONVEYOR_BAUD=115200
python "rayen mansali bellehy emchy.py"
```

On Linux, **Tesseract** is taken from **`PATH`** (no extra path configuration).

---

### Step 13 — If the app does not see the camera

```bash
cd ~/hifi_matcher
source .venv/bin/activate
```

Then:

```bash
python3 - <<'PY'
import cv2
for i in (0, 1, 2):
    cap = cv2.VideoCapture(i)
    ok, fr = cap.read()
    print(i, "opened:", cap.isOpened(), "read:", ok, "shape:", None if not ok else fr.shape)
    cap.release()
PY
ls -l /dev/video* 2>/dev/null || true
```

Try the index that prints **`read: True`**. USB webcams are often index **0**.

---

## Manual install (only if you cannot use `setup_pi.sh`)

```bash
sudo apt update
sudo apt install -y \
  python3 python3-pip python3-venv python3-tk \
  tesseract-ocr libtesseract-dev \
  libzbar0 \
  python3-opencv \
  libatlas-base-dev libjpeg-dev libpng-dev libopenjp2-7 \
  libglib2.0-0

cd ~/hifi_matcher
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then continue from **Step 12**.

---

## STM32 conveyor — what the Pi sends (reference)

| Matcher result | Serial line |
|----------------|-------------|
| **Match** | `RUN` + newline |
| **Mismatch** | `STOP` + newline |

| Environment variable | Meaning |
|------------------------|---------|
| **`HIFI_CONVEYOR_SERIAL`** | Device path (`/dev/serial0`, `/dev/ttyACM0`, …). Empty = disabled. |
| **`HIFI_CONVEYOR_BAUD`** | Default **115200** (must match CubeMX). |

Firmware details: **`stm32_conveyor/INTEGRATION.txt`**.

---

## Performance (after it runs)

- Use good **light** and **focus** on the label.
- Prefer moderate resolution if you tune the app; close unused programs.
- Use a **fan** on the Pi if the CPU throttles during live OCR.

---

## Troubleshooting

| Problem | What to do |
|---------|------------|
| **`No module named 'tkinter'`** | `sudo apt install -y python3-tk` then re-run **`setup_pi.sh`** or install manually |
| **`TesseractNotFoundError`** | `sudo apt install -y tesseract-ocr` — run `which tesseract` |
| **`No module named 'serial'`** | `source .venv/bin/activate` then `pip install -r requirements.txt` |
| **Camera preview fails** | Repeat Steps **4** and **7**; check CSI ribbon; try USB camera |
| **App opens but no camera** | **Step 13** |
| **`import cv2` fails** | `sudo apt install -y python3-opencv` |
| **No `/dev/serial0` or permission denied** | Repeat Steps **5** and **8** (reboot after `raspi-config`; re-login after `dialout`) |
| **Nucleo ignores RUN/STOP** | Same baud (**115200**); correct wiring; STM32 **`HAL_UART_Receive_IT`** loop — **`INTEGRATION.txt`** |

---

## Optional: start the app automatically on boot

Do this **only** after **Step 12** works reliably every time. Use a **systemd** service or a **Desktop Autostart** `.desktop` file.

---

## Optional: extra Python packages

Not installed by default `requirements.txt`:

```bash
source .venv/bin/activate
pip install ultralytics
```

---

## Project files (reference)

| File | Purpose |
|------|---------|
| **`setup_pi.sh`** | Installs apt packages, creates `.venv`, `pip install -r requirements.txt` |
| **`run_matcher.sh`** | Activates `.venv`, sets default conveyor serial env, starts the app |
| **`requirements.txt`** | Python dependencies |
| **`stm32_conveyor/INTEGRATION.txt`** | CubeMX pins, UART callback pattern, wiring notes |
