# RunPod Serverless worker — LTX-Video hero clips

Генерирует короткие 9:16 AI-клипы (gold coins / jackpot / money rain) для `igaming-video-pro`.

## Деплой на RunPod Serverless
1. Залить этот репозиторий на GitHub (нужен repo-scoped токен/доступ).
2. RunPod → **Serverless → New Endpoint → Import Git Repository** → выбрать репо.
3. GPU: **48 GB** (L40S/A6000) рекоменд., или 24 GB (4090) — медленнее из-за offload.
4. **Network Volume** 50 GB смонтировать (модель кэшируется в `/runpod-volume/hf`, чтобы не качать на каждом cold-start).
5. Active workers `0`, Max `1`, FlashBoot on, Container disk ~20 GB.

## API
`POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync`
Header: `Authorization: Bearer <RUNPOD_API_KEY>`
Body: см. `test_input.json` → ответ `{ "output": { "mp4_base64": "...", "fps": 24 } }`.

## Параметры input
- `prompt` (обяз.), `negative_prompt`
- `width`/`height` кратны 32 (default 704×1216, в пайплайне кропится в 1080×1920)
- `frames` — (n−1)%8==0 (default 97 ≈ 4 c)
- `steps` — distilled: 6–10 (default 8); `fps` (default 24)

Первый вызов на новом воркере качает модель в Network Volume (~несколько минут), дальше быстро.
