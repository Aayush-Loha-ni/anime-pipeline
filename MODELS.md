# Models you must supply

The pipeline is code-only. It cannot render until you place these files in your own local installs. Names below are the exact filenames the config and workflow look for. Check each model's license before commercial use.

## ComfyUI — still images

Put in `ComfyUI-Shared/models/`:

| Folder | Filename |
|---|---|
| `diffusion_models/` | `qwen_image_2.1_Q6_K.gguf` |
| `text_encoders/` | `qwen3vl_8b_int8_convrot.safetensors` |
| `vae/` | `qwen_image_2.1_vae_bf16.safetensors` |

Install the Qwen-Image 2.1 GGUF loader and matching CLIP/VAE custom nodes in ComfyUI.

## ComfyUI — motion (MiniMax H3 Ref2VA int8)

Put in `ComfyUI-Shared/models/`:

| Folder | Filename |
|---|---|
| `diffusion_models/` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| `text_encoders/` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| `vae/` | `minimax_h3_video_vae_int8_convrot.safetensors` |
| `vae/` | `minimax_h3_audio_vae_fp32.safetensors` |
| `loras/` | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` |
| `loras/` | `MiniMax-H3-Ref2VA-Acc-8Step.safetensors` |
| `loras/` | `h3-realism-people-t2v-i2v-r2v.safetensors` |
| `loras/` | `camera_motion_h3_lora_v1_3000_pruned.safetensors` |

Also install the EchoMimic lip-sync custom nodes. `hermes-skill/workflows/prompt2video.json` is a reference workflow showing which node loads which file.

## llama.cpp — script writing and visual QA

Put in your llama.cpp `models/` folder and point `setup.py` at them:

- A script/instruction GGUF (the original used a Gemma-4-26B-A4B IQ4_XS quant).
- A vision GGUF plus its `mmproj` file (the original used Qwen3VL-8B-Instruct Q4_K_M).

The script GGUF and vision model share port 8080 and are swapped per phase by the pipeline.

## Voice

`qwen-tts` downloads `Qwen/Qwen3-TTS-12Hz-1.7B-Base` from Hugging Face on first use. You must also record your own short clean sample and give its exact transcript in `cosy-server/ref.json` — an empty transcript stops the voice server on purpose.

## Optional: LoRA training

Only if you set `lora_training.enabled` to true: Qwen-Image 2.1 transformer weights, text encoder, VAE and ai-toolkit. The default `false` needs none of these.

## Verify

After placing files, run:

```
python verify_setup.py
```

It reports exactly what is still missing before you start the dashboard.
