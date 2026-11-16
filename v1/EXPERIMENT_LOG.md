# TeleopQ v1 — experiment log (Nov 14, 2026, Pacific time)

| Time | Event |
|---|---|
| ~1:20 PM | First GPU attempt: Brev L4 24 GB with 100 GB disk. Launchable setup script failed (likely out of disk). Instance deleted. |
| 1:42 PM | Second attempt: Brev L40S 48 GB with 256 GB disk. Setup completed (Isaac Sim 6.0.1, Isaac Lab 3.0.0-beta2). |
| ~2:00 PM | Downloaded NVIDIA's annotated GR-1 seeds (`00_download_seed.sh`). |
| ~2:05 PM | Smoke test: 10/12 demos generated (83.3%), 4.8 MB. |
| ~2:10 PM | Started 300-demo generation with 20 parallel envs. |
| ~2:45 PM | Generation finished: 300/388 = 77.3% yield, 145 MB. Started BC training. |
| 2:56 PM | Training epoch 105/2000, L1 loss 7.2e-5, ~1.1 s/epoch, 3.2 GB GPU memory. |
| 3:07 PM | Epoch 676, L1 loss 1.8e-5. |
| 3:18 PM | Epoch 1272, L1 loss 6.6e-6. |
| 3:30 PM | Epoch 1857, L1 loss 4.8e-6. |
| 3:38 PM | Training finished (2000 epochs, no errors). Started 50-rollout evaluation of `model_epoch_2000.pth`. |
| 3:48 PM | **Evaluation finished: 48/50 = 96% success.** v1 complete. |

## Problems and fixes

- L4 + 100 GB disk: launchable setup failed; the template needs ~100 GB, so the default 256 GB disk is required.
- `generate_dataset.py --help` printed nothing under the launcher; running the smoke test directly worked.
- The VS Code web terminal did not open with Ctrl+`; use the Command Palette ("Terminal: Create New Terminal").
