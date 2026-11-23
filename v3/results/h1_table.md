Seeds: 56; mean yield 0.499

| Scorer | Spearman rho with yield | 95% bootstrap CI |
|---|---|---|
| TeleopQ q = z(V)-z(L)+z(S) | 0.250 | [0.004, 0.476] |
| TeleopQ without V (z(S)-z(L)) | 0.160 | [-0.066, 0.385] |
| Smoothness S (SPARC) | -0.293 | [-0.528, -0.044] |
| Visibility V alone | 0.219 | [-0.053, 0.458] |
| Latency (-L) alone | 0.447 | [0.204, 0.646] |
| Action high-freq power (PSD, neg) | -0.129 | [-0.389, 0.128] |
| kNN trajectory typicality | 0.354 | [0.100, 0.560] |
| Episode length (shorter=better) | 0.255 | [-0.023, 0.488] |
| Oracle (injected severity) | 0.440 | [0.204, 0.618] |

**Key test (H1):** rho(q) - rho(S) = 0.543, 95% CI [0.240, 0.833] -> H1 supported

Partial rho(V, yield | S, length) = 0.149; rho(length, yield) = 0.255
