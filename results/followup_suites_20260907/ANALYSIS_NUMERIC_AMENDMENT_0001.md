# Numerical precision correction

The independent report review found one incorrect supplementary direction label after Suite 1 completed its first analysis and reproduction.

The affected result is the exchanged judge view for added salience, model `seq_AthenB_o0.0_M_A_s0`, preference B. Exact integer arithmetic gives a 95% interval of [-0.15625, 0]. Floating-point arithmetic returned an upper endpoint of -8.67e-19. A strict comparison with zero incorrectly classified that result as negative.

The corrected analysis treats magnitudes below 1e-12 as numerical zero for interval endpoints and direction classification. This tolerance is far below the resolution of the planned response counts. It changes no hypothesis, comparison, model inclusion rule, bootstrap draw count, random seed, or evidence label. It corrects numerical classification near zero.

The original analysis, reproduction, source files, and completion receipt remain under `suite1/analysis_before_numeric_fix/`. The corrected final analysis receives a new manifest, independent reproduction, and completion receipt. The main prompt-order and retention conclusions remain unchanged.

Suite 2 had already started after the original Suite 1 completion receipt. This correction affects analysis only. The frozen training, generation, and judge code remain unchanged.
