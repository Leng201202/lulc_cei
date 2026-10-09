# OEM baseline vs CEI fine-tuning (normal vs Agriculture-aware sampling)

## OA / mIoU / mF1

| Model | OA | mIoU | mF1 | Status |
| --- | --- | --- | --- | --- |
| OEM baseline | 0.6381 | 0.4529 | 0.6055 | cached |
| CEI fine-tuned (normal sampling) | 0.7661 | 0.5201 | 0.6406 | cached |
| CEI fine-tuned (Agriculture-aware sampling) | 0.6417 | 0.4539 | 0.6039 | cached |

## Agriculture focus

| Model | Agriculture IoU | Correctly Agriculture | ->Rangeland | ->Tree |
| --- | --- | --- | --- | --- |
| OEM baseline | 0.303 | 38.6% | 44.5% | 13.4% |
| CEI fine-tuned (normal sampling) | 0.124 | 90.5% | 1.6% | 6.9% |
| CEI fine-tuned (Agriculture-aware sampling) | 0.314 | 77.9% | 9.2% | 12.6% |

## Per-class IoU

| Model | Rangeland | Agriculture | Tree | Water | Building | Road | Non-vegetated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OEM baseline | 0.300 | 0.303 | 0.608 | 0.403 | 0.679 | 0.625 | 0.252 |
| CEI fine-tuned (normal sampling) | 0.324 | 0.124 | 0.875 | 0.754 | 0.660 | 0.663 | 0.242 |
| CEI fine-tuned (Agriculture-aware sampling) | 0.330 | 0.314 | 0.609 | 0.404 | 0.656 | 0.670 | 0.195 |

## Per-class F1

| Model | Rangeland | Agriculture | Tree | Water | Building | Road | Non-vegetated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OEM baseline | 0.462 | 0.465 | 0.756 | 0.575 | 0.809 | 0.769 | 0.402 |
| CEI fine-tuned (normal sampling) | 0.489 | 0.220 | 0.933 | 0.860 | 0.795 | 0.797 | 0.389 |
| CEI fine-tuned (Agriculture-aware sampling) | 0.496 | 0.478 | 0.757 | 0.575 | 0.792 | 0.802 | 0.326 |
