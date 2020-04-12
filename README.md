# MNIST Digit Recognition

Training a custom CNN with less than 6.5k parameters to achieve over 99.4% validation accuracy within 10 epochs.

### Model Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5            [-1, 8, 24, 24]             576
              ReLU-6            [-1, 8, 24, 24]               0
       BatchNorm2d-7            [-1, 8, 24, 24]              16
           Dropout-8            [-1, 8, 24, 24]               0
            Conv2d-9            [-1, 8, 22, 22]              72
           Conv2d-10           [-1, 16, 22, 22]             128
             ReLU-11           [-1, 16, 22, 22]               0
      BatchNorm2d-12           [-1, 16, 22, 22]              32
          Dropout-13           [-1, 16, 22, 22]               0
           Conv2d-14           [-1, 16, 20, 20]             144
           Conv2d-15           [-1, 16, 20, 20]             256
             ReLU-16           [-1, 16, 20, 20]               0
      BatchNorm2d-17           [-1, 16, 20, 20]              32
          Dropout-18           [-1, 16, 20, 20]               0
        MaxPool2d-19           [-1, 16, 10, 10]               0
           Conv2d-20            [-1, 8, 10, 10]             128
             ReLU-21            [-1, 8, 10, 10]               0
      BatchNorm2d-22            [-1, 8, 10, 10]              16
          Dropout-23            [-1, 8, 10, 10]               0
           Conv2d-24             [-1, 16, 8, 8]           1,152
             ReLU-25             [-1, 16, 8, 8]               0
      BatchNorm2d-26             [-1, 16, 8, 8]              32
          Dropout-27             [-1, 16, 8, 8]               0
           Conv2d-28             [-1, 16, 8, 8]           2,304
             ReLU-29             [-1, 16, 8, 8]               0
      BatchNorm2d-30             [-1, 16, 8, 8]              32
          Dropout-31             [-1, 16, 8, 8]               0
           Conv2d-32             [-1, 16, 6, 6]             144
           Conv2d-33             [-1, 32, 6, 6]             512
             ReLU-34             [-1, 32, 6, 6]               0
      BatchNorm2d-35             [-1, 32, 6, 6]              64
          Dropout-36             [-1, 32, 6, 6]               0
           Conv2d-37             [-1, 32, 4, 4]             288
           Conv2d-38             [-1, 10, 4, 4]             320
             ReLU-39             [-1, 10, 4, 4]               0
      BatchNorm2d-40             [-1, 10, 4, 4]              20
          Dropout-41             [-1, 10, 4, 4]               0
        AvgPool2d-42             [-1, 10, 1, 1]               0
           Conv2d-43             [-1, 10, 1, 1]             100
================================================================
Total params: 6,456
Trainable params: 6,456
Non-trainable params: 0
----------------------------------------------------------------
```
### Training Logs
```
EPOCH: 1
Loss=0.16230268776416779 Batch_id=468 Accuracy=92.45: 100%|██████████| 469/469 [00:14<00:00, 31.39it/s]
Test set: Average loss: 0.0553, Accuracy: 9810/10000 (98.10%)

EPOCH: 2
Loss=0.13705630600452423 Batch_id=468 Accuracy=97.74: 100%|██████████| 469/469 [00:14<00:00, 31.35it/s]
Test set: Average loss: 0.0485, Accuracy: 9835/10000 (98.35%)

EPOCH: 3
Loss=0.04203194007277489 Batch_id=468 Accuracy=98.12: 100%|██████████| 469/469 [00:14<00:00, 31.72it/s]
Test set: Average loss: 0.0348, Accuracy: 9897/10000 (98.97%)

EPOCH: 4
Loss=0.059701353311538696 Batch_id=468 Accuracy=98.78: 100%|██████████| 469/469 [00:14<00:00, 31.97it/s]
Test set: Average loss: 0.0212, Accuracy: 9933/10000 (99.33%)

EPOCH: 5
Loss=0.004027942661195993 Batch_id=468 Accuracy=98.85: 100%|██████████| 469/469 [00:14<00:00, 31.29it/s]
Test set: Average loss: 0.0218, Accuracy: 9930/10000 (99.30%)

EPOCH: 6
Loss=0.05641213431954384 Batch_id=468 Accuracy=98.96: 100%|██████████| 469/469 [00:15<00:00, 30.45it/s]
Test set: Average loss: 0.0220, Accuracy: 9930/10000 (99.30%)

EPOCH: 7
Loss=0.0268141720443964 Batch_id=468 Accuracy=98.98: 100%|██████████| 469/469 [00:15<00:00, 29.54it/s]
Test set: Average loss: 0.0200, Accuracy: 9945/10000 (99.45%)

EPOCH: 8
Loss=0.040901269763708115 Batch_id=468 Accuracy=99.04: 100%|██████████| 469/469 [00:15<00:00, 29.90it/s]
Test set: Average loss: 0.0200, Accuracy: 9937/10000 (99.37%)

EPOCH: 9
Loss=0.018276089802384377 Batch_id=468 Accuracy=99.04: 100%|██████████| 469/469 [00:16<00:00, 27.99it/s]
Test set: Average loss: 0.0208, Accuracy: 9943/10000 (99.43%)

EPOCH: 10
Loss=0.02205638773739338 Batch_id=468 Accuracy=99.03: 100%|██████████| 469/469 [00:16<00:00, 28.50it/s]
Test set: Average loss: 0.0204, Accuracy: 9941/10000 (99.41%)

EPOCH: 11
Loss=0.016721585765480995 Batch_id=468 Accuracy=99.03: 100%|██████████| 469/469 [00:16<00:00, 28.96it/s]
Test set: Average loss: 0.0203, Accuracy: 9940/10000 (99.40%)

EPOCH: 12
Loss=0.053530577570199966 Batch_id=468 Accuracy=99.05: 100%|██████████| 469/469 [00:15<00:00, 29.32it/s]
Test set: Average loss: 0.0202, Accuracy: 9941/10000 (99.41%)

EPOCH: 13
Loss=0.016529506072402 Batch_id=468 Accuracy=99.06: 100%|██████████| 469/469 [00:15<00:00, 35.65it/s]
Test set: Average loss: 0.0204, Accuracy: 9944/10000 (99.44%)

EPOCH: 14
Loss=0.020957177504897118 Batch_id=468 Accuracy=99.04: 100%|██████████| 469/469 [00:15<00:00, 34.07it/s]
Test set: Average loss: 0.0204, Accuracy: 9938/10000 (99.38%)

EPOCH: 15
Loss=0.019895777106285095 Batch_id=468 Accuracy=98.99: 100%|██████████| 469/469 [00:15<00:00, 29.87it/s]
Test set: Average loss: 0.0200, Accuracy: 9940/10000 (99.40%)

```
