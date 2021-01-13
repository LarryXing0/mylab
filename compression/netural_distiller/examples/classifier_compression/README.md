# Image Classifiers Compression

This is Distiller's main example application for compressing image classification models.

- [Image Classifiers Compression](#image-classifiers-compression)
  - [Usage](#usage)
  - [Compression Methods](#compression-methods)
    - [Sparsity - Pruning and Regularization](#sparsity---pruning-and-regularization)
    - [Quantization](#quantization)
    - [Knowledge Distillation](#knowledge-distillation)
  - [Models Supported](#models-supported)
  - [Datasets Supported](#datasets-supported)
  - [Re-usable Image Classification Code](#re-usable-image-classification-code)

## Usage

Please see the [docs](https://intellabs.github.io/distiller/usage.html) for usage details. In addition, run `compress_classifier.py -h` to show the extensive list of command-line options available.

## Compression Methods

**Follow the links for more details on each method and experiment results.**

### Sparsity - Pruning and Regularization

A non-exhaustive list of the methods implemented:

- [AGP](https://github.com/IntelLabs/distiller/tree/master/examples/agp-pruning)
- [DropFilter](https://github.com/IntelLabs/distiller/tree/master/examples/drop_filter)
- [Lottery-Ticket Hypothesis](https://github.com/IntelLabs/distiller/tree/master/examples/lottery_ticket)
- [Network Surgery](https://github.com/IntelLabs/distiller/tree/master/examples/network_surgery)
- [Network Trimming](https://github.com/IntelLabs/distiller/tree/master/examples/network_trimming)
- [Hybrids](https://github.com/IntelLabs/distiller/tree/master/examples/hybrid): These are examples where multiple pruning strategies are combined.

### Quantization

- [Post-training quantization](https://github.com/IntelLabs/distiller/tree/master/examples/quantization/post_train_quant/command_line.md) based on the TensorFlow quantization scheme (originally GEMMLOWP) with additional capabilities.
  - Optimizing post-training quantization parameters with the [LAPQ](https://arxiv.org/abs/1911.07190) method - see [example YAML](https://github.com/IntelLabs/distiller/blob/master/examples/quantization/post_train_quant/resnet18_imagenet_post_train_lapq.yaml) file for details.
- [Quantization-aware training](https://github.com/IntelLabs/distiller/tree/master/examples/quantization/quant_aware_train): TensorFlow scheme, DoReFa, PACT

### Knowledge Distillation

See details in the [docs](https://intellabs.github.io/distiller/schedule.html#knowledge-distillation), and these YAML schedules training ResNet on CIFAR-10 with knowledge distillation: [FP32](https://github.com/IntelLabs/distiller/tree/master/examples/quantization/fp32_baselines/preact_resnet_cifar_base_fp32.yaml) ; [DoReFa](https://github.com/IntelLabs/distiller/tree/master/examples/quantization/quant_aware_train/preact_resnet_cifar_dorefa.yaml).

## Models Supported

The sample app integrates with [TorchVision](https://pytorch.org/docs/master/torchvision/models.html#classification) and [Cadene's pre-trained models](https://github.com/Cadene/pretrained-models.pytorch). Barring specific issues, any model from these two repositories can be specified from the command line and used.

We've implemented additional models, which can be found [here](https://github.com/IntelLabs/distiller/tree/master/distiller/models).

## Datasets Supported

The application supports ImageNet, CIFAR-10 and MNIST.

The `compress_classifier.py` application will download the CIFAR-10 and MNIST datasets automatically the first time you try to use them (thanks to TorchVision).  The example invocations used  throughout Distiller's documentation assume that you have downloaded the images to directory `distiller/../data.cifar10`, but you can place the images anywhere you want (you tell `compress_classifier.py` where the dataset is located - or where you want the application to download the dataset to - using a command-line parameter).

ImageNet needs to be [downloaded](http://image-net.org/download) manually, due to copyright issues.  Facebook has created a [set of scripts](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) to help download and extract the dataset.

Again, the Distiller documentation assumes the following directory structure for the datasets, but this is just a suggestion:
```
distiller
  examples
    classifier_compression
data.imagenet/
    train/
    val/
data.cifar10/
    cifar-10-batches-py/
        batches.meta
        data_batch_1
        data_batch_2
        data_batch_3
        data_batch_4
        data_batch_5
        readme.html
        test_batch
data.mnist/
    MNIST/
        processed/
        raw/
```

## Re-usable Image Classification Code

We borrow the main flow code from PyTorch's ImageNet classification training sample application ([see here](https://github.com/pytorch/examples/tree/master/imagenet)). Much of the flow was refactored into a class called `ClassifierCompressor`, which can be re-used to build different scripts that perform image classifiers compression. Its implementation can be found in [`distiller/apputils/image_classifier.py`](https://github.com/IntelLabs/distiller/tree/master/distiller/apputils/image_classifier.py).  

The [AMC auto-compression](https://github.com/IntelLabs/distiller/tree/master/examples/auto_compression/amc) sample is another application that uses this building block.


```
python compression_classifier.py -h
usage: compress_classifier.py [-h] [--arch ARCH] [-j N] [--epochs N] [-b N]
                              [--lr LR] [--momentum M] [--weight-decay W]
                              [--print-freq N] [--verbose]
                              [--resume-from PATH | --exp-load-weights-from PATH]
                              [--pretrained] [--reset-optimizer] [-e]
                              [--activation-stats PHASE [PHASE ...]]
                              [--activation-histograms PORTION_OF_TEST_SET]
                              [--masks-sparsity] [--param-hist]
                              [--summary {sparsity,compute,model,modules,png,png_w_params}]
                              [--export-onnx [EXPORT_ONNX]]
                              [--compress [COMPRESS]]
                              [--sense {element,filter,channel}]
                              [--sense-range SENSITIVITY_RANGE SENSITIVITY_RANGE SENSITIVITY_RANGE]
                              [--deterministic] [--seed SEED] [--gpus DEV_ID]
                              [--cpu] [--name NAME] [--out-dir OUTPUT_DIR]
                              [--validation-split VALIDATION_SPLIT]
                              [--effective-train-size EFFECTIVE_TRAIN_SIZE]
                              [--effective-valid-size EFFECTIVE_VALID_SIZE]
                              [--effective-test-size EFFECTIVE_TEST_SIZE]
                              [--confusion]
                              [--num-best-scores NUM_BEST_SCORES]
                              [--load-serialized] [--thinnify]
                              [--quantize-eval] [--qe-mode QE_MODE]
                              [--qe-mode-acts QE_MODE_ACTS]
                              [--qe-mode-wts QE_MODE_WTS]
                              [--qe-bits-acts NUM_BITS]
                              [--qe-bits-wts NUM_BITS]
                              [--qe-bits-accum NUM_BITS]
                              [--qe-clip-acts QE_CLIP_ACTS]
                              [--qe-clip-n-stds QE_CLIP_N_STDS]
                              [--qe-no-clip-layers LAYER_NAME [LAYER_NAME ...]]
                              [--qe-no-quant-layers LAYER_NAME [LAYER_NAME ...]]
                              [--qe-per-channel]
                              [--qe-scale-approx-bits NUM_BITS]
                              [--qe-save-fp-weights] [--qe-convert-pytorch]
                              [--qe-pytorch-backend {fbgemm,qnnpack}]
                              [--qe-stats-file PATH | --qe-dynamic | --qe-calibration PORTION_OF_TEST_SET | --qe-config-file PATH]
                              [--qe-lapq] [--lapq-maxiter LAPQ_MAXITER]
                              [--lapq-maxfev LAPQ_MAXFEV]
                              [--lapq-method LAPQ_METHOD]
                              [--lapq-basinhopping]
                              [--lapq-basinhopping-niter LAPQ_BASINHOPPING_NITER]
                              [--lapq-init-mode LAPQ_INIT_MODE]
                              [--lapq-init-method LAPQ_INIT_METHOD]
                              [--lapq-eval-size LAPQ_EVAL_SIZE]
                              [--lapq-eval-memoize-dataloader]
                              [--lapq-search-clipping]
                              [--save-untrained-model]
                              [--earlyexit_lossweights [EARLYEXIT_LOSSWEIGHTS [EARLYEXIT_LOSSWEIGHTS ...]]]
                              [--earlyexit_thresholds [EARLYEXIT_THRESHOLDS [EARLYEXIT_THRESHOLDS ...]]]
                              [--kd-teacher ARCH] [--kd-pretrained]
                              [--kd-resume PATH] [--kd-temperature TEMP]
                              [--kd-distill-wt WEIGHT]
                              [--kd-student-wt WEIGHT]
                              [--kd-teacher-wt WEIGHT]
                              [--kd-start-epoch EPOCH_NUM] [--greedy]
                              [--greedy-ft-epochs GREEDY_FT_EPOCHS]
                              [--greedy-target-density GREEDY_TARGET_DENSITY]
                              [--greedy-pruning-step GREEDY_PRUNING_STEP]
                              [--greedy-finetuning-policy {constant,linear-grow}]
                              DATASET_DIR

Distiller image classification model compression

positional arguments:
  DATASET_DIR           path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | alexnet_bn | bninception
                        | cafferesnet101 | densenet121 | densenet161 |
                        densenet169 | densenet201 | dpn107 | dpn131 | dpn68 |
                        dpn68b | dpn92 | dpn98 | fbresnet152 | googlenet |
                        inception_v3 | inceptionresnetv2 | inceptionv3 |
                        inceptionv4 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 |
                        mnasnet1_3 | mobilenet | mobilenet_025 | mobilenet_050
                        | mobilenet_075 | mobilenet_v1_dropout | mobilenet_v2
                        | nasnetalarge | nasnetamobile | plain20_cifar |
                        plain20_cifar_nobn | pnasnet5large | polynet |
                        preact_resnet101 | preact_resnet110_cifar |
                        preact_resnet110_cifar_conv_ds | preact_resnet152 |
                        preact_resnet18 | preact_resnet20_cifar |
                        preact_resnet20_cifar_conv_ds | preact_resnet32_cifar
                        | preact_resnet32_cifar_conv_ds | preact_resnet34 |
                        preact_resnet44_cifar | preact_resnet44_cifar_conv_ds
                        | preact_resnet50 | preact_resnet56_cifar |
                        preact_resnet56_cifar_conv_ds | resnet101 |
                        resnet110_cifar_earlyexit | resnet1202_cifar_earlyexit
                        | resnet152 | resnet18 | resnet20_cifar |
                        resnet20_cifar_earlyexit | resnet32_cifar |
                        resnet32_cifar_earlyexit | resnet34 | resnet44_cifar |
                        resnet44_cifar_earlyexit | resnet50 |
                        resnet50_earlyexit | resnet56_cifar |
                        resnet56_cifar_earlyexit | resnext101_32x4d |
                        resnext101_32x8d | resnext101_64x4d | resnext50_32x4d
                        | se_resnet101 | se_resnet152 | se_resnet50 |
                        se_resnext101_32x4d | se_resnext50_32x4d | senet154 |
                        shufflenet_v2_x0_5 | shufflenet_v2_x1_0 |
                        shufflenet_v2_x1_5 | shufflenet_v2_x2_0 |
                        simplenet2_cifar | simplenet_cifar | simplenet_mnist |
                        simplenet_v2_mnist | squeezenet1_0 | squeezenet1_1 |
                        vgg11 | vgg11_bn | vgg11_bn_cifar | vgg11_cifar |
                        vgg13 | vgg13_bn | vgg13_bn_cifar | vgg13_cifar |
                        vgg16 | vgg16_bn | vgg16_bn_cifar | vgg16_cifar |
                        vgg19 | vgg19_bn | vgg19_bn_cifar | vgg19_cifar |
                        wide_resnet101_2 | wide_resnet50_2 | xception
                        (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run (default: 90
  -b N, --batch-size N  mini-batch size (default: 256)
  --print-freq N, -p N  print frequency (default: 10)
  --verbose, -v         Emit debug log messages
  -e, --evaluate        evaluate model on test set
  --activation-stats PHASE [PHASE ...], --act-stats PHASE [PHASE ...]
                        collect activation statistics on phases: train, valid,
                        and/or test (WARNING: this slows down training)
  --activation-histograms PORTION_OF_TEST_SET, --act-hist PORTION_OF_TEST_SET
                        Run the model in evaluation mode on the specified
                        portion of the test dataset and generate activation
                        histograms. NOTE: This slows down evaluation
                        significantly
  --masks-sparsity      print masks sparsity table at end of each epoch
  --param-hist          log the parameter tensors histograms to file (WARNING:
                        this can use significant disk space)
  --summary {sparsity,compute,model,modules,png,png_w_params}
                        sparsityprint a summary of the model, and exit -
                        options: | computeprint a summary of the model, and
                        exit - options: | modelprint a summary of the model,
                        and exit - options: | modulesprint a summary of the
                        model, and exit - options: | pngprint a summary of the
                        model, and exit - options: | png_w_params
  --export-onnx [EXPORT_ONNX]
                        export model to ONNX format
  --compress [COMPRESS]
                        configuration file for pruning the model (default is
                        to use hard-coded schedule)
  --sense {element,filter,channel}
                        test the sensitivity of layers to pruning
  --sense-range SENSITIVITY_RANGE SENSITIVITY_RANGE SENSITIVITY_RANGE
                        an optional parameter for sensitivity testing
                        providing the range of sparsities to test. This is
                        equivalent to creating sensitivities =
                        np.arange(start, stop, step)
  --deterministic, --det
                        Ensure deterministic execution for re-producible
                        results.
  --seed SEED           seed the PRNG for CPU, CUDA, numpy, and Python
  --gpus DEV_ID         Comma-separated list of GPU device IDs to be used
                        (default is to use all available devices)
  --cpu                 Use CPU only. Flag not set => uses GPUs according to
                        the --gpus flag value.Flag set => overrides the --gpus
                        flag
  --name NAME, -n NAME  Experiment name
  --out-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Path to dump logs and checkpoints
  --validation-split VALIDATION_SPLIT, --valid-size VALIDATION_SPLIT, --vs VALIDATION_SPLIT
                        Portion of training dataset to set aside for
                        validation
  --effective-train-size EFFECTIVE_TRAIN_SIZE, --etrs EFFECTIVE_TRAIN_SIZE
                        Portion of training dataset to be used in each epoch.
                        NOTE: If --validation-split is set, then the value of
                        this argument is applied AFTER the train-validation
                        split according to that argument
  --effective-valid-size EFFECTIVE_VALID_SIZE, --evs EFFECTIVE_VALID_SIZE
                        Portion of validation dataset to be used in each
                        epoch. NOTE: If --validation-split is set, then the
                        value of this argument is applied AFTER the train-
                        validation split according to that argument
  --effective-test-size EFFECTIVE_TEST_SIZE, --etes EFFECTIVE_TEST_SIZE
                        Portion of test dataset to be used in each epoch
  --confusion           Display the confusion matrix
  --num-best-scores NUM_BEST_SCORES
                        number of best scores to track and report (default: 1)
  --load-serialized     Load a model without DataParallel wrapping it
  --thinnify            physically remove zero-filters and create a smaller
                        model
  --save-untrained-model
                        Save the randomly-initialized model before training
                        (useful for lottery-ticket method)
  --earlyexit_lossweights [EARLYEXIT_LOSSWEIGHTS [EARLYEXIT_LOSSWEIGHTS ...]]
                        List of loss weights for early exits (e.g.
                        --earlyexit_lossweights 0.1 0.3)
  --earlyexit_thresholds [EARLYEXIT_THRESHOLDS [EARLYEXIT_THRESHOLDS ...]]
                        List of EarlyExit thresholds (e.g.
                        --earlyexit_thresholds 1.2 0.9)
  --greedy              greedy filter pruning

Optimizer arguments:
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)

Resuming arguments:
  --resume-from PATH    path to latest checkpoint. Use to resume paused
                        training session.
  --exp-load-weights-from PATH
                        path to checkpoint to load weights from (excluding
                        other fields) (experimental)
  --pretrained          use pre-trained model
  --reset-optimizer     Flag to override optimizer if resumed from checkpoint.
                        This will reset epochs count.

Post-Training Quantization Arguments:
  --quantize-eval, --qe
                        Apply linear quantization to model before evaluation.
                        Applicable only if --evaluate is also set
  --qe-mode QE_MODE, --qem QE_MODE
                        Default linear quantization mode (for weights and
                        activations). Choices: sym | sym_restr | asym_s |
                        asym_u
  --qe-mode-acts QE_MODE_ACTS, --qema QE_MODE_ACTS
                        Linear quantization mode for activations. Overrides
                        --qe-mode`. Choices: sym | sym_restr | asym_s | asym_u
  --qe-mode-wts QE_MODE_WTS, --qemw QE_MODE_WTS
                        Linear quantization mode for Weights. Overrides --qe-
                        mode`. Choices: sym | sym_restr | asym_s | asym_u
  --qe-bits-acts NUM_BITS, --qeba NUM_BITS
                        Number of bits for quantization of activations. Use 0
                        to not quantize activations. Default value is 8
  --qe-bits-wts NUM_BITS, --qebw NUM_BITS
                        Number of bits for quantization of weights. Use 0 to
                        not quantize weights. Default value is 8
  --qe-bits-accum NUM_BITS
                        Number of bits for quantization of the accumulator
  --qe-clip-acts QE_CLIP_ACTS, --qeca QE_CLIP_ACTS
                        Activations clipping mode. Choices: none | avg | n_std
                        | gauss | laplace
  --qe-clip-n-stds QE_CLIP_N_STDS
                        When qe-clip-acts is set to 'n_std', this is the
                        number of standard deviations to use
  --qe-no-clip-layers LAYER_NAME [LAYER_NAME ...], --qencl LAYER_NAME [LAYER_NAME ...]
                        List of layer names for which not to clip activations.
                        Applicable only if --qe-clip-acts is not 'none'
  --qe-no-quant-layers LAYER_NAME [LAYER_NAME ...], --qenql LAYER_NAME [LAYER_NAME ...]
                        List of layer names for which to skip quantization.
  --qe-per-channel, --qepc
                        Enable per-channel quantization of weights (per output
                        channel)
  --qe-scale-approx-bits NUM_BITS, --qesab NUM_BITS
                        Enables scale factor approximation using integer
                        multiply + bit shift, using this number of bits the
                        integer multiplier
  --qe-save-fp-weights  Allow weights requantization.
  --qe-convert-pytorch, --qept
                        Convert the model to PyTorch native post-train
                        quantization modules
  --qe-pytorch-backend {fbgemm,qnnpack}
                        When --qe-convert-pytorch is set, specifies the
                        PyTorch quantization backend to use
  --qe-stats-file PATH  Path to YAML file with pre-made calibration stats
  --qe-dynamic          Apply dynamic quantization
  --qe-calibration PORTION_OF_TEST_SET
                        Run the model in evaluation mode on the specified
                        portion of the test dataset and collect statistics
  --qe-config-file PATH
                        Path to YAML file containing configuration for
                        PostTrainRLinearQuantizer (if present, all other --qe*
                        arguments are ignored)
  --qe-lapq, --qe-coordinate-search
                        Optimize post-training quantization parameters using
                        LAPQ method

Post-Training Quantization Auto-Optimization (LAPQ) Arguments:
  --lapq-maxiter LAPQ_MAXITER
                        Max iteration for minimization method.
  --lapq-maxfev LAPQ_MAXFEV
                        Max iteration for minimization method.
  --lapq-method LAPQ_METHOD
                        Minimization method used by scip.optimize.minimize.
  --lapq-basinhopping, --lapq-bh
                        Use scipy.optimize.basinhopping stochastic global
                        minimum search.
  --lapq-basinhopping-niter LAPQ_BASINHOPPING_NITER, --lapq-bh-niter LAPQ_BASINHOPPING_NITER
                        Number of iterations for the basinhopping algorithm.
  --lapq-init-mode LAPQ_INIT_MODE
                        The mode of quant initalization. Choices:
                        NONE|AVG|LAPLACE|GAUSS|L1|L2|L3
  --lapq-init-method LAPQ_INIT_METHOD
                        If --lapq-init-mode was specified as L1/L2/L3, this
                        specifies the method of minimization.
  --lapq-eval-size LAPQ_EVAL_SIZE
                        Portion of test dataset to use for evaluation
                        function.
  --lapq-eval-memoize-dataloader
                        Stores the input batch in memory to optimize
                        performance.
  --lapq-search-clipping
                        Search on clipping values instead of scale/zero_point.

Knowledge Distillation Training Arguments:
  --kd-teacher ARCH     Model architecture for teacher model
  --kd-pretrained       Use pre-trained model for teacher
  --kd-resume PATH      Path to checkpoint from which to load teacher weights
  --kd-temperature TEMP, --kd-temp TEMP
                        Knowledge distillation softmax temperature
  --kd-distill-wt WEIGHT, --kd-dw WEIGHT
                        Weight for distillation loss (student vs. teacher soft
                        targets)
  --kd-student-wt WEIGHT, --kd-sw WEIGHT
                        Weight for student vs. labels loss
  --kd-teacher-wt WEIGHT, --kd-tw WEIGHT
                        Weight for teacher vs. labels loss
  --kd-start-epoch EPOCH_NUM
                        Epoch from which to enable distillation

Greedy Pruning:
  --greedy-ft-epochs GREEDY_FT_EPOCHS
                        number of epochs to fine-tune each discovered network
  --greedy-target-density GREEDY_TARGET_DENSITY
                        target density of the network we are seeking
  --greedy-pruning-step GREEDY_PRUNING_STEP
                        size of each pruning step (as a fraction in [0..1])
  --greedy-finetuning-policy {constant,linear-grow}
                        policy used for determining how long to fine-tune

```
