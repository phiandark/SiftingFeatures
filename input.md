#List of input file keys and brief explanation

The input file is a simple `.ini` file, with sections marked as `[Section]` and one field per line. Lines can be commented with `#`.
Here we will list all sections and possible keywords (many of which are optional, with deafult assumed if missing).
Some basic checks are performed, but be careful about parameters consistency, especially with optional experiments, as not all possible combinations have been tested.
Please see folder `inputs` for sample common inputs.

##[IO]
Information about where to save output files.
The code produces 3 output files: `_out.dat`, containing the real time training progress, `_itdata_`*x*`.pkl`, containing summary data about iteration *x*, and `_findata.pkl`, containing a final summary of the whole IMP.

- `save_dir` -- The directory where to save all outputs.
- `prefix` -- Prefix for the output, the default "IMP" produces `IMP_out.dat` etc.

##[Data]
Information about the training dataset. For now only supports downsampled ImageNet data from `http://www.image-net.org/`.

- `dataset` -- Dataset of choice, only supports `ImageNet32` (default) and `ImageNet64`.
- `dataloc` -- Location of the dataset files as unpacked from download.
- `preload` -- *(boolean)* Whether to preprocess the whole dataset or just load the files. Preprocessing the dataset (default) takes considerably longer to load and uses more memory, but is much faster at training time and very efficient if many epocs are trained.
- `data_ratio` -- *(float in [0,1])* Fraction of the dataset to load, 1.0 (default) loads the whole dataset.
- `reclass` -- Options to modify the original 1000 classes. 3 comma-separated string formats are available:
    - `n` : (default) indicates no modification (original classes).
    - `m,`*x* : indicates to take classes modulo *x*, e.g. `m,10` as done in the paper.
    - `f,`*x*`,`*filename* : indicates to consider *x* categories and apply a superclass list found in file *filename* (expected as a comma-separated list of 1000 superclass identifiers). To reproduce the paper use `f,10,classes.dat` with the supplied file.

##[Network]
All the options to define the network and training parameters. Besides fully connected and cross-entropy layers, a limited form of convolutional layers is available.

- `arch` -- The architecture of the network as a colon-separated string of layers. The network input shape is taken from the data, each layer fully connected layer is just defined by its size, and the final cross entropy layer spans the total number of categories. Simple convolutional and max-pool layers are also available. As an example, the string `f1024:f1024:f1024:x` produces the main network considered in the paper. The notation is the following:
    - `f`*x* -- Adds a fully connected layer of size *x*, with batch normalization and ReLU activation. The previous layer will be flattened if not 1 dimensional.
    - `x` -- Adds a final cross-entropy layer of the size of the number of categories.
    - `c`*x*`,`*y*`,`*z* -- Adds a convolutional layer with features of size (*x*,*x*), in *y* channels, with strides (*z*,*z*) (optional), `VALID` padding and ReLU nonlinearity.
    - `p`*x*`,`*y* -- Adds a max-pool layer with pools of size (*x*,*x*) and strides (*y*,*y*).
- `steps` -- *(int)* Total number of training steps.
- `batch_size` -- *(int)* Number of examples per mini-batch.
- `val_size` -- *(int)* Number of examples to use for validation (constant, loaded at initialization).
- `validate_step` -- *(int)* Number of steps between validations.
- `minimizer` -- Type of minimizer. Possible options: stochastic gradient descent `SGD` (default) and `Adam`.
- `learning_rate` -- *(float)* Learning rate for the minimizer.
- `init_sparsity` -- *(float in [0,1])* Optionally, sparsity (random) of the initial mask of the network (default 0 indicate a dense starting network).
- `batch_norm_beta` -- *(float in [0,1])* Batch normalization uses an exponential moving average over the batches examples for evaluation of the validation set during training. This is the parameter controlling the average decay (defaults to 0.99).
- `restart` -- *(boolean)* Whether to restart from a previously saved step. In this cases some of the following are required.
- `rest_it` -- *(int)* Number of iteration to consider as the first after restart (for naming consistency).
- `rest_weights_file` -- Name (or complete path) of the file to load the weights from (typically step 0 of the previous run, defaults to *save_dir*`/IMP_itdata_0.pkl`).
- `rest_mask_file` -- Name (or complete path) of the file to compute the starting mask from (typically step *rest_it*-1 of the previous run, defaults to *save_dir*`/IMP_itdata_`*(rest_it-1)*`.pkl`)
- `rest_file_step` -- *(int)* Which checkpoint of the loaded iteration to use to compute the masl (typically the last one, as the default -1, but might be -2 if training was run for longer).

##[IMP]
All options about the iterative magnitude pruning (IMP) procedure: what to prune, by how much and which iterations to consider. The training stops when either stop criterion is met.

- `prune` -- Colon-separated string of `T` (true) or `F` (false) indicating which layers to prune. If missing or shorter than `arch`, all (remaining) layers are taken as True (convolutional layers can also be pruned, max-pool layers cannot be pruned, but require an entry, which will be ignored). For example `T:T:T:F` reproduces the pruning used in the paper.
- `prune_ratio` -- *(float in [0,1])* Fraction of remaining weights to prune at each iteration (the default `0.3` prunes 30% of weights per iteration as used in the paper).
- `weights_step` -- *(int)* Step at which to save weights for rewinding in the first iteration.
- `prune_step` -- *(int)* Optionally, step to consider at each iteration for construction of the mask (defaults to the last training step).
- `stop_ratio` -- *(float in [0,1])* Possible stop criterion: stop when at least one layer has less than this fraction of nodes completely disconnected (default 0.0 stops when a layer is completely pruned).
- `max_iterations` -- *(int)* Possible stop criterion: stop after this number of iterations (default 1000 should not be relevant in general).

