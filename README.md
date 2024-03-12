# GAN Project

This project uses Generative Adversarial Networks (GANs) for text generation. GANs are a type of artificial intelligence model that consists of two parts: a generator that creates new data instances, and a discriminator that attempts to distinguish between real and fake data.

## Scripts

### enc-full.py

This script encodes the files in the `/lib/world` directory into a format that can be used by PyTorch. Each file is treated as a single text.

### enc-split.py

This script also encodes the files in the `/lib/world` directory, but it treats each session within a file as a separate text.

### testgen.py

This script is used to test the current state of the generator.

### gan.py

This script trains the GAN using the encoded texts. It requires a `fake.pt` file, which should contain fake texts for training. The script has several advantages:

- It allows you to specify the number of epochs, the number of samples, and the noise size through command-line arguments.
- It uses PyTorch's DataSets and DataLoader to reduce memory usage.
- It saves the training statistics at the end of each epoch.
- It can save the model locally or on the cloud (Requires HuggingFace Token).

## Usage

1) Edit you prefered encoder script to the path of your `lib/world/` directory. `enc-session.py` is the recomended version since it generates smaller texts.
2) Place your `fake.pt` in the same directory.
3) Edit `gan.py` to change the `types` for the script.
4) Run `python gan.py` to run the training script. Use `--help` to see the avaiable arguments.
5) At the end anytime of the training script, use `python testgen.py` for testing the generator.

### Note

To use cloud saving edit your token on `gan.py`

## License

Because this project uses files from the TBAmud (CircleMud) area, the TBAmud license is included in the repository.

Please note that this project is for educational and/or research purposes and should not be used for commercial purposes without the appropriate permissions.

