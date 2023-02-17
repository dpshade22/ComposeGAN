# Midi Music Composition with a Genetic Algorithm GAN

This project uses a Generative Adversarial Network (GAN) to create new MIDI music compositions. The GAN is trained on a dataset of MIDI files, and then used to generate new compositions that resemble the ones in the training set. The generated compositions are then fed through a genetic algorithm to further refine them and make them more human-like.

## Dependencies
- mido
- pandas
- numpy
- scikit-learn
- keras
- tensorflow
- smtplib
- email

## How to Use
To use this project, follow these steps:



Structure
The project has the following structure:

```markdown
- assets/
  - csvs/
    - fakeCSVs/
    - realCSVs/
  - fakeMusic/
  - realMusic/
- models/
- src/
  - GAN/
    - __init__.py
    - discriminator.py
    - generator.py
  - genetic_algorithm/
    - __init__.py
    - crossover.py
    - mutation.py
    - selection.py
  - __init__.py
  - main.py
  - midi_processing.py
  - model_training.py
  - model_utils.py
- app/
  - __init__.py
  - app.py
- README.md
```
- `assets/` folder contains the MIDI files and CSV files used for training the GAN.
- `models/` folder contains the trained GAN model.
- `src/` folder contains the main source code.
  - `GAN/` folder contains the code for the GAN model.
  - `genetic_algorithm/` folder contains the code for the genetic algorithm.
  - `main.py` is the main file to train the GAN and generate new compositions.
  - `midi_processing.py` contains functions for processing MIDI files.
  - `model_training.py` contains functions for training the GAN.
  - `model_utils.py` contains utility functions for loading and saving models.
- `app/` folder contains the Streamlit app code.
- `README.md` contains the project documentation.