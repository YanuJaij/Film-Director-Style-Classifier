# Film Director Style Classifier

An image classifier that tries to identify a film director's visual style from a single movie frame. The six directors were chosen specifically because they each have a strong and distinct visual identity, making them interesting candidates for a vision-based classifier.

Frames were sourced from [film-grab.com](https://film-grab.com) and the model was trained on stills that best represented each director's cinematography, colour palette and compositional style. The app is a simple Flask web interface where you upload a frame and it returns a confidence score for each director.

## Directors

| Director | Films trained on |
|----------|------------------|
| Wes Anderson | *The Grand Budapest Hotel*, *Moonrise Kingdom*, *The Royal Tenenbaums*, *Rushmore*, *Fantastic Mr. Fox* |
| David Fincher | *Se7en*, *Fight Club*, *Zodiac*, *Gone Girl*, *The Social Network* |
| Stanley Kubrick | *The Shining*, *2001: A Space Odyssey*, *A Clockwork Orange*, *Full Metal Jacket*, *Barry Lyndon* |
| Denis Villeneuve | *Blade Runner 2049*, *Arrival*, *Sicario*, *Dune*, *Prisoners* |
| Tim Burton | *Edward Scissorhands*, *Beetlejuice*, *Sleepy Hollow*, *Batman (1989)*, *Big Fish* |
| Guillermo del Toro | *Pan's Labyrinth*, *The Shape of Water*, *Crimson Peak*, *The Devil's Backbone*, *Hellboy* |

## Model

Built on ResNet-18 with a dropout layer (0.4) added before the final classifier, fine-tuned on the frames dataset. Images are resized to **224×224** and normalised before being passed to the model.

Training ran for **30 epochs**, and the best model was saved at **epoch 20** with a validation accuracy of **71.8%**. Training accuracy climbed close to 100% by the end, which points to some overfitting given the relatively small dataset size.

## Tech

- Python + PyTorch for model training and inference
- ResNet-18 (`torchvision`) as the base architecture
- Flask for the web app
- Pillow (PIL) for image preprocessing

## Project Structure

```text
Film-Director-Style-Classifier/
├── app.py                           # Flask app and prediction logic
├── train_director_classifier.py     # Model training script
├── training_curves.png              # Loss and accuracy curves
└── README.md
```

The model weights (`best_model.pth`) are not included due to file size. To reproduce them, download the dataset and run:

```bash
python train_director_classifier.py
```

The dataset is not included in the repository either. You can download it here: [Google Drive folder](https://drive.google.com/drive/folders/1X3wFMbJvjeWVrIjeBBTCRjfEVTXcK9mL?usp=drive_link)

## Setup

```bash
git clone https://github.com/YanuJaij/Film-Director-Style-Classifier.git
cd Film-Director-Style-Classifier
pip install flask torch torchvision pillow
```

Add `best_model.pth` to the project root, then run:

```bash
python app.py
```

Open <http://127.0.0.1:5000> in your browser.

## Results

**Best validation accuracy:** **71.8%** (epoch 20/30)

This is a reasonable result given the nature of the task. A director's visual style is not consistent across their entire filmography since cinematography choices, colour grading, and framing vary between films and collaborators. The model picks up on strong stylistic signals like Wes Anderson's symmetrical compositions and pastel palettes, or Kubrick's one-point perspective shots, but naturally struggles when a frame does not lean heavily into those signatures.

## Limitations

The training accuracy reached nearly 100% while validation accuracy plateaued around 71%, suggesting the model overfitted to the specific frames in the training set rather than fully generalising. A larger and more varied dataset would likely improve performance.

It is also worth noting that visual style in film is not solely determined by the director. The cinematographer, production designer, lighting, costume design, and colour grading all contribute significantly, so the model is effectively learning a combination of these visual influences rather than the director's style alone.
