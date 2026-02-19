# Παρουσίαση Game Learning

15 διαφάνειες για Fictitious Play & Reinforcement Learning (αντλώντας από το report).

## Περιεχόμενα

- `slide_1.html` … `slide_15.html` — Οι διαφάνειες
- `speaker_notes.md` — Τι να πεις σε κάθε slide

## Παραγωγή HTML & PDF

```bash
# Compose → project_presentation.html εδώ στο game-learning/
python presentation/compose_slides.py -d presentation/game-learning

# HTML → PDF (απαιτεί playwright, img2pdf)
python presentation/html_to_pdf.py
```

Με Docker:

```bash
docker-compose run --rm game-learning python presentation/compose_slides.py -d presentation/game-learning
docker-compose run --rm game-learning python presentation/html_to_pdf.py
```

Το `project_presentation.html` και το `project_presentation.pdf` δημιουργούνται στο φάκελο `game-learning/`.
