# Arabic Spell Checker: A Semantic-Aware Hybrid Approach

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2402.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2402.XXXXX)

A highly customizable Arabic spell checker that combines rule-based approaches with neural networks to provide superior contextual understanding and dialect adaptation.

## Features

- **Hybrid Architecture**: Combines rule-based filtering with neural verification for optimal accuracy
- **Context-Aware Corrections**: Uses WordNet and VerbNet for semantic disambiguation
- **Dialect Support**: Works with Modern Standard Arabic (MSA), Egyptian, Gulf, Levantine, and Maghrebi dialects
- **Domain Adaptation**: Easily adaptable to specialized domains like medical and legal texts
- **High Accuracy**: Achieves 93.4% F1 score across domains, outperforming existing systems
- **Efficient Deployment**: Optimized inference with quantization options for production use

## Performance Highlights

| System | Character Error Rate (%) | Context F1 | Dialect F1 |
|--------|--------------------------|------------|------------|
| AraSpell | 1.24 | 0.912 | 0.723 |
| **Ours** | **1.86** | **0.934** | **0.847** |

## Installation

```bash
# Clone the repository
git clone https://github.com/arabic-spellcheck/asc2024.git
cd asc2024

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Download required resources
python scripts/download_resources.py
```

## Quick Start

```python
from arabic_spellcheck import ArabicSpellChecker

# Initialize the spell checker
spell_checker = ArabicSpellChecker()

# Basic spell checking
text = "هاذا النص يحتوي علي بعض الأخطاء الإملائيه."
corrected_text = spell_checker.correct(text)
print(corrected_text)
# Output: "هذا النص يحتوي على بعض الأخطاء الإملائية."

# Dialect-specific spell checking
egyptian_text = "انا عايز اروح السينما بكره"
msa_text = spell_checker.correct(egyptian_text, dialect="egyptian", target="msa")
print(msa_text)
# Output: "أنا أريد أن أذهب إلى السينما غدا"

# Domain-specific spell checking (e.g., medical)
medical_text = "المريض يعاني من ألم في القفص الصدري"
corrected_medical = spell_checker.correct(medical_text, domain="medical")
print(corrected_medical)
```

## Advanced Usage

### Working with Dialects

```python
# Configure dialect handling
spell_checker = ArabicSpellChecker(
    dialects=["egyptian", "gulf"],
    default_dialect="egyptian"
)

# Correct with dialect detection
text = "انا بدي اروح ع البيت هلق"  # Levantine Arabic
corrected = spell_checker.correct(text, detect_dialect=True)
```

### Domain Adaptation

```python
# Fine-tune for a specific domain
spell_checker.adapt_domain(
    domain_name="legal",
    data_path="data/legal_samples.txt",
    method="qlora",  # Uses QLoRA for efficient adaptation
    num_epochs=3
)

# Use the adapted model
legal_text = "وفقًا للماده الخامسه من القنون المدني"
corrected = spell_checker.correct(legal_text, domain="legal")
```

### Neural Verification Configuration

```python
# Configure the neural verification component
spell_checker = ArabicSpellChecker(
    confidence_threshold=0.9,  # Adjust confidence threshold (default: 0.85)
    use_wordnet=True,          # Enable WordNet for semantic disambiguation
    use_verbnet=True,          # Enable VerbNet for verb form resolution
    gnn_layers=3               # Number of GNN layers
)
```

## System Architecture

The system employs a three-stage processing pipeline:

1. **Rule-Based Filtering**: 45 morphological rules handle systematic errors using finite-state transducers.
2. **Neural Verification**: A DeBERTa-GNN ensemble resolves homographs with confidence thresholding.
3. **Active Refinement**: Human validation for low-confidence cases to reduce false positives.

![Architecture Diagram](docs/images/architecture.png)

## Evaluation Results

### Cross-Dialect Performance (F1 Scores)

| Dialect | F1 | 95% CI |
|---------|-----|-------|
| Egyptian | 0.924 | [0.915, 0.933] |
| Levantine | 0.901 | [0.891, 0.911] |
| Gulf | 0.889 | [0.878, 0.900] |

### Domain-Specific Performance (F1)

| Domain | Ours | AraSpell |
|--------|------|----------|
| Medical | 0.894 | 0.812 |
| Legal | 0.901 | 0.834 |

## Web Demo

A web demo is available at [https://arabic-spellcheck.github.io/demo](https://arabic-spellcheck.github.io/demo).

Or run it locally:

```bash
cd web_demo
pip install -r requirements.txt
python app.py
```

The demo will be available at http://localhost:5000.

## Training Your Own Models

```bash
# Basic training
python scripts/train.py --model hybrid --batch_size 64 --lr 3e-5 \
    --data_dir ./data/qalb-extended --gnn_layers 3

# Advanced training with curriculum
python scripts/train.py --model hybrid --batch_size 64 --curriculum \
    --phase1_steps 500000 --phase1_lr 3e-5 \
    --phase2_steps 300000 --phase2_lr 1e-5 \
    --data_dir ./data/qalb-extended --gnn_layers 3 \
    --optimizer lion --beta1 0.95 --beta2 0.98
```

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@inproceedings{ibrahim2024accurate,
  title={An Accurate and Highly Customizable Arabic Spell Checker: A Semantic-Aware Hybrid Approach with Domain Adaptation},
  author={Ibrahim, Mossab and Gervás, Pablo and Méndez, Gonzalo},
  booktitle={Proceedings of the International Conference on Computational Creativity (ICCC)},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- QALB-2015 Extended Corpus
- Arabic WordNet 3.0
- NADI 2023 Dataset
- The authors would like to thank the annotators who participated in the evaluation process.

## Contact

- Mossab Ibrahim - [mibrahim@ucm.es](mailto:mibrahim@ucm.es)
- Pablo Gervás - [pgervas@ucm.es](mailto:pgervas@ucm.es)
- Gonzalo Méndez - [gmendez@ucm.es](mailto:gmendez@ucm.es)

For questions about the code or the paper, please open an issue in this repository.
