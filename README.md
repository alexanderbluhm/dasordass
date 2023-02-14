# dasordass

AI NLP model to correct the usage of "das" and "dass" in German sentences.  
The model architecture consists of a BERT model, pretrained exclusively on German language ([HugginfaceModel](https://huggingface.co/bert-base-german-cased)), and a simple classifier layer on top.
Docker image generated using [cog](https://github.com/replicate/cog) and deployed to [fly.io](https://fly.io/) for inference (CPU, 2GB RAM).


## Training
Trained for 10min on an RTX 2070 Super on 20k sentences from the `alexanderbluhm/wiki_sentences_de_2k` dataset.
The dataset consists of sentences containing "das" or "dass" from the first 2,000 [de wikipedia](https://huggingface.co/datasets/wikipedia) documents, split with [spaCy](https://spacy.io/).

## Deployment

Adapted from: https://til.simonwillison.net/fly/fly-docker-registry

- Create an empty fly.io application using `flyctl launch`
- Build image using cog and tag it like `registry.fly.io/your-app-name:unique-image-tag`: `cog build -t registry.fly.io/your-app-name:unique-tag-for-your-image`
- Run: `flyctl auth docker`
- Push to the registry: `docker push registry.fly.io/your-app-name:unique-image-tag`
- Deploy: `flyctl deploy --image registry.fly.io/your-app-name:unique-image-tag`
- Adjust RAM amount in case of a memory error (2GB required for this model)