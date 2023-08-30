import os
import datasets

_CITATION = """
"""

_DESCRIPTION = """\
BabyLM data
"""
_HOMEPAGE = "https://babylm.github.io/"
_LICENSE = "????"
_DATA_URL = "./babylm_data"


class babyLMConfig(datasets.BuilderConfig):
    """BuilderConfig for babyLM."""

    def __init__(self, data_url, **kwargs):
        """BuilderConfig for babyLM
        Args:
          data_url: `string`, url to the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(
            version=datasets.Version(
                "1.0.0",
            ),
            **kwargs,
        )
        self.data_url = data_url


class babyLM(datasets.GeneratorBasedBuilder):
    """TODO: Short description of dataset dataset."""
    DATA_SOURCES = [
            'aochildes', 'bnc_spoken', 'cbt', 'children_stories',
            'gutenberg', 'open_subtitles', 'qed',  'simple_wikipedia',
            'switchboard',  'wikipedia']
    VERSION = datasets.Version("0.0.0")
    BUILDER_CONFIGS = [
            babyLMConfig(
                name="babyLM-10M",
                data_url=os.path.join(_DATA_URL, 'babylm_10M'),
                description="Raw level dataset: the raw tokens before the addition of <unk> tokens. 10M tokens.",
            ),
            babyLMConfig(
                name="babyLM-100M",
                data_url=os.path.join(_DATA_URL, 'babylm_100M'),
                description="Raw level dataset: the raw tokens before the addition of <unk> tokens. 100M tokens.",
            ),
            ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "text": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        ret_list = [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_folder": os.path.join(_DATA_URL, "babylm_test"), "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_folder": os.path.join(_DATA_URL, "babylm_dev"), "split": "dev"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_folder": self.config.data_url, "split": "train"},
            ),
        ]
        return ret_list

    def _generate_examples(self, data_folder, split):
        """Yields examples."""
        all_data_files = [
                os.path.join(data_folder, f'{source}.{split}')
                for source in self.DATA_SOURCES]
        all_lines = []
        for data_file in all_data_files:
            with open(data_file, encoding="utf-8") as f:
                all_lines.extend(f.readlines())
        for idx, row in enumerate(all_lines):
            if row.strip():
                yield idx, {"text": row}
            else:
                yield idx, {"text": ""}