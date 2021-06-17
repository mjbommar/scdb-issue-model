import json
import tarfile

# Packages
from typing import Iterable

import gensim
import spacy
import tika
import tika.parser


class TarTokenExtractor(Iterable):
    """
    Iterable tar file token extraction for gensim input
    """

    def __init__(self, tar_file_name: str, spacy_model: str = "en_core_web_sm", return_member=False,
                 sample_size: int = None, progress_step: int = 1000):
        """
        Constructor with tar file name
        :param tar_file_name:
        """
        self.tar_file_name = tar_file_name
        self.nlp = nlp = spacy.load(spacy_model)
        self.return_member = return_member
        self.sample_size = sample_size
        self.progress_step = progress_step

    def __iter__(self):
        n = 0
        with tarfile.open(self.tar_file_name, 'r:gz') as scotus_tar:
            if self.sample_size is None:
                sample = scotus_tar.getmembers()
            else:
                sample = scotus_tar.getmembers()[0:self.sample_size]

            for member in sample:
                if n % self.progress_step == 0:
                    print(f"completed: {n}/{self.sample_size}")
                n += 1

                # set member ID
                member_id = int(member.name.replace(".json", ""))

                # read the file and parse to JSON
                try:
                    # get the contents and convert HTML to plain text
                    file_data = json.loads(scotus_tar.extractfile(member).read())
                except Exception as e:
                    print(f"unable to parse {member.name} as JSON", e)

                try:
                    tika_response = tika.parser.from_buffer(f"<html>{file_data['html']}</html>")
                    file_content = tika_response['content']
                except Exception as e:
                    print(f"unable to extract {member.name} text with tika", e)

                # process text and get tokens
                try:
                    file_doc = self.nlp(file_content)
                    file_tokens = [token.lemma_.lower() for token in file_doc if
                                   not token.is_stop and not token.is_space and not token.is_punct]
                except Exception as e:
                    # print(f"unable to parse file {member.name}")
                    continue

                if self.return_member:
                    yield file_tokens, member.name
                else:
                    yield file_tokens
