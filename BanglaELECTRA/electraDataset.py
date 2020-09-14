import logging
import os
import pickle
import random
import linecache
import json

import torch
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset

from tqdm.auto import tqdm


logger = logging.getLogger(__name__)



class LazyElectraDatasetV2(Dataset):
    def __init__(self, tokenizer, args, file_path, mode, block_size=512):
        """Dataset abstraction compliant with simpletransformers `dataset_class` with lazy loading.

        Args:
            tokenizer ([type]): [description]
            args ([type]): [description]
            file_path (str): Input file path.
            mode (str): "train"/"dev"
            block_size (int, optional): Optional target sequence length. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.current_sentences = []
        self.current_length = 0
        self.block_size = block_size
        self.example_count = 0
        self.cached_features_file = ""

        assert os.path.isfile(file_path)
        if hasattr(args, 'max_seq_length'):
            self.block_size = args.max_seq_length
        
        _, filename = os.path.split(file_path)
        self.cached_features_file = os.path.join(args.cache_dir, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename)

        if (
                os.path.exists(self.cached_features_file) and 
                (
                    (not args.reprocess_input_data and not args.no_cache) or
                    (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
                )
            ):
            logger.info(" Lazy Loading features from cached file %s", self.cached_features_file)
            self.example_count = self.get_example_count(self.cached_features_file)
            

        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            outfp = open(self.cached_features_file, 'w')
            with open(file_path, errors='surrogateescape') as f:
                for line in tqdm(f):
                    line = line.strip()
                    example = self.add_line(line)
                    if example:
                        print(example, file=outfp)

                example = self.add_line("")
                if example:
                    print(example, file=outfp)
            
            logger.info(" Saved features into cached file %s", self.cached_features_file)
            outfp.close()
            self.example_count = self.get_example_count(self.cached_features_file)
            
            

    @staticmethod
    def get_example_count(data_file):
        "Returns no of examples present in the cache file"
        with open(data_file, encoding="utf-8", errors='surrogateescape') as f:
            for line_idx, _ in enumerate(f, 1):
                pass

        return line_idx

    
    def __len__(self):
        return self.example_count

    def __getitem__(self, idx):
        line = linecache.getline(self.cached_features_file, idx + 1).rstrip()
        example = json.loads(line)
        input_ids, input_mask, segment_ids = example[:]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(input_mask, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long)
        )

    def add_line(self, line):
        """Adds a line of text to the current example being built."""
        line = line.strip().replace("\n", " ")
        if (not line) and self.current_length != 0:  # empty lines separate docs
            return self.create_example()
        tokens = self.tokenizer.tokenize(line)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if not self.current_sentences:
            self.current_sentences.append(token_ids)
            self.current_length += len(token_ids)
            if self.current_length >= self.block_size:
                return self.create_example()
        else:
            if self.current_length + len(token_ids) >= self.block_size:
                return self.create_example(token_ids)

            self.current_sentences.append(token_ids)
            self.current_length += len(token_ids)

        return None

    def create_example(self, new_sentence_tokens=None):
        """Creates a pre-training example from the current list of sentences."""
        probability = random.random()
        # small chance to only have one segment as in classification tasks
        if probability < 0.1:
            first_segment_target_length = 100000
            second_segment_target_length = -1
        elif 0.1 <= probability < 0.2:
            # small chance to have 3 segments like RACE
            first_segment_target_length = (self.block_size - 4) // 3
            second_segment_target_length = first_segment_target_length
        else:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (self.block_size - 3) // 2
            second_segment_target_length = 100000

        first_segment = []
        second_segment = []
        third_segment = []
        for sentence in self.current_sentences:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (
                    len(first_segment) == 0 or
                    len(first_segment) + len(sentence) < first_segment_target_length or
                    (   
                        len(second_segment) == 0 and
                        len(first_segment) < first_segment_target_length and
                        random.random() < 0.5
                    )
                ):
                first_segment += sentence
            else:
                # the sentence goes to the second segment if (1) the second segment is
                # empty, (2) the sentence doesn't put the second segment over length or
                # (3) 50% of the time when it does put the second segment over length
                if (
                        len(second_segment) == 0 or
                        len(second_segment) + len(sentence) < second_segment_target_length or
                        (   
                            len(third_segment) == 0 and
                            len(second_segment) < second_segment_target_length and
                            random.random() < 0.5
                        )
                    ):
                    second_segment += sentence
                else:
                    third_segment += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[:self.block_size - 2]
        second_segment = second_segment[:max(0, self.block_size - len(first_segment) - 3)]
        third_segment = third_segment[:max(0, self.block_size - len(first_segment) - len(second_segment) - 4)]

        
        # combine three segments into a example
        sep_token = self.tokenizer.sep_token_id
        cls_token = self.tokenizer.cls_token_id
        pad_token = self.tokenizer.pad_token_id

        input_ids = [cls_token] + first_segment + [sep_token]
        segment_ids = [0] * len(input_ids)
        
        if second_segment:
            input_ids += second_segment + [sep_token]
            segment_ids += [1] * (len(second_segment) + 1)
        if third_segment:
            input_ids += third_segment + [sep_token]
            segment_ids += [2] * (len(third_segment) + 1)
        

        input_mask = [1] * len(input_ids)
        input_ids += [pad_token] * (self.block_size - len(input_ids))
        input_mask += [0] * (self.block_size - len(input_mask))
        # segment id for masked tokens wouldnt matter
        segment_ids += [0] * (self.block_size - len(segment_ids))
        
        assert len(input_ids) == len(input_mask) == len(segment_ids)
        # prepare to start building the next example
        self.current_sentences = []
        self.current_length = 0

        if new_sentence_tokens:
            self.current_sentences.append(new_sentence_tokens)
            self.current_length += len(new_sentence_tokens) 

        return [input_ids, input_mask, segment_ids]
    


class LazyElectraDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, mode, block_size=512):
        """Dataset abstraction compliant with simpletransformers `dataset_class` with lazy loading.

        Args:
            tokenizer ([type]): [description]
            args ([type]): [description]
            file_path (str): Input file path.
            mode (str): "train"/"dev"
            block_size (int, optional): Optional target sequence length. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.current_sentences = []
        self.current_length = 0
        self.block_size = block_size
        self.example_count = 0
        self.cached_features_file = ""

        assert os.path.isfile(file_path)
        if hasattr(args, 'max_seq_length'):
            self.block_size = args.max_seq_length
        
        _, filename = os.path.split(file_path)
        self.cached_features_file = os.path.join(args.cache_dir, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename)

        if (
                os.path.exists(self.cached_features_file) and 
                (
                    (not args.reprocess_input_data and not args.no_cache) or
                    (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
                )
            ):
            logger.info(" Lazy Loading features from cached file %s", self.cached_features_file)
            self.example_count = self.get_example_count(self.cached_features_file)
            

        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            outfp = open(self.cached_features_file, 'w')
            with open(file_path, errors='surrogateescape') as f:
                for line in tqdm(f):
                    line = line.strip()
                    example = self.add_line(line)
                    if example:
                        print(example, file=outfp)

                example = self.add_line("")
                if example:
                    print(example, file=outfp)
            
            logger.info(" Saved features into cached file %s", self.cached_features_file)
            outfp.close()
            self.example_count = self.get_example_count(self.cached_features_file)
            
            

    @staticmethod
    def get_example_count(data_file):
        "Returns no of examples present in the cache file"
        with open(data_file, encoding="utf-8", errors='surrogateescape') as f:
            for line_idx, _ in enumerate(f, 1):
                pass

        return line_idx

    
    def __len__(self):
        return self.example_count

    def __getitem__(self, idx):
        line = linecache.getline(self.cached_features_file, idx + 1).rstrip()
        example = json.loads(line)
        input_ids, input_mask, segment_ids = example[:]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(input_mask, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long)
        )

    def add_line(self, line):
        """Adds a line of text to the current example being built."""
        line = line.strip().replace("\n", " ")
        if (not line) and self.current_length != 0:  # empty lines separate docs
            return self.create_example()
        tokens = self.tokenizer.tokenize(line)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if not self.current_sentences:
            self.current_sentences.append(token_ids)
            self.current_length += len(token_ids)
            if self.current_length >= self.block_size:
                return self.create_example()
        else:
            if self.current_length + len(token_ids) >= self.block_size:
                return self.create_example(token_ids)

            self.current_sentences.append(token_ids)
            self.current_length += len(token_ids)

        return None

    def create_example(self, new_sentence_tokens=None):
        """Creates a pre-training example from the current list of sentences."""
        # small chance to only have one segment as in classification tasks
        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (self.block_size - 3) // 2

        first_segment = []
        second_segment = []
        for sentence in self.current_sentences:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (
                    len(first_segment) == 0 or
                    len(first_segment) + len(sentence) < first_segment_target_length or
                    (   
                        len(second_segment) == 0 and
                        len(first_segment) < first_segment_target_length and
                        random.random() < 0.5
                    )
                ):
                first_segment += sentence
            else:
                second_segment += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[:self.block_size - 2]
        second_segment = second_segment[:max(0, self.block_size - len(first_segment) - 3)]

        
        # combine two segments into a example
        sep_token = self.tokenizer.sep_token_id
        cls_token = self.tokenizer.cls_token_id
        pad_token = self.tokenizer.pad_token_id

        input_ids = [cls_token] + first_segment + [sep_token]
        segment_ids = [0] * len(input_ids)
        
        if second_segment:
            input_ids += second_segment + [sep_token]
            segment_ids += [1] * (len(second_segment) + 1)

        input_mask = [1] * len(input_ids)
        input_ids += [pad_token] * (self.block_size - len(input_ids))
        input_mask += [0] * (self.block_size - len(input_mask))
        # segment id for masked tokens wouldnt matter
        segment_ids += [0] * (self.block_size - len(segment_ids))
        
        assert len(input_ids) == len(input_mask) == len(segment_ids)
        # prepare to start building the next example
        self.current_sentences = []
        self.current_length = 0

        if new_sentence_tokens:
            self.current_sentences.append(new_sentence_tokens)
            self.current_length += len(new_sentence_tokens) 

        return [input_ids, input_mask, segment_ids]
    



class ElectraDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, mode, block_size=512):
        """Dataset abstraction compliant with simpletransformers `dataset_class`. Doesnt support lazy loading.

        Args:
            tokenizer ([type]): [description]
            args ([type]): [description]
            file_path (str): Input file path.
            mode (str): "train"/"dev"
            block_size (int, optional): Optional target sequence length. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.current_sentences = []
        self.current_length = 0
        self.block_size = block_size
        self.examples = []

        assert os.path.isfile(file_path)
        if hasattr(args, 'max_seq_length'):
            self.block_size = args.max_seq_length
        
        _, filename = os.path.split(file_path)
        cached_features_file = os.path.join(args.cache_dir, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename)

        if (
                os.path.exists(cached_features_file) and 
                (
                    (not args.reprocess_input_data and not args.no_cache) or
                    (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
                )
            ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            with open(file_path, errors='surrogateescape') as f:
                for line in tqdm(f):
                    line = line.strip()
                    example = self.add_line(line)
                    if example:
                        self.examples.append(example)

                example = self.add_line("")
                if example:
                    self.examples.append(example)
            
            logger.info(" Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.examples[idx][0], dtype=torch.long),
            torch.tensor(self.examples[idx][1], dtype=torch.long),
            torch.tensor(self.examples[idx][2], dtype=torch.long)
        )

    def add_line(self, line):
        """Adds a line of text to the current example being built."""
        line = line.strip().replace("\n", " ")
        if (not line) and self.current_length != 0:  # empty lines separate docs
            return self.create_example()
        tokens = self.tokenizer.tokenize(line)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if not self.current_sentences:
            self.current_sentences.append(token_ids)
            self.current_length += len(token_ids)
            if self.current_length >= self.block_size:
                return self.create_example()
        else:
            if self.current_length + len(token_ids) >= self.block_size:
                return self.create_example(token_ids)

            self.current_sentences.append(token_ids)
            self.current_length += len(token_ids)

        return None

    def create_example(self, new_sentence_tokens=None):
        """Creates a pre-training example from the current list of sentences."""
        # small chance to only have one segment as in classification tasks
        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (self.block_size - 3) // 2

        first_segment = []
        second_segment = []
        for sentence in self.current_sentences:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (
                    len(first_segment) == 0 or
                    len(first_segment) + len(sentence) < first_segment_target_length or
                    (   
                        len(second_segment) == 0 and
                        len(first_segment) < first_segment_target_length and
                        random.random() < 0.5
                    )
                ):
                first_segment += sentence
            else:
                second_segment += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[:self.block_size - 2]
        second_segment = second_segment[:max(0, self.block_size - len(first_segment) - 3)]

        
        # combine two segments into a example
        sep_token = self.tokenizer.sep_token_id
        cls_token = self.tokenizer.cls_token_id
        pad_token = self.tokenizer.pad_token_id

        input_ids = [cls_token] + first_segment + [sep_token]
        segment_ids = [0] * len(input_ids)
        
        if second_segment:
            input_ids += second_segment + [sep_token]
            segment_ids += [1] * (len(second_segment) + 1)

        input_mask = [1] * len(input_ids)
        input_ids += [pad_token] * (self.block_size - len(input_ids))
        input_mask += [0] * (self.block_size - len(input_mask))
        # segment id for masked tokens wouldnt matter
        segment_ids += [0] * (self.block_size - len(segment_ids))
        
        assert len(input_ids) == len(input_mask) == len(segment_ids)
        # prepare to start building the next example
        self.current_sentences = []
        self.current_length = 0

        if new_sentence_tokens:
            self.current_sentences.append(new_sentence_tokens)
            self.current_length += len(new_sentence_tokens) 

        return [input_ids, input_mask, segment_ids]
    
    




if __name__ == "__main__":
    pass