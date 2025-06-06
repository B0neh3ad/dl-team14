# Copyright 2024 Daniel Franzen and Jan Disselhoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import numpy as np
from numpy.random import randint
from tqdm import tqdm
from collections import OrderedDict


class ArcDataset(object):
    def __init__(self, challenge, solutions={}, keys=None, is_fake=False, is_orig=False):
        if keys is None:
            self.keys = []
            for k, v in challenge.items():
                reply_num = len(v['test'])
                self.keys.extend([f'{k}_{i}' for i in range(reply_num)] if reply_num else [k])
            self.keys = sorted(self.keys)
        else:
            self.keys = [k for k in keys]
        base_keys = set(map(self.get_base_key, self.keys))
        self.challenge = {k: challenge[k] for k in base_keys}
        self.solutions = {k: solutions[k] for k in base_keys if k in solutions}
        self.is_orig = is_fake
        self.is_orig = is_orig

    @classmethod
    def load_from_json_dl(cls, path, size=6, seed=42, shuffle=True, eval=False):  # loader for arc
        np.random.seed(seed)
        keys = [[] for _ in range(1000 // (1 + size) if not eval else 1)]
        challenge = {}
        solutions = {}

        for fname in tqdm(sorted(os.listdir(path)), desc="load arc dataset"):
            if not fname.endswith(".json"):
                continue

            key = fname[:-5]
            with open(os.path.join(path, f'{key}.json')) as f:
                tasks = np.random.permutation(json.load(f)).tolist()

            n = len(tasks) // (1 + size) if not eval else 1
            for epoch in range(n):
                next_size_with_test = 1 + size
                base_key = f'arc-{key}{epoch:02x}'
                keys[epoch].append(f'{base_key}_0')
                challenge[base_key] = {'train': [], 'test': []}
                solutions[base_key] = reply = []
                for _ in range(next_size_with_test):
                    if not len(tasks):
                        raise RuntimeError('Not enough examples - generate more re-arc examples or reduce epochs.')
                    challenge[base_key]['train'].append({k: v for k, v in tasks.pop().items()})
                challenge[base_key]['test'].append(challenge[base_key]['train'].pop())
                solutions[base_key].append(challenge[base_key]['test'][-1].pop('output'))

        if shuffle:
            keys = [np.random.permutation(epoch) for epoch in keys]
        keys = [k for epoch in keys for k in epoch]
        print("** Dataset prepared **")
        print("Number of tasks:", len(keys))
        return cls(keys=keys, challenge=challenge, solutions=solutions, is_orig=True)

    def change_keys(self, keys):
        return self.__class__(challenge=self.challenge, solutions=self.solutions, keys=keys)

    def split(self, n, split_seed, **kwargs):
        assert self.is_orig, 'Must be run on original dataset.'
        keys = sorted(self.challenge.keys())
        if split_seed == 'len':
            keys = self.sort_keys_by_len(keys=keys, **kwargs)
        else:
            assert isinstance(split_seed, int)
            assert not kwargs
            np.random.seed(split_seed)
            keys = np.random.permutation(keys)
        split_datasets = []
        for new_keys in np.array_split(keys, n):
            new_challenge = {k: self.challenge[k] for k in new_keys}
            split_datasets.append(self.__class__(challenge=new_challenge, solutions=self.solutions, is_orig=True))
        return split_datasets

    @staticmethod
    def get_base_key_and_reply_num(key):
        key_num = key.split('.', 1)[0]
        base_key, reply_num = key_num.split('_') if '_' in key_num else (key_num, -1)
        return base_key, int(reply_num)

    @classmethod
    def get_base_key(cls, key):
        return cls.get_base_key_and_reply_num(key)[0]

    def grouped_keys(self):
        grouped_keys = OrderedDict()
        for key in self.keys:
            base_key, reply_num = self.get_base_key_and_reply_num(key)
            if base_key not in grouped_keys:
                grouped_keys[base_key] = []
            while len(grouped_keys[base_key])<=reply_num:
                grouped_keys[base_key].append([])
            grouped_keys[base_key][reply_num].append(key)
        return grouped_keys

    def move_test_to_train(self):
        assert self.is_orig, 'Must be run on original dataset.'
        new_challenge = {}
        for k, v in self.challenge.items():
            new_challenge[k] = {
                'train': v['train'] + [{**t, 'output': self.solutions[k][i]} for i, t in enumerate(v['test'])],
                'test': []
            }
        return self.__class__(challenge=new_challenge, is_orig=self.is_orig)

    @staticmethod
    def permute_array(a, descriptor, invert=False):
        permutation = [int(i) for i in descriptor if str(i).isdigit()]
        assert sorted(permutation) == list(range(10))
        a = np.asarray(a)
        assert a.ndim == 2
        if invert: permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
        return a

    @classmethod
    def transform_array(cls, array, transforms, apply_perm=True, invert=False):
        if array is None: return None
        array = np.asarray(array)
        if invert: transforms = transforms[::-1]
        for tf in transforms:
            if tf == 'tp':
                array = np.swapaxes(array, 0, 1)
            if tf == 'rt':
                array = np.rot90(np.rot90(np.rot90(array)) if invert else array)
            if apply_perm and tf.startswith('perm'):
                array = cls.permute_array(array, tf, invert=invert)
        return array

    @classmethod
    def fmt_array(cls, array, lines_sep, tf=None):
        if tf is not None:
            array = cls.transform_array(array, tf)
        return lines_sep.join(''.join(map(str, row)) for row in array)

    @classmethod
    def fmt_input(cls, array, query_beg, reply_beg, **kwargs):
        return query_beg + cls.fmt_array(array, **kwargs) + reply_beg

    @classmethod
    def fmt_output(cls, array, reply_end, **kwargs):
        return cls.fmt_array(array, **kwargs) + reply_end

    @classmethod
    def fmt_train(cls, train_ex, preprompt, query_beg, reply_beg, reply_end, **kwargs):
        examples = [cls.fmt_input(x['input'], query_beg, reply_beg, **kwargs) +
                    cls.fmt_output(x['output'], reply_end, **kwargs) for x in train_ex]
        return preprompt + ''.join(examples)

    def fmt_task(self, key, preprompt, query_beg, reply_beg, reply_end, reply=True, **kwargs):
        key_num, *tf = key.split('.')
        base_key, reply_num = self.get_base_key_and_reply_num(key_num)
        data_train = self.challenge[base_key]['train']
        data_query = self.challenge[base_key]['test']
        if reply is True:
            reply = self.solutions[base_key][reply_num] if base_key in self.solutions and reply_num >= 0 else None
        elif reply is not None:
            assert reply_num >= 0
        for t in tf:
            if t.startswith('ex'):
                data_train = [data_train[int(i)] for i in t[2:].split('-')]
        ret = dict(key=key)
        ret['train'] = self.fmt_train(data_train, preprompt, query_beg, reply_beg, reply_end, tf=tf, **kwargs)
        ret['query'] = self.fmt_input(data_query[reply_num]['input'], query_beg, reply_beg, tf=tf, **kwargs) if reply_num >= 0 else ''
        ret['input'] = ret['train'] + ret['query'] if reply_num >= 0 else ''
        if reply is not None:
            ret['reply'] = self.fmt_output(reply, reply_end, tf=tf, **kwargs)
        ret['text'] = ret['train'] + (ret['query'] + ret['reply'] if reply is not None else '')
        return ret

    def get_task(self, key, max_tokens=None, len_name=None, **kwargs):
        while True:
            fmt = self.fmt_task(key, **kwargs)
            if max_tokens is None or self.count_tokens(fmt[len_name]) <= max_tokens:
                break
            if not key.split('.')[-1].startswith('ex'):
                base_key = self.get_base_key(key)
                key = f"{key}.ex{'-'.join(map(str, range(len(self.challenge[base_key]['train']))))}"
            key_split = key.split('.')
            key_split[-1] = '-'.join(key_split[-1].split('-')[:-1])
            assert len(key_split[-1]) > 2 and key_split[-1].startswith('ex')
            key = '.'.join(key_split)
        return key, fmt

    def repeat(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        new_keys = []
        for i in range(n):
            new_keys.extend(self.keys if seed is None else np.random.permutation(self.keys))
        return self.change_keys(new_keys)

    @staticmethod
    def count_tokens(data, replace_special=re.compile('<[^<]*>')):
        replaced = replace_special.sub('x', data)  # replace '<...>' by a single char to count special tokens only once
        return len(replaced)

    @classmethod
    def max_new_tokens(cls, reply_end, lines_sep, max_size=30, safety_margin=1, **_):
        max_sized_reply = np.zeros([max_size, max_size], dtype=int)
        fmt = cls.fmt_output(max_sized_reply, reply_end=reply_end, lines_sep=lines_sep)
        return cls.count_tokens(fmt) + safety_margin

    def get_length(self, key, len_name, max_of_transposed=False, max_tokens=None, **fmt_opts):
        if not fmt_opts:
            fmt_opts = dict(preprompt='', query_beg='', reply_beg='', reply_end='', lines_sep='')
            length = self.count_tokens(self.fmt_task(key, **fmt_opts)[len_name])
        else:
            length = self.count_tokens(self.fmt_task(key, **fmt_opts)[len_name])
            if max_of_transposed:
                length = max(length, self.count_tokens(self.fmt_task(f'{key}.tp', fmt_opts)[len_name]))
            length += 1  # for bos token
        return length

    def sort_keys_by_len(self, keys, reverse=False, **kwargs):
        lengths = [(key, self.get_length(key, **kwargs)) for key in keys]
        return [x[0] for x in sorted(lengths, reverse=reverse, key=lambda x: x[1])]

    def sorted_by_len(self,**kwargs):
        return self.change_keys(self.sort_keys_by_len(self.keys, **kwargs))

    def convert_with_token_limit(self, **kwargs):
        out_list = []
        new_keys = []
        for key in tqdm(self.keys, desc='convert dataset'):
            key, fmt = self.get_task(key, **kwargs)
            new_keys.append(key)
            out_list.append(fmt)
        print("formatted dataset size:", len(out_list))
        print("max tokens:", max(map(lambda x: self.count_tokens(x['text']), out_list)))
        print("max size:", max(map(lambda x: self.get_length(x['key'], **kwargs), out_list)))
        return out_list, self.change_keys(new_keys)

    def as_list(self, **kwargs):
        return self.convert_with_token_limit(**kwargs)[0]

    @staticmethod
    def rand_perm(n, sep=None, keep_zero=False):
        permutation = np.random.permutation(n).tolist()
        if keep_zero:
            permutation = [0] + [x for x in permutation if x != 0]
        return permutation if sep is None else sep.join(map(str, permutation))

    def augment_keys(self, keys, tp=False, rt=False, n=1, perm=False, keep_background=False, shfl_ex=False):
        keys = [k + n * '.tp' for n in range(2) for k in keys] if tp == 'all' else keys
        keys = [k + n * '.rt' for n in range(4) for k in keys] if rt == 'all' else keys
        keys = [k + bool(tp) * randint(0, 2) * '.tp' for k in keys] if tp != 'all' else keys
        keys = [k + bool(rt) * randint(0, 4) * '.rt' for k in keys] if rt != 'all' else keys
        keys = keys * n  # repeat n times
        keys = [k + bool(perm) * ('.perm' + self.rand_perm(10, '', keep_background)) for k in keys]
        n_ex = lambda k: len(self.challenge[self.get_base_key(k)]['train'])
        keys = [k + bool(shfl_ex) * ('.ex' + self.rand_perm(n_ex(k), '-')) for k in keys]
        return keys

    def augment(self, seed, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return self.change_keys([k for key in self.keys for k in self.augment_keys([key], **kwargs)])

    def decode(self, text, lines_sep, key=None):
        correct, info = None, 'unknown'
        try:
            data = [[int(x) for x in row if x.isdigit()] for row in text.split(lines_sep)]
            data = [row for row in data if len(row)]
            data = np.array(data, dtype=int)
            assert data.ndim == 2 and all(0 < x <= 30 for x in data.shape)
        except:
            data = None
            correct, info = False, 'cant_decode'
        if key is not None and data is not None:
            key_num, *transforms = key.split('.')
            base_key, reply_num = self.get_base_key_and_reply_num(key_num)
            data = self.transform_array(data, transforms, invert=True)
            correct_solution = self.solutions.get(base_key)
            if correct_solution is None:
                info = 'sol_unknown'
            else:
                correct_solution = np.asarray(correct_solution[reply_num])
                if np.array_equal(correct_solution, data):
                    correct, info = True, 'ALL_CORRECT'
                else:
                    correct, info = False, ('bad_content' if correct_solution.shape == data.shape else 'bad_xy_size')
        return data, correct, info

