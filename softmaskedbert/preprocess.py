# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

'''
Ernie preprocess script.
'''

import os
import argparse
import src.dataset.get_dataset as get_dataset

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="soft-masked bert preprocess")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Eval batch size, default is 2")
    parser.add_argument("--eval_data_file_path", type=str, default="./dataset/dev_processed.json",
                        help="Data path, it is better to use absolute path")
    parser.add_argument('--result_path', type=str, default='./preprocess_result/', help='result path')
    args_opt = parser.parse_args()

    # if args_opt.eval_data_file_path == "":
    #     raise ValueError("'eval_data_file_path' must be set when do evaluation task")
    return args_opt


if __name__ == "__main__":
    args = parse_args()
    ds = get_dataset(fp=args.eval_data_file_path, max_seq_len=512)
    print(ds.dataset_size)
    wrong_ids_path = os.path.join(args.result_path, "00_data")
    original_tokens_path = os.path.join(args.result_path, "01_data")
    original_tokens_mask_path = os.path.join(args.result_path, "02_data")
    correct_tokens_path = os.path.join(args.result_path, "03_data")
    correct_tokens_mask_path = os.path.join(args.result_path, "04_data")
    original_token_type_ids_path = os.path.join(args.result_path, "05_data")
    correct_token_type_ids_path = os.path.join(args.result_path, "06_data")
    os.makedirs(wrong_ids_path)
    os.makedirs(original_tokens_path)
    os.makedirs(original_tokens_mask_path)
    os.makedirs(correct_tokens_path)
    os.makedirs(correct_tokens_mask_path)
    os.makedirs(original_token_type_ids_path)
    os.makedirs(correct_token_type_ids_path)

    for idx, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        wrong_ids = data["wrong_ids"]
        # print(wrong_ids.shape)
        original_tokens = data["original_tokens"]
        # print(original_tokens.shape)
        original_tokens_mask = data["original_tokens_mask"]
        # print(original_tokens_mask.shape)
        correct_tokens = data["correct_tokens"]
        # print(correct_tokens.shape)
        correct_tokens_mask = data["correct_tokens_mask"]
        # print(correct_tokens_mask.shape)
        original_token_type_ids = data["original_token_type_ids"]
        # print(original_token_type_ids.shape)
        correct_token_type_ids = data["correct_token_type_ids"]
        # print(correct_token_type_ids.shape)

        file_name = "batch_" + str(args.eval_batch_size) + "_" + str(idx) + ".bin"
        wrong_ids_file_path = os.path.join(wrong_ids_path, file_name)
        wrong_ids.tofile(wrong_ids_file_path)
        original_tokens_file_path = os.path.join(original_tokens_path, file_name)
        original_tokens.tofile(original_tokens_file_path)
        original_tokens_mask_file_path = os.path.join(original_tokens_mask_path, file_name)
        original_tokens_mask.tofile(original_tokens_mask_file_path)
        correct_tokens_file_path = os.path.join(correct_tokens_path, file_name)
        correct_tokens.tofile(correct_tokens_file_path)
        correct_tokens_mask_file_path = os.path.join(correct_tokens_mask_path, file_name)
        correct_tokens_mask.tofile(correct_tokens_mask_file_path)
        original_token_type_ids_file_path = os.path.join(original_token_type_ids_path, file_name)
        original_token_type_ids.tofile(original_token_type_ids_file_path)
        correct_token_type_ids_file_path = os.path.join(correct_token_type_ids_path, file_name)
        correct_token_type_ids.tofile(correct_token_type_ids_file_path)
    print("=" * 20, "export bin files finished", "=" * 20)
