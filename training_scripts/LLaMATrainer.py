import torch, gc
import logging
from tqdm import tqdm
from trl import SFTTrainer
from transformers.trainer_utils import EvalPrediction
from unsloth import FastLanguageModel, is_bfloat16_supported
import os, json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLaMATrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def save_output(self, output_file, outputs):
        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=4)
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        max_seq_length = self.max_seq_length[0] if type(self.max_seq_length) == type(()) else self.max_seq_length
        logger.info("Running evaluation loop...")
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference

        # Decoder-only models must be LEFT-padded for batched generation. The eval
        # collator otherwise right-pads (this is the "right-padding was detected"
        # warning), which makes generate() continue from pad positions and emit
        # garbage that no longer matches the expected `[Event(...)]` output format.
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        outputs = []
        all_decoder_ids = []
        picked_dataset = self.gt_val_data if metric_key_prefix == "eval" else self.gt_test_data
        current_epoch = int(self.state.epoch) if self.state.epoch else "final"
        output_dir = self.args.output_dir
        output_file = os.path.join(output_dir, f"{self.dataset_name}_{metric_key_prefix}_results_epoch_{current_epoch}.json")
        os.makedirs(output_dir, exist_ok=True)

        #########
        eval_loader = self.get_eval_dataloader()

        batch_size = self.args.per_device_eval_batch_size
        count = 0
        with torch.no_grad():
            for batch_idx, eval_set in tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Hold on, :D, doing {metric_key_prefix} ..."):
                # Prepare batched input_ids
                input_ids = eval_set["input_ids"].to(self.model.device)
                # Pass the attention mask so padded positions are ignored during generation.
                attention_mask = eval_set["attention_mask"].to(self.model.device)
                # Generate outputs for the entire batch
                decoder_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=250, pad_token_id = self.tokenizer.pad_token_id)
                # Decode all outputs in the batch
                decoded_output = self.tokenizer.batch_decode(decoder_ids, skip_special_tokens=True)
                # Decode the input sentences in the batch
                input_sentences = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                # Collect ground truths and outputs
                for i in range(len(eval_set["input_ids"])):
                    count += 1
                    dataset_idx = batch_idx * batch_size + i  # Correct index calculation
                    ground_truth = picked_dataset[dataset_idx]["conversations"][-1]["gt"]
                    input_sentence = input_sentences[i]
                    prediction = decoded_output[i]    
                    # Append the results
                    outputs.append({"Input": input_sentence, "Prediction": prediction, "Label": ground_truth})
                    # Print the results
                    # Append to outputs
                    # outputs.append({"Input": input_sentence, "Prediction": prediction, "Label": ground_truth})
                # Save output after processing the batch
                self.save_output(output_file, outputs)
                # Manage CUDA memory
                torch.cuda.empty_cache()
                gc.collect()

        #########
        logger.info("Custom evaluation complete")
        logger.info(f"Saved evaluation results to {output_file}")
        eval_pred = EvalPrediction(predictions=outputs, label_ids=all_decoder_ids)
        output_metrics = self.compute_metrics(outputs)#### fix me
        output_file_numbers = os.path.join(output_dir, f"{self.dataset_name}_val_numbers_epoch_{current_epoch}.json")
        self.save_output(output_file_numbers, output_metrics)
        for key in list(output_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output_metrics[f"{metric_key_prefix}_{key}"] = output_metrics.pop(key)
        self.log(output_metrics)
        del eval_pred
        del outputs
        del all_decoder_ids
        torch.cuda.empty_cache()
        gc.collect()
        # Restore the original padding side so training (loss computed over
        # right-padded batches) is unaffected by the left-padding used for generation.
        self.tokenizer.padding_side = original_padding_side
        FastLanguageModel.for_training(self.model)
        return output_metrics