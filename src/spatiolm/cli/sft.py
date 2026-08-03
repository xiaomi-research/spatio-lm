from swift.llm import LazyLLMDataset
from swift.llm.train.sft import SwiftSft
from swift.utils import get_logger, get_model_parameter_info
from transformers import AutoModelForDepthEstimation

from torch.utils.data import ConcatDataset
from spatiolm.swift import Seq2Seq3DTrainer, Training3DArguments, load_torch_dataset

logger = get_logger()


class SwiftSft3D(SwiftSft):
    args_class = Training3DArguments
    args: args_class

    def run(self):
        args = self.args

        train_dataset, val_dataset = self._prepare_dataset()

        if args.task_type == "seq_cls":
            args.problem_type = args.problem_type or getattr(
                self.model.config, "problem_type", None
            )
            logger.info(f"args.problem_type: {args.problem_type}")
        args.save_args()

        data_collator = self._get_data_collator()
        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(
            self.args, self.model, template=self.template, train_dataset=train_dataset
        )
        logger.info(f"model: {self.model}")
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg["model_parameter_info"] = model_parameter_info
        logger.info(f"model_parameter_info: {model_parameter_info}")

        # trainer_cls = TrainerFactory.get_trainer_cls(args)
        # trainer = trainer_cls(
        trainer = Seq2Seq3DTrainer(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)

    def _get_trainer_kwargs(self):
        old_kwargs = super()._get_trainer_kwargs()
        args = self.args

        teacher3d = None
        if hasattr(args, "teacher3d"):
            teacher3d = AutoModelForDepthEstimation.from_pretrained(args.teacher3d).to(
                self.model.device
            )

        return {"teacher3d": teacher3d, **old_kwargs}

    def _prepare_dataset(self):
        train_dataset = self._prepare_torch_dataset()
        val_dataset = None

        if len(self.args.dataset) > 0:
            hf_train_dataset, val_dataset = super()._prepare_dataset()
            if train_dataset is not None:
                train_dataset = ConcatDataset([train_dataset, hf_train_dataset])
            else:
                train_dataset = hf_train_dataset

        return train_dataset, val_dataset

    def _prepare_torch_dataset(self):
        torch_dataset, hf_dataset = [], []
        for dat in self.args.dataset:
            if dat.startswith("torch::"):
                torch_dataset.append(dat)
            else:
                hf_dataset.append(dat)

        if len(torch_dataset) == 0:
            return None

        # Update original dataset, remove torch.dataset
        self.args.dataset = hf_dataset

        dataset = load_torch_dataset(torch_dataset)
        return LazyLLMDataset(
            dataset,
            self.template.encode,
            strict=self.args.strict,
            random_state=self.args.data_seed,
        )


if __name__ == "__main__":
    SwiftSft3D().main()
