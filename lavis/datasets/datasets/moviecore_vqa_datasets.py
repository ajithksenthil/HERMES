"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset


class MovieCoreVQADataset(VideoQADataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,
        ann_paths,
        num_frames,
        trail_percentage,
        prompt="",
        split="train",
    ):
        self.vis_root = vis_root

        self.annotation = {}
        for ann_path in ann_paths:
            self.annotation.update(json.load(open(ann_path)))
        self.question_id_list = list(self.annotation.keys())
        self.question_id_list.sort()
        self.fps = 10

        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt

        # Percentage of frames to ignore at the beginning and end of the video
        self.trail_percentage = trail_percentage

    def __getitem__(self, index):
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]

        # Percentage of frames to ignore at the beginning and end of the video
        trailing_frames = int(ann["frame_length"] * self.trail_percentage)

        # Divide the range into num_frames segments and select a random index from each segment
        segment_list = np.linspace(
            trailing_frames,
            ann["frame_length"] - trailing_frames,
            self.num_frames + 1,
            dtype=int,
        )
        segment_start_list = segment_list[:-1]
        segment_end_list = segment_list[1:]
        selected_frame_index = []
        for start, end in zip(segment_start_list, segment_end_list):
            if start == end:
                selected_frame_index.append(start)
            else:
                selected_frame_index.append(np.random.randint(start, end))

        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(
                os.path.join(
                    self.vis_root,
                    ann["video"],
                    "frame{:06d}.jpg".format(frame_index + 1),
                )
            ).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)

        question = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        answer = self.text_processor(ann["answer"])

        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": ann["question_id"],
        }

    def __len__(self):
        return len(self.question_id_list)


class MovieCoreVQAEvalDataset(MovieCoreVQADataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,
        ann_paths,
        num_frames,
        trail_percentage,
        prompt,
        split="test",
    ):
        super().__init__(
            vis_processor,
            text_processor,
            vis_root,
            ann_paths,
            num_frames,
            trail_percentage,
            prompt,
            split="test",
        )

    def __getitem__(self, index):
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]

        # Percentage of frames to ignore at the beginning and end of the video
        trailing_frames = int(ann["frame_length"] * self.trail_percentage)

        selected_frame_index = (
            np.rint(
                np.linspace(
                    trailing_frames,
                    ann["frame_length"] - trailing_frames,
                    self.num_frames,
                )
            )
            .astype(int)
            .tolist()
        )
        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(
                os.path.join(
                    self.vis_root,
                    ann["video"],
                    "frame{:06d}.jpg".format(frame_index + 1),
                )
            ).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)

        question = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        answer = self.text_processor(ann["answer"])

        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": ann["question_id"],
        }
