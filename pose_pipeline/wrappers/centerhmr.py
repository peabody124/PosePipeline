import os
import sys
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torchvision


def centerhmr_parse_video(video_path, centerhmr_python_path, output_video_path=None):

    old_args = sys.argv

    # CenterHMR uses a pretty awkward interface
    sys.argv = [
        "centerhmr_parse_video.py",
        "--gpu=0",
        "--gmodel-path=" + os.path.join(centerhmr_python_path, "trained_models/pw3d_81.8_58.6.pkl"),
        "--configs_yml=" + os.path.join(centerhmr_python_path, "src/configs/basic_test_video.yml"),
    ]

    from base import Base, Visualizer

    class Parser(Base):
        def __init__(self):
            super(Parser, self).__init__()
            self.set_up_smplx()
            self._build_model()
            self.generator.eval()
            self.vis_size = [1024, 1024, 3]  # [1920,1080]
            self.visualizer = Visualizer(
                model_type=self.model_type, resolution=self.vis_size, input_size=self.input_size, with_renderer=True
            )

        def single_image_forward(self, image):
            image_size = image.shape[:2][::-1]
            image_org = Image.fromarray(image)

            resized_image_size = (float(self.input_size) / max(image_size) * np.array(image_size) // 2 * 2).astype(int)[
                ::-1
            ]
            padding = tuple((self.input_size - resized_image_size)[::-1] // 2)
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize([*resized_image_size], interpolation=3),
                    torchvision.transforms.Pad(padding, fill=0, padding_mode="constant"),
                ]
            )
            image = torch.from_numpy(np.array(transform(image_org))).unsqueeze(0).contiguous().float()
            if "-1" not in self.gpu:
                image = image.cuda()
            outputs, centermaps, heatmap_AEs, _, reorganize_idx = self.net_forward(
                None, self.generator, image, mode="test"
            )
            outputs.update({"input_image": image, "reorganize_idx": reorganize_idx})
            return outputs

        def process_video(self, video_file_path=None, output_file_name=None):

            cap = cv2.VideoCapture(video_file_path)
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            results = []
            writer = None

            for frame_id in tqdm(range(video_length)):

                ret, frame = cap.read()
                if frame is None:
                    break

                with torch.no_grad():
                    outputs = self.single_image_forward(frame[:, :, ::-1])

                # move data structures to CPU and append it to the results output array
                def cpu(x):
                    if isinstance(x, dict):
                        return {k: cpu(v) for k, v in x.items()}
                    elif isinstance(x, torch.Tensor):
                        return x.cpu().numpy()
                    else:
                        return x

                outputs_cpu = {k: cpu(v) for k, v in outputs.items()}
                outputs_cpu.pop("input_image")
                results.append(outputs_cpu)

                if "verts" in outputs.keys() and output_file_name is not None:
                    vis_dict = {"image_org": outputs["input_image"].cpu()}
                    vis_eval_results = self.visualizer.visulize_result_onorg(
                        outputs["verts"], outputs["verts_camed"], vis_dict, reorganize_idx=outputs["reorganize_idx"]
                    )
                    result_frame = vis_eval_results[0]

                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                        writer = cv2.VideoWriter(
                            output_file_name, fourcc, fps, (result_frame.shape[1], result_frame.shape[0])
                        )

                # writer.write(result_frame)

            if writer is not None:
                writer.release()

            return results

    centerhmr_parser = Parser()
    results = centerhmr_parser.process_video(video_path, output_video_path)

    sys.argv = old_args

    return results
